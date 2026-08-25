"""E3 — competition at fixed distance: does confusable context cost accuracy on its own?

Needle-in-a-haystack setups confound two variables. Burying the answer makes it both *far from
the query* and *surrounded by plausible alternatives*, and no published design separates them.
E1 isolated the first (evidence displaced k turns back, fill fixed, question byte-identical) and
E1c showed the cost is causally mediated by the evidence's attention mass. E3 isolates the second:
the evidence stays **local** (distance 0), fill and turn count are held fixed, and the only thing
that moves is how confusable the accumulated context is with the current case.

Arms (see ``tasks/e3_competition_brief.md`` for why these are DDXPlus and not the MMLU arms the
parent brief named -- measured, the MMLU instrument does not separate near_dup from same_subject):

    disjoint  | context cases sharing 0 of the probe's 5 candidate pathologies
    random    | context cases sampled uniformly (the natural stream, ~0.75 shared)
    near_dup  | context cases sharing >=3, with a *different* gold (~3.65 shared)

All three arms are DDXPlus cases, so format and in-context-learning affordance are constant and
only confusability moves. Keeping the parent brief's MMLU ``unrelated`` arm would have confounded
competition with ICL -- a medical-case context teaches the task, MMLU filler does not, and this
paper's thesis is that ICL is what holds accuracy up.

The design is **paired**: every probe is asked once per arm against its own three contexts, so the
bootstrap resamples probes and each arm sees the identical item set. E1_MECHANISM.md's power note
is the reason -- independent arms at n=192 give a CI half-width of ~0.12, which could not resolve
the effects this program produces.

Context cases are answered in the transcript with their **gold** letter, so context answer quality
is not a between-arm variable.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import (
    RowAppender,
    extract_mcq_answer,
    generate_with_entropy,
    per_head_rows,
    render_prompt,
)

from src.probes.context_fatigue.attention_capture import (
    SelectiveAttentionCapture,
    mean_attention_row,
)
from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_phrase_spans,
    locate_token_span,
    locate_turn_spans,
    measure_span_share,
    select_hot_token_spans,
    span_share,
)
from src.probes.context_fatigue.context_assembly import (
    ArmSpec,
    OverflowGuard,
    assemble_transcript,
    select_by_option_overlap,
)
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    load_evidence_db,
    load_probe_pool,
)

INTRO = ("You are reviewing a stream of patient cases. Each one is self-contained. "
         "Answer each with the single most likely diagnosis.")
REFERENT = "For the patient described earlier"  # identical to E1, so `local` is comparable
ACK = "Noted."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--n-probes", type=int, default=384)
    p.add_argument("--n-context", type=int, default=8,
                   help="context cases per arm; 8 puts fill at ~0.69, matching E1")
    p.add_argument("--min-overlap", type=int, default=3,
                   help="shared options required of a near_dup context case")
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e3_competition")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-layer", type=int, default=24)
    p.add_argument("--per-head", action="store_true",
                   help="also write heads.csv: one row per probe x arm x head x layer. Implies "
                        "--attention-only, since the per-head shares come off the same forward.")
    p.add_argument("--head-layers", type=int, nargs="+", default=None,
                   help="layers to record per-head shares at (default: every layer). Extra layers "
                        "are free -- they are read off the same forward as the reference layer.")
    p.add_argument("--attention-only", action="store_true",
                   help="skip generation; measure the evidence span's attention share only. "
                        "Answers whether competition drains the evidence's mass (folding it into "
                        "the same account as distance) or is an independent channel.")
    p.add_argument("--close-arms", action="store_true",
                   help="E3c: paired near_dup arms closing (scale 0) every context occurrence of "
                        "the probe's option names, against a size-matched random-closure control "
                        "and the natural near_dup and random arms. Requires eager attention. "
                        "Tests whether competition's cost is carried by *reading* the competitor "
                        "instances at generation time (brief: tasks/e3c_competitor_close_brief.md).")
    p.add_argument("--measured-close", type=float, default=None, metavar="K",
                   help="E3c': add a closure arm built from the *measured* hot tokens — the "
                        "context-body tokens the final position actually reads most, ranked by "
                        "the all-layer-mean attention row — with token budget K x the verbatim "
                        "closure's per-probe token count, plus its own size-matched random "
                        "control. Implies --close-arms; the verbatim arms stay in as the "
                        "within-session anchor. Attacks the closure residual: rescue beyond the "
                        "verbatim arm's means the residual was instrument slack, not prefill "
                        "interference (brief: tasks/per_token_capture_brief.md).")
    p.add_argument("--store-rows", action="store_true",
                   help="store each probe's final-position attention row (all-layer/head mean, "
                        "float16) plus span metadata under rows/ in the out dir — the "
                        "per-token capture program's Stage-0 artifact.")
    p.add_argument("--preflight", action="store_true",
                   help="run two probes end-to-end, write them, and exit")
    return p.parse_args()


def size_matched_control_spans(spans, lo, hi, rng, avoid):
    """Random spans with the same count and widths as ``spans``, inside ``[lo, hi)``.

    Each drawn span must overlap neither ``avoid`` nor previously drawn controls — the
    size-matched random-closure control both E3c and E3c' compare their closure arms against.
    """
    taken = list(avoid)
    controls = []
    for a, b in spans:
        width = b - a
        for _ in range(200):
            s = rng.randrange(lo, hi - width)
            if all(s + width <= x or s >= y for x, y in taken):
                controls.append((s, s + width))
                taken.append((s, s + width))
                break
    return sorted(controls)


def build_context_turns(cases, n_options):
    """Prior turns: each context case as a user turn, answered with its gold letter."""
    turns = [{"role": "user", "content": INTRO}, {"role": "assistant", "content": "Understood."}]
    for case in cases:
        turns.append({"role": "user", "content": case["vignette"]
                      + format_case_question(case["options"], n_options)})
        turns.append({"role": "assistant", "content": case["gold"]})
    return turns


def main():
    args = parse_args()
    if args.per_head:
        args.attention_only = True
    if args.measured_close is not None:
        args.close_arms = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = out_dir / "rows"
    if args.store_rows:
        rows_dir.mkdir(exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        # closure biases the additive mask, which sdpa optimizes away on a causal-only prompt
        **({"attn_implementation": "eager"} if args.close_arms else {})).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    evidence_db = load_evidence_db()
    # Identical pool, seed and construction as E1, so the `disjoint`/`random` arms are directly
    # comparable to E1's `local` (0.464). A control arm that misses it means the harness drifted.
    pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  probe pool {len(pool)} DDXPlus cases", flush=True)

    n_probes = 2 if args.preflight else args.n_probes
    rng = random.Random(args.seed)
    probes = rng.sample(pool, min(n_probes, len(pool)))

    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=args.headroom)
    records, leaks, starved = [], 0, 0
    turns_path = out_dir / "turns.csv"
    capture_layers = sorted({args.reference_layer, *(args.head_layers if args.head_layers
                                                     else range(len(model.model.layers)))})
    heads = RowAppender(out_dir / "heads.csv") if args.per_head else None

    for idx, probe in enumerate(probes):
        question = format_case_question(probe["options"], args.n_options, referent=REFERENT)
        probe_options = set(probe["options"])

        # Select every arm's context *before* generating any of them: if one arm is starved the
        # probe is dropped from all three, so the paired item set stays identical across arms.
        contexts = {}
        try:
            for arm in ArmSpec.overlap_arms():
                contexts[arm] = select_by_option_overlap(
                    pool, probe, arm=arm, n=args.n_context, seed=args.seed + idx,
                    min_overlap=args.min_overlap)
        except ValueError:
            starved += 1
            continue

        built_arms, skip_probe = {}, False
        for arm, cases in contexts.items():
            leaks += sum(1 for c in cases if c["pathology"] == probe["pathology"])
            prior = build_context_turns(cases, args.n_options)
            built = assemble_transcript(prior, evidence=probe["vignette"],
                                        question=question, distance=0, ack=ACK)
            rendered = render_prompt(tokenizer, built.turns, is_chat)
            # Charge the whole transcript against the window: a truncated near-full item loses its
            # own options and scores as a spurious error, manufacturing the effect under study.
            if not guard.fits(rendered, used=0, index=idx):
                skip_probe = True
                break
            built_arms[arm] = (rendered, cases)
        if skip_probe:
            continue

        if args.close_arms:
            rendered_nd, _ = built_arms["near_dup"]
            rendered_rand, _ = built_arms["random"]
            vig_start = rendered_nd.rindex(probe["vignette"])
            comp_spans = locate_phrase_spans(
                tokenizer, rendered_nd, probe["options"][:args.n_options],
                region=(0, vig_start))
            if not comp_spans:
                starved += 1
                continue
            ids = tokenizer(rendered_nd, return_tensors="pt").input_ids.to(args.device)
            evid_start = locate_token_span(tokenizer, rendered_nd, probe["vignette"])[0]
            intro_end = locate_token_span(tokenizer, rendered_nd, "Understood.")[1]
            comp_tokens = sum(b - a for a, b in comp_spans)
            hot_cols = {}
            if args.measured_close is not None or args.store_rows:
                # One capture forward serves the E3b rider (competitor union share), the
                # measured hot-token ranking, and the stored row.
                capture = SelectiveAttentionCapture(model,
                                                    list(range(len(model.model.layers))))
                capture.enabled = True
                with torch.no_grad():
                    model(ids)
                capture.remove()
                comp_share = float(sum(
                    sum(span_share(capture.captured[li], s) for s in comp_spans)
                    for li in capture.captured) / len(capture.captured))
                row = mean_attention_row(capture.captured)
                del capture
            else:
                # E3b rider: the competitor spans' union share before closure, all-layer mean,
                # from one capture forward on the transcript the closure arms generate from.
                comp_share = measure_span_share(model, ids, comp_spans,
                                                list(range(len(model.model.layers))))
                row = None
            if args.store_rows:
                np.savez_compressed(
                    rows_dir / f"probe_{idx}.npz",
                    row=row.numpy().astype(np.float16),
                    input_ids=ids[0].cpu().numpy().astype(np.int32),
                    meta=json.dumps({"probe": idx, "arm": "near_dup",
                                     "comp_spans": comp_spans, "evid_start": evid_start,
                                     "intro_end": intro_end,
                                     "pathology": probe["pathology"]}))
            # Size-matched random-closure control: same span count and sizes, sampled in the same
            # context region, overlapping neither the competitor spans nor each other.
            rng_probe = random.Random(args.seed * 1000 + idx)
            rand_spans = size_matched_control_spans(comp_spans, intro_end, evid_start,
                                                    rng_probe, avoid=comp_spans)
            close_arms = [("near_dup", rendered_nd, None),
                          ("near_dup_comp_close", rendered_nd, comp_spans),
                          ("near_dup_rand_close", rendered_nd, rand_spans),
                          ("random", rendered_rand, None)]
            if args.measured_close is not None:
                # E3c': close what the final position measurably reads in the context body,
                # at K x the verbatim closure's token budget, ranked by received mass. Two
                # variants: `hot` takes the row as measured (preflight showed it is dominated
                # by chat-template glue and turn boundaries — the E6/E7 precedent channel),
                # `hotc` restricts candidates to the context cases' own content (vignettes,
                # questions, demonstrated answer letters), so it can only close
                # competitor-side reading.
                budget = max(1, int(round(args.measured_close * comp_tokens)))
                hot_spans = select_hot_token_spans(row, budget,
                                                   region=(intro_end, evid_start))
                _, cases_nd = built_arms["near_dup"]
                prior_turns = build_context_turns(cases_nd, args.n_options)
                content_spans = locate_turn_spans(
                    tokenizer, rendered_nd, [t["content"] for t in prior_turns])
                glue, cursor = [], intro_end
                for a, b in content_spans[2:]:  # the case turns; INTRO/ack stay excluded
                    if a > cursor:
                        glue.append((cursor, a))
                    cursor = max(cursor, b)
                if cursor < evid_start:
                    glue.append((cursor, evid_start))
                hotc_spans = select_hot_token_spans(row, budget,
                                                    region=(intro_end, evid_start),
                                                    exclude=glue)
                hot_rand_spans = size_matched_control_spans(
                    hot_spans, intro_end, evid_start, rng_probe, avoid=hot_spans)
                hotc_rand_spans = size_matched_control_spans(
                    hotc_spans, intro_end, evid_start, rng_probe, avoid=hotc_spans)

                def span_mass(spans):
                    return float(sum(float(row[a:b].sum()) for a, b in spans))
                overlap = sum(max(0, min(b, y) - max(a, x))
                              for a, b in hot_spans for x, y in comp_spans)
                hot_cols = {"hot_tokens": sum(b - a for a, b in hot_spans),
                            "hotc_tokens": sum(b - a for a, b in hotc_spans),
                            "hot_comp_overlap_tokens": overlap,
                            "hot_mass": span_mass(hot_spans),
                            "hotc_mass": span_mass(hotc_spans),
                            "comp_mass": span_mass(comp_spans)}
                close_arms += [("near_dup_hot_close", rendered_nd, hot_spans),
                               ("near_dup_hot_rand_close", rendered_nd, hot_rand_spans),
                               ("near_dup_hotc_close", rendered_nd, hotc_spans),
                               ("near_dup_hotc_rand_close", rendered_nd, hotc_rand_spans)]
            for arm, rendered, spans in close_arms:
                if spans:
                    with SpanAttentionClamp(model, span=spans, scale=0.0):
                        resp, ctx_len, entropy, _ = generate_with_entropy(
                            model, tokenizer, rendered, args.device, args.max_new, args.max_ctx)
                else:
                    resp, ctx_len, entropy, _ = generate_with_entropy(
                        model, tokenizer, rendered, args.device, args.max_new, args.max_ctx)
                pred = extract_mcq_answer(resp) if resp else None
                records.append({
                    "probe": idx, "arm": arm, **hot_cols,
                    "comp_spans": len(comp_spans), "comp_tokens": comp_tokens,
                    "comp_share_alllayer": comp_share,
                    "closed_tokens": sum(b - a for a, b in spans) if spans else 0,
                    "ctx_tokens": ctx_len, "context_fill": round(ctx_len / args.max_ctx, 4),
                    "pathology": probe["pathology"], "gold": probe["gold"], "pred": pred,
                    "correct": bool(pred == probe["gold"]), "parsed": pred is not None,
                    "mean_entropy": entropy, "response": (resp or "")[:200],
                })
            torch.cuda.empty_cache()
            pd.DataFrame(records).to_csv(turns_path, index=False)
            if (idx + 1) % 25 == 0:
                df = pd.DataFrame(records)
                acc = df.groupby("arm")["correct"].mean().to_dict()
                print(f"  [{idx + 1}/{len(probes)}] "
                      + "  ".join(f"{a}={v:.3f}" for a, v in sorted(acc.items())), flush=True)
            continue

        for arm, (rendered, cases) in built_arms.items():
            attn_cols = {}
            if args.attention_only:
                ids = tokenizer(rendered, return_tensors="pt").input_ids.to(args.device)
                capture = SelectiveAttentionCapture(model, capture_layers)
                capture.enabled = True
                with torch.no_grad():
                    model(ids)
                capture.remove()
                attn = capture.captured[args.reference_layer]
                spans = {"evidence": locate_token_span(tokenizer, rendered, probe["vignette"]),
                         "question": locate_token_span(tokenizer, rendered, question)}
                attn_cols = {"evidence_share": span_share(attn, spans["evidence"]),
                             "question_share": span_share(attn, spans["question"])}
                if heads is not None:
                    for li in capture_layers:
                        heads.extend(per_head_rows(
                            capture.captured[li], spans, probe=idx, arm=arm, layer=li,
                            pathology=probe["pathology"]))
                resp, ctx_len, entropy, pred = None, int(ids.shape[1]), None, None
            else:
                resp, ctx_len, entropy, _ = generate_with_entropy(
                    model, tokenizer, rendered, args.device, args.max_new, args.max_ctx)
                pred = extract_mcq_answer(resp) if resp else None
            overlaps = [len(probe_options & set(c["options"])) for c in cases]
            records.append({**attn_cols,
                "probe": idx, "arm": arm,
                "mean_shared_options": sum(overlaps) / len(overlaps),
                "n_context": len(cases),
                "ctx_tokens": ctx_len, "context_fill": round(ctx_len / args.max_ctx, 4),
                "pathology": probe["pathology"], "gold": probe["gold"], "pred": pred,
                "correct": bool(pred == probe["gold"]), "parsed": pred is not None,
                "mean_entropy": entropy, "response": (resp or "")[:200],
            })
            torch.cuda.empty_cache()

        pd.DataFrame(records).to_csv(turns_path, index=False)  # killed run keeps completed probes
        if (idx + 1) % 25 == 0:
            df = pd.DataFrame(records)
            acc = df.groupby("arm")["correct"].mean().to_dict()
            print(f"  [{idx + 1}/{len(probes)}] "
                  + "  ".join(f"{a}={v:.3f}" for a, v in sorted(acc.items())), flush=True)

    if heads is not None:
        heads.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    summary = {
        "model": args.model,
        "arms": (sorted(df["arm"].unique()) if args.close_arms and len(df)
                 else ArmSpec.overlap_arms()),
        "close_arms": args.close_arms,
        "measured_close": args.measured_close,
        "store_rows": args.store_rows,
        "n_context": args.n_context,
        "min_overlap": args.min_overlap,
        "n_probes_requested": n_probes,
        "n_probes_used": int(df["probe"].nunique()) if len(df) else 0,
        "gold_leaks": leaks,
        "starved_probes": starved,
        "overflow": guard.report(n_seen=len(probes)),
        "by_arm": (df.groupby("arm")
                     .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                          **({"closed_tokens": ("closed_tokens", "mean"),
                              "comp_share_alllayer": ("comp_share_alllayer", "mean")}
                             if args.close_arms
                             else {"mean_shared_options": ("mean_shared_options", "mean")}),
                          **({"hot_tokens": ("hot_tokens", "mean"),
                              "hotc_tokens": ("hotc_tokens", "mean"),
                              "hot_comp_overlap_tokens": ("hot_comp_overlap_tokens", "mean"),
                              "hot_mass": ("hot_mass", "mean"),
                              "hotc_mass": ("hotc_mass", "mean"),
                              "comp_mass": ("comp_mass", "mean")}
                             if args.measured_close is not None else {}),
                          mean_fill=("context_fill", "mean"),
                          parse_rate=("parsed", "mean"),
                          **({"evidence_share": ("evidence_share", "mean"),
                              "question_share": ("question_share", "mean")}
                             if args.attention_only else {}))
                     .reset_index().to_dict("records")) if len(df) else [],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nE3 COMPETITION SWEEP — {args.model}\n{'='*72}")
    print(f"probes used {summary['n_probes_used']}  |  gold leaks {leaks} (must be 0)  |  "
          f"starved {starved}  |  overflow skips {summary['overflow']['n_skipped']}")
    for row in summary["by_arm"]:
        detail = (f"closed_tok={row['closed_tokens']:.1f}  comp_share={row['comp_share_alllayer']:.4f}"
                  if args.close_arms else f"shared_opts={row['mean_shared_options']:.2f}")
        print(f"  {row['arm']:20s} n={row['n']:4d}  acc={row['accuracy']:.3f}  {detail}  "
              f"fill={row['mean_fill']:.3f}  parsed={row['parse_rate']:.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
