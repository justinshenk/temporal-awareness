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

from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture
from src.probes.context_fatigue.attention_clamp import locate_token_span, span_share
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
    p.add_argument("--preflight", action="store_true",
                   help="run two probes end-to-end, write them, and exit")
    return p.parse_args()


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
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
        "arms": ArmSpec.overlap_arms(),
        "n_context": args.n_context,
        "min_overlap": args.min_overlap,
        "n_probes_requested": n_probes,
        "n_probes_used": int(df["probe"].nunique()) if len(df) else 0,
        "gold_leaks": leaks,
        "starved_probes": starved,
        "overflow": guard.report(n_seen=len(probes)),
        "by_arm": (df.groupby("arm")
                     .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                          mean_shared_options=("mean_shared_options", "mean"),
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
        print(f"  {row['arm']:9s} n={row['n']:4d}  acc={row['accuracy']:.3f}  "
              f"shared_opts={row['mean_shared_options']:.2f}  fill={row['mean_fill']:.3f}  "
              f"parsed={row['parse_rate']:.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
