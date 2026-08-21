"""E6 — does an answer format specified in the system prompt survive irrelevant accumulation?

E5 established two things. Clamping the system span's attention share from 0.165 to 0.050 breaks
compliance (0.99 -> 0.03) while accuracy barely moves, so the mass is causally necessary. And
accumulation drives that share from 0.166 to 0.021 on its own. Compliance nonetheless survived
accumulation in every earlier arm — because the accumulated context *demonstrated* the behaviour,
and the model copies the last assistant turn's format exactly (720/720 replies were one character
long when the prior turn was one character).

E6 removes the demonstration. The system prompt specifies a clinical answer format; the
accumulated context is **non-medical** MMLU, so nothing in it ever exhibits that format and
in-context learning cannot supply it. The comparison is no context versus a full window of
irrelevant context, with intermediate depths so the trend is visible rather than inferred from
two points.

    uv run python scripts/context_fatigue/run_format_erosion.py --preflight
    uv run python scripts/context_fatigue/run_format_erosion.py --n-probes 40
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
    MEDICAL_SUBJECTS,
    generate_with_entropy,
    load_code_filler_pool,
    load_filler_pool,
    load_gsm8k_filler_pool,
    render_prompt,
)

from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_token_span,
    locate_turn_spans,
    measure_multi_span_shares,
    measure_span_share,
    solve_span_scale,
)
from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    load_evidence_db,
    load_probe_pool,
)
from src.probes.context_fatigue.instruction_checks import (
    CLINICAL_FORMAT_SYSTEM,
    check_clinical_format,
)

DEPTHS = [0, 3, 7, 14, 21, 28, 35]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=128)  # the format needs room to be produced
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    p.add_argument("--n-probes", type=int, default=40)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filler", choices=["code", "gsm8k", "mmlu"], default="code",
                   help="what accumulates. `mmlu` is 4-option multiple choice, so every filler "
                        "turn demonstrates answering an options question with a bare letter -- "
                        "the very shape that competes with the system prompt's format, which "
                        "makes it a structural-copy control rather than neutral filler. `code` "
                        "and `gsm8k` are free-form with no options anywhere, and their long "
                        "replies fill the window in far fewer turns.")
    p.add_argument("--filler-max-new", type=int, default=200,
                   help="generation budget for a filler turn; free-form filler needs room")
    p.add_argument("--include-medical-filler", action="store_true",
                   help="do NOT exclude medical MMLU subjects from the filler. Off by default: "
                        "15 of 57 subjects are medical-adjacent, so leaving them in gives the "
                        "model in-domain practice at the task the probe measures.")
    p.add_argument("--out-dir", default="results/context_fatigue/e6_format_erosion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--recovery", action="store_true",
                   help="at the deepest depth, additionally run recovery arms: clamp the system "
                        "span back up to its no-context share, restate the policy in the latest "
                        "user turn, or both. Requires eager attention for the clamp.")
    p.add_argument("--close-arms", action="store_true",
                   help="at the deepest depth, close or dose-match the attention channel to the "
                        "filler answer spans (fa_close / fa_matched), with filler-question "
                        "closure (fq_close) as the control. Requires eager attention.")
    p.add_argument("--record-spans", action="store_true",
                   help="also write spans.csv: the final position's attention share on every "
                        "turn (system, each filler question/answer, the probe), from the same "
                        "captured forward the system share already uses")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        # the clamp needs an explicit additive mask, which sdpa optimizes away on a causal-only
        # prompt; only pay for eager when a recovery arm will actually use it
        **({"attn_implementation": "eager"} if args.recovery or args.close_arms else {})).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    reference_layers = list(range(len(model.model.layers)))
    evidence_db = load_evidence_db()
    need = max(args.depths) + 2
    if args.filler == "mmlu":
        excluded = None if args.include_medical_filler else MEDICAL_SUBJECTS
        filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens, excluded)
    elif args.filler == "gsm8k":
        filler_pool = load_gsm8k_filler_pool(need * 4, args.seed)
    else:
        filler_pool = load_code_filler_pool(need * 4, args.seed)
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    subjects = sorted({f["subject"] for f in filler_pool})
    print(f"  filler '{args.filler}': {len(filler_pool)} items over {len(subjects)} subject(s) | "
          f"probes {len(probe_pool)}", flush=True)

    depths = args.depths[:2] if args.preflight else args.depths
    n_probes = 2 if args.preflight else args.n_probes

    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=args.headroom)
    rng = random.Random(args.seed)
    filler = rng.sample(filler_pool, min(max(depths) + 2, len(filler_pool)))
    probes = rng.sample(probe_pool, n_probes)

    # One accumulation, snapshotted at every depth, so the *same* probes are asked at each depth
    # against the *same* prefix. Depth is then the only thing that moves.
    base = [{"role": "system", "content": CLINICAL_FORMAT_SYSTEM}]
    snapshots, conv = {}, list(base)
    for i, item in enumerate(filler):
        if i in depths:
            snapshots[i] = list(conv)
        if len(snapshots) == len(depths):
            break
        conv = conv + [{"role": "user", "content": item["text"]}]
        # MCQ filler is answered with a bare letter (its own convention); free-form filler keeps
        # the model's full reply, which is what makes it accumulate quickly and demonstrate no
        # answer shape at all.
        budget = 8 if args.filler == "mmlu" else args.filler_max_new
        resp, _, _, _ = generate_with_entropy(
            model, tokenizer, render_prompt(tokenizer, conv, is_chat),
            args.device, budget, args.max_ctx)
        answer = (resp or "A").strip()
        conv = conv + [{"role": "assistant",
                        "content": answer[:8] if args.filler == "mmlu" else answer}]
    for d in depths:
        snapshots.setdefault(d, list(conv))

    records, span_records, skipped = [], [], 0
    turns_path = out_dir / "turns.csv"
    for depth in depths:
        prefix = snapshots[depth]
        for pi, probe in enumerate(probes):
            question = format_case_question(probe["options"], args.n_options, answer_cue=False)
            turns = prefix + [{"role": "user", "content": probe["vignette"] + question}]
            text = render_prompt(tokenizer, turns, is_chat)
            if not guard.fits(text, used=0, index=pi):
                skipped += 1
                continue
            # System-span attention on the same input, before generation. The hypothesis this
            # tests: in-context structure competes with the system prompt for attention, so a
            # learnable filler stream should drain the system span while an unlearnable one does
            # not -- and compliance should follow the mass, not the token count.
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            if args.record_spans:
                spans = locate_turn_spans(tokenizer, text, [t["content"] for t in turns])
                shares = measure_multi_span_shares(model, ids, spans, reference_layers)
                kinds = (["system"]
                         + ["filler_q" if t["role"] == "user" else "filler_a"
                            for t in turns[1:-1]]
                         + ["probe"])
                span_records += [
                    {"depth": depth, "probe": pi, "turn": ti, "kind": kind,
                     "n_tokens": b - a, "ctx_tokens": int(ids.shape[1]), "share": share}
                    for ti, (kind, (a, b), share) in enumerate(zip(kinds, spans, shares))]
                sys_span, sys_share = spans[0], shares[0]
            else:
                sys_span = locate_token_span(tokenizer, text, CLINICAL_FORMAT_SYSTEM)
                sys_share = measure_span_share(model, ids, sys_span, reference_layers)
            # The span is a fixed number of tokens while the context grows, so raw share falls by
            # arithmetic alone. Enrichment -- share divided by the span's share of tokens -- is
            # what says whether the model is actually consulting the instruction less.
            sys_tokens = sys_span[1] - sys_span[0]
            sys_enrichment = sys_share / (sys_tokens / int(ids.shape[1]))
            resp, ctx_len, entropy, _ = generate_with_entropy(
                model, tokenizer, text, args.device, args.max_new, args.max_ctx)
            graded = check_clinical_format(resp or "", probe["vignette"],
                                           options=probe["options"][:args.n_options])
            records.append({
                "depth": depth, "probe": pi,
                "ctx_tokens": ctx_len, "fill": round(ctx_len / args.max_ctx, 4),
                "gold": probe["gold"], "pred": graded["answer"],
                "correct": bool(graded["answer"] == probe["gold"]),
                "parsed": graded["answer"] is not None,
                "response_chars": len(resp or ""),
                "system_share": sys_share,
                "system_tokens": sys_tokens,
                "system_enrichment": sys_enrichment,
                "mean_entropy": entropy,
                **{k: graded[k] for k in ("has_answer", "has_supporting", "n_symptoms",
                                          "grounded_fraction", "fully_compliant")},
                "response": resp or "",
            })
        pd.DataFrame(records).to_csv(turns_path, index=False)
        if span_records:
            pd.DataFrame(span_records).to_csv(out_dir / "spans.csv", index=False)
        d = pd.DataFrame(records)
        cur = d[d.depth == depth]
        print(f"  depth {depth:2d}: fill={cur['fill'].mean():.3f}  "
              f"sys_share={cur['system_share'].mean():.4f}  "
              f"enrich={cur['system_enrichment'].mean():.2f}  "
              f"compliant={cur['fully_compliant'].mean():.3f}  "
              f"acc={cur['correct'].mean():.3f}  chars={cur['response_chars'].mean():.0f}",
              flush=True)
        torch.cuda.empty_cache()

    if args.recovery:
        deepest = max(depths)
        # Target: the share the system span held with no accumulated context at all. Restoring to
        # anything else would be restoring to an arbitrary level.
        target = float(pd.DataFrame(records).query("depth == 0")["system_share"].mean())
        print(f"\nrecovery at depth {deepest}: restoring system share to its depth-0 value "
              f"{target:.4f}", flush=True)
        prefix = snapshots[deepest]
        for pi, probe in enumerate(probes):
            question = format_case_question(probe["options"], args.n_options, answer_cue=False)
            for arm in ("upclamp", "refresh", "both"):
                # `refresh` restates the policy in the latest user turn -- the context-refresh
                # intervention the repo already defines for canaries, applied to a real format.
                user = ((CLINICAL_FORMAT_SYSTEM + "\n\n") if arm in ("refresh", "both") else "")
                turns = prefix + [{"role": "user", "content": user + probe["vignette"] + question}]
                text = render_prompt(tokenizer, turns, is_chat)
                if not guard.fits(text, used=0, index=pi):
                    continue
                ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
                span = locate_token_span(tokenizer, text, CLINICAL_FORMAT_SYSTEM)
                if arm in ("upclamp", "both"):
                    scale, achieved = solve_span_scale(
                        model, ids, span=span, target_share=target,
                        reference_layer=reference_layers, tol=1e-3)
                    with SpanAttentionClamp(model, span=span, scale=scale):
                        resp, ctx_len, entropy, _ = generate_with_entropy(
                            model, tokenizer, text, args.device, args.max_new, args.max_ctx)
                else:
                    achieved = measure_span_share(model, ids, span, reference_layers)
                    resp, ctx_len, entropy, _ = generate_with_entropy(
                        model, tokenizer, text, args.device, args.max_new, args.max_ctx)
                graded = check_clinical_format(resp or "", probe["vignette"],
                                           options=probe["options"][:args.n_options])
                records.append({
                    "depth": deepest, "probe": pi, "recovery_arm": arm,
                    "ctx_tokens": ctx_len, "fill": round(ctx_len / args.max_ctx, 4),
                    "gold": probe["gold"], "pred": graded["answer"],
                    "correct": bool(graded["answer"] == probe["gold"]),
                    "parsed": graded["answer"] is not None,
                    "response_chars": len(resp or ""), "system_share": achieved,
                    "mean_entropy": entropy,
                    **{k: graded[k] for k in ("has_answer", "has_supporting", "n_symptoms",
                                              "grounded_fraction", "fully_compliant")},
                    "response": resp or "",
                })
            torch.cuda.empty_cache()
        pd.DataFrame(records).to_csv(turns_path, index=False)
        rec = pd.DataFrame(records)
        rec = rec[rec.get("recovery_arm").notna()] if "recovery_arm" in rec else rec
        for arm, g in rec.groupby("recovery_arm"):
            print(f"  {arm:>9s}: share={g['system_share'].mean():.4f}  "
                  f"compliant={g['fully_compliant'].mean():.3f}  "
                  f"acc={g['correct'].mean():.3f}", flush=True)

    if args.close_arms:
        deepest = max(depths)
        prefix = snapshots[deepest]
        print(f"\nexemplar-close at depth {deepest}: intervening on the filler channels, "
              f"system span untouched", flush=True)
        for pi, probe in enumerate(probes):
            question = format_case_question(probe["options"], args.n_options, answer_cue=False)
            turns = prefix + [{"role": "user", "content": probe["vignette"] + question}]
            text = render_prompt(tokenizer, turns, is_chat)
            if not guard.fits(text, used=0, index=pi):
                continue
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            spans = locate_turn_spans(tokenizer, text, [t["content"] for t in turns])
            fa = [s for s, t in zip(spans[1:-1], turns[1:-1], strict=True)
                  if t["role"] == "assistant"]
            fq = [s for s, t in zip(spans[1:-1], turns[1:-1], strict=True)
                  if t["role"] == "user"]
            ctx = int(ids.shape[1])
            # Size-matched causal control: one random token inside each filler question — the
            # same count and span size as the answer letters, so if fa_close works through span
            # geometry rather than content, this arm works too.
            r1 = random.Random(args.seed * 1000 + pi)
            rand1 = sorted((j, j + 1) for j in (r1.randrange(a, b) for a, b in fq))
            for arm, target_spans, scale in (("fa_close", fa, 0.0),
                                             ("fa_matched", fa, None),
                                             ("fq_close", fq, 0.0),
                                             ("rand1_close", rand1, 0.0)):
                if scale is None:
                    # Dose control: bring the answer spans' union share to the per-token level
                    # code-arm answers get (enrichment 0.3) rather than to zero — closure and
                    # near-ablation read differently (E2a), so both arms exist.
                    tok = sum(b - a for a, b in target_spans)
                    scale, achieved = solve_span_scale(
                        model, ids, span=target_spans, target_share=0.3 * tok / ctx,
                        reference_layer=reference_layers, tol=5e-4)
                else:
                    achieved = 0.0
                with SpanAttentionClamp(model, span=target_spans, scale=scale):
                    resp, ctx_len, entropy, _ = generate_with_entropy(
                        model, tokenizer, text, args.device, args.max_new, args.max_ctx)
                graded = check_clinical_format(resp or "", probe["vignette"],
                                               options=probe["options"][:args.n_options])
                records.append({
                    "depth": deepest, "probe": pi, "recovery_arm": arm,
                    "closed_share": achieved,
                    "ctx_tokens": ctx_len, "fill": round(ctx_len / args.max_ctx, 4),
                    "gold": probe["gold"], "pred": graded["answer"],
                    "correct": bool(graded["answer"] == probe["gold"]),
                    "parsed": graded["answer"] is not None,
                    "response_chars": len(resp or ""),
                    "mean_entropy": entropy,
                    **{k: graded[k] for k in ("has_answer", "has_supporting", "n_symptoms",
                                              "grounded_fraction", "fully_compliant")},
                    "response": resp or "",
                })
            torch.cuda.empty_cache()
        pd.DataFrame(records).to_csv(turns_path, index=False)
        rec = pd.DataFrame(records)
        rec = rec[rec["recovery_arm"].notna()] if "recovery_arm" in rec else rec
        for arm, g in rec.groupby("recovery_arm"):
            print(f"  {arm:>10s}: closed_share={g['closed_share'].mean():.4f}  "
                  f"compliant={g['fully_compliant'].mean():.3f}  "
                  f"acc={g['correct'].mean():.3f}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    if "recovery_arm" not in df:
        df["recovery_arm"] = None
    by_depth = {
        int(depth): {
            "n": int(len(g)), "fill": float(g["fill"].mean()),
            "system_share": float(g["system_share"].mean()),
            "system_enrichment": float(g["system_enrichment"].mean()),
            "fully_compliant": float(g["fully_compliant"].mean()),
            "has_answer": float(g["has_answer"].mean()),
            "has_supporting": float(g["has_supporting"].mean()),
            "mean_symptoms": float(g["n_symptoms"].mean()),
            "grounded_fraction": float(g["grounded_fraction"].mean()),
            "accuracy": float(g["correct"].mean()),
            "parse_rate": float(g["parsed"].mean()),
            "response_chars": float(g["response_chars"].mean()),
        }
        for depth, g in df[df["recovery_arm"].isna()].groupby("depth")
    }
    (out_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "system_prompt": CLINICAL_FORMAT_SYSTEM,
        "filler": args.filler,
        "filler_excludes_medical": not args.include_medical_filler,
        "filler_subjects": len(subjects), "depths": depths, "n_probes": n_probes,
        "overflow_skips": skipped, "by_depth": by_depth}, indent=2))

    print("\n" + "=" * 78)
    print(f"E6 FORMAT EROSION UNDER IRRELEVANT CONTEXT — {args.model}")
    print("=" * 78)
    print(f"probes {n_probes} | filler '{args.filler}' over {len(subjects)} subject(s) | "
          f"overflow skips {skipped}")
    print(f"{'depth':>6} {'fill':>6} {'sysShare':>9} {'enrich':>7} {'compliant':>10} {'answer:':>8} "
          f"{'supporting:':>12} {'symptoms':>9} {'grounded':>9} {'acc':>6} {'chars':>6}")
    for depth, s in sorted(by_depth.items()):
        print(f"{depth:>6} {s['fill']:>6.3f} {s['system_share']:>9.4f} "
              f"{s['system_enrichment']:>7.2f} {s['fully_compliant']:>10.3f} "
              f"{s['has_answer']:>8.3f} {s['has_supporting']:>12.3f} "
              f"{s['mean_symptoms']:>9.2f} {s['grounded_fraction']:>9.3f} "
              f"{s['accuracy']:>6.3f} {s['response_chars']:>6.0f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
