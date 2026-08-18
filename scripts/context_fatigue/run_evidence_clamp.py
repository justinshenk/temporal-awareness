"""E1c — the decisive test of dilution as the mechanism behind the distance effect.

E1 established that moving the evidence back costs 0.19–0.21 accuracy at identical fill, and the
attention addendum established that the evidence's attention share falls with distance
(0.041 → 0.012, a 3.3x drain). Those two facts are consistent with dilution but do not establish
it: distance and share move together by construction, so a positional or retrieval account
predicts the same pair of curves.

This driver breaks the tie. Each probe is presented in the **local** arm — evidence adjacent to the
question, nothing moved — and its evidence span is clamped down to **the very same item's measured
`back_20` share**. Position is held at local; only the mass changes.

    if clamped-local accuracy ≈ back_20 accuracy  → the mass drain explains the distance effect
    if clamped-local accuracy ≈ local accuracy    → it does not; distance acts through something else

The matched design matters: the target is per-item, taken from that item's own deep-arm
measurement, so the comparison does not depend on an average share standing in for a distribution.
"""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import render_prompt
from run_distance_sweep import (
    ARMS,
    INTRO,
    REFERENT,
    load_filler_pool,
    load_probe_pool,
)

from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_token_span,
    measure_span_share,
    solve_span_scale,
)
from src.probes.context_fatigue.context_assembly import assemble_transcript
from src.probes.context_fatigue.ddxplus_cases import format_case_question, load_evidence_db

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--reference-layer", type=int, default=24)
    p.add_argument("--donor-arm", default="back_20",
                   help="arm whose per-item evidence share becomes the clamp target")
    p.add_argument("--clamp-arm", default="local",
                   help="arm whose evidence span is clamped to the donor's share. "
                        "local+donor back_20 tests sufficiency (starve mass, reproduce the "
                        "deficit); back_20+donor local tests necessity (restore mass, rescue it)")
    p.add_argument("--depths", type=int, nargs="+", default=[21, 28, 35, 42])
    p.add_argument("--probes-per-cell", type=int, default=8)
    p.add_argument("--n-sessions", type=int, default=6)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e1c_evidence_clamp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--levels", type=float, nargs="+", default=None,
                   help="sweep mode: clamp the evidence share at --clamp-arm to each of these "
                        "absolute levels instead of to a donor arm's per-item share. Locates the "
                        "knee in the share->accuracy curve that reconciles E1c with E1e's C2.")
    p.add_argument("--n-probes", type=int, default=192, help="sweep mode only")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def letter_token_ids(tokenizer):
    ids = {}
    for letter in OPTION_LETTERS:
        variants = {tokenizer.encode(f, add_special_tokens=False)[0]
                    for f in (letter, f" {letter}")
                    if tokenizer.encode(f, add_special_tokens=False)}
        ids[letter] = sorted(variants)
    return ids


def score_forced_choice(logits, letter_ids):
    last = logits[0, -1]
    return max(OPTION_LETTERS, key=lambda ltr: max(float(last[i]) for i in letter_ids[ltr]))


def padded(tokenizer, text, pad_id, device):
    base = tokenizer(text, return_tensors="pt").input_ids
    ids = torch.cat([torch.full((1, 1), pad_id, dtype=base.dtype), base], dim=1).to(device)
    attn = torch.ones_like(ids)
    attn[0, 0] = 0
    return ids, attn


def run_sweep(args, model, tokenizer, pad_id, is_chat, letter_ids, out_dir):
    """Dose-response on the *evidence* span at a fixed position.

    E1c cut evidence share 0.041 -> 0.012 at ``local`` and cost 0.21 accuracy. E1e's C2 cut it
    0.029 -> 0.010 at ``back_5`` and cost nothing. Both hold only if the share->accuracy curve has a
    knee between those values: steep above, flat below. This sweep locates it.
    """
    import random

    evidence_db = load_evidence_db()
    filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens)
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  filler {len(filler_pool)} | probes {len(probe_pool)}", flush=True)

    levels = args.levels[:2] if args.preflight else args.levels
    n_probes = 1 if args.preflight else args.n_probes
    records = []
    turns_path = out_dir / "turns.csv"

    for idx in range(n_probes):
        rng = random.Random(args.seed + idx)
        probe = probe_pool[rng.randrange(len(probe_pool))]
        question = format_case_question(probe["options"], args.n_options, referent=REFERENT)
        prior = [{"role": "user", "content": INTRO},
                 {"role": "assistant", "content": "Understood."}]
        for f in rng.sample(filler_pool, 35):
            prior += [{"role": "user", "content": f["text"]},
                      {"role": "assistant", "content": f["gold"]}]

        built = assemble_transcript(prior, evidence=probe["vignette"], question=question,
                                    distance=ARMS[args.clamp_arm])
        text = render_prompt(tokenizer, built.turns, is_chat)
        if len(tokenizer.encode(text)) + 24 > args.max_ctx:
            continue
        ids, attn = padded(tokenizer, text, pad_id, args.device)
        s, e = locate_token_span(tokenizer, text, probe["vignette"])
        span = (s + 1, e + 1)

        natural = measure_span_share(model, ids, span, args.reference_layer, attn)
        with torch.no_grad():
            pred = score_forced_choice(model(ids, attention_mask=attn).logits, letter_ids)
        base = {"probe": idx, "gold": probe["gold"], "pathology": probe["pathology"],
                "ctx_tokens": int(ids.shape[1])}
        records.append({**base, "level": "natural", "target_share": None,
                        "achieved_share": natural, "scale": 1.0, "pred": pred,
                        "correct": pred == probe["gold"]})

        for target in levels:
            if target >= natural:      # only clamp downward; upward is a different question
                continue
            scale, achieved = solve_span_scale(
                model, ids, span=span, target_share=target,
                reference_layer=args.reference_layer, tol=1e-4, attention_mask=attn)
            with torch.no_grad(), SpanAttentionClamp(model, span=span, scale=scale):
                pr = score_forced_choice(model(ids, attention_mask=attn).logits, letter_ids)
            records.append({**base, "level": f"{target:.3f}", "target_share": target,
                            "achieved_share": achieved, "scale": scale, "pred": pr,
                            "correct": pr == probe["gold"]})
        pd.DataFrame(records).to_csv(turns_path, index=False)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n_probes} probes, {len(records)} rows", flush=True)
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    by = (df.groupby("level").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                  achieved=("achieved_share", "mean"),
                                  scale=("scale", "median"))
            .reset_index().to_dict("records"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model": args.model, "mode": "evidence_share_sweep",
                   "clamp_arm": args.clamp_arm, "levels": levels, "by_level": by}, f, indent=2)
    print(f"\n{'='*66}\nEVIDENCE-SHARE SWEEP at {args.clamp_arm} — L{args.reference_layer}\n{'='*66}")
    for r in sorted(by, key=lambda r: -r["achieved"]):
        print(f"  {r['level']:>8s}  achieved={r['achieved']:.4f}  n={r['n']:4d}  "
              f"acc={r['accuracy']:.3f}  scale={r['scale']:.4f}")
    print(f"\nSaved to {out_dir}/")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None
    letter_ids = letter_token_ids(tokenizer)

    if args.levels:
        run_sweep(args, model, tokenizer, pad_id, is_chat, letter_ids, out_dir)
        return

    evidence_db = load_evidence_db()
    filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens)
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  filler {len(filler_pool)} | probes {len(probe_pool)}", flush=True)

    depths = args.depths[:1] if args.preflight else args.depths
    sessions = 1 if args.preflight else args.n_sessions
    per_cell = 1 if args.preflight else args.probes_per_cell
    donor_distance = ARMS[args.donor_arm]

    import random
    records = []
    turns_path = out_dir / "turns.csv"

    for session in range(sessions):
        rng = random.Random(args.seed + 1000 * session)
        filler = rng.sample(filler_pool, min(max(depths) + 5, len(filler_pool)))
        probes = rng.sample(probe_pool, min(per_cell * len(depths), len(probe_pool)))

        conv = [{"role": "user", "content": INTRO},
                {"role": "assistant", "content": "Understood."}]
        snapshots = {}
        for item in filler:
            conv = conv + [{"role": "user", "content": item["text"]},
                           {"role": "assistant", "content": item["gold"]}]
            n_user = sum(1 for t in conv if t["role"] == "user") - 1
            if n_user in depths and n_user not in snapshots:
                snapshots[n_user] = list(conv)
            if len(snapshots) == len(depths):
                break

        for di, depth in enumerate(depths):
            base = snapshots.get(depth)
            if base is None:
                continue
            for probe in probes[di * per_cell:(di + 1) * per_cell]:
                question = format_case_question(probe["options"], args.n_options,
                                                referent=REFERENT)
                built = {}
                for arm, dist in ((args.clamp_arm, ARMS[args.clamp_arm]),
                                  (args.donor_arm, donor_distance)):
                    try:
                        t = assemble_transcript(base, evidence=probe["vignette"],
                                                question=question, distance=dist)
                    except ValueError:
                        built = {}
                        break
                    built[arm] = render_prompt(tokenizer, t.turns, is_chat)
                if not built:
                    continue

                row = {"session": session, "filler_turns": depth,
                       "pathology": probe["pathology"], "gold": probe["gold"]}
                shares, preds = {}, {}
                spans, inputs = {}, {}
                for arm, text in built.items():
                    if len(tokenizer.encode(text)) + 16 > args.max_ctx:
                        break
                    ids, attn = padded(tokenizer, text, pad_id, args.device)
                    s, e = locate_token_span(tokenizer, text, probe["vignette"])
                    span = (s + 1, e + 1)
                    spans[arm], inputs[arm] = span, (ids, attn)
                    shares[arm] = measure_span_share(model, ids, span,
                                                    args.reference_layer, attn)
                    with torch.no_grad():
                        preds[arm] = score_forced_choice(
                            model(ids, attention_mask=attn).logits, letter_ids)
                if len(shares) < 2:
                    continue

                # the intervention: local position, donor-arm evidence mass
                target = shares[args.donor_arm]
                ids, attn = inputs[args.clamp_arm]
                scale, achieved = solve_span_scale(
                    model, ids, span=spans[args.clamp_arm], target_share=target,
                    reference_layer=args.reference_layer, tol=1e-4, attention_mask=attn)
                with torch.no_grad(), SpanAttentionClamp(model, span=spans[args.clamp_arm],
                                                         scale=scale):
                    pred_clamped = score_forced_choice(
                        model(ids, attention_mask=attn).logits, letter_ids)

                for arm in built:
                    records.append({**row, "condition": arm, "evidence_share": shares[arm],
                                    "target_share": None, "scale": 1.0,
                                    "pred": preds[arm], "correct": preds[arm] == probe["gold"]})
                records.append({**row, "condition": f"{args.clamp_arm}_clamped", "evidence_share": achieved,
                                "target_share": target, "scale": scale,
                                "pred": pred_clamped,
                                "correct": pred_clamped == probe["gold"]})
                pd.DataFrame(records).to_csv(turns_path, index=False)
                torch.cuda.empty_cache()
        print(f"  [s{session}] {len(records)} rows", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    by = (df.groupby("condition").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                      share=("evidence_share", "mean"))
            .reset_index().to_dict("records"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model": args.model, "donor_arm": args.donor_arm,
                   "reference_layer": args.reference_layer, "by_condition": by}, f, indent=2)
    print(f"\n{'='*64}\nE1c EVIDENCE CLAMP — {args.clamp_arm} position, "
          f"{args.donor_arm} mass\n{'='*64}")
    for r in by:
        print(f"  {r['condition']:16s} n={r['n']:4d} acc={r['accuracy']:.3f} "
              f"evidence_share={r['share']:.4f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
