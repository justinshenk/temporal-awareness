"""E2 — causal attention-mass dose-response: where is the floor, and does mass explain the dip?

The paper says accuracy is flat over the range accumulation happened to traverse (current-query
share ≈0.35 → ≈0.15) and infers headroom from that. E2 stops inferring and *sets* the mass.

**E2a (find the floor).** On cold-start contexts the current query's post-softmax share is clamped
down through {0.30, 0.20, 0.15, 0.10, 0.05, 0.02} and accuracy is measured at each level. A plateau
followed by a cliff locates the floor; the margin between the cliff and the ≈0.15 accumulation
actually reaches *is* the headroom claim, as a number.

**E2b (does mass rescue the dip?).** On the deep-fill stream the share is clamped back *up* to its
cold-start level. If the −14 point top-bin dip does not recover, the dip is not a dilution effect.

Two mechanics worth knowing
---------------------------
*The mask must exist.* Under sdpa a purely causal mask is optimized away to ``None`` before it
reaches ``self_attn``, and the clamp has nothing to bias. One **left-pad token, masked out**, forces
transformers to build an explicit mask. Loading with ``attn_implementation="eager"`` would also
work but materializes [1, H, N, N] per layer and OOMs a 32 GB card at 4k. The padded input is used
for *every* condition, the unclamped baseline included, so comparisons are internally exact; the
sdpa mask path shifts logits by ~2% of scale versus an unpadded forward (argmax agreement 97.7%,
last-token argmax unchanged).

*Scoring is forced-choice.* Accuracy is read from the next-token logits restricted to the option
letters rather than from free generation. That is deterministic, costs one forward instead of a
generation, and removes parse failures — which in E1 accounted for 19% of responses and would
otherwise confound a dose-response with a verbosity response.
"""

import argparse
import ast
import gc
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import generate_with_entropy, render_prompt
from run_evidence_clamp import _normalise_layers

from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_token_span,
    measure_span_share,
    solve_span_scale,
)
from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import format_case_mcq, load_evidence_db

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
# The paper's own DDXPlus composition, reproduced so the mass trajectory matches the one E2 is
# reasoning about: system prompt + accumulated cases. Measured share@L24 falls 0.253 (cold start)
# -> 0.186 -> 0.159 -> 0.156 -> 0.149 (8 prior cases, 69% fill), i.e. the ~0.15 that accumulation
# actually reaches. An MMLU-filler context instead sits flat at ~0.16 regardless of fill, which
# would give the sweep nothing to be measured against.
SYSTEM_PROMPT = "You are a doctor."
LEVELS = [0.30, 0.20, 0.15, 0.10, 0.05, 0.02]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--reference-layer", type=int, nargs="+", default=[24],
                   help="layer(s) the span share is read from; the clamp biases every layer "
                        "regardless. Default [24] reproduces the committed runs exactly; pass "
                        "every layer for the pooled readout.")
    p.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    p.add_argument("--cold-start-cases", type=int, default=0,
                   help="prior DDXPlus cases before the probe; 0 = coldest start")
    p.add_argument("--n-items", type=int, default=110)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e2a_mass_clamp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["e2a", "e2b"], default="e2a")
    # E2b (dip rescue) on the random-subject MMLU stream
    p.add_argument("--fill-target", type=float, default=0.93)
    p.add_argument("--deep-fill", type=float, default=0.80)
    p.add_argument("--cold-fill", type=float, default=0.20)
    p.add_argument("--n-sessions", type=int, default=14)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--preflight", action="store_true")
    return _normalise_layers(p.parse_args())


def letter_token_ids(tokenizer):
    """Token ids for each option letter, both bare and space-prefixed."""
    ids = {}
    for letter in OPTION_LETTERS:
        variants = set()
        for form in (letter, f" {letter}"):
            enc = tokenizer.encode(form, add_special_tokens=False)
            if enc:
                variants.add(enc[0])
        ids[letter] = sorted(variants)
    return ids


def score_forced_choice(logits, letter_ids):
    """Argmax over the option letters at the final position."""
    last = logits[0, -1]
    return max(OPTION_LETTERS, key=lambda ltr: max(float(last[i]) for i in letter_ids[ltr]))


def build_probe_pool(evidence_db, args):
    path = Path(__file__).resolve().parents[2] / "data" / "context_fatigue" / "ddxplus_test.csv"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("aai530-group6/ddxplus", "test.csv", repo_type="dataset")
    df = pd.read_csv(path, nrows=4000)
    rng = random.Random(args.seed)
    probes = []
    for _, row in df.iterrows():
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        options = [d[0] for d in ddx[:args.n_options]]
        if len(options) < args.n_options or row["PATHOLOGY"] not in options:
            continue
        rng.shuffle(options)
        probes.append({
            "case": format_case_mcq(row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
                                    row["EVIDENCES"], evidence_db, options, args.n_options),
            "gold": OPTION_LETTERS[options.index(row["PATHOLOGY"])],
            "pathology": row["PATHOLOGY"],
        })
    return probes


MMLU_LABELS = ["A", "B", "C", "D"]
MMLU_INTRO = ("Answer each multiple-choice question. Reply with only the letter of the correct "
              "option.")


def format_mmlu(question, choices):
    return (question + "\n"
            + "".join(f"{MMLU_LABELS[i]}) {o}\n" for i, o in enumerate(choices))
            + "\nReply with only the letter (A, B, C, or D).")


def padded_inputs(tokenizer, text, pad_id, device):
    """Tokenize with one masked left-pad token so sdpa builds an explicit, biasable mask."""
    base = tokenizer(text, return_tensors="pt").input_ids
    ids = torch.cat([torch.full((1, 1), pad_id, dtype=base.dtype), base], dim=1).to(device)
    attn = torch.ones_like(ids)
    attn[0, 0] = 0
    return ids, attn


def run_e2b(args, model, tokenizer, pad_id, is_chat, letter_ids, out_dir):
    """Clamp the query share back *up* at deep fill and ask whether the dip recovers.

    Each deep-fill turn is clamped to **its own session's** mean cold-start share, so the target is
    a within-session control rather than a number imported from another run.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    pool = [{"text": format_mmlu(r["question"], r["choices"]),
             "gold": MMLU_LABELS[r["answer"]], "subject": r["subject"]} for r in ds]
    print(f"  MMLU pool {len(pool)}", flush=True)

    guard = OverflowGuard(count_tokens=lambda s: len(tokenizer.encode(s)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=16)
    records = []
    n_seen = 0
    sessions = 1 if args.preflight else args.n_sessions
    turns_path = out_dir / "turns.csv"

    for session in range(sessions):
        rng = random.Random(args.seed + 1000 * session + 7)
        order = list(range(len(pool)))
        rng.shuffle(order)
        conv = [{"role": "user", "content": MMLU_INTRO},
                {"role": "assistant", "content": "Understood. I'll reply with only the letter."}]
        cold_shares = []

        for qi in order:
            used = len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))
            fill = used / args.max_ctx
            if fill > args.fill_target:
                break
            item = pool[qi]
            n_seen += 1
            if not guard.fits(item["text"], used=used, index=n_seen):
                continue

            conv_q = conv + [{"role": "user", "content": item["text"]}]
            text = render_prompt(tokenizer, conv_q, is_chat)
            ids, attn = padded_inputs(tokenizer, text, pad_id, args.device)
            start, end = locate_token_span(tokenizer, text, item["text"])
            span = (start + 1, end + 1)

            share = measure_span_share(model, ids, span, args.reference_layer, attn)
            with torch.no_grad():
                pred = score_forced_choice(model(ids, attention_mask=attn).logits, letter_ids)
            # The committed deep-fill artifact scored by *generating* and extracting a letter,
            # counting unparseable output as wrong. Scoring both ways on the same forward makes the
            # scoring rule the only difference, which is what decides whether the published top-bin
            # dip is an accuracy effect or a response-format artifact.
            resp, _, _, _ = generate_with_entropy(model, tokenizer, text, args.device,
                                                  args.max_new, args.max_ctx)
            m = re.search(r"\b([A-D])\b", (resp or "").upper())
            pred_gen = m.group(1) if m else None
            row = {"session": session, "context_fill": round(fill, 4),
                   "ctx_tokens": int(ids.shape[1]), "subject": item["subject"],
                   "gold": item["gold"], "natural_share": share,
                   "pred": pred, "correct": pred == item["gold"],
                   "pred_gen": pred_gen, "correct_gen": pred_gen == item["gold"],
                   "parsed_gen": pred_gen is not None, "response": (resp or "")[:120],
                   "condition": "natural", "target_share": None, "achieved_share": share}
            records.append(row)
            if fill < args.cold_fill:
                cold_shares.append(share)

            # the rescue arm: only at deep fill, and only once a cold-start baseline exists
            if fill >= args.deep_fill and cold_shares:
                target = float(sum(cold_shares) / len(cold_shares))
                if target > share:
                    scale, achieved = solve_span_scale(
                        model, ids, span=span, target_share=target,
                        reference_layer=args.reference_layer, tol=1e-3, attention_mask=attn)
                    with torch.no_grad(), SpanAttentionClamp(model, span=span, scale=scale):
                        rpred = score_forced_choice(
                            model(ids, attention_mask=attn).logits, letter_ids)
                    records.append({**row, "condition": "rescued", "pred": rpred,
                                    "correct": rpred == item["gold"],
                                    "target_share": target, "achieved_share": achieved})

            conv = conv_q + [{"role": "assistant", "content": pred}]
            pd.DataFrame(records).to_csv(turns_path, index=False)
            torch.cuda.empty_cache()

        done = [r for r in records if r["session"] == session]
        print(f"  [s{session}] {len(done)} rows, cold-share mean="
              f"{(sum(cold_shares)/len(cold_shares) if cold_shares else float('nan')):.3f}",
              flush=True)

    df = pd.DataFrame(records)
    nat = df[df.condition == "natural"]
    deep = nat[nat.context_fill >= args.deep_fill]["correct"]
    rest = nat[nat.context_fill < args.deep_fill]["correct"]
    res = df[df.condition == "rescued"]["correct"]
    summary = {
        "model": args.model, "mode": "e2b", "reference_layer": args.reference_layer,
        "deep_fill": args.deep_fill, "n_sessions": sessions,
        "overflow": guard.report(n_seen=n_seen),
        "n_deep": int(len(deep)), "n_rest": int(len(rest)), "n_rescued": int(len(res)),
        "acc_deep": float(deep.mean()) if len(deep) else None,
        "acc_rest": float(rest.mean()) if len(rest) else None,
        "acc_rescued": float(res.mean()) if len(res) else None,
        "dip": float(deep.mean() - rest.mean()) if len(deep) and len(rest) else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*62}\nE2b DIP RESCUE — {args.model} @ L{args.reference_layer}\n{'='*62}")
    print(f"  below {args.deep_fill:.0%} fill : n={summary['n_rest']:4d} acc={summary['acc_rest']}")
    print(f"  >= {args.deep_fill:.0%} natural : n={summary['n_deep']:4d} acc={summary['acc_deep']}")
    print(f"  >= {args.deep_fill:.0%} rescued : n={summary['n_rescued']:4d} acc={summary['acc_rescued']}")
    print(f"  dip (deep - rest) = {summary['dip']}")
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

    if args.mode == "e2b":
        run_e2b(args, model, tokenizer, pad_id, is_chat, letter_ids, out_dir)
        return

    evidence_db = load_evidence_db()
    probe_pool = build_probe_pool(evidence_db, args)
    print(f"  probes {len(probe_pool)}", flush=True)

    levels = args.levels[:2] if args.preflight else args.levels
    n_items = 1 if args.preflight else args.n_items

    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=8, headroom=16)
    rng = random.Random(args.seed)
    records = []
    turns_path = out_dir / "turns.csv"

    for idx in range(n_items):
        probe = probe_pool[idx % len(probe_pool)]
        conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        for prior in rng.sample(probe_pool, args.cold_start_cases):
            conv += [{"role": "user", "content": prior["case"]},
                     {"role": "assistant", "content": prior["gold"]}]
        conv += [{"role": "user", "content": probe["case"]}]

        text = render_prompt(tokenizer, conv, is_chat)
        if not guard.fits(text, used=0, index=idx):
            continue
        base_ids = tokenizer(text, return_tensors="pt").input_ids
        # one masked left-pad token, so sdpa receives an explicit mask the clamp can bias
        ids = torch.cat([torch.full((1, 1), pad_id, dtype=base_ids.dtype), base_ids],
                        dim=1).to(args.device)
        attn = torch.ones_like(ids)
        attn[0, 0] = 0

        start, end = locate_token_span(tokenizer, text, probe["case"])
        span = (start + 1, end + 1)  # +1 for the pad

        natural = measure_span_share(model, ids, span, args.reference_layer, attn)
        with torch.no_grad():
            pred_natural = score_forced_choice(
                model(ids, attention_mask=attn).logits, letter_ids)
        records.append({"item": idx, "level": "natural", "target_share": None,
                        "achieved_share": natural, "scale": 1.0,
                        "ctx_tokens": int(ids.shape[1]),
                        "fill": round(int(ids.shape[1]) / args.max_ctx, 4),
                        "gold": probe["gold"], "pred": pred_natural,
                        "correct": pred_natural == probe["gold"]})

        for target in levels:
            scale, achieved = solve_span_scale(
                model, ids, span=span, target_share=target,
                reference_layer=args.reference_layer, tol=1e-3, attention_mask=attn)
            with torch.no_grad(), SpanAttentionClamp(model, span=span, scale=scale):
                pred = score_forced_choice(model(ids, attention_mask=attn).logits, letter_ids)
            records.append({"item": idx, "level": f"{target:.2f}", "target_share": target,
                            "achieved_share": achieved, "scale": scale,
                            "ctx_tokens": int(ids.shape[1]),
                            "fill": round(int(ids.shape[1]) / args.max_ctx, 4),
                            "gold": probe["gold"], "pred": pred,
                            "correct": pred == probe["gold"]})
        pd.DataFrame(records).to_csv(turns_path, index=False)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{n_items} items, {len(records)} rows", flush=True)
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    by_level = (df.groupby("level")
                  .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                       achieved=("achieved_share", "mean"))
                  .reset_index().to_dict("records"))
    summary = {"model": args.model, "reference_layer": args.reference_layer,
               "levels": levels, "n_items": n_items, "n_rows": len(df),
               "overflow": guard.report(n_seen=n_items), "by_level": by_level}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*62}\nE2a MASS CLAMP — {args.model} @ L{args.reference_layer}\n{'='*62}")
    for row in by_level:
        print(f"  level {row['level']:>8s}  achieved={row['achieved']:.3f}  "
              f"n={row['n']:4d}  acc={row['accuracy']:.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
