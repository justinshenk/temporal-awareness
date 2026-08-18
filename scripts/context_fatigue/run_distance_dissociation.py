"""E1e — is the distance penalty about tokens or about turns?

E1 confounds the two: `back_20` puts the evidence 20 turns *and* ~1,600 tokens before the question.
A positional account (RoPE decay is the obvious candidate) predicts the penalty tracks **token**
distance; an interference account predicts it tracks the **number of intervening exchanges**. E1c/E1d
showed mass removal is sufficient for the penalty but mass restoration recovers only ~32% of it, so
the residual needs attributing, and this is the experiment that attributes it.

Design — a partial 2x2 in (gap turns) x (filler length), with the infeasible cell dropped:

    arm            gap turns   filler length   gap tokens
    local              0            —              0
    turns5_short       5         ~75 tok        ~400
    turns5_long        5        ~320 tok       ~1600
    turns20_short     20         ~75 tok       ~1600

Two contrasts fall out:

* ``turns5_long`` vs ``turns20_short`` — **matched gap tokens**, 5 turns against 20.
  A difference here means turn count matters beyond token distance (interference).
* ``turns5_short`` vs ``turns5_long`` — **matched gap turns**, ~400 tokens against ~1600.
  A difference here means token distance matters beyond turn count (positional).

Total context is equalized across arms by padding each one with *leading* short filler, so fill is
held fixed and only the composition of the gap differs. Scoring is forced-choice, as in E1c/E1d.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import render_prompt
from run_evidence_clamp import letter_token_ids, score_forced_choice

from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture
from src.probes.context_fatigue.attention_clamp import locate_token_span, span_share
from src.probes.context_fatigue.context_assembly import assemble_transcript
from src.probes.context_fatigue.ddxplus_cases import format_case_question, load_evidence_db
from run_distance_sweep import INTRO, MMLU_LABELS, REFERENT, format_mmlu, load_probe_pool

# arm -> (gap turns, filler bucket)
ARMS = {
    "local": (0, "short"),
    "turns5_short": (5, "short"),
    "turns5_long": (5, "long"),
    "turns20_short": (20, "short"),
}
BUCKETS = {"short": (60, 90), "long": (250, 450)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--reference-layer", type=int, default=24)
    p.add_argument("--target-tokens", type=int, default=3000,
                   help="total context every arm is padded to, so fill is matched")
    p.add_argument("--n-probes", type=int, default=192)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e1e_dissociation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def load_buckets(tokenizer):
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = {k: [] for k in BUCKETS}
    for row in ds:
        text = format_mmlu(row["question"], row["choices"])
        n = len(tokenizer.encode(text))
        for name, (lo, hi) in BUCKETS.items():
            if lo <= n <= hi:
                out[name].append({"text": text, "gold": MMLU_LABELS[row["answer"]], "n": n})
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None
    letter_ids = letter_token_ids(tokenizer)

    evidence_db = load_evidence_db()
    buckets = load_buckets(tokenizer)
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  filler short {len(buckets['short'])} | long {len(buckets['long'])} | "
          f"probes {len(probe_pool)}", flush=True)

    def n_tok(conv):
        return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))

    n_probes = 1 if args.preflight else args.n_probes
    records = []
    turns_path = out_dir / "turns.csv"

    for idx in range(n_probes):
        rng = random.Random(args.seed + idx)
        probe = probe_pool[rng.randrange(len(probe_pool))]
        question = format_case_question(probe["options"], args.n_options, referent=REFERENT)

        for arm, (gap_turns, bucket) in ARMS.items():
            gap = rng.sample(buckets[bucket], gap_turns) if gap_turns else []
            gap_conv = []
            for g in gap:
                gap_conv += [{"role": "user", "content": g["text"]},
                             {"role": "assistant", "content": g["gold"]}]

            # pad with leading short filler until the whole thing reaches target_tokens
            leading = []
            pool = rng.sample(buckets["short"], min(60, len(buckets["short"])))
            for f in pool:
                trial = ([{"role": "user", "content": INTRO},
                          {"role": "assistant", "content": "Understood."}]
                         + leading + [{"role": "user", "content": f["text"]},
                                      {"role": "assistant", "content": f["gold"]}])
                projected = (n_tok(trial + gap_conv)
                             + len(tokenizer.encode(probe["vignette"] + question)) + 24)
                if projected > args.target_tokens:
                    break
                leading += [{"role": "user", "content": f["text"]},
                            {"role": "assistant", "content": f["gold"]}]

            prior = ([{"role": "user", "content": INTRO},
                      {"role": "assistant", "content": "Understood."}]
                     + leading + gap_conv)
            built = assemble_transcript(prior, evidence=probe["vignette"], question=question,
                                        distance=gap_turns)
            text = render_prompt(tokenizer, built.turns, is_chat)
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            if ids.shape[1] + 24 > args.max_ctx:
                continue

            ev_span = locate_token_span(tokenizer, text, probe["vignette"])
            q_span = locate_token_span(tokenizer, text, question)
            capture = SelectiveAttentionCapture(model, [args.reference_layer])
            capture.enabled = True
            with torch.no_grad():
                logits = model(ids).logits
            capture.remove()
            attn = capture.captured[args.reference_layer]
            pred = score_forced_choice(logits, letter_ids)

            records.append({
                "probe": idx, "arm": arm, "gap_turns": gap_turns, "filler": bucket,
                "gap_tokens": q_span[0] - ev_span[1],
                "ctx_tokens": int(ids.shape[1]),
                "context_fill": round(int(ids.shape[1]) / args.max_ctx, 4),
                "evidence_share": span_share(attn, ev_span),
                "question_share": span_share(attn, q_span),
                "gold": probe["gold"], "pred": pred, "correct": pred == probe["gold"],
            })
            torch.cuda.empty_cache()
        pd.DataFrame(records).to_csv(turns_path, index=False)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n_probes} probes, {len(records)} rows", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    by = (df.groupby("arm").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                gap_tokens=("gap_tokens", "mean"),
                                ctx=("ctx_tokens", "mean"), fill=("context_fill", "mean"),
                                ev_share=("evidence_share", "mean"))
            .reset_index().to_dict("records"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model": args.model, "arms": {k: list(v) for k, v in ARMS.items()},
                   "target_tokens": args.target_tokens, "by_arm": by}, f, indent=2)

    print(f"\n{'='*74}\nE1e TOKENS vs TURNS — {args.model} @ L{args.reference_layer}\n{'='*74}")
    for r in by:
        print(f"  {r['arm']:14s} n={r['n']:4d} gap_tok={r['gap_tokens']:7.0f} "
              f"ctx={r['ctx']:6.0f} fill={r['fill']:.2f} ev_share={r['ev_share']:.4f} "
              f"acc={r['accuracy']:.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
