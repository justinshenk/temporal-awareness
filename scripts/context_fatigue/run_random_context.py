"""Does context rot lower accuracy once in-context learning can't compensate?

In the homogeneous DDXPlus stream accuracy stays flat/rises with context because
the model in-context-learns the single repeating task. This experiment removes
that crutch: each accumulated question is drawn **uniformly at random from all 57
MMLU subjects**, so the prior context carries *no predictive information* about
the next question (only the constant MCQ answer format is learnable, not the
task/domain). We contrast it with a **coherent** stream (one subject), where ICL
*can* help, on the same model with matched lengths.

Question: with ICL deterred, does per-case accuracy decay as the conversation
fills — i.e. is there genuine context rot in the *score*, not just in attention?

Multi-session pooling (a 4096-ctx session only holds ~15-25 short MCQs) gives the
per-case accuracy curve statistical power.
"""

import argparse
import gc
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import generate_with_entropy, render_prompt

LABELS = ["A", "B", "C", "D"]
INTRO = "Answer each multiple-choice question. Reply with only the letter of the correct option."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--modes", nargs="+", default=["random", "coherent"])
    p.add_argument("--coherent-subject", default="high_school_psychology")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--fill-target", type=float, default=0.88)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--n-sessions", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/random_context")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def format_q(q, choices):
    return (q + "\n" + "".join(f"{LABELS[i]}) {o}\n" for i, o in enumerate(choices))
            + "\nReply with only the letter (A, B, C, or D).")


def extract_letter(text):
    if not text:
        return None
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MMLU (all subjects)...")
    allds = load_dataset("cais/mmlu", "all", split="test")
    pool = [{"question": r["question"], "choices": [r["choices"][i] for i in range(4)],
             "gold": LABELS[r["answer"]], "subject": r["subject"]} for r in allds]
    coherent_pool = [q for q in pool if q["subject"] == args.coherent_subject]
    print(f"  {len(pool)} questions across {len({q['subject'] for q in pool})} subjects | "
          f"coherent subject '{args.coherent_subject}': {len(coherent_pool)}")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    def n_tokens(conv):
        return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))

    records = []
    for mode in args.modes:
        src = pool if mode == "random" else coherent_pool
        for session in range(args.n_sessions):
            rng = random.Random(args.seed + 1000 * session + (7 if mode == "random" else 0))
            order = list(range(len(src)))
            rng.shuffle(order)
            conv = [{"role": "user", "content": INTRO},
                    {"role": "assistant", "content": "Understood. I'll reply with only the letter."}]
            sess = []
            for pos, qi in enumerate(order):
                ctx = n_tokens(conv)
                if ctx / args.max_ctx > args.fill_target:
                    break
                q = src[qi]
                q_text = format_q(q["question"], q["choices"])
                # Near-full context, a long question would be RIGHT-truncated by the tokenizer —
                # losing its own options and scoring as a spurious error in exactly the top fill
                # bin. Skip anything that cannot fit whole (with generation headroom) instead.
                if ctx + len(tokenizer.encode(q_text)) + args.max_new + 16 > args.max_ctx:
                    continue
                conv.append({"role": "user", "content": q_text})
                resp, _, _, _ = generate_with_entropy(
                    model, tokenizer, render_prompt(tokenizer, conv, is_chat),
                    args.device, args.max_new, args.max_ctx)
                pred = extract_letter(resp)
                correct = pred == q["gold"] if pred else False
                records.append({"mode": mode, "session": session, "position": pos,
                                "context_fill": round(ctx / args.max_ctx, 4),
                                "subject": q["subject"], "gold": q["gold"], "pred": pred,
                                "correct": correct})
                conv.append({"role": "assistant", "content": (resp or "A")[:8]})
                sess.append(correct)
                torch.cuda.empty_cache()
            print(f"  [{mode}] session {session+1}/{args.n_sessions}: {len(sess)} q, "
                  f"acc={np.mean(sess):.2f}", flush=True)
            pd.DataFrame(records).to_csv(out_dir / "turns.csv", index=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    analyze(pd.DataFrame(records), args.model, out_dir)


def analyze(df, model_name, out_dir):
    print(f"\n{'='*72}\nACCURACY vs CONTEXT FILL — does rot lower the score when ICL is deterred?\n{'='*72}")
    print(f"model: {model_name}")
    summary = []
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    for mode in df["mode"].unique():
        m = df[df["mode"] == mode]
        n_total = len(m)
        slope = np.corrcoef(m["correct"], m["context_fill"])[0, 1] if m["context_fill"].std() > 0 else float("nan")
        print(f"\n[{mode}]  n={n_total}, overall acc={m['correct'].mean():.2f}, "
              f"corr(correct, fill)={slope:+.2f}")
        print(f"   {'fill':>10s} {'acc':>6s} {'n':>5s}")
        for lo, hi in bins:
            b = m[(m["context_fill"] >= lo) & (m["context_fill"] < hi)]
            if len(b):
                print(f"   {lo:>4.0%}-{hi:<4.0%} {b['correct'].mean():>6.2f} {len(b):>5d}")
                summary.append({"mode": mode, "fill_bin": f"{lo:.0%}-{hi:.0%}",
                                "accuracy": b["correct"].mean(), "n": len(b)})
        # early vs late within session (position), to compare conversation progression
        early = m[m["position"] < 5]["correct"].mean()
        late = m[m["position"] >= 5]["correct"].mean()
        print(f"   position <5 acc={early:.2f}  vs  >=5 acc={late:.2f}  (Δ={late-early:+.2f})")
    pd.DataFrame(summary).to_csv(out_dir / "accuracy_by_fill.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIf 'random' decays with fill (negative slope) while 'coherent' stays flat/positive,\n"
          f"that isolates genuine context rot in the score, normally masked by in-context learning.")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
