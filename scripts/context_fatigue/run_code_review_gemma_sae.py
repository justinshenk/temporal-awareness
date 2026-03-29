"""
Code Review Fatigue + SAE — Gemma 2 9B IT

Progressive code review tasks across Python snippets.
Each snippet: explain → find bug → suggest fix.
Later tasks reference earlier code to test context retrieval.

Tracks entropy, hedging, response quality, and SAE features at layer 20.
"""

import json, re, random, gc, argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--out-dir", default="results/code_review_gemma_sae")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain",
    "likely", "unlikely", "somewhat", "arguably", "i guess",
]

CODE_SNIPPETS = [
    {
        "name": "fibonacci",
        "code": '''def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)''',
        "bug": "Exponential time complexity — no memoization. Will hang for n > 35.",
    },
    {
        "name": "find_duplicates",
        "code": '''def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)''',
        "bug": "No bug per se, but returns duplicates in arbitrary order (set). Could cause non-deterministic behavior in tests.",
    },
    {
        "name": "bank_account",
        "code": '''class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount
        return self.balance

    def transfer(self, other, amount):
        self.withdraw(amount)
        other.deposit(amount)''',
        "bug": "No validation — withdraw allows negative balance, deposit accepts negative amounts, transfer not atomic (if deposit fails after withdraw, money is lost).",
    },
    {
        "name": "csv_parser",
        "code": '''def parse_csv(text):
    rows = []
    for line in text.split("\\n"):
        if line.strip():
            rows.append(line.split(","))
    return rows

def get_column(rows, col_idx):
    return [row[col_idx] for row in rows]''',
        "bug": "Doesn't handle quoted fields with commas. get_column will IndexError if any row is shorter than col_idx.",
    },
    {
        "name": "binary_search",
        "code": '''def binary_search(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1''',
        "bug": "Off-by-one: uses exclusive high bound but returns -1 even when target is at the boundary. Also integer overflow risk in (low + high) // 2 for very large arrays.",
    },
]


def build_tasks():
    """Build progressive code review tasks."""
    tasks = []

    for i, snippet in enumerate(CODE_SNIPPETS):
        # Task 1: Explain
        tasks.append({
            "type": "explain",
            "difficulty": "easy",
            "snippet_idx": i,
            "prompt": f"Review this Python code and explain what it does:\n\n```python\n{snippet['code']}\n```",
            "references_prior": False,
        })

        # Task 2: Find bug
        tasks.append({
            "type": "find_bug",
            "difficulty": "medium",
            "snippet_idx": i,
            "prompt": f"Can you identify any bugs or issues in the `{snippet['name']}` code I just showed you?",
            "references_prior": True,
        })

        # Task 3: Suggest fix
        tasks.append({
            "type": "suggest_fix",
            "difficulty": "hard",
            "snippet_idx": i,
            "prompt": f"How would you fix the issues you found in `{snippet['name']}`? Describe your approach.",
            "references_prior": True,
        })

    # Cross-reference tasks (reference multiple earlier snippets)
    tasks.append({
        "type": "compare",
        "difficulty": "hard",
        "snippet_idx": -1,
        "prompt": "Compare the error handling approach in the BankAccount class versus the csv_parser. Which is more robust and why?",
        "references_prior": True,
    })
    tasks.append({
        "type": "reason",
        "difficulty": "hard",
        "snippet_idx": -1,
        "prompt": "If I wanted to use the binary_search function to search through the output of find_duplicates, what would I need to do first and why?",
        "references_prior": True,
    })
    tasks.append({
        "type": "summarize",
        "difficulty": "medium",
        "snippet_idx": -1,
        "prompt": "Summarize all the bugs and issues you've found across all five code snippets we reviewed.",
        "references_prior": True,
    })

    return tasks


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


def check_quality(response):
    r = response.lower()
    return {
        "has_code": "```" in response or "def " in response or "return " in response,
        "has_specific": bool(re.search(
            r'\b(bug|error|fix|issue|wrong|incorrect|should be|instead of|overflow|negative|validation)\b', r)),
        "references_prior": bool(re.search(
            r'\b(earlier|previous|before|originally|fibonacci|find_duplicates|bank|csv|binary)\b', r)),
        "is_generic": r.startswith(("sure", "certainly", "of course", "great question")),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    print("Loading IT SAE (layer 20)...")
    sae = SAE.from_pretrained(release="gemma-scope-9b-it-res",
                               sae_id="layer_20/width_131k/average_l0_81")
    sae = sae.to(args.device).eval()
    print(f"  SAE: d_sae={sae.cfg.d_sae}")

    captured = {}
    def hook_fn(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().float()
    hook = model.model.layers[20].register_forward_hook(hook_fn)

    def count_tokens(conv):
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))

    def generate(conv):
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]
        ctx_len = input_ids.shape[1]
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        if eff_max < 10:
            return None, ctx_len, [], []
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=eff_max, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        new_tokens = out.sequences[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        entropies, logprobs = [], []
        for si, score in enumerate(out.scores):
            if si >= len(new_tokens): break
            logp = torch.log_softmax(score.float(), dim=-1)
            probs = torch.softmax(score.float(), dim=-1)
            entropies.append(-(probs * logp).sum(dim=-1).item())
            logprobs.append(logp[0, new_tokens[si].item()].item())
        return response, ctx_len, entropies, logprobs

    tasks = build_tasks()
    print(f"\n{'='*70}")
    print(f"Code Review Fatigue — {args.model}")
    print(f"{len(tasks)} tasks, {len(CODE_SNIPPETS)} snippets")
    print(f"{'='*70}\n")

    # Gemma: no system role
    conversation = [
        {"role": "user", "content": "You are a senior Python developer doing code review. Explain issues clearly in plain English."},
        {"role": "model", "content": "Understood. I'll review code and explain issues clearly."},
    ]
    all_turns = []
    all_sae_feats = []

    for ti, task in enumerate(tasks):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > 0.88:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        conversation.append({"role": "user", "content": task["prompt"]})

        response, ctx_len, entropies, logprobs = generate(conversation)
        if response is None:
            break

        # SAE
        act = captured["act"].to(args.device)
        with torch.no_grad():
            sae_feat = sae.encode(act.unsqueeze(0))[0].cpu()
        all_sae_feats.append(sae_feat)

        mean_ent = float(np.mean(entropies)) if entropies else 0
        hedges = detect_hedging(response)
        quality = check_quality(response)

        turn = {
            "turn": ti,
            "task_type": task["type"],
            "difficulty": task["difficulty"],
            "snippet_idx": task["snippet_idx"],
            "references_prior": task["references_prior"],
            "context_tokens": ctx_len,
            "context_fill": round(ctx_len / args.max_ctx, 4),
            "mean_entropy": mean_ent,
            "hedging_detected": len(hedges) > 0,
            "num_hedge_words": len(hedges),
            **quality,
            "num_sae_active": int((sae_feat > 0).sum().item()),
            "response_length": len(response),
            "response": response[:400],
        }
        all_turns.append(turn)

        flags = []
        if quality["has_code"]: flags.append("CODE")
        if quality["has_specific"]: flags.append("SPEC")
        if quality["references_prior"]: flags.append("REF")
        if quality["is_generic"]: flags.append("GEN")

        print(f"  T{ti:2d} {task['type']:12s} ctx={ctx_len/args.max_ctx:5.1%} | "
              f"ent={mean_ent:.3f} sae={int((sae_feat>0).sum()):3d} "
              f"[{','.join(flags)}]")

        conversation.append({"role": "model", "content": response})
        torch.cuda.empty_cache()

    hook.remove()

    # ── Save & Analyze ──────────────────────────────────────────────
    df = pd.DataFrame(all_turns)
    df.to_csv(out_dir / "turns.csv", index=False)

    sae_tensor = torch.stack(all_sae_feats) if all_sae_feats else torch.zeros(0)
    torch.save({"sae_features": sae_tensor, "turns": all_turns}, out_dir / "sae_data.pt")

    n = len(df)
    print(f"\n{'='*70}")
    print(f"RESULTS — {n} turns")
    print(f"{'='*70}")

    # By task type
    print(f"\nBy task type:")
    for tt in ["explain", "find_bug", "suggest_fix", "compare", "reason", "summarize"]:
        sub = df[df["task_type"] == tt]
        if sub.empty: continue
        print(f"  {tt:15s}: ent={sub['mean_entropy'].mean():.4f} "
              f"specific={sub['has_specific'].mean()*100:.0f}% "
              f"refs_prior={sub['references_prior_y'].mean()*100:.0f}% " if 'references_prior_y' in sub.columns else
              f"  {tt:15s}: ent={sub['mean_entropy'].mean():.4f} "
              f"specific={sub['has_specific'].mean()*100:.0f}% "
              f"n={len(sub)}")

    # Entropy trend
    print(f"\nEntropy by context fill:")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        sub = df[(df["context_fill"] >= lo) & (df["context_fill"] < hi)]
        if sub.empty: continue
        print(f"  {lo:.0%}-{hi:.0%}: ent={sub['mean_entropy'].mean():.4f} "
              f"sae_active={sub['num_sae_active'].mean():.0f} "
              f"specific={sub['has_specific'].mean()*100:.0f}% n={len(sub)}")

    # SAE feature drift
    if len(all_sae_feats) >= 4:
        early = sae_tensor[:3]
        late = sae_tensor[-3:]
        early_act = (early > 0).float().mean(0)
        late_act = (late > 0).float().mean(0)
        diff = (late_act - early_act).abs()
        topk = diff.topk(10)
        print(f"\nTop 10 SAE features that change most (early vs late):")
        for val, idx in zip(topk.values, topk.indices):
            e = early_act[idx].item()
            l = late_act[idx].item()
            d = "↑" if l > e else "↓"
            print(f"  F{idx.item():6d}: early={e:.2f} late={l:.2f} {d}")

    with open(out_dir / "conversation.jsonl", "w") as f:
        for msg in conversation:
            f.write(json.dumps(msg) + "\n")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
