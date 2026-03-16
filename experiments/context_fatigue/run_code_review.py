"""
Code Review Accumulation + Activation Drift

Single conversation accumulating code review tasks (explain, reason, modify,
find_bug, review). Tracks entropy, logprobs, hedging, and response quality
as context fills.

Optionally runs twice (code visible vs code stripped) to measure activation
drift between conditions.

Outputs a single merged dataset (CSV + Parquet).
"""

import json, re, gc, argparse
from pathlib import Path
import torch, numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--max-tasks", type=int, default=None,
                   help="Limit number of tasks (for testing)")
    p.add_argument("--data", default="../runpod_data/RQ3/experiment/data/code_review_tasks_v2.json")
    p.add_argument("--out-dir", default="results_code_review")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compare-drift", action="store_true",
                   help="Run twice (code visible vs stripped) and compare activations")
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are a senior Python developer doing code review. "
    "Do not include code in your responses. "
    "Explain everything in plain English only."
)

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain", "likely",
    "unlikely", "somewhat", "arguably", "i guess", "it appears", "in my opinion",
    "hard to say", "difficult to determine", "suggests", "suggestive",
    "consider", "may be",
]


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


def strip_code(text):
    text = re.sub(r'```[\s\S]*?```', '[code omitted]', text)
    text = re.sub(
        r'(def \w+\(.*?\):.*?)(?=\n\n|\Z)', '[code omitted]',
        text, flags=re.DOTALL,
    )
    return text


def check_quality(response):
    r = response.lower()
    return {
        "has_code": '```' in response or 'def ' in response or 'return ' in response,
        "has_specific": bool(re.search(
            r'\b(bug|error|fix|issue|wrong|incorrect|should be|instead of)\b', r)),
        "references_prior": bool(re.search(
            r'\b(earlier|previous|before|originally|we changed|we modified|'
            r'you asked|I modified)\b', r)),
        "is_generic": r.startswith(('sure', 'certainly', 'of course', 'great question')),
    }


class Runner:
    def __init__(self, args):
        self.args = args
        print(f"Loading {args.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device,
            attn_implementation="eager",
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # activation hooks (for drift comparison)
        self.activation_store = {}
        self.hooks = []
        if args.compare_drift:
            for i, layer in enumerate(self.model.model.layers):
                self.hooks.append(layer.register_forward_hook(self._make_hook(i)))
            print(f"Registered {len(self.hooks)} activation hooks.")
        print("Ready.\n")

    def _make_hook(self, layer_idx):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.activation_store[layer_idx] = h[:, -1, :].detach().cpu().float()
        return fn

    def count_tokens(self, conv):
        return len(self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)))

    def generate(self, conv):
        self.activation_store.clear()
        input_text = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to(self.args.device)
        ctx_len = input_ids.shape[1]
        eff_max = min(self.args.max_new, self.args.max_ctx - ctx_len - 1)
        if eff_max < 10:
            return None, ctx_len, [], [], {}

        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=eff_max, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id)

        new_tokens = out.sequences[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entropies, logprobs = [], []
        for si, score in enumerate(out.scores):
            if si >= len(new_tokens):
                break
            logp = torch.log_softmax(score.float(), dim=-1)
            probs = torch.softmax(score.float(), dim=-1)
            entropies.append(-(probs * logp).sum(dim=-1).item())
            logprobs.append(logp[0, new_tokens[si].item()].item())

        vecs = {li: self.activation_store[li].squeeze().clone()
                for li in self.activation_store}

        return response, ctx_len, entropies, logprobs, vecs

    def run_conversation(self, tasks, do_strip_code, label):
        print(f"\n{'='*70}")
        print(f"RUN: {label}")
        print(f"Context: {self.args.max_ctx:,} | Tasks: {len(tasks)}")
        print(f"{'='*70}\n")

        conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        turns = []

        for ti, task in enumerate(tasks):
            ctx_now = self.count_tokens(conv)
            if ctx_now / self.args.max_ctx > 0.85:
                print(f"\n>>> Context {ctx_now/self.args.max_ctx:.0%} full — stopping.")
                break

            conv.append({"role": "user", "content": task["prompt"]})
            response, ctx_len, entropies, logprobs, vecs = self.generate(conv)

            if response is None:
                print(f"  >>> Context exhausted at task {ti}")
                conv.pop()
                break

            mean_ent = float(np.mean(entropies)) if entropies else 0
            mean_lp = float(np.mean(logprobs)) if logprobs else 0
            hedges = detect_hedging(response)
            fill = ctx_len / self.args.max_ctx
            quality = check_quality(response)

            turn_data = {
                "turn": ti,
                "task_idx": ti,
                "turn_type": task["type"],
                "difficulty": task["difficulty"],
                "is_followup": task.get("is_followup", False),
                "context_tokens": ctx_len,
                "context_fill": round(fill, 4),
                "mean_entropy": mean_ent,
                "mean_logprob": mean_lp,
                "median_entropy": float(np.median(entropies)) if entropies else 0,
                "std_entropy": float(np.std(entropies)) if entropies else 0,
                "hedging_detected": len(hedges) > 0,
                "num_hedge_words": len(hedges),
                "hedge_words": hedges,
                "num_generated_tokens": len(entropies),
                **quality,
                "response_preview": response[:200],
                "_vecs": vecs,
            }
            turns.append(turn_data)

            h_flag = f" [{len(hedges)}hw]" if hedges else ""
            flags = []
            if quality["has_code"]:
                flags.append("CODE")
            if quality["has_specific"]:
                flags.append("SPECIFIC")
            if quality["references_prior"]:
                flags.append("RECALLS")
            if quality["is_generic"]:
                flags.append("GENERIC")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            print(f"T{ti:>2} {task['type']:<12} ctx={ctx_len:,} ({fill:.0%}) | "
                  f"ent={mean_ent:.4f}{h_flag}{flag_str}")
            print(f"    \"{response[:90]}...\"")

            assistant_content = strip_code(response) if do_strip_code else response
            conv.append({"role": "assistant", "content": assistant_content})
            torch.cuda.empty_cache()

        return turns, conv

    def cleanup(self):
        for h in self.hooks:
            h.remove()


def build_turns_df(turns, model_id, max_ctx, condition="default"):
    rows = []
    for t in turns:
        row = {k: v for k, v in t.items() if k != "_vecs"}
        row["hedge_words"] = "|".join(row["hedge_words"])
        row["model"] = model_id
        row["max_context_tokens"] = max_ctx
        row["condition"] = condition
        rows.append(row)
    return pd.DataFrame(rows)


def build_drift_df(turns_v1, turns_v3, model_id):
    n = min(len(turns_v1), len(turns_v3))
    rows = []
    for i in range(n):
        r1, r3 = turns_v1[i], turns_v3[i]
        cos_sims, norms_1, norms_3 = [], [], []
        for li in r1["_vecs"]:
            if li in r3["_vecs"]:
                v1, v3 = r1["_vecs"][li], r3["_vecs"][li]
                cos = torch.nn.functional.cosine_similarity(
                    v1.unsqueeze(0), v3.unsqueeze(0)).item()
                cos_sims.append(cos)
                norms_1.append(v1.norm().item())
                norms_3.append(v3.norm().item())
        mc = float(np.mean(cos_sims)) if cos_sims else 0
        dn = float(np.mean(norms_3) - np.mean(norms_1)) if norms_1 else 0

        rows.append({
            "model": model_id,
            "turn": i,
            "turn_type": r1["turn_type"],
            "context_fill": r1["context_fill"],
            "v1_entropy": r1["mean_entropy"],
            "v3_entropy": r3["mean_entropy"],
            "cosine_similarity": mc,
            "delta_norm": dn,
        })

        print(f"T{i:>2} {r1['turn_type']:<12} fill={r1['context_fill']:.0%} | "
              f"v1_ent={r1['mean_entropy']:.4f} v3_ent={r3['mean_entropy']:.4f} | "
              f"cos={mc:.4f} dnorm={dn:+.1f}")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        tasks = json.load(f)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    runner = Runner(args)

    if args.compare_drift:
        # Run 1: code visible
        turns_v1, conv_v1 = runner.run_conversation(
            tasks, do_strip_code=False, label="V1: code visible")
        gc.collect()
        torch.cuda.empty_cache()

        # Run 2: code stripped
        turns_v3, conv_v3 = runner.run_conversation(
            tasks, do_strip_code=True, label="V3: code stripped")
        gc.collect()
        torch.cuda.empty_cache()

        # Turns dataset — both conditions stacked
        df_v1 = build_turns_df(turns_v1, args.model, args.max_ctx, "code_visible")
        df_v3 = build_turns_df(turns_v3, args.model, args.max_ctx, "code_stripped")
        df_turns = pd.concat([df_v1, df_v3], ignore_index=True)

        # Drift dataset
        print(f"\n{'='*70}")
        print(f"ACTIVATION DRIFT: V1 vs V3")
        print(f"{'='*70}\n")
        df_drift = build_drift_df(turns_v1, turns_v3, args.model)
        df_drift.to_csv(out_dir / "code_review_activation_drift.csv", index=False)
        df_drift.to_parquet(out_dir / "code_review_activation_drift.parquet", index=False)
        print(f"\nSaved drift: {len(df_drift)} rows")

        # Save both conversations
        for label, conv in [("v1_code_visible", conv_v1), ("v3_code_stripped", conv_v3)]:
            with open(out_dir / f"conversation_{label}.jsonl", "w") as f:
                for msg in conv:
                    f.write(json.dumps(msg) + "\n")

        all_turns = turns_v1 + turns_v3
    else:
        # Single run with code stripped (standard)
        turns, conv = runner.run_conversation(
            tasks, do_strip_code=True, label="Code review (code stripped)")

        df_turns = build_turns_df(turns, args.model, args.max_ctx, "code_stripped")

        with open(out_dir / "conversation.jsonl", "w") as f:
            for msg in conv:
                f.write(json.dumps(msg) + "\n")

        all_turns = turns

    # Save turns dataset
    df_turns.to_csv(out_dir / "code_review_turns.csv", index=False)
    df_turns.to_parquet(out_dir / "code_review_turns.parquet", index=False)

    # Summary
    summary = {
        "model": args.model,
        "max_context": args.max_ctx,
        "tasks_total": len(tasks),
        "turns_completed": len(all_turns),
        "hedging_rate": (sum(1 for t in all_turns if t["hedging_detected"])
                         / len(all_turns)) if all_turns else 0,
        "mean_entropy": float(np.mean([t["mean_entropy"] for t in all_turns])),
        "mean_logprob": float(np.mean([t["mean_logprob"] for t in all_turns])),
        "compare_drift": args.compare_drift,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {summary['turns_completed']} turns completed of {summary['tasks_total']} tasks")
    print(f"{'='*70}")
    print(f"Hedging rate: {summary['hedging_rate']:.1%}")
    print(f"Mean entropy: {summary['mean_entropy']:.4f}")
    print(f"Mean logprob: {summary['mean_logprob']:.4f}")
    print(f"\nSaved to {out_dir}/")
    print(f"  code_review_turns.csv  ({len(df_turns)} rows x {len(df_turns.columns)} cols)")

    runner.cleanup()


if __name__ == "__main__":
    main()
