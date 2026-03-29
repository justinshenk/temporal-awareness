"""
DDXPlus MCQ Accumulation — Context Fatigue Experiment

Present patient profile + 5 candidate diagnoses (from the DDx).
Model picks the most likely diagnosis from the options.
All cases accumulate in one conversation until context fills (~92%).

This gives a tractable task (~70%+ baseline expected) while keeping
the clinical reasoning domain.
"""

import json, re, ast, argparse, random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── config ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--fill-target", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--out-dir", default="results/ddxplus_mcq")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reverse", action="store_true",
                   help="Reverse case order (control for question difficulty)")
    return p.parse_args()


SYSTEM_PROMPT = "You are a doctor."

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain", "likely",
    "unlikely", "somewhat", "arguably", "i guess", "it appears", "in my opinion",
    "hard to say", "difficult to determine", "suggests", "suggestive",
    "consider", "may be",
]
REFUSAL_PATTERNS = [
    r"i('m| am) (not able|unable|cannot)",
    r"i can('t|not)",
    r"as an ai",
    r"i('m| am) sorry",
    r"i apologize",
]


# ── evidence helpers ────────────────────────────────────────────────────

def load_evidence_db(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        value_meanings = {}
        for vk, vv in info.get("value_meaning", {}).items():
            if isinstance(vv, dict):
                value_meanings[vk] = vv.get("en", str(vv))
            else:
                value_meanings[vk] = str(vv)
        db[code] = {
            "question": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "data_type": info.get("data_type", "B"),
            "value_meanings": value_meanings,
        }
    return db


def decode_evidence(ev_str: str, evidence_db: dict) -> tuple[list[str], list[str]]:
    evs = ast.literal_eval(ev_str)
    symptoms = []
    antecedents = []

    grouped: dict[str, list[str]] = {}
    for ev in evs:
        if "@" in ev:
            base, val = ev.split("@", 1)
            base = base.strip().rstrip("_")
            val = val.strip()
            grouped.setdefault(base, []).append(val)
        else:
            grouped[ev] = []

    for code, values in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        q = info["question"]
        statement = q.replace("Do you have ", "Has ").replace("Are you ", "Is ")
        statement = statement.rstrip("?").rstrip(".")

        if info["data_type"] == "B":
            text = f"Yes — {statement}"
        elif info["data_type"] == "M" and values:
            decoded_vals = []
            for v in values:
                meaning = info["value_meanings"].get(v, v)
                if meaning and meaning != "NA":
                    decoded_vals.append(meaning)
            if decoded_vals:
                text = f"{statement}: {', '.join(decoded_vals)}"
            else:
                text = f"Yes — {statement}"
        elif info["data_type"] == "C" and values:
            text = f"{statement}: {', '.join(values)}"
        else:
            text = f"Yes — {statement}"

        if info["is_antecedent"]:
            antecedents.append(text)
        else:
            symptoms.append(text)

    return symptoms, antecedents


def format_case_mcq(age: int, sex: str, initial_ev: str,
                     evidence_str: str, evidence_db: dict,
                     options: list[str], n_options: int) -> str:
    sex_full = "Male" if sex == "M" else "Female"
    chief = evidence_db.get(initial_ev, {}).get("question", initial_ev)
    chief = chief.replace("Do you have ", "").replace("?", "").strip()

    symptoms, antecedents = decode_evidence(evidence_str, evidence_db)

    lines = [f"Patient: {age}-year-old {sex_full}"]
    lines.append(f"Chief complaint: {chief}")

    if symptoms:
        lines.append("Symptoms:")
        for s in symptoms:
            lines.append(f"  - {s}")
    if antecedents:
        lines.append("History:")
        for a in antecedents:
            lines.append(f"  - {a}")

    lines.append("\nMost likely diagnosis:")
    for i, opt in enumerate(options[:n_options]):
        lines.append(f"{OPTION_LABELS[i]}) {opt}")
    lines.append("\nAnswer:")

    return "\n".join(lines)


# ── helpers ─────────────────────────────────────────────────────────────

def extract_mcq_answer(text: str) -> str | None:
    text = text.strip().upper()
    if text and text[0] in "ABCDE":
        return text[0]
    m = re.search(r'\b([ABCDE])\b', text)
    return m.group(1) if m else None


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


def detect_refusal(t):
    return [p for p in REFUSAL_PATTERNS if re.search(p, t.lower())]


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_db = load_evidence_db("release_evidences.json")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def count_tokens(conv):
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))

    def generate(conv):
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]
        ctx_len = input_ids.shape[1]
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        if eff_max < 5:
            return None, ctx_len, [], []

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=eff_max, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id)

        new_tokens = out.sequences[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entropies, logprobs = [], []
        for si, score in enumerate(out.scores):
            if si >= len(new_tokens):
                break
            logp = torch.log_softmax(score.float(), dim=-1)
            probs = torch.softmax(score.float(), dim=-1)
            entropies.append(-(probs * logp).sum(dim=-1).item())
            logprobs.append(logp[0, new_tokens[si].item()].item())

        return response, ctx_len, entropies, logprobs

    # Load cases
    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")

    # Filter to cases where gold is in top-N DDx
    rng = random.Random(args.seed)
    valid_indices = []
    for i in range(len(ds)):
        row = ds[i]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        names = [d[0] for d in ddx[:args.n_options]]
        if row["PATHOLOGY"] in names:
            valid_indices.append(i)

    rng.shuffle(valid_indices)
    if args.max_cases:
        valid_indices = valid_indices[:args.max_cases]
    if args.reverse:
        valid_indices = valid_indices[::-1]

    print(f"Valid cases (gold in top-{args.n_options}): {len(valid_indices)}"
          f"{' [REVERSED]' if args.reverse else ''}")

    print(f"\n{'='*70}")
    print(f"DDXPlus MCQ — {args.model}")
    print(f"Context: {args.max_ctx:,} | Fill target: {args.fill_target:.0%} | "
          f"Options: {args.n_options}")
    print(f"{'='*70}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_turns = []
    correct_total = 0
    answered_total = 0

    for case_num, idx in enumerate(valid_indices):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > args.fill_target:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        row = ds[idx]
        pathology = row["PATHOLOGY"]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        option_names = [d[0] for d in ddx[:args.n_options]]

        # Shuffle options so gold isn't always in same position
        shuffled = list(enumerate(option_names))
        rng.shuffle(shuffled)
        shuffled_names = [name for _, name in shuffled]

        # Find gold position
        gold_pos = next(i for i, name in enumerate(shuffled_names) if name == pathology)
        gold_letter = OPTION_LABELS[gold_pos]

        case_text = format_case_mcq(
            row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
            row["EVIDENCES"], evidence_db, shuffled_names, args.n_options)

        conversation.append({"role": "user", "content": case_text})

        response, ctx_len, entropies, logprobs = generate(conversation)
        if response is None:
            print(f"  >>> Context exhausted at case {case_num}")
            break

        pred = extract_mcq_answer(response)
        is_correct = pred == gold_letter if pred else False
        if is_correct:
            correct_total += 1
        answered_total += 1

        mean_ent = float(np.mean(entropies)) if entropies else 0
        mean_lp = float(np.mean(logprobs)) if logprobs else 0
        hedges = detect_hedging(response)

        try:
            pred_name = shuffled_names[OPTION_LABELS.index(pred)] if pred and pred in OPTION_LABELS[:args.n_options] else None
        except (ValueError, IndexError):
            pred_name = None

        turn_data = {
            "turn": case_num,
            "dataset_idx": idx,
            "age": row["AGE"],
            "sex": row["SEX"],
            "pathology": pathology,
            "gold_letter": gold_letter,
            "pred_letter": pred,
            "pred_name": pred_name,
            "correct": is_correct,
            "context_tokens": ctx_len,
            "context_fill": round(ctx_len / args.max_ctx, 4),
            "mean_entropy": mean_ent,
            "mean_logprob": mean_lp,
            "median_entropy": float(np.median(entropies)) if entropies else 0,
            "std_entropy": float(np.std(entropies)) if entropies else 0,
            "hedging_detected": len(hedges) > 0,
            "num_hedge_words": len(hedges),
            "hedge_words": hedges,
            "refusal_detected": len(detect_refusal(response)) > 0,
            "num_generated_tokens": len(entropies),
            "response": response[:200],
        }
        all_turns.append(turn_data)
        conversation.append({"role": "assistant", "content": response})

        running_acc = correct_total / answered_total * 100
        h_flag = f" [{len(hedges)}hw]" if hedges else ""
        if (case_num + 1) % 25 == 0 or case_num < 5:
            print(f"  C{case_num+1:4d} ctx={ctx_len:6,} ({ctx_len/args.max_ctx:5.1%}) | "
                  f"ent={mean_ent:.4f}{h_flag} | "
                  f"pred={pred}({str(pred_name)[:20] if pred_name else '?':20s}) "
                  f"gold={gold_letter}({pathology[:20]:20s}) "
                  f"{'✓' if is_correct else '✗'} | acc={running_acc:.1f}%")

        torch.cuda.empty_cache()

    # ── save outputs ────────────────────────────────────────────────────
    turn_df = pd.DataFrame(all_turns)
    turn_df["hedge_words"] = turn_df["hedge_words"].apply(lambda x: "|".join(x))
    turn_df.insert(0, "model", args.model)
    turn_df.insert(1, "max_context_tokens", args.max_ctx)
    turn_df.insert(2, "experiment", "ddxplus_mcq")

    turn_df.to_csv(out_dir / "ddxplus_mcq_turns.csv", index=False)
    turn_df.to_parquet(out_dir / "ddxplus_mcq_turns.parquet", index=False)

    with open(out_dir / "conversation.jsonl", "w") as f:
        for msg in conversation:
            f.write(json.dumps(msg) + "\n")

    n_total = len(all_turns)
    n_correct = sum(1 for t in all_turns if t["correct"])

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_stats = {}
    for lo, hi in bins:
        in_bin = [t for t in all_turns if lo <= t["context_fill"] < hi]
        if in_bin:
            bc = sum(1 for t in in_bin if t["correct"])
            bin_stats[f"{lo:.0%}-{hi:.0%}"] = {
                "count": len(in_bin),
                "correct": bc,
                "accuracy": bc / len(in_bin),
                "mean_entropy": float(np.mean([t["mean_entropy"] for t in in_bin])),
                "hedging_rate": sum(1 for t in in_bin if t["hedging_detected"]) / len(in_bin),
            }

    summary = {
        "model": args.model,
        "max_context": args.max_ctx,
        "seed": args.seed,
        "n_options": args.n_options,
        "cases_completed": n_total,
        "correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0,
        "hedging_rate": sum(1 for t in all_turns if t["hedging_detected"]) / n_total if n_total else 0,
        "mean_entropy": float(np.mean([t["mean_entropy"] for t in all_turns])),
        "mean_logprob": float(np.mean([t["mean_logprob"] for t in all_turns])),
        "accuracy_by_context_bin": bin_stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUMMARY — {n_total} cases, {n_correct} correct = "
          f"{n_correct/n_total*100:.1f}%")
    print(f"{'='*70}")
    print(f"Hedging rate: {summary['hedging_rate']:.1%}")
    print(f"Mean entropy: {summary['mean_entropy']:.4f}")
    print(f"\nAccuracy by context fill:")
    for bin_name, stats in bin_stats.items():
        print(f"  {bin_name:10s}: {stats['accuracy']:5.1%} "
              f"({stats['correct']}/{stats['count']}) "
              f"ent={stats['mean_entropy']:.4f} "
              f"hedge={stats['hedging_rate']:.1%}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
