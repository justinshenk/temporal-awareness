"""
Mixed Task Fatigue Experiment

Randomly interleaves DDXPlus medical MCQ and MMLU math MCQ in a single
accumulating conversation. Tests whether fatigue effects (entropy collapse,
accuracy degradation) are task-specific or universal.

Hypotheses:
  - If ICL drives accuracy: medical accuracy should drop when pattern is broken
  - If entropy collapse is universal: both tasks should show it
  - If pattern matching is task-specific: math accuracy should be independent
"""

import json, re, ast, argparse, random, gc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "Answer the following questions."
OPTION_LABELS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--max-new", type=int, default=1000)
    p.add_argument("--fill-target", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/mixed_fatigue")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain", "likely",
    "unlikely", "somewhat", "arguably", "i guess", "it appears", "in my opinion",
    "hard to say", "difficult to determine",
]


# ── evidence helpers ────────────────────────────────────────────────────

def load_evidence_db(path):
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        vm = {vk: (vv.get("en", str(vv)) if isinstance(vv, dict) else str(vv))
              for vk, vv in info.get("value_meaning", {}).items()}
        db[code] = {
            "question": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "data_type": info.get("data_type", "B"),
            "value_meanings": vm,
        }
    return db


def decode_evidence(ev_str, evidence_db):
    evs = ast.literal_eval(ev_str)
    symptoms, antecedents = [], []
    grouped = {}
    for ev in evs:
        if "@" in ev:
            b, v = ev.split("@", 1)
            grouped.setdefault(b.strip().rstrip("_"), []).append(v.strip())
        else:
            grouped[ev] = []
    for code, vals in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        stmt = info["question"].replace("Do you have ", "Has ").replace("Are you ", "Is ").rstrip("?.")
        if info["data_type"] == "B":
            t = f"Yes — {stmt}"
        elif info["data_type"] == "M" and vals:
            dec = [info["value_meanings"].get(v, v) for v in vals if info["value_meanings"].get(v, v) != "NA"]
            t = f"{stmt}: {', '.join(dec)}" if dec else f"Yes — {stmt}"
        elif info["data_type"] == "C" and vals:
            t = f"{stmt}: {', '.join(vals)}"
        else:
            t = f"Yes — {stmt}"
        (antecedents if info["is_antecedent"] else symptoms).append(t)
    return symptoms, antecedents


def format_ddxplus_mcq(row, shuffled_names, evidence_db):
    age = row["AGE"]
    sex = "Male" if row["SEX"] == "M" else "Female"
    initial_ev = row["INITIAL_EVIDENCE"]
    chief = evidence_db.get(initial_ev, {}).get("question", initial_ev).replace("Do you have ", "").replace("?", "").strip()
    symptoms, antecedents = decode_evidence(row["EVIDENCES"], evidence_db)
    lines = [f"Patient: {age}-year-old {sex}", f"Chief complaint: {chief}"]
    if symptoms:
        lines.append("Symptoms:")
        lines.extend(f"  - {s}" for s in symptoms)
    if antecedents:
        lines.append("History:")
        lines.extend(f"  - {a}" for a in antecedents)
    lines.append("\nMost likely diagnosis:")
    lines.extend(f"{OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(shuffled_names[:5]))
    lines.append("\nAnswer:")
    return "\n".join(lines)


def format_math_mcq(question, choices):
    lines = [question]
    for i, opt in enumerate(choices):
        lines.append(f"{OPTION_LABELS[i]}) {opt}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


def extract_mcq_answer(text):
    text = text.strip().upper()
    if text and text[0] in "ABCDE":
        return text[0]
    m = re.search(r'\b([ABCDE])\b', text)
    return m.group(1) if m else None


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


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

    # ── Load medical cases ──────────────────────────────────────────
    print("Loading DDXPlus...")
    ds_med = load_dataset("aai530-group6/ddxplus", split="test")
    rng = random.Random(args.seed)

    med_items = []
    for i in range(len(ds_med)):
        row = ds_med[i]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        names = [d[0] for d in ddx[:5]]
        if row["PATHOLOGY"] in names:
            med_items.append(i)
    rng.shuffle(med_items)

    # ── Load science/psych questions (70%+ baseline) ─────────────────
    print("Loading MMLU science/psych...")
    math_items = []
    for subj in ["high_school_psychology", "college_biology",
                  "high_school_biology", "nutrition"]:
        ds_other = load_dataset("cais/mmlu", subj, split="test")
        for row in ds_other:
            math_items.append({
                "question": row["question"],
                "choices": [row["choices"][i] for i in range(4)],
                "gold": OPTION_LABELS[row["answer"]],
                "subject": subj,
            })
    rng.shuffle(math_items)

    # ── Build randomized mixed queue ────────────────────────────────
    queue = []
    med_idx, math_idx = 0, 0
    case_rng = random.Random(args.seed)

    # Prepare medical cases
    for i in range(min(200, len(med_items))):
        idx = med_items[i]
        row = ds_med[idx]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        opts = [d[0] for d in ddx[:5]]
        shuffled = list(enumerate(opts))
        case_rng.shuffle(shuffled)
        sn = [n for _, n in shuffled]
        gp = next(j for j, n in enumerate(sn) if n == row["PATHOLOGY"])
        gl = OPTION_LABELS[gp]
        text = format_ddxplus_mcq(row, sn, evidence_db)
        queue.append({"task": "medical", "text": text, "gold": gl,
                       "pathology": row["PATHOLOGY"]})

    # Prepare math cases
    for item in math_items[:200]:
        text = format_math_mcq(item["question"], item["choices"])
        queue.append({"task": "science", "text": text, "gold": item["gold"],
                       "subject": item["subject"]})

    # Shuffle everything together
    rng.shuffle(queue)

    print(f"Queue: {len(queue)} items "
          f"({sum(1 for q in queue if q['task']=='medical')} medical, "
          f"{sum(1 for q in queue if q['task']=='science')} science)")

    print(f"\n{'='*70}")
    print(f"Mixed Task Fatigue — {args.model}")
    print(f"Context: {args.max_ctx:,} | Fill target: {args.fill_target:.0%}")
    print(f"{'='*70}\n")

    # ── Run ─────────────────────────────────────────────────────────
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_turns = []

    for q_idx, q in enumerate(queue):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > args.fill_target:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        conversation.append({"role": "user", "content": q["text"]})

        response, ctx_len, entropies, logprobs = generate(conversation)
        if response is None:
            break

        pred = extract_mcq_answer(response)
        is_correct = pred == q["gold"] if pred else False
        mean_ent = float(np.mean(entropies)) if entropies else 0
        mean_lp = float(np.mean(logprobs)) if logprobs else 0
        hedges = detect_hedging(response)

        turn = {
            "turn": q_idx,
            "task": q["task"],
            "gold": q["gold"],
            "pred": pred,
            "correct": is_correct,
            "context_tokens": ctx_len,
            "context_fill": round(ctx_len / args.max_ctx, 4),
            "mean_entropy": mean_ent,
            "mean_logprob": mean_lp,
            "hedging_detected": len(hedges) > 0,
            "num_generated_tokens": len(entropies),
            "response_length": len(response),
            "response": response[:300],
        }
        if q["task"] == "medical":
            turn["pathology"] = q.get("pathology", "")
        else:
            turn["subject"] = q.get("subject", "")

        all_turns.append(turn)
        conversation.append({"role": "assistant", "content": response})

        if (q_idx + 1) % 5 == 0 or q_idx < 3:
            med_turns = [t for t in all_turns if t["task"] == "medical"]
            math_turns = [t for t in all_turns if t["task"] == "science"]
            med_acc = sum(t["correct"] for t in med_turns) / len(med_turns) * 100 if med_turns else 0
            math_acc = sum(t["correct"] for t in math_turns) / len(math_turns) * 100 if math_turns else 0
            print(f"  Q{q_idx+1:3d} [{q['task']:7s}] ctx={ctx_len/args.max_ctx:5.1%} | "
                  f"ent={mean_ent:.4f} pred={pred} gold={q['gold']} "
                  f"{'✓' if is_correct else '✗'} | "
                  f"med_acc={med_acc:.0f}% sci_acc={math_acc:.0f}%")

        torch.cuda.empty_cache()

    # ── Save & Summarize ────────────────────────────────────────────
    df = pd.DataFrame(all_turns)
    df.to_csv(out_dir / "mixed_turns.csv", index=False)

    with open(out_dir / "conversation.jsonl", "w") as f:
        for msg in conversation:
            f.write(json.dumps(msg) + "\n")

    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    print(f"\n{'='*70}")
    print("MIXED FATIGUE RESULTS")
    print(f"{'='*70}")

    for task in ["medical", "science"]:
        tdf = df[df["task"] == task]
        if tdf.empty:
            continue
        acc = tdf["correct"].mean() * 100
        print(f"\n{task.upper()} — {len(tdf)} questions, {acc:.1f}% overall")
        print(f"  {'Ctx Fill':12s} {'Accuracy':>10s} {'Entropy':>10s} {'Hedging':>10s} {'n':>5s}")
        for lo, hi in bins:
            sub = tdf[(tdf["context_fill"] >= lo) & (tdf["context_fill"] < hi)]
            if sub.empty:
                continue
            print(f"  {lo:.0%}-{hi:.0%}       "
                  f"{sub['correct'].mean()*100:9.1f}% "
                  f"{sub['mean_entropy'].mean():10.4f} "
                  f"{sub['hedging_detected'].mean()*100:9.1f}% "
                  f"{len(sub):5d}")

    # Cross-task comparison
    print(f"\nEntropy comparison:")
    for task in ["medical", "science"]:
        tdf = df[df["task"] == task]
        if len(tdf) < 4:
            continue
        early = tdf[tdf["context_fill"] < 0.3]["mean_entropy"].mean()
        late = tdf[tdf["context_fill"] >= 0.7]["mean_entropy"].mean()
        print(f"  {task}: early={early:.4f} late={late:.4f} delta={late-early:+.4f}")

    summary = {
        "model": args.model,
        "total_questions": len(df),
        "medical_count": len(df[df["task"] == "medical"]),
        "science_count": len(df[df["task"] == "science"]),
        "medical_accuracy": df[df["task"] == "medical"]["correct"].mean() if len(df[df["task"] == "medical"]) > 0 else 0,
        "science_accuracy": df[df["task"] == "science"]["correct"].mean() if len(df[df["task"] == "science"]) > 0 else 0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
