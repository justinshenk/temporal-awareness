"""
MedQA Interactive Accumulation

For each MedQA case:
  1. Present patient demographics + chief complaint only
  2. Model can ask for more info (labs, imaging, vitals, history, etc.)
  3. We provide whatever they ask for from the case data
  4. Once model gives a concrete answer, move on
  5. All cases accumulate in one conversation until context fills

Tracks entropy, logprobs, hedging per turn as context grows.
Outputs a single merged dataset (turns + case-level info).
"""

import json, re, gc, argparse
from pathlib import Path
import torch, numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── config ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--max-followups", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None,
                   help="Limit number of cases (for testing)")
    p.add_argument("--data", default="../runpod_data/RQ3/experiment/data/medqa_rich_subset.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="cuda")
    return p.parse_args()

SYSTEM_PROMPT = (
    "You are a doctor. You will be presented with patient cases. "
    "You may ask for labs, imaging, vitals, history, or exam findings. "
    "When ready, state your diagnosis and recommended next step."
)

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


# ── detection helpers ───────────────────────────────────────────────────

def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]

def detect_refusal(t):
    return [p for p in REFUSAL_PATTERNS if re.search(p, t.lower())]

def has_concrete_answer(text):
    text_l = text.lower()
    asking = any(p in text_l for p in [
        "would you like", "can you provide", "could you provide", "do you have",
        "i would need", "i need more", "please provide", "can i get",
        "what are the", "are there any", "could i see", "i'd like to",
        "let me ask", "i would like to know", "what does the", "what is the",
        "can you tell", "is there", "any history of", "what were",
        "i recommend the following tests", "i suggest we order",
        "let's get", "we should order", "i'd recommend ordering",
    ])
    diagnosing = any(p in text_l for p in [
        "diagnosis is", "diagnosed with", "consistent with", "this is",
        "the patient has", "i believe this is", "most likely",
        "i would treat", "treatment is", "recommend", "administer",
        "start the patient on", "prescribe", "the condition is",
        "this presentation is", "suggestive of", "indicates",
    ])
    return diagnosing and not asking


# ── case parsing ────────────────────────────────────────────────────────

def extract_chief_complaint(case):
    text = case["question"]
    sentences = re.split(r'(?<=[a-z])\.\s+(?=[A-Z])', text)
    first = sentences[0].strip()
    if len(first) < 80 and len(sentences) > 1:
        first = first + ". " + sentences[1].strip()
    for cutoff in [
        "Past medical", "past medical", "PMH", "Current medications",
        "His vital", "Her vital", "Vital signs", "vital signs",
        "On physical", "Physical exam", "Laboratory", "She has no",
        "He has no significant", "No significant",
    ]:
        idx = first.find(cutoff)
        if idx > 30:
            first = first[:idx].strip().rstrip(',').rstrip('.')
            break
    return first + "."

def build_info_response(case, model_request):
    full_text = case["question"]
    req = model_request.lower()
    parts = []

    wants_labs = any(w in req for w in [
        "lab", "blood", "cbc", "bmp", "cmp", "electrolyte", "sodium",
        "potassium", "creatinine", "bun", "glucose", "hemoglobin",
        "hematocrit", "wbc", "platelet", "bilirubin", "ast", "alt",
        "alkaline", "albumin", "troponin", "inr", "pt ", "ptt", "calcium",
    ])
    wants_vitals = any(w in req for w in [
        "vital", "temperature", "blood pressure", "pulse", "heart rate",
        "respiratory", "oxygen", "bp", "temp",
    ])
    wants_imaging = any(w in req for w in [
        "imaging", "x-ray", "xray", "ct ", "mri", "ultrasound", "echo",
        "ecg", "ekg", "chest", "radiograph", "scan",
    ])
    wants_exam = any(w in req for w in [
        "exam", "physical", "auscult", "palpat", "inspect", "finding",
    ])
    wants_history = any(w in req for w in [
        "history", "past medical", "pmh", "medication", "med ",
        "surgical", "family", "social", "allerg",
    ])
    wants_all = any(w in req for w in [
        "all", "everything", "full", "complete", "more information",
        "additional",
    ])

    lines = full_text.split('\n')
    lab_lines = [
        l.strip() for l in lines
        if any(u in l for u in [
            'mEq/L', 'mg/dL', 'U/L', 'mm3', 'g/dL', 'mmol/L',
            'mIU/mL', 'ng/mL', 'mcg/dL', 'IU/ml', '%', 'x 10',
        ])
    ]

    if wants_labs or wants_all:
        if lab_lines:
            parts.append("Laboratory results:\n" + "\n".join(lab_lines))
        else:
            parts.append("No specific lab values available for this case.")

    vital_patterns = [
        r'temperature\s+[\d.]+[°\s]*[CF]?\s*\([^)]+\)',
        r'blood pressure\s+[\d/]+\s*mm\s*Hg',
        r'pulse\s+[\d/]+\s*/?\s*min',
        r'respir\w+\s+(?:rate\s+)?[\d]+/min',
        r'oxygen saturation\s+[\d]+%[^.]*',
    ]
    if wants_vitals or wants_all:
        vitals_found = []
        for vp in vital_patterns:
            m = re.search(vp, full_text, re.IGNORECASE)
            if m:
                vitals_found.append(m.group(0))
        if vitals_found:
            parts.append("Vital signs: " + ", ".join(vitals_found))

    if wants_imaging or wants_all:
        for sent in re.split(r'[.\n]', full_text):
            if any(kw.lower() in sent.lower() for kw in [
                "x-ray", "xray", "CT ", "MRI", "ultrasound", "echo",
                "ECG", "EKG", "radiograph", "scan",
            ]):
                parts.append(sent.strip())

    if wants_exam or wants_all:
        for sent in re.split(r'[.\n]', full_text):
            if any(kw.lower() in sent.lower() for kw in [
                "physical exam", "on exam", "examination is",
                "cardiac exam", "lungs", "abdomen", "extremit",
            ]):
                parts.append(sent.strip())

    if wants_history or wants_all:
        for sent in re.split(r'[.\n]', full_text):
            if any(kw.lower() in sent.lower() for kw in [
                "past medical", "medication", "current med",
                "history is", "family history", "social history",
            ]):
                parts.append(sent.strip())

    if not parts:
        parts.append("Here is the complete case information:\n" + full_text)

    return "\n".join(parts)


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Ready.\n")

    def count_tokens(conv):
        return len(tokenizer.encode(
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)))

    def generate(conv):
        input_text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(args.device)
        ctx_len = input_ids.shape[1]
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        if eff_max < 10:
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

    # load cases
    with open(args.data) as f:
        cases = json.load(f)

    def difficulty_score(c):
        score = c["lab_count"] / 16 + c["question_length"] / 2200
        if c["meta_info"] == "step2&3":
            score += 0.3
        return score

    cases.sort(key=difficulty_score)
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"{'='*70}")
    print(f"MedQA INTERACTIVE — {args.model}")
    print(f"Context: {args.max_ctx:,} | Cases: {len(cases)} | Max followups: {args.max_followups}")
    print(f"{'='*70}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_turns = []
    case_meta = []
    global_turn = 0

    for case_idx, case in enumerate(cases):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > 0.92:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        chief_complaint = extract_chief_complaint(case)
        correct = case["answer_idx"]

        print(f"\n{'─'*60}")
        print(f"CASE {case_idx+1} (idx={case['idx']}, answer={correct}) | "
              f"ctx={ctx_now:,} ({ctx_now/args.max_ctx:.0%})")
        print(f"{'─'*60}")

        conversation.append({"role": "user", "content": f"New patient:\n\n{chief_complaint}"})

        answered = False
        model_answer = None
        case_turns = []

        for followup in range(args.max_followups + 1):
            response, ctx_len, entropies, logprobs = generate(conversation)
            if response is None:
                print("  >>> Context exhausted mid-case")
                break

            mean_ent = float(np.mean(entropies)) if entropies else 0
            mean_lp = float(np.mean(logprobs)) if logprobs else 0
            hedges = detect_hedging(response)

            turn_type = "followup" if followup > 0 else "initial"
            if has_concrete_answer(response):
                turn_type = "answer"
                model_answer = response
                answered = True

            turn_data = {
                "global_turn": global_turn,
                "case_idx": case_idx,
                "case_followup": followup,
                "turn_type": turn_type,
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
                "response_preview": response[:200],
            }
            all_turns.append(turn_data)
            case_turns.append(turn_data)
            global_turn += 1

            h_flag = f" [{len(hedges)}hw]" if hedges else ""
            print(f"  T{followup} ctx={ctx_len:,} ({ctx_len/args.max_ctx:.0%}) | "
                  f"ent={mean_ent:.4f} lp={mean_lp:.4f}{h_flag} | "
                  f"{turn_type.upper()}")
            print(f"    \"{response[:90]}...\"")

            conversation.append({"role": "assistant", "content": response})

            if answered:
                break

            info = build_info_response(case, response)
            conversation.append({"role": "user", "content": info})
            torch.cuda.empty_cache()

        if not answered:
            conversation.append({
                "role": "user",
                "content": "Based on what you know, what is your diagnosis "
                           "and recommended next step?"
            })
            response, ctx_len, entropies, logprobs = generate(conversation)
            if response:
                model_answer = response
                answered = True
                mean_ent = float(np.mean(entropies)) if entropies else 0
                mean_lp = float(np.mean(logprobs)) if logprobs else 0
                hedges = detect_hedging(response)
                turn_data = {
                    "global_turn": global_turn,
                    "case_idx": case_idx,
                    "case_followup": followup + 1,
                    "turn_type": "forced_answer",
                    "context_tokens": ctx_len,
                    "context_fill": round(ctx_len / args.max_ctx, 4),
                    "mean_entropy": mean_ent,
                    "mean_logprob": mean_lp,
                    "median_entropy": float(np.median(entropies)) if entropies else 0,
                    "std_entropy": float(np.std(entropies)) if entropies else 0,
                    "hedging_detected": len(hedges) > 0,
                    "num_hedge_words": len(hedges),
                    "hedge_words": hedges,
                    "refusal_detected": False,
                    "num_generated_tokens": len(entropies),
                    "response_preview": response[:200],
                }
                all_turns.append(turn_data)
                case_turns.append(turn_data)
                global_turn += 1
                print(f"  FORCED ctx={ctx_len:,} | ent={mean_ent:.4f}")
                print(f"    \"{response[:90]}...\"")
                conversation.append({"role": "assistant", "content": response})

        print(f"  Correct: {correct} — {case['answer']}")

        case_meta.append({
            "case_idx": case_idx,
            "question_idx": case["idx"],
            "correct_answer": case["answer"],
            "correct_answer_idx": correct,
            "model_answer": model_answer[:200] if model_answer else None,
            "num_followups": len(case_turns),
            "final_context_fill": case_turns[-1]["context_fill"] if case_turns else 0,
        })
        torch.cuda.empty_cache()

    # ── build merged dataset ────────────────────────────────────────────
    case_df = pd.DataFrame(case_meta)
    turn_df = pd.DataFrame(all_turns)
    # stringify hedge_words list for csv
    turn_df["hedge_words"] = turn_df["hedge_words"].apply(lambda x: "|".join(x))

    merged = turn_df.merge(
        case_df[["case_idx", "question_idx", "correct_answer",
                 "correct_answer_idx", "model_answer", "num_followups",
                 "final_context_fill"]],
        on="case_idx", how="left",
    )
    merged.insert(0, "model", args.model)
    merged.insert(1, "max_context_tokens", args.max_ctx)
    merged.insert(2, "experiment", "medqa_interactive")

    merged.to_csv(out_dir / "medqa_interactive.csv", index=False)
    merged.to_parquet(out_dir / "medqa_interactive.parquet", index=False)

    # also save conversation + summary
    with open(out_dir / "conversation.jsonl", "w") as f:
        for msg in conversation:
            f.write(json.dumps(msg) + "\n")

    n_cases = len(case_meta)
    n_answered = sum(1 for c in case_meta if c["model_answer"] is not None)
    summary = {
        "model": args.model,
        "max_context": args.max_ctx,
        "cases_completed": n_cases,
        "answered": n_answered,
        "total_turns": len(all_turns),
        "hedging_rate": (sum(1 for t in all_turns if t["hedging_detected"])
                         / len(all_turns)) if all_turns else 0,
        "mean_entropy": float(np.mean([t["mean_entropy"] for t in all_turns])),
        "mean_logprob": float(np.mean([t["mean_logprob"] for t in all_turns])),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── print summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY — {n_cases} cases, {n_answered} answered, {len(all_turns)} turns")
    print(f"{'='*70}")
    print(f"Hedging rate: {summary['hedging_rate']:.1%}")
    print(f"Mean entropy: {summary['mean_entropy']:.4f}")
    print(f"Mean logprob: {summary['mean_logprob']:.4f}")
    print(f"\nSaved to {out_dir}/")
    print(f"  medqa_interactive.csv       ({len(merged)} rows x {len(merged.columns)} cols)")
    print(f"  medqa_interactive.parquet")
    print(f"  conversation.jsonl")
    print(f"  summary.json")


if __name__ == "__main__":
    main()
