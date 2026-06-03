"""Instruction-Adherence Decay under Context Accumulation (Phase 1, behavioral).

Accumulates DDXPlus MCQ cases in one conversation (the existing context-fatigue
harness) while a fixed, checkable "canary" instruction sits in the system prompt.
At every turn we record task correctness AND whether the instruction was obeyed,
both vs context fill. The question: does instruction adherence decay as context
fills even though task accuracy stays flat?

Three arms (see ``src/probes/context_fatigue/instruction_checks.py``) separate a
genuine adherence decay from (a) the model imitating its own non-compliant history
(``forced`` teacher-forces a compliant history) and (b) pure positional distance
from the system prompt (``refresh`` re-states the canary in the latest user turn).

Run from the repo root via ``-m`` (Qwen is not gated):

    uv run python -m scripts.context_fatigue.run_instruction_adherence \
        --model Qwen/Qwen2.5-7B-Instruct

    # fast smoke test
    uv run python -m scripts.context_fatigue.run_instruction_adherence \
        --max-cases 6 --max-ctx 2048 --instructions prefix_marker --arms forced
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.context_fatigue._cf_common import (
    generate_with_entropy,
    load_evidence_db,
    render_prompt,
)
from src.probes.context_fatigue.instruction_checks import (
    ARMS,
    INSTRUCTIONS,
    fill_bin_stats,
    history_assistant_for,
    pearson,
    system_prompt_for,
    user_content_for,
)
from src.probes.lora_icl.ddxplus_cases import build_cases, select_valid_indices

BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

# Parseable answer format, enforced in EVERY arm so task correctness is read
# independently of the canary (the canary marker must not be an option letter).
# Kept brief so a bare answer fits in a small token budget (no truncation).
TASK_FORMAT = ("Answer briefly, then give your final answer on its own line as "
               "'ANSWER: X' where X is the letter (A, B, C, D, or E) of the best option.")


def extract_answer(response: str) -> str | None:
    """Letter from the 'ANSWER: X' line; fall back to the last standalone A–E."""
    m = re.findall(r'ANSWER:\s*([A-Ea-e])', response)
    if m:
        return m[-1].upper()
    m = re.findall(r'\b([A-E])\b', response.upper())
    return m[-1] if m else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--fill-target", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--base-system", default="You are a doctor.")
    p.add_argument("--instructions", default="prefix_marker,suffix_ok",
                   help="comma-separated keys of INSTRUCTIONS")
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of baseline,forced,refresh")
    p.add_argument("--out-dir", default="results/context_fatigue/instruction_adherence")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_pass(model, tokenizer, cases, spec, arm, args, is_chat):
    """One accumulation pass for a single (instruction, arm). Returns turn dicts."""
    def count_tokens(conv):
        return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))

    effective_base = f"{args.base_system}\n\n{TASK_FORMAT}"
    conversation = [{"role": "system",
                     "content": system_prompt_for(spec, arm, effective_base)}]
    turns = []
    for case_num, case in enumerate(cases):
        if count_tokens(conversation) / args.max_ctx > args.fill_target:
            break

        conversation.append({"role": "user",
                             "content": user_content_for(spec, arm, case.prompt_text)})
        text = render_prompt(tokenizer, conversation, is_chat)
        response, ctx_len, mean_ent, n_gen = generate_with_entropy(
            model, tokenizer, text, args.device, args.max_new, args.max_ctx)
        if response is None:
            conversation.pop()  # could not generate; drop the dangling user turn
            break

        pred = extract_answer(response)
        obeyed = spec.check_obeyed(response)
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        turns.append({
            "instruction": spec.name,
            "arm": arm,
            "turn": case_num,
            "source_index": case.source_index,
            "gold_letter": case.gold_letter,
            "pred_letter": pred,
            "correct": int(pred == case.gold_letter),
            "obeyed": int(obeyed),
            "violation": int(not obeyed),
            "context_tokens": ctx_len,
            "context_fill": round(ctx_len / args.max_ctx, 4),
            "mean_entropy": mean_ent,
            "num_generated_tokens": n_gen,
            "truncated": int(eff_max > 0 and n_gen >= eff_max),
            "response": response[:200],
        })
        conversation.append({"role": "assistant",
                             "content": history_assistant_for(spec, arm, response)})
    return turns


def summarize(turns):
    fills = [t["context_fill"] for t in turns]
    viol = [t["violation"] for t in turns]
    corr = [t["correct"] for t in turns]
    n = len(turns)
    return {
        "n": n,
        "violation_rate": sum(viol) / n if n else None,
        "accuracy": sum(corr) / n if n else None,
        "corr_violation_fill": pearson(fills, viol),
        "corr_correct_fill": pearson(fills, corr),
        "truncated_rate": sum(t["truncated"] for t in turns) / n if n else None,
        "violation_by_fill": fill_bin_stats(turns, "violation", BINS),
        "accuracy_by_fill": fill_bin_stats(turns, "correct", BINS),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instructions = [INSTRUCTIONS[k] for k in args.instructions.split(",")]
    arms = args.arms.split(",")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    evidence_db = load_evidence_db()
    print("Loading DDXPlus test set ...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid = select_valid_indices(ds, args.n_options)
    random.Random(args.seed).shuffle(valid)
    if args.max_cases:
        valid = valid[: args.max_cases]
    cases = build_cases(ds, valid, evidence_db, args.n_options, args.seed)
    print(f"Built {len(cases)} cases (gold in top-{args.n_options}).")

    summary = {"model": args.model, "seed": args.seed, "max_ctx": args.max_ctx,
               "fill_target": args.fill_target, "runs": {}}
    for spec in instructions:
        for arm in arms:
            print(f"\n=== {spec.name} / {arm} ===")
            turns = run_pass(model, tokenizer, cases, spec, arm, args, is_chat)
            df = pd.DataFrame(turns)
            df.to_csv(out_dir / f"{spec.name}_{arm}.csv", index=False)
            s = summarize(turns)
            summary["runs"][f"{spec.name}_{arm}"] = s
            print(f"  n={s['n']} acc={s['accuracy']:.3f} viol={s['violation_rate']:.3f} "
                  f"corr(viol,fill)={s['corr_violation_fill']:+.3f} "
                  f"corr(correct,fill)={s['corr_correct_fill']:+.3f} "
                  f"trunc={s['truncated_rate']:.2f}")
            torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print("\nDissociation check (corr with context fill):")
    print(f"{'run':32s} {'viol↑':>8s} {'acc~0':>8s}")
    for name, s in summary["runs"].items():
        print(f"{name:32s} {s['corr_violation_fill']:+8.3f} {s['corr_correct_fill']:+8.3f}")


if __name__ == "__main__":
    main()
