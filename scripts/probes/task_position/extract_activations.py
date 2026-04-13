"""Extract per-token residual streams + task-position labels on DDXPlus.

Runs Gemma-9B-IT through a multi-case DDXPlus trace, capturing residual
streams at specified layers for every token. Saves activations and per-token
labels to disk.

When --eval-correctness is passed, the script additionally records per-case
option-letter probabilities and derives predictions via argmax, saving
everything to correctness.json next to activations.pt.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import (
    OPTION_LABELS,
    SYSTEM_PROMPT,
    format_case_mcq,
    load_evidence_db,
)
from src.probes.extraction import PerTokenResidualCapture
from src.probes.task_position.labels import label_trace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--fill-target", type=float, default=0.90)
    p.add_argument("--n-traces", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default="0,10,20,30,41")
    p.add_argument(
        "--evidence-db",
        default="data/context_fatigue/release_evidences.json",
    )
    p.add_argument(
        "--out-dir",
        default="results/probes/task_position/gemma-9b-it",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--eval-correctness",
        action="store_true",
        help="Record per-case option-letter probabilities and argmax predictions.",
    )
    return p.parse_args()


def _apply_template(tokenizer, conversation: list[dict]) -> str:
    """Apply chat template, injecting system prompt into first user turn if needed.

    Gemma-2 does not support the system role. When the first message is a
    system turn, its content is prepended to the first user turn instead.
    Returns an empty string if no non-system messages exist yet.
    """
    messages = list(conversation)
    system_content = None
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages = messages[1:]
    if not messages:
        return ""
    if system_content and messages[0]["role"] == "user":
        messages[0] = {
            "role": "user",
            "content": f"{system_content}\n\n{messages[0]['content']}",
        }
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _apply_template_with_generation_prompt(tokenizer, conversation: list[dict]) -> str:
    """Like _apply_template but adds a generation prompt at the end."""
    messages = list(conversation)
    system_content = None
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages = messages[1:]
    if not messages:
        return ""
    if system_content and messages[0]["role"] == "user":
        messages[0] = {
            "role": "user",
            "content": f"{system_content}\n\n{messages[0]['content']}",
        }
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_trace(
    tokenizer,
    ds,
    valid_indices,
    evidence_db,
    max_ctx,
    fill_target,
    rng,
    model=None,
    option_token_ids=None,
    device="cuda",
):
    """Build one trace: accumulate DDXPlus cases until context fills.

    When model is None, inserts gold letters as assistant turns (original
    behaviour). When model is provided, records option-letter probabilities
    and derives predictions via argmax. The assistant turn always uses the
    gold letter so activation context is consistent across cases.

    Returns:
        tokens: list[int], the tokenized concatenated conversation
        case_start_tokens: list[int], token indices where each case begins
        correctness_records: list[dict] | None — None when model is None
    """
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    case_start_tokens: list[int] = []
    correctness_records: list[dict] | None = [] if model is not None else None

    case_rng = random.Random(rng.randint(0, 2**31 - 1))
    indices = list(valid_indices)
    case_rng.shuffle(indices)

    case_index = 0
    for idx in indices:
        text_before = _apply_template(tokenizer, conversation)
        if text_before:
            ids_before = tokenizer(text_before, return_tensors="pt").input_ids[0]
            n_before = ids_before.shape[0]
        else:
            n_before = 0

        if n_before / max_ctx > fill_target:
            break

        row = ds[idx]
        pathology = row["PATHOLOGY"]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        option_names = [d[0] for d in ddx[:5]]
        shuffled = list(option_names)
        case_rng.shuffle(shuffled)
        gold_letter = OPTION_LABELS[shuffled.index(pathology)]

        case_text = format_case_mcq(
            row["AGE"],
            row["SEX"],
            row["INITIAL_EVIDENCE"],
            row["EVIDENCES"],
            evidence_db,
            shuffled,
        )

        case_start_tokens.append(n_before)
        conversation.append({"role": "user", "content": case_text})

        if model is not None:
            # Build prompt with generation prompt appended
            prompt_text = _apply_template_with_generation_prompt(
                tokenizer, conversation
            )
            prompt_ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

            # Single forward pass for option-letter logits at the last prompt token
            with torch.no_grad():
                logits = model(prompt_ids, use_cache=False).logits
            last_logits = logits[0, -1, :]
            option_ids_list = [option_token_ids[letter] for letter in OPTION_LABELS]
            option_logits = last_logits[option_ids_list]
            option_probs_tensor = torch.softmax(option_logits, dim=0)
            option_probs = {
                letter: option_probs_tensor[i].item()
                for i, letter in enumerate(OPTION_LABELS)
            }

            pred = max(option_probs, key=option_probs.get)

            # Compute prediction_site: token offset in the FINAL trace where
            # the model would emit case i's answer.  Tokenise prompt_text the
            # same way final_ids is computed (no add_special_tokens=False) so
            # the prefix length is measured consistently.
            prefix_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
            prediction_site = len(prefix_ids) - 1

            correctness_records.append(
                {
                    "case_index": case_index,
                    "gold": gold_letter,
                    "pred": pred,
                    "correct": pred == gold_letter,
                    "option_probs": option_probs,
                    "prediction_site": prediction_site,
                }
            )

        assistant_content = gold_letter

        conversation.append({"role": "assistant", "content": assistant_content})
        case_index += 1

    final_text = _apply_template(tokenizer, conversation)
    final_ids = tokenizer(final_text, return_tensors="pt").input_ids[0].tolist()

    # The case_start_tokens were measured BEFORE appending each case, using
    # conversation state without add_generation_prompt. After concatenation
    # they may be slightly off due to chat-template structural tokens. Re-fit:
    # the first boundary is 0 only if the full trace starts that way; we
    # compensate by taking the boundaries as measured. For robustness we
    # prepend 0 if the first recorded boundary is non-zero.
    if case_start_tokens and case_start_tokens[0] != 0:
        case_start_tokens = [0] + case_start_tokens

    # Drop any boundaries >= final trace length (can happen if tokenization
    # compacts trailing whitespace at the end)
    case_start_tokens = [b for b in case_start_tokens if b < len(final_ids)]

    return final_ids, case_start_tokens, correctness_records


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    evidence_db = load_evidence_db(args.evidence_db)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    option_token_ids: dict[str, int] = {}
    for letter in OPTION_LABELS:
        ids_with_space = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids_with_space) == 1:
            option_token_ids[letter] = ids_with_space[0]
        else:
            option_token_ids[letter] = tokenizer.encode(
                letter, add_special_tokens=False
            )[0]

    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid_indices = [
        i
        for i in range(len(ds))
        if ds[i]["PATHOLOGY"]
        in [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:5]]
    ]

    capture = PerTokenResidualCapture(model, layers=layers)

    eval_model = model if args.eval_correctness else None

    trace_rng = random.Random(args.seed)
    all_traces = []
    all_correctness: dict[str, list[dict]] = {}

    for trace_i in range(args.n_traces):
        print(f"\nTrace {trace_i + 1}/{args.n_traces}: building...")
        tokens, case_boundaries, correctness_records = build_trace(
            tokenizer,
            ds,
            valid_indices,
            evidence_db,
            args.max_ctx,
            args.fill_target,
            trace_rng,
            model=eval_model,
            option_token_ids=option_token_ids,
            device=args.device,
        )
        n_tokens = len(tokens)
        n_cases = len(case_boundaries)
        print(f"  tokens={n_tokens} cases={n_cases}")

        if correctness_records is not None:
            n_correct = sum(r["correct"] for r in correctness_records)
            print(f"  correctness: {n_correct}/{len(correctness_records)}")
            all_correctness[str(trace_i)] = correctness_records

        input_ids = torch.tensor([tokens], device=args.device)
        capture.clear()
        with capture.capturing(), torch.no_grad():
            _ = model(input_ids, use_cache=False)

        acts_by_layer = {li: capture.captured[li].clone() for li in layers}
        labels = label_trace(trace_length=n_tokens, case_boundaries=case_boundaries)

        all_traces.append(
            {
                "trace_id": trace_i,
                "tokens": tokens,
                "case_boundaries": case_boundaries,
                "labels": labels.to_dict(),
                "activations": acts_by_layer,
            }
        )

        del input_ids, _
        torch.cuda.empty_cache()
        gc.collect()

    capture.remove()

    out_file = out_dir / "activations.pt"
    torch.save({"layers": layers, "traces": all_traces}, out_file)
    print(f"\nSaved {len(all_traces)} traces to {out_file}")

    if args.eval_correctness:
        correctness_file = out_dir / "correctness.json"
        with open(correctness_file, "w") as f:
            json.dump(all_correctness, f, indent=2)
        print(f"Saved correctness.json to {correctness_file}")


if __name__ == "__main__":
    main()
