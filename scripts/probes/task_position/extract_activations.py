"""Extract per-token residual streams + task-position labels on DDXPlus.

Runs Gemma-9B-IT through a multi-case DDXPlus trace, capturing residual
streams at specified layers for every token. Saves activations and per-token
labels to disk.
"""

from __future__ import annotations

import argparse
import ast
import gc
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


def build_trace(tokenizer, ds, valid_indices, evidence_db, max_ctx, fill_target, rng):
    """Build one trace: accumulate DDXPlus cases until context fills.

    Returns:
        tokens: list[int], the tokenized concatenated conversation
        case_start_tokens: list[int], token indices where each case begins (first is 0)
    """
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    case_start_tokens: list[int] = []

    case_rng = random.Random(rng.randint(0, 2**31 - 1))
    indices = list(valid_indices)
    case_rng.shuffle(indices)

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
        conversation.append({"role": "assistant", "content": gold_letter})

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

    return final_ids, case_start_tokens


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

    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid_indices = [
        i
        for i in range(len(ds))
        if ds[i]["PATHOLOGY"]
        in [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:5]]
    ]

    capture = PerTokenResidualCapture(model, layers=layers)

    trace_rng = random.Random(args.seed)
    all_traces = []

    for trace_i in range(args.n_traces):
        print(f"\nTrace {trace_i + 1}/{args.n_traces}: building...")
        tokens, case_boundaries = build_trace(
            tokenizer,
            ds,
            valid_indices,
            evidence_db,
            args.max_ctx,
            args.fill_target,
            trace_rng,
        )
        n_tokens = len(tokens)
        n_cases = len(case_boundaries)
        print(f"  tokens={n_tokens} cases={n_cases}")

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


if __name__ == "__main__":
    main()
