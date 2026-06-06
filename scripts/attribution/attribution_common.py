"""Shared model loading, GSM8K data, CoT generation, and accuracy eval for the
primal-ridge attribution scripts (collect / fit / steer)."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    gsm8k_gold_answer,
    metamath_prompt,
    numeric_match,
)


def load_base_and_lora(cfg) -> tuple:
    """Load tokenizer, base model, and the LoRA-wrapped model on one base instance.

    ``base`` and the adapter share weights/layers, so a single ``PerTokenResidualCapture``
    on ``base`` sees both forwards; toggle with ``lora.disable_adapter()``.
    """
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=cfg["device"]).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter"]).eval()
    return tok, base, lora


def gsm8k_problems(split: str, n: int, skip: int = 0) -> list[tuple[str, float]]:
    """Return ``n`` (question, gold_float) pairs from a GSM8K split, skipping the first ``skip``."""
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for i in range(skip, min(skip + n, len(ds))):
        out.append((ds[i]["question"], gsm8k_gold_answer(ds[i]["answer"])))
    return out


def prompt_token_ids(tokenizer, question: str, device) -> torch.Tensor:
    ids = tokenizer(metamath_prompt(question), return_tensors="pt").input_ids
    return ids.to(device)


@torch.no_grad()
def generate_cot_ids(model, tokenizer, question: str, device, max_new: int) -> tuple[torch.Tensor, int]:
    """Greedy-generate a CoT; return ``(full_ids (1,L), prompt_len)`` for teacher-forcing."""
    prompt_ids = prompt_token_ids(tokenizer, question, device)
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out, prompt_ids.shape[1]


@torch.no_grad()
def manifold_bases(acc_dir, layers, k: int, device, which: str = "base") -> dict[int, torch.Tensor]:
    """Top-``k`` manifold basis per layer from the stored accumulators (float32 columns).

    ``which`` selects the token-second-moment manifold: ``base`` (Σaaᵀ), ``lora``
    (Σ(a+δ)(a+δ)ᵀ), or ``union`` (top-k of base⊕lora). Used as the projection subspace for
    the steering manifold probes. Returns ``{layer: V (d, k)}``.
    """
    bases = {}
    for l in layers:
        acc = GramAccumulator.from_state_dict(torch.load(Path(acc_dir) / f"train_L{l}.pt"), device=device)
        bases[l] = acc.manifold_basis(k, which).to(torch.float32)
    return bases


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, problems: list[tuple[str, float]], device, max_new: int) -> float:
    """Greedy-generate from the MetaMath prompt and score parsed answers against gold."""
    correct = 0
    for question, gold in problems:
        prompt_ids = prompt_token_ids(tokenizer, question, device)
        out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        if numeric_match(extract_pred_number(text), gold):
            correct += 1
    return correct / len(problems)
