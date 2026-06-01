"""Train a LoRA adapter on single-case DDXPlus MCQ (clean, no context).

This is the "finetuning" arm of the LoRA-vs-ICL subspace comparison: the adapter
learns the DDXPlus task from weight updates alone, with no in-context examples.
A `--shuffle-labels` control trains on permuted answers to test task-specificity.

Usage:
    HF_TOKEN=... uv run python -m scripts.lora_icl.train_ddxplus_lora \
        --config configs/lora_icl/ddxplus_gemma_lora.yaml
    HF_TOKEN=... uv run python -m scripts.lora_icl.train_ddxplus_lora \
        --config configs/lora_icl/ddxplus_gemma_lora.yaml --shuffle-labels

The adapter is written to `output.adapter_dir` (or `output.shuffled_adapter_dir`
when `--shuffle-labels`). Activations/adapters are gitignored; only configs and
the writeup are committed.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import (
    build_cases,
    chat_messages,
    disjoint_split,
    permute_labels,
    select_valid_indices,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_example(tokenizer, prompt_text: str, gold_letter: str, max_seq_len: int):
    """Tokenize one case; mask the prompt so loss falls only on the answer letter."""
    prompt_ids = tokenizer.apply_chat_template(
        chat_messages(prompt_text), add_generation_prompt=True, tokenize=True,
    )
    answer_ids = tokenizer(gold_letter, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
    return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id: int):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="train on permuted answers (task-specificity control)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    out_dir = Path(
        cfg["output"]["shuffled_adapter_dir"] if args.shuffle_labels
        else cfg["output"]["adapter_dir"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DDXPlus train split + evidence DB...")
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["train_split"])
    valid = select_valid_indices(ds, cfg["data"]["n_options"])
    train_idx, _ = disjoint_split(
        valid, cfg["data"]["n_train_cases"], cfg["data"]["n_eval_cases"], cfg["seed"]
    )
    cases = build_cases(ds, train_idx, evidence_db, cfg["data"]["n_options"], cfg["seed"])
    if args.shuffle_labels:
        cases = permute_labels(cases, cfg["seed"])
    print(f"Built {len(cases)} training cases (shuffle_labels={args.shuffle_labels})")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        encode_example(tokenizer, c.prompt_text, c.gold_letter, cfg["train"]["max_seq_len"])
        for c in cases
    ]

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float32,
        device_map=args.device,
    )
    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    loader = DataLoader(
        examples, batch_size=cfg["train"]["micro_batch_size"], shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=float(cfg["train"]["lr"])
    )
    accum = cfg["train"]["grad_accum"]

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for i, batch in enumerate(loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            if (i + 1) % accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item() * accum:.4f}")
        optim.step()
        optim.zero_grad()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
