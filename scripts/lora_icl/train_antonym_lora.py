"""Train a LoRA adapter on the antonym mapping (the in-weights route).

Bare ``word: antonym`` format — the same format as the ICL prompts in the FV experiment, so
the LoRA's weight-shift and the in-context function vector live at a comparable prediction site.
Trains on the train split of the curated pairs; the held-out split is used for FV + eval.

Usage:
    HF_TOKEN=... uv run python -m scripts.lora_icl.train_antonym_lora \
        --config configs/lora_icl/antonym_fv_gemma.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.lora_icl.antonym_data import antonym_split
from scripts.lora_icl.train_ddxplus_lora import collate, set_seed

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tokenizer, word: str, antonym: str):
    """Bare 'word:' prompt, ' antonym' target; loss only on the antonym + eos."""
    prompt_ids = tokenizer(f"{word}:", add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(f" {antonym}", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    out_dir = Path(cfg["adapter_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train, _ = antonym_split(cfg["n_train"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [encode(tokenizer, w, a) for w, a in train]

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device)
    model = get_peft_model(model, LoraConfig(
        r=cfg["train"]["rank"], lora_alpha=cfg["train"]["alpha"], lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.print_trainable_parameters()
    model.train()

    loader = DataLoader(examples, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=float(cfg["train"]["lr"]))
    for epoch in range(cfg["train"]["epochs"]):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total += loss.item()
        print(f"epoch {epoch}: mean loss {total / len(loader):.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved antonym adapter to {out_dir}")


if __name__ == "__main__":
    main()
