"""Qualitative side-by-side: base vs. LoRA greedy generation on one GSM8K problem.

    uv run python -m scripts.attribution.show_generation \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--index 0 --max-new 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    gsm8k_problems,
    load_base_and_lora,
    prompt_token_ids,
)
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match


@torch.no_grad()
def generate_text(model, tokenizer, question, device, max_new) -> str:
    prompt_ids = prompt_token_ids(tokenizer, question, device)
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=400)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device, max_new = cfg["device"], args.max_new

    tokenizer, base, lora = load_base_and_lora(cfg)
    question, gold = gsm8k_problems(cfg["eval"]["split"], 1, skip=args.index)[0]

    with lora.disable_adapter():
        base_text = generate_text(base, tokenizer, question, device, max_new)
    lora_text = generate_text(lora, tokenizer, question, device, max_new)

    bar = "=" * 80
    print(f"{bar}\nQUESTION (test #{args.index}):\n{question}\n\nGOLD: {gold}\n{bar}")
    print(f"\n----- BASE (adapter disabled) -----\n{base_text}")
    print(f"  -> parsed={extract_pred_number(base_text)}  correct={numeric_match(extract_pred_number(base_text), gold)}")
    print(f"\n----- LoRA (MetaMath adapter) -----\n{lora_text}")
    print(f"  -> parsed={extract_pred_number(lora_text)}  correct={numeric_match(extract_pred_number(lora_text), gold)}")


if __name__ == "__main__":
    main()
