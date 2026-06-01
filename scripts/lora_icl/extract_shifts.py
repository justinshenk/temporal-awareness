"""Extract ICL and LoRA activation shifts at the DDXPlus prediction site.

For each held-out eval case, capture the residual stream at the final prompt
token (the position from which the model emits its answer letter) under three
conditions, all on the same case text:

    base_clean : base model, single clean case (the shared baseline)
    base_icl   : base model, case preceded by accumulated DDXPlus Q&A (ICL)
    lora_clean : LoRA model, single clean case (finetuning)

and write the two shift sets, both relative to the shared baseline:

    icl_shift  = base_icl  - base_clean
    lora_shift = lora_clean - base_clean

Run once for the real adapter (writes icl_shift + lora_shift_real), then again
with --tag shuffled --skip-icl for the shuffled-label control adapter.

Usage:
    HF_TOKEN=... uv run python -m scripts.lora_icl.extract_shifts \
        --config configs/lora_icl/ddxplus_gemma_lora.yaml \
        --adapter results/lora_icl/adapter --tag real
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import (
    build_cases,
    chat_messages,
    disjoint_split,
    icl_messages,
    select_valid_indices,
)
from src.probes.lora_icl.shift_extraction import last_token_residual, stack_shift_set


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def capture_site(model, capture, input_ids, device) -> dict[int, np.ndarray]:
    """Run a forward pass and return the final-token residual per captured layer."""
    capture.clear()
    with capture.capturing():
        model(torch.tensor([input_ids], device=device), use_cache=False)
    return last_token_residual(capture.captured)


def clean_ids(tokenizer, prompt_text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        chat_messages(prompt_text), add_generation_prompt=True, tokenize=True,
    )


def icl_ids(tokenizer, fillers, eval_prompt: str, max_ctx: int, fill_target: float) -> list[int]:
    """Build an ICL conversation: filler Q&A turns then the eval case."""
    messages = icl_messages(
        tokenizer, fillers, chat_messages(eval_prompt), max_ctx, fill_target
    )
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--tag", default="real", help="output label for the lora shift (real/shuffled)")
    ap.add_argument("--skip-icl", action="store_true",
                    help="reuse the previously saved base_clean/icl_shift")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    shifts_dir = Path(cfg["output"]["shifts_dir"])
    shifts_dir.mkdir(parents=True, exist_ok=True)

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["eval_split"])
    valid = select_valid_indices(ds, cfg["data"]["n_options"])
    train_like, eval_idx = disjoint_split(
        valid, cfg["data"]["n_train_cases"], cfg["data"]["n_eval_cases"], cfg["seed"]
    )
    eval_cases = build_cases(ds, eval_idx, evidence_db, cfg["data"]["n_options"], cfg["seed"])
    fillers = build_cases(ds, train_like, evidence_db, cfg["data"]["n_options"], cfg["seed"] + 1)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    capture = PerTokenResidualCapture(base, layers)

    base_clean, base_icl = [], []
    for c in eval_cases:
        base_clean.append(capture_site(base, capture, clean_ids(tokenizer, c.prompt_text), args.device))
        if not args.skip_icl:
            ids = icl_ids(tokenizer, fillers, c.prompt_text,
                          cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"])
            base_icl.append(capture_site(base, capture, ids, args.device))

    # LoRA pass: hooks live on the shared base decoder layers, so they fire here too.
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()
    lora_clean = [
        capture_site(lora_model, capture, clean_ids(tokenizer, c.prompt_text), args.device)
        for c in eval_cases
    ]
    capture.remove()

    for layer in layers:
        lora_shift = stack_shift_set(base_clean, lora_clean, layer)
        np.save(shifts_dir / f"lora_shift_{args.tag}_L{layer}.npy", lora_shift)
        if not args.skip_icl:
            icl_shift = stack_shift_set(base_clean, base_icl, layer)
            np.save(shifts_dir / f"icl_shift_L{layer}.npy", icl_shift)

    meta = {
        "base_model": cfg["base_model"], "layers": layers,
        "n_eval": len(eval_cases), "tag": args.tag,
        "hidden_dim": int(base_clean[0][layers[0]].shape[0]),
    }
    (shifts_dir / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote shift sets to {shifts_dir} ({meta})")


if __name__ == "__main__":
    main()
