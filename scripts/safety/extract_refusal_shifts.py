"""Does DDXPlus medical adaptation erode safety — by weights or by activations?

On held-out harmful prompts, measure both behavior (refusal rate) and the residual
shift along the refusal direction, under three conditions:

    base_clean : base model, harmful prompt alone (baseline refusal)
    base_icl   : base model, harmful prompt preceded by DDXPlus medical Q&A context
    lora_clean : the DDXPlus-finetuned LoRA, harmful prompt alone

Shifts (relative to the shared base_clean baseline):
    icl_shift  = resid(base, harmful WITH medical ICL) - resid(base, harmful clean)
    lora_shift = resid(DDXPlus-lora, harmful clean)    - resid(base, harmful clean)

The refusal direction r (per layer) is fit from separate harmful/harmless prompts on
the base model. run_safety_comparison.py then projects the shifts onto r.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.extract_refusal_shifts \
        --config configs/safety/ddxplus_safety_gemma.yaml
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
from src.probes.lora_icl.ddxplus_cases import build_cases, icl_messages, select_valid_indices
from src.probes.lora_icl.shift_extraction import last_token_residual, stack_shift_set
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def user_turn(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


@torch.no_grad()
def capture_resid(model, capture, ids, device) -> dict[int, np.ndarray]:
    capture.clear()
    with capture.capturing():
        model(torch.tensor([ids], device=device), use_cache=False)
    return last_token_residual(capture.captured)


@torch.no_grad()
def generate_reply(model, tokenizer, ids, device, max_new) -> str:
    out = model.generate(
        torch.tensor([ids], device=device), max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)


def prompt_ids(tokenizer, messages) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return ids if isinstance(ids, list) else ids["input_ids"]  # Qwen returns a BatchEncoding


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    shifts_dir = Path(cfg["output"]["shifts_dir"])
    shifts_dir.mkdir(parents=True, exist_ok=True)

    # Prompt sets: harmful split into direction-fit + held-out eval; harmless for the fit.
    harmful = load_harmful()
    harmless = load_harmless()
    n_dir, n_eval = cfg["direction"]["n_harmful"], cfg["eval"]["n_harmful"]
    harmful_fit = harmful[:n_dir]
    harmful_eval = harmful[n_dir : n_dir + n_eval]
    harmless_fit = harmless[: cfg["direction"]["n_harmless"]]

    # DDXPlus medical filler cases for the ICL context.
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])[: cfg["ddxplus"]["n_filler"]]
    fillers = build_cases(ds, valid, evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    capture = PerTokenResidualCapture(base, layers)
    mc = cfg["extract"]["max_ctx"]
    ft = cfg["extract"]["icl_fill_target"]
    max_new = cfg["eval"]["max_new"]

    # 1. Refusal direction r_layer from harmful/harmless fit prompts (base, clean).
    harmful_fit_res = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device)
                       for p in harmful_fit]
    harmless_fit_res = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device)
                        for p in harmless_fit]
    for layer in layers:
        h = np.stack([r[layer] for r in harmful_fit_res])
        s = np.stack([r[layer] for r in harmless_fit_res])
        np.save(shifts_dir / f"refusal_dir_L{layer}.npy", refusal_direction(h, s))

    # 2. Eval prompts: base_clean + base_icl residuals & generations.
    base_clean, base_icl = [], []
    refusals = {"base": [], "icl": [], "lora": []}
    for p in harmful_eval:
        clean_ids = prompt_ids(tokenizer, user_turn(p))
        base_clean.append(capture_resid(base, capture, clean_ids, args.device))
        refusals["base"].append(generate_reply(base, tokenizer, clean_ids, args.device, max_new))

        icl_ids = prompt_ids(tokenizer, icl_messages(tokenizer, fillers, user_turn(p), mc, ft))
        base_icl.append(capture_resid(base, capture, icl_ids, args.device))
        refusals["icl"].append(generate_reply(base, tokenizer, icl_ids, args.device, max_new))

    # 3. LoRA (DDXPlus adapter) on the same shared base layers: lora_clean residuals & generations.
    lora_model = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()
    lora_clean = []
    for p in harmful_eval:
        clean_ids = prompt_ids(tokenizer, user_turn(p))
        lora_clean.append(capture_resid(lora_model, capture, clean_ids, args.device))
        refusals["lora"].append(generate_reply(lora_model, tokenizer, clean_ids, args.device, max_new))
    capture.remove()

    for layer in layers:
        np.save(shifts_dir / f"icl_shift_L{layer}.npy", stack_shift_set(base_clean, base_icl, layer))
        np.save(shifts_dir / f"lora_shift_L{layer}.npy", stack_shift_set(base_clean, lora_clean, layer))

    Path(cfg["output"]["refusal_json"]).write_text(json.dumps(refusals, indent=2))
    meta = {"base_model": cfg["base_model"], "adapter": cfg["adapter_dir"], "layers": layers,
            "n_eval": len(harmful_eval), "hidden_dim": int(base_clean[0][layers[0]].shape[0])}
    (shifts_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote shifts + refusal direction to {shifts_dir}; refusals to {cfg['output']['refusal_json']}")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
