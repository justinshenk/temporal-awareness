"""Does the DDXPlus-distilled map destabilize an unrelated task (MMLU)?

Fit the distilled map W a_base ≈ Δh_L (LoRA shift) on DDXPlus, then steer the BASE model on
MMLU and measure accuracy vs no-steer. If MMLU drops, the transfer isn't task-specific — it
broadly perturbs the model. (Specificity / collateral-damage check.)

    HF_HUB_OFFLINE=1 uv run python -m scripts.safety.run_mmlu_destab \
        --config configs/safety/route_safety_gemma.yaml --adapter results/lora_icl/adapter
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed
from scripts.safety.run_lora_distill import ridge_maps
from src.probes.safety.steering_hook import LinearConditionalSteerHook
from src.probes.safety.mcq_icl import chat_mcq, mcq_cases, parse4
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/lora_icl/adapter")
    ap.add_argument("--n-fit", type=int, default=100)
    ap.add_argument("--n-mmlu", type=int, default=50)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, max_new = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["eval"]["max_new"]
    alphas = [float(a) for a in args.alphas.split(",")]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    fit = build_cases(ds, valid[:args.n_fit], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=cfg["seed"])
    mm_eval = mcq_cases(mm, range(args.n_mmlu))

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(model, msgs):
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    base_fit = [resid(base, chat_messages(c.prompt_text)) for c in fit]  # BEFORE wrapping
    A = {L: np.stack([b[L] for b in base_fit]) for L in layers}
    lora = PeftModel.from_pretrained(base, args.adapter).eval()
    lora_fit = [resid(lora, chat_messages(c.prompt_text)) for c in fit]
    capture.remove()
    Delta = {L: np.stack([lo[L] for lo in lora_fit]) - A[L] for L in layers}
    maps = ridge_maps(A, Delta, layers, args.lam)

    def mmlu_acc(hook_factory=None):
        h = hook_factory() if hook_factory else None
        correct = n = 0
        for c in mm_eval:
            ids = prompt_ids(tokenizer, chat_mcq(c.prompt_text))
            if len(ids) > mc - max_new:
                continue
            pred = parse4(generate_reply(base, tokenizer, ids, args.device, max_new))
            if pred is not None:
                n += 1
                correct += int(pred == c.gold_letter)
        if h:
            h.remove()
        return correct / n if n else float("nan")

    results = {}
    with lora.disable_adapter():                       # base behavior + (optionally) the distilled map
        results["mmlu_base"] = mmlu_acc()
        print(f"  MMLU base (no steer): {results['mmlu_base']:.3f}")
        for a in alphas:
            results[f"mmlu_distill_a{a}"] = mmlu_acc(lambda a=a: LinearConditionalSteerHook(base, maps, a))
            print(f"  MMLU + distill map a={a}: {results[f'mmlu_distill_a{a}']:.3f}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mmlu_destab.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir}/mmlu_destab.json")


if __name__ == "__main__":
    main()
