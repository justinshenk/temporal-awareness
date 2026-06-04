"""Distill a LoRA finetune into a linear activation map on the base model.

Fit, per layer, W a_base ≈ Δh_L  where Δh_L = resid(LoRA, x) - resid(base, x) is the LoRA's
own activation shift (ridge, dual form), on DDXPlus fit cases. Then steer the BASE model with
α·(W a) and ask:
  - does it transfer the task competence (toward the LoRA's accuracy)?
  - does it also copy the LoRA's refusal erosion? (the LoRA shift has an r-component)
  - does fitting on the r-orthogonalized LoRA shift give competence WITHOUT erosion?

Conditions: base, the LoRA itself (reference), distilled map, r-orthogonalized distilled map.

    uv run python -m scripts.safety.run_lora_distill --config configs/safety/route_safety_qwen.yaml \
        --adapter results/safety/qwen_sweep/adapter_d600
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

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.safety.steering_hook import LinearConditionalSteerHook
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless


def ridge_maps(A, Delta, layers, lam):
    maps = {}
    for L in layers:
        AL = A[L].astype(np.float64)
        K = AL @ AL.T
        reg = lam * np.trace(K) / K.shape[0]
        C = np.linalg.solve(K + reg * np.eye(K.shape[0]), Delta[L].astype(np.float64))
        maps[L] = (torch.tensor(AL.T), torch.tensor(C))
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--n-fit", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-harmful", type=int, default=25)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, max_new = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["eval"]["max_new"]
    alphas = [float(a) for a in args.alphas.split(",")]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]
    h_rfit, s_rfit = load_harmful()[:nh], load_harmless()[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    n_o = cfg["ddxplus"]["n_options"]
    fit = build_cases(ds, valid[:args.n_fit], evidence_db, n_o, cfg["seed"])
    ev = build_cases(ds, valid[args.n_fit:args.n_fit + args.n_eval], evidence_db, n_o, cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(model, msgs):
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    # r direction (base, clean) — BEFORE wrapping
    hr = [resid(base, user_turn(p)) for p in h_rfit]
    sr = [resid(base, user_turn(p)) for p in s_rfit]
    r = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in layers}
    rhat = {L: r[L] / np.linalg.norm(r[L]) for L in layers}

    # base activations on the fit cases (input to the map) — BEFORE wrapping
    base_fit = [resid(base, chat_messages(c.prompt_text)) for c in fit]
    A = {L: np.stack([b[L] for b in base_fit]) for L in layers}

    lora = PeftModel.from_pretrained(base, args.adapter).eval()
    lora_fit = [resid(lora, chat_messages(c.prompt_text)) for c in fit]
    capture.remove()
    # Δh_L = LoRA shift, and its r-orthogonalized version
    Delta = {L: np.stack([lo[L] for lo in lora_fit]) - A[L] for L in layers}
    Delta_orth = {L: Delta[L] - np.outer(Delta[L] @ rhat[L], rhat[L]) for L in layers}

    maps_distill = ridge_maps(A, Delta, layers, args.lam)
    maps_safe = ridge_maps(A, Delta_orth, layers, args.lam)

    def acc_ref(model, hook_factory=None):
        h = hook_factory() if hook_factory else None
        a = ddxplus_accuracy(model, tokenizer, ev, args.device, mc)[0]
        r_ = refusal_rate([generate_reply(model, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                           for p in h_eval])
        if h:
            h.remove()
        return {"task_acc": a, "refusal": r_}

    results = {}
    with lora.disable_adapter():
        results["base"] = acc_ref(lora)                                   # true base
    results["lora_finetune"] = acc_ref(lora)                              # the LoRA itself (reference)
    for a in alphas:
        with lora.disable_adapter():                                     # distilled map on the BASE model
            results[f"distill_a{a}"] = acc_ref(lora, lambda a=a: LinearConditionalSteerHook(base, maps_distill, a))
            results[f"distill_safe_a{a}"] = acc_ref(lora, lambda a=a: LinearConditionalSteerHook(base, maps_safe, a))
        print(f"  a={a}: distill={results[f'distill_a{a}']} safe={results[f'distill_safe_a{a}']}")

    print(f"  base={results['base']} | lora={results['lora_finetune']}")
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora_distill.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir}/lora_distill.json")


if __name__ == "__main__":
    main()
