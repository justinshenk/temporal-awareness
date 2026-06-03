"""Route-dependent safety cost, swept over dose: weight (LoRA) vs activation (ICL).

For each dose of each route we measure, on a held-out harmful set, both the behavioral
safety cost and where the activation move points:

  - ΔRefusal   = base_refusal - condition_refusal   (positive = erosion)
  - task_gain  = DDXPlus accuracy(condition) - accuracy(base)   (the matched-dose x-axis)
  - per layer: projection of the activation shift onto the refusal direction r
               (negative = toward compliance) and cos(mean_shift, r)

Hypothesis: ΔRefusal tracks movement along r (the refusal axis), which the weight route
has and the activation route lacks — even at equal task_gain. Plotting ΔRefusal vs the
refusal projection should collapse both routes onto one line; vs task_gain they diverge.

Reuses the single-point harness helpers (extract_refusal_shifts) + the projection helpers.
Adapters must be trained first (train_ddxplus_lora.py --n-train-cases/--adapter-dir).

    uv run python -m scripts.safety.run_route_safety_sweep \
        --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import (
    capture_resid,
    generate_reply,
    prompt_ids,
    set_seed,
    user_turn,
)
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import (
    build_cases,
    chat_messages,
    icl_messages,
    select_valid_indices,
)
from src.probes.lora_icl.shift_extraction import stack_shift_set
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import project_onto, refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless

import torch


def parse_letter(text: str):
    t = text.strip().upper()
    if t and t[0] in "ABCDE":
        return t[0]
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m else None


def ddxplus_accuracy(model, tokenizer, cases, device, layers_max_ctx,
                     fillers=None, k=0, fill_target=0.9, max_new=6):
    """MCQ accuracy; k>0 prepends k DDXPlus demonstrations (the ICL route)."""
    correct = n = 0
    for c in cases:
        final = chat_messages(c.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, layers_max_ctx, fill_target) \
            if k and fillers else final
        ids = prompt_ids(tokenizer, msgs)
        if len(ids) > layers_max_ctx - max_new:
            continue
        pred = parse_letter(generate_reply(model, tokenizer, ids, device, max_new))
        if pred is not None:
            n += 1
            correct += int(pred == c.gold_letter)
    return (correct / n if n else float("nan")), n


def condition_metrics(resids, refusals, base_resids, base_rate, layers, r_by_layer):
    """ΔRefusal + per-layer refusal-axis projection for one condition."""
    rate = refusal_rate(refusals)
    per_layer, mean_shift = [], {}
    for L in layers:
        shift = stack_shift_set(base_resids, resids, L)
        mean_shift[L] = shift.mean(axis=0)
        per_layer.append({
            "layer": L,
            "refusal_proj": float(np.mean(project_onto(shift, r_by_layer[L]))),
            "cos_r": vector_cosine(shift.mean(axis=0), r_by_layer[L]),
        })
    return rate, per_layer, mean_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    layers = cfg["extract"]["layers"]
    mc, ft = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    max_new = cfg["eval"]["max_new"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful, harmless = load_harmful(), load_harmless()
    n_dir, n_eval = cfg["direction"]["n_harmful"], cfg["eval"]["n_harmful"]
    harmful_fit = harmful[:n_dir]
    harmful_eval = harmful[n_dir:n_dir + n_eval]          # held-out, disjoint from the r-fit
    harmless_fit = harmless[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf, nt = cfg["ddxplus"]["n_filler"], cfg["ddxplus"]["n_task_eval"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    task_eval = build_cases(ds, valid[nf:nf + nt], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    # Refusal direction r (base, clean harmful vs harmless).
    h_res = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in harmful_fit]
    s_res = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in harmless_fit]
    r_by_layer = {L: refusal_direction(np.stack([x[L] for x in h_res]),
                                       np.stack([x[L] for x in s_res])) for L in layers}

    # Baseline (base, clean harmful) — the shared reference for all shifts.
    base_resids = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in harmful_eval]
    base_refusals = [generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new) for p in harmful_eval]
    base_rate = refusal_rate(base_refusals)
    base_acc, _ = ddxplus_accuracy(base, tokenizer, task_eval, args.device, mc)
    print(f"base refusal={base_rate:.3f}  base DDXPlus acc={base_acc:.3f}")

    conditions = []

    # ── activation route: many-shot ICL ──────────────────────────────────
    for k in cfg["doses"]["icl_shots"]:
        resids, refus, n_shots = [], [], None
        for p in harmful_eval:
            msgs = icl_messages(tokenizer, fillers[:k], user_turn(p), mc, ft)
            n_shots = (len(msgs) - 1) // 2  # actual demonstrations packed
            ids = prompt_ids(tokenizer, msgs)
            resids.append(capture_resid(base, capture, ids, args.device))
            refus.append(generate_reply(base, tokenizer, ids, args.device, max_new))
        rate, per_layer, _ = condition_metrics(resids, refus, base_resids, base_rate, layers, r_by_layer)
        acc, _ = ddxplus_accuracy(base, tokenizer, task_eval, args.device, mc, fillers, k, ft)
        conditions.append({"route": "icl", "dose": k, "n_shots": n_shots,
                           "refusal_rate": rate, "delta_refusal": base_rate - rate,
                           "task_acc": acc, "task_gain": acc - base_acc, "per_layer": per_layer})
        print(f"  ICL k={k} (~{n_shots} shots): refusal={rate:.3f} dRefusal={base_rate-rate:+.3f} acc={acc:.3f}")

    # ── weight route: LoRA adapters ──────────────────────────────────────
    lora_model = None
    for i, d in enumerate(cfg["doses"]["lora"]):
        if not Path(d["adapter_dir"]).exists():
            print(f"  [skip] adapter missing: {d['adapter_dir']}")
            continue
        name = f"d{i}"
        if lora_model is None:
            lora_model = PeftModel.from_pretrained(base, d["adapter_dir"], adapter_name=name)
        else:
            lora_model.load_adapter(d["adapter_dir"], adapter_name=name)
        lora_model.set_adapter(name)
        resids = [capture_resid(lora_model, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in harmful_eval]
        refus = [generate_reply(lora_model, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new) for p in harmful_eval]
        rate, per_layer, _ = condition_metrics(resids, refus, base_resids, base_rate, layers, r_by_layer)
        acc, _ = ddxplus_accuracy(lora_model, tokenizer, task_eval, args.device, mc)
        conditions.append({"route": "lora", "dose": d["cases"], "n_shots": None,
                           "refusal_rate": rate, "delta_refusal": base_rate - rate,
                           "task_acc": acc, "task_gain": acc - base_acc, "per_layer": per_layer})
        print(f"  LoRA cases={d['cases']}: refusal={rate:.3f} dRefusal={base_rate-rate:+.3f} acc={acc:.3f}")
    capture.remove()

    result = {"base_model": cfg["base_model"], "base_refusal": base_rate, "base_acc": base_acc,
              "n_eval": len(harmful_eval), "layers": layers, "conditions": conditions}
    (out_dir / "route_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_dir}/route_sweep.json  ({len(conditions)} conditions)")


if __name__ == "__main__":
    main()
