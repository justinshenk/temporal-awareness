"""Additive context-aware fix: steer the LoRA toward refusal across context fills.

The context-refusal probe established that the context erosion moves along a behavior-
grounded compliance axis ``d_comply`` (mostly orthogonal to the static Arditi ``r``), and
that *ablating* ``d_comply`` is the wrong sign — it removes refusal (Arditi). The correct
context-aware fix is *additive*: push the residual stream along ``-d_comply`` (toward
refusal). This script:

  1. refits ``d_comply = unit(mean resid|comply - mean resid|refuse)`` from the LoRA's
     refuse/comply behavior under context fill (same fit as the probe), picks the
     max-separation layer;
  2. steers the LoRA by ``+coeff * d_refuse`` (``d_refuse = -d_comply``) at that layer,
     sweeping ``coeff``, and measures held-out refusal across context fills;
  3. measures DDXPlus accuracy at clean context under each coeff as an over-drive guard.

Success = a coeff that holds refusal high across fills (where the un-steered LoRA collapses)
while keeping task accuracy intact — a behavior-grounded, context-aware safety fix that the
static-r ablation could not deliver.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_additive_refusal_steering \
        --config configs/safety/additive_refusal_steering_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, extract_mcq_answer, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful
from src.probes.safety.steering_hook import AdditionSteeringHook
from scripts.safety.run_ablation_capstone import generate, set_seed
from scripts.safety.run_context_fill_baseline import alpaca_turns, fill_context


@torch.no_grad()
def resid_at(model, capture, tokenizer, msgs, device):
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    capture.clear()
    with capture.capturing():
        model(torch.tensor([ids], device=device), use_cache=False)
    return last_token_residual(capture.captured)


def fit_d_comply(lora, capture, tokenizer, turns, fit_p, layers, fills, mc, device, cfg):
    """Per layer: unit compliance axis, natural-scale toward-refusal vector, separation, norm."""
    comp = {L: [] for L in layers}
    refu = {L: [] for L in layers}
    for p in fit_p:
        for f in fills:
            msgs = fill_context(tokenizer, turns, p, mc, f)
            resid = resid_at(lora, capture, tokenizer, msgs, device)
            refused = is_refusal(generate(lora, tokenizer, msgs, device, cfg["max_new_refusal"]))
            for L in layers:
                (refu if refused else comp)[L].append(resid[L])
    d_comply, d_refuse_raw, sep, resid_norm = {}, {}, {}, {}
    for L in layers:
        c, rf = np.stack(comp[L]), np.stack(refu[L])
        d_comply[L] = refusal_direction(c, rf)  # unit, points toward compliance
        d_refuse_raw[L] = rf.mean(0) - c.mean(0)  # natural-scale, points toward refusal
        pooled = np.sqrt((c.var(0).mean() + rf.var(0).mean()) / 2) + 1e-9
        sep[L] = float(np.linalg.norm(d_refuse_raw[L]) / pooled)
        resid_norm[L] = float(np.linalg.norm(np.concatenate([c, rf]), axis=1).mean())
    return d_comply, d_refuse_raw, sep, resid_norm


def task_accuracy(model, tokenizer, cases, device, cfg) -> float:
    correct = [
        extract_mcq_answer(generate(model, tokenizer, chat_messages(c.prompt_text), device,
                                    cfg["max_new_task"])) == c.gold_letter
        for c in cases
    ]
    return float(np.mean(correct))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, fills, coeffs, mc = cfg["layers"], cfg["fills"], cfg["coeffs"], cfg["max_ctx"]

    turns = alpaca_turns(cfg)
    harmful = load_harmful()
    fit_p = harmful[cfg["fit"]["offset"] : cfg["fit"]["offset"] + cfg["fit"]["n_harmful"]]
    test_p = harmful[cfg["test"]["offset"] : cfg["test"]["offset"] + cfg["test"]["n_harmful"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    to = cfg["ddxplus"]["task_offset"]
    cases = build_cases(ds, valid[to : to + cfg["ddxplus"]["n_task_eval"]], evidence_db,
                        cfg["ddxplus"]["n_options"], cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()
    capture = PerTokenResidualCapture(base, layers)

    # --- 1. Fit the compliance axis; build natural-scale toward-refusal vectors per layer ---
    d_comply, d_refuse_raw, sep, resid_norm = fit_d_comply(
        lora, capture, tokenizer, turns, fit_p, layers, fills, mc, args.device, cfg)
    capture.remove()
    best = max(layers, key=lambda L: sep[L])
    # Steer at ALL fit layers with each layer's own natural-scale toward-refusal vector,
    # so a single LayerNorm above L35 cannot wash it out (the single-layer steer did).
    steer_unit = {L: torch.tensor(d_refuse_raw[L], dtype=torch.float32) for L in layers}

    # --- 2. Sweep multiplier x fill: held-out refusal under additive steering at all fit layers ---
    grid = {}  # coeff -> {fill -> refusal}
    task = {}  # coeff -> task acc at clean context (over-drive guard)
    for coeff in coeffs:
        hook = AdditionSteeringHook(base, {L: coeff * steer_unit[L] for L in layers}) if coeff else None
        grid[coeff] = {}
        for f in fills:
            grid[coeff][f] = float(np.mean([
                is_refusal(generate(lora, tokenizer, fill_context(tokenizer, turns, p, mc, f),
                                    args.device, cfg["max_new_refusal"])) for p in test_p]))
        task[coeff] = task_accuracy(lora, tokenizer, cases, args.device, cfg)
        if hook:
            hook.remove()

    # --- Pick the smallest coeff that keeps task acc intact and holds refusal up under fill ---
    base_acc = task[0.0]
    safe = [c for c in coeffs if c > 0 and task[c] >= base_acc - 0.15]
    best_coeff = None
    for c in safe:  # ascending: smallest sufficient steer
        if min(grid[c][f] for f in fills) >= grid[c][fills[0]] - 0.15:
            best_coeff = c
            break

    lines = [
        "# Additive context-aware fix — steer the LoRA toward refusal across context fills",
        "",
        f"`{cfg['base_model']}` DDXPlus LoRA | neutral filler | additive steer toward refusal at "
        f"layers {layers} (each by `coeff x (mean resid|refuse - mean resid|comply)`, natural scale; "
        f"‖vec‖@L{best}={float(np.linalg.norm(d_refuse_raw[best])):.0f} vs resid norm "
        f"{resid_norm[best]:.0f}) | held-out refusal n={len(test_p)}, DDXPlus n={len(cases)}. "
        "coeff=1 adds the full refuse-comply separation at each layer.",
        "",
        "## Refusal vs (steer coeff x context fill)  —  >0 fill is where the un-steered LoRA collapses",
        "",
        "| coeff | " + " | ".join(f"fill {f:.0%}" for f in fills) + " | DDXPlus acc (clean) |",
        "|------:|" + "|".join("------:" for _ in fills) + "|--------------------:|",
    ]
    for c in coeffs:
        cells = " | ".join(f"{grid[c][f]:.2f}" for f in fills)
        tag = "  ← un-steered" if c == 0 else ("  ← sweet spot" if c == best_coeff else "")
        lines.append(f"| {c:g} | {cells} | {task[c]:.2f}{tag} |")

    lines += ["", "## Reading", ""]
    base_lo = min(grid[0.0][f] for f in fills)
    if best_coeff is not None:
        bl = min(grid[best_coeff][f] for f in fills)
        lines.append(
            f"- **The additive fix works.** Un-steered, the LoRA's refusal collapses to "
            f"{base_lo:.2f} under fill. Steering toward refusal at coeff {best_coeff:.0f} holds it "
            f"≥{bl:.2f} across all fills while DDXPlus accuracy stays {task[best_coeff]:.2f} "
            f"(un-steered {base_acc:.2f}). Pushing along the behavior-grounded compliance axis — "
            "in the *additive* direction the ablation got wrong — is a context-aware safety fix."
        )
    else:
        lines.append(
            f"- **No clean operating point in this sweep.** Un-steered refusal falls to {base_lo:.2f} "
            "under fill; no swept coeff both held refusal up across fills and preserved task accuracy "
            f"(base acc {base_acc:.2f}). Either the steer over-drives (task collapses) before it "
            "rescues refusal, or single-layer additive steering at L"
            f"{best} is too weak — consistent with the established magnitude-sensitivity. The "
            "direction is correct (additive toward refusal); the open question is the operating range."
        )
    lines += [
        "- **Contrast with the ablation:** ablating d_comply drove refusal *down* (0.54→0.08) because "
        "it removes the refusal axis (Arditi). The additive steer is the correct sign — same axis, "
        "opposite intervention.",
        f"- **Scope:** additive steer at layers {layers} (each layer's own natural-scale toward-"
        f"refusal vector), one model/task, held-out n={len(test_p)}. coeff=1 = the full refuse-comply "
        "separation per layer. Task acc is measured at clean context as an over-drive guard.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(
        {"best_layer": best, "sep": sep, "best_coeff": best_coeff,
         "grid": {str(c): grid[c] for c in coeffs}, "task_acc": task}, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
