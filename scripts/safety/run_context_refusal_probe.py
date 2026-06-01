"""Fit a context-specific compliance direction and test it against the static r.

The drift stress test showed the static Arditi refusal direction r does NOT track the
context/finetuning behavioral change. This fits a behavior-grounded direction instead:

  d_comply = mean resid(model COMPLIED) - mean resid(model REFUSED)

on the DDXPlus LoRA across context fills (neutral filler), then:
  1. compares d_comply to r (cos) — are they different directions?
  2. checks whether the context-fill drift projects onto d_comply (and tracks the
     refusal rate across fills) better than onto r;
  3. ablates d_comply (best layer, all layers) on held-out prompts and measures whether
     it prevents the context refusal collapse.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_context_refusal_probe \
        --config configs/safety/context_refusal_probe_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.ablation_hook import DirectionalAblationHook
from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.refusal_direction import project_onto, refusal_direction
from src.probes.safety.safety_data import load_harmful
from scripts.safety.run_ablation_capstone import generate, set_seed
from scripts.safety.run_context_fill_baseline import alpaca_turns, fill_context


@torch.no_grad()
def resid_at(model, capture, tokenizer, msgs, device):
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    capture.clear()
    with capture.capturing():
        model(torch.tensor([ids], device=device), use_cache=False)
    return last_token_residual(capture.captured)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, fills, mc = cfg["layers"], cfg["fills"], cfg["max_ctx"]
    r = {L: np.load(Path(cfg["refusal_dir"]) / f"refusal_dir_L{L}.npy") for L in layers}

    turns = alpaca_turns(cfg)
    harmful = load_harmful()
    fit_p = harmful[cfg["fit"]["offset"] : cfg["fit"]["offset"] + cfg["fit"]["n_harmful"]]
    test_p = harmful[cfg["test"]["offset"] : cfg["test"]["offset"] + cfg["test"]["n_harmful"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()
    capture = PerTokenResidualCapture(base, layers)

    # --- 1. Fit d_comply from LoRA refuse/comply behavior under context ---
    comp = {L: [] for L in layers}
    refu = {L: [] for L in layers}
    drift_by_fill = {f: {L: [] for L in layers} for f in fills}
    refusal_fit = {f: [] for f in fills}
    for p in fit_p:
        resid_by_fill = {}
        for f in fills:
            msgs = fill_context(tokenizer, turns, p, mc, f)
            resid_by_fill[f] = resid_at(lora, capture, tokenizer, msgs, args.device)
            refused = is_refusal(generate(lora, tokenizer, msgs, args.device, cfg["max_new_refusal"]))
            refusal_fit[f].append(refused)
            for L in layers:
                (refu if refused else comp)[L].append(resid_by_fill[f][L])
        clean = resid_by_fill[fills[0]]
        for f in fills:
            for L in layers:
                drift_by_fill[f][L].append(resid_by_fill[f][L] - clean[L])

    d_comply, sep = {}, {}
    for L in layers:
        c, rf = np.stack(comp[L]), np.stack(refu[L])
        d_comply[L] = refusal_direction(c, rf)  # mean(comply) - mean(refuse), unit
        pooled = np.sqrt((c.var(0).mean() + rf.var(0).mean()) / 2) + 1e-9
        sep[L] = float(np.linalg.norm(c.mean(0) - rf.mean(0)) / pooled)
    best = max(layers, key=lambda L: sep[L])

    # --- 2. Compare directions + which one the drift tracks ---
    rows = []
    for f in fills:
        rec = {"fill": f, "refusal": float(np.mean(refusal_fit[f]))}
        for L in layers:
            D = np.stack(drift_by_fill[f][L])
            rec[f"drift_dcomply_L{L}"] = float(np.mean(project_onto(D, d_comply[L])))
            rec[f"drift_r_L{L}"] = float(np.mean(project_onto(D, r[L])))
        rows.append(rec)
    cos_dr = {L: vector_cosine(d_comply[L], r[L]) for L in layers}

    # --- 3. Causal: ablate d_comply (best layer) across all layers, held-out prompts ---
    causal = []
    for f in fills:
        rec = {"fill": f}
        rec["lora"] = float(np.mean([
            is_refusal(generate(lora, tokenizer, fill_context(tokenizer, turns, p, mc, f),
                                args.device, cfg["max_new_refusal"])) for p in test_p]))
        hook = DirectionalAblationHook(base, torch.tensor(d_comply[best]))
        rec["ablate"] = float(np.mean([
            is_refusal(generate(lora, tokenizer, fill_context(tokenizer, turns, p, mc, f),
                                args.device, cfg["max_new_refusal"])) for p in test_p]))
        hook.remove()
        causal.append(rec)
    capture.remove()

    lines = [
        "# Context-specific compliance direction vs the static refusal direction",
        "",
        f"`{cfg['base_model']}` DDXPlus LoRA | neutral filler | d_comply fit on "
        f"{len(fit_p)} prompts' refuse/comply behavior across fills | best layer L{best} "
        f"(separation {sep[best]:.2f}).",
        "",
        "## d_comply vs static r (cosine per layer)",
        "",
        "| Layer | cos(d_comply, r) | refuse/comply separation |",
        "|------:|-----------------:|-------------------------:|",
        *[f"| {L} | {cos_dr[L]:+.3f} | {sep[L]:.2f} |" for L in layers],
        "",
        "## Does the drift track d_comply or r? (signed projection; >0 = toward compliance)",
        "",
        f"| fill | LoRA refusal | drift·d_comply@L{best} | drift·r@L{best} |",
        "|-----:|-------------:|----------------------:|----------------:|",
        *[f"| {x['fill']:.0%} | {x['refusal']:.2f} | {x[f'drift_dcomply_L{best}']:+.1f} | "
          f"{x[f'drift_r_L{best}']:+.1f} |" for x in rows],
        "",
        "## Causal: ablate d_comply across fills (held-out prompts)",
        "",
        "| fill | LoRA refusal | + ablate d_comply |",
        "|-----:|-------------:|------------------:|",
        *[f"| {x['fill']:.0%} | {x['lora']:.2f} | {x['ablate']:.2f} |" for x in causal],
        "",
        "## Reading",
        "",
        "- If cos(d_comply, r) is low, the behavior-grounded direction is genuinely different from "
        "the static Arditi r — explaining why r did not track the context drift.",
        "- If drift·d_comply rises toward compliance as refusal falls (while drift·r does not), the "
        "context erosion lives along this behavior-fit direction, not the static one.",
        "- If ablating d_comply holds refusal high across fills, it is the causal context-refusal "
        "axis — a context-aware fix where the static-r story was incomplete.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(
        {"cos_d_r": cos_dr, "sep": sep, "best_layer": best, "drift": rows, "causal": causal}, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
