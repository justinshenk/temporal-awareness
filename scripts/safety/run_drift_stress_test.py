"""Stress-test the context-fill representation drift.

On harmful prompts with neutral (Alpaca) filler, as context fills measure — for base and
the DDXPlus LoRA — the prediction-site drift relative to the no-context baseline:

  - drift magnitude       (how far the representation moves)
  - drift · r̂  (signed)    (does the drift point toward compliance, i.e. along −r?)
  - cos(drift, r)          (is the drift *aligned* with the refusal axis, or just large?)
  - behavioral refusal     (does behavior flip?)
  - output entropy         (overconfidence)

The point: if refusal stays flat at the ceiling but drift·r grows negative with fill, the
drift is NOT benign — it is sub-threshold erosion the binary metric cannot see.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_drift_stress_test \
        --config configs/safety/drift_stress_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.distribution_metrics import softmax_entropy
from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.refusal_direction import project_onto
from src.probes.safety.safety_data import load_harmful
from scripts.safety.run_ablation_capstone import generate, set_seed
from scripts.safety.run_context_fill_baseline import alpaca_turns, fill_context


@torch.no_grad()
def forward_resid_entropy(model, capture, ids, device):
    capture.clear()
    with capture.capturing():
        out = model(torch.tensor([ids], device=device), use_cache=False)
    resid = last_token_residual(capture.captured)
    entropy = softmax_entropy(out.logits[0, -1].float().cpu().numpy())
    return resid, entropy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, fills = cfg["layers"], cfg["fills"]
    r = {L: np.load(Path(cfg["refusal_dir"]) / f"refusal_dir_L{L}.npy") for L in layers}

    turns = alpaca_turns(cfg)
    ho = cfg["harmful"]["offset"]
    harmful = load_harmful()[ho : ho + cfg["harmful"]["n"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()
    capture = PerTokenResidualCapture(base, layers)
    mc = cfg["max_ctx"]

    results = {}
    for model_name in ("base", "lora"):
        ctx = lora.disable_adapter() if model_name == "base" else nullcontext()
        with ctx:
            drift = {(f, L): [] for f in fills for L in layers}
            beh = {f: {"ref": [], "ent": []} for f in fills}
            for p in harmful:
                resid_by_fill = {}
                for f in fills:
                    msgs = fill_context(tokenizer, turns, p, mc, f)
                    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
                    resid, ent = forward_resid_entropy(lora, capture, ids, args.device)
                    resid_by_fill[f] = resid
                    beh[f]["ent"].append(ent)
                    beh[f]["ref"].append(is_refusal(generate(lora, tokenizer, msgs, args.device,
                                                              cfg["max_new_refusal"])))
                clean = resid_by_fill[fills[0]]
                for f in fills:
                    for L in layers:
                        drift[(f, L)].append(resid_by_fill[f][L] - clean[L])
            for f in fills:
                results[f"{model_name}|{f}"] = {
                    "refusal": float(np.mean(beh[f]["ref"])),
                    "entropy": float(np.mean(beh[f]["ent"])),
                }
                for L in layers:
                    D = np.stack(drift[(f, L)])
                    results[f"{model_name}|{f}|L{L}"] = {
                        "drift_norm": float(np.mean(np.linalg.norm(D, axis=1))),
                        "drift_dot_r": float(np.mean(project_onto(D, r[L]))),
                        "cos_drift_r": vector_cosine(D.mean(0), r[L]) if f != fills[0] else 0.0,
                    }
            print(f"{model_name}: " + " | ".join(
                f"f{f}: ref {results[f'{model_name}|{f}']['refusal']:.2f} "
                f"ent {results[f'{model_name}|{f}']['entropy']:.2f} "
                f"drift·r@35 {results[f'{model_name}|{f}|L35']['drift_dot_r']:+.1f}" for f in fills))
    capture.remove()

    lines = ["# Context-fill representation-drift stress test", "",
             f"`{cfg['base_model']}` | neutral Alpaca filler | {len(harmful)} harmful prompts | "
             f"drift = resid(fill) − resid(clean) at the prediction site | r̂ = refusal direction.", ""]
    for model_name in ("base", "lora"):
        lines += [f"## {model_name}", "",
                  "| fill | refusal | entropy | ‖drift‖@L35 | drift·r̂@L35 | cos(drift,r)@L35 |",
                  "|-----:|--------:|--------:|------------:|------------:|-----------------:|"]
        for f in fills:
            b = results[f"{model_name}|{f}"]
            d = results[f"{model_name}|{f}|L35"]
            lines.append(f"| {f:.0%} | {b['refusal']:.2f} | {b['entropy']:.2f} | "
                         f"{d['drift_norm']:.1f} | {d['drift_dot_r']:+.1f} | {d['cos_drift_r']:+.3f} |")
        lines.append("")

    lines += [
        "## Reading",
        "",
        "- **Benign drift** ⇒ as fill grows: refusal flat, entropy flat, and crucially drift·r̂ ≈ 0 / "
        "cos(drift,r) ≈ 0 — the representation moves but NOT along the refusal axis.",
        "- **Sub-threshold erosion** ⇒ refusal still flat (ceiling) but drift·r̂ grows negative and "
        "cos(drift,r) trends negative — the representation is latently sliding toward compliance "
        "before behavior flips.",
        "- Compare base vs LoRA: if the LoRA's drift·r̂ is far more negative (and its refusal "
        "collapses) while base stays ~0, the finetune amplifies the *safety-relevant* component of "
        "the same context drift.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
