"""Q3/Q4: keep only the parallel (to ICL) vs orthogonal component of the LoRA shift.

Steers the BASE model (no LoRA weights to confound) by per-layer steering vectors built
from the mean DDXPlus LoRA shift, decomposed against the mean ICL direction:
    full        = mean LoRA shift               (par + perp)
    parallel    = component along the ICL direction (the shared / task part)
    orthogonal  = the remainder                 (the finetune-only part)
and measures DDXPlus accuracy + refusal under each — the functional answer to
"what happens if you keep only the parallel / orthogonal component."

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_keep_component_steering \
        --config configs/safety/keep_component_gemma.yaml
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

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import build_cases, select_valid_indices
from src.probes.lora_icl.subspace_metrics import parallel_perp
from src.probes.safety.safety_data import load_harmful
from src.probes.safety.steering_hook import AdditionSteeringHook
from scripts.safety.run_ablation_capstone import evaluate, set_seed


def build_steer_vectors(cfg) -> dict[str, dict[int, torch.Tensor]]:
    shifts = Path(cfg["medical_shifts"])
    full, parallel, orthogonal = {}, {}, {}
    for layer in cfg["layers"]:
        icl = np.load(shifts / f"icl_shift_L{layer}.npy").mean(axis=0)
        lora = np.load(shifts / f"lora_shift_real_L{layer}.npy").mean(axis=0)
        par, perp = parallel_perp(lora, icl)
        full[layer] = torch.tensor(lora, dtype=torch.float32)
        parallel[layer] = torch.tensor(par[0], dtype=torch.float32)
        orthogonal[layer] = torch.tensor(perp[0], dtype=torch.float32)
    return {"full": full, "parallel": parallel, "orthogonal": orthogonal}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    steer = build_steer_vectors(cfg)

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    to = cfg["ddxplus"]["task_offset"]
    cases = build_cases(ds, valid[to : to + cfg["ddxplus"]["n_task_eval"]], evidence_db,
                        cfg["ddxplus"]["n_options"], cfg["seed"])
    ho = cfg["eval"]["harmful_offset"]
    harmful = load_harmful()[ho : ho + cfg["eval"]["n_harmful"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    # We evaluate the BASE model (adapter loaded but kept disabled throughout).
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()

    results = {}
    with lora.disable_adapter():
        results["base (no steer)"] = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
        for name in ("full", "parallel", "orthogonal"):
            hook = AdditionSteeringHook(base, steer[name])
            results[f"base + steer {name}"] = evaluate(
                lora, tokenizer, harmful, cases, args.device, cfg)
            hook.remove()

    lines = [
        "# Keep-only-component steering — what each LoRA component does (base model)",
        "",
        f"Base `{cfg['base_model']}` steered by the mean DDXPlus LoRA shift at layers "
        f"{cfg['layers']} | {len(cases)} medical, {len(harmful)} harmful | LoRA reference: "
        "acc 1.00, refusal 0.84.",
        "",
        "| Steering | DDXPlus acc (task↑) | refusal rate |",
        "|----------|--------------------:|-------------:|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['task_acc']:.3f} | {r['refusal_rate']:.3f} |")

    b = results["base (no steer)"]
    par = results["base + steer parallel"]
    orth = results["base + steer orthogonal"]
    steered_refusals = [r["refusal_rate"] for k, r in results.items() if "steer" in k]
    refusal_confounded = max(steered_refusals) < 0.05  # all collapsed → magnitude artifact
    lines += ["", "## Reading", ""]
    if refusal_confounded:
        lines.append(
            "- **CONFOUND — refusal uninformative:** every steering condition drove refusal to ~0, "
            "a non-specific magnitude artifact (large late-layer shifts over-drive the model). The "
            "safety read needs a coefficient sweep; do not interpret the refusal column."
        )
    lines += [
        f"- **Parallel-only** (ICL-shared): DDXPlus acc {b['task_acc']:.2f}→{par['task_acc']:.2f}.",
        f"- **Orthogonal-only** (finetune-only): DDXPlus acc {b['task_acc']:.2f}→{orth['task_acc']:.2f}.",
        "- Steering at the 5 extracted late layers only (not all 42), coeff 1.0 — an approximate, "
        "possibly over-driven sufficiency test; the `full` row shows how well it reproduces the LoRA.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
