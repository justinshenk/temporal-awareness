"""Coefficient sweep for keep-only-component steering.

Steers base by {full, parallel, orthogonal} × {coeff} and measures DDXPlus accuracy +
refusal, to (a) find a non-over-driven regime where refusal is not destroyed for every
condition, and (b) read the parallel-vs-orthogonal effect at matched, sane magnitude.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_keep_component_sweep \
        --config configs/safety/keep_component_sweep_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import build_cases, select_valid_indices
from src.probes.safety.safety_data import load_harmful
from src.probes.safety.steering_hook import AdditionSteeringHook
from scripts.safety.run_ablation_capstone import evaluate, set_seed
from scripts.safety.run_keep_component_steering import build_steer_vectors


def scaled(vectors: dict[int, torch.Tensor], c: float) -> dict[int, torch.Tensor]:
    return {li: v * c for li, v in vectors.items()}


def grid(results, key, components, coeffs) -> list[str]:
    lines = ["| component | " + " | ".join(f"c={c}" for c in coeffs) + " |",
             "|" + "---|" * (len(coeffs) + 1)]
    for comp in components:
        cells = " | ".join(f"{results[(comp, c)][key]:.2f}" for c in coeffs)
        lines.append(f"| {comp} | {cells} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    steer = build_steer_vectors(cfg)
    components = ["full", "parallel", "orthogonal"]
    coeffs = cfg["coeffs"]

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
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()

    results = {}
    with lora.disable_adapter():
        ref = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
        for comp in components:
            for c in coeffs:
                hook = AdditionSteeringHook(base, scaled(steer[comp], c))
                results[(comp, c)] = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
                hook.remove()
                r = results[(comp, c)]
                print(f"{comp} c={c}: acc {r['task_acc']:.2f} refusal {r['refusal_rate']:.2f}")

    lines = [
        "# Keep-only-component steering — coefficient sweep (base model)",
        "",
        f"Base `{cfg['base_model']}` | layers {cfg['layers']} | {len(cases)} medical, "
        f"{len(harmful)} harmful | base (no steer): acc {ref['task_acc']:.2f}, "
        f"refusal {ref['refusal_rate']:.2f} | LoRA ref: acc 1.00, refusal 0.84.",
        "",
        "## DDXPlus accuracy (task↑)",
        "",
        *grid(results, "task_acc", components, coeffs),
        "",
        "## Refusal rate (safer↑)",
        "",
        *grid(results, "refusal_rate", components, coeffs),
        "",
        "## Reading",
        "",
        "- Find the largest coeff where refusal is NOT destroyed for all components; read the "
        "parallel-vs-orthogonal accuracy gap there.",
        "- If orthogonal > parallel on accuracy across sane coeffs, the task capability is in the "
        "LoRA-orthogonal component, and the ICL-shared (parallel) direction is a context-mode signal "
        "that does not carry the answer.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(
        json.dumps({f"{c}|{k}": v for (c, k), v in results.items()}, indent=2)
    )
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
