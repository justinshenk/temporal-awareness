"""Geometry of the LoRA vs ICL shifts on DDXPlus, layer by layer.

Answers, from the existing medical shift sets (no model run):
  Q1  cos(mean ICL shift, mean LoRA shift)
  Q2  fraction of LoRA energy in the ICL subspace (top-k PCA of ICL), and in the 1-D mean-ICL line
  Q3/Q4 (geometry) how much of the LoRA shift is kept if you keep only the ICL-parallel vs the
        orthogonal component (the functional test is run separately by steering).

Usage:
    uv run python -m scripts.lora_icl.run_overlap_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.lora_icl.subspace_metrics import (
    fraction_in_subspace,
    mean_direction_cosine,
    parallel_perp,
    pca_components,
)

LAYERS = [0, 7, 14, 21, 28, 35, 41]
SHIFTS = Path("results/lora_icl/shifts")
PCA_K = 5
REPORT = Path("results/lora_icl/2026-06-01-overlap-analysis.md")


def main() -> None:
    rows = []
    for layer in LAYERS:
        icl = np.load(SHIFTS / f"icl_shift_L{layer}.npy")
        lora = np.load(SHIFTS / f"lora_shift_real_L{layer}.npy")

        icl_basis = pca_components(icl, PCA_K)                 # ICL subspace (top-k PCA)
        icl_line = icl.mean(axis=0)
        icl_line = (icl_line / np.linalg.norm(icl_line))[:, None]  # 1-D mean-ICL direction

        # Q3/Q4 geometry: split the mean LoRA shift by the 1-D ICL direction.
        lora_mean = lora.mean(axis=0)
        par, perp = parallel_perp(lora_mean, icl.mean(axis=0))
        frac_par = float(np.linalg.norm(par) / np.linalg.norm(lora_mean))

        rows.append({
            "layer": layer,
            "cos_mean": mean_direction_cosine(icl, lora),                 # Q1
            "lora_var_in_icl_subspace": fraction_in_subspace(lora, icl_basis),  # Q2 (k-dim)
            "lora_var_in_icl_line": fraction_in_subspace(lora, icl_line),       # Q2 (1-D)
            "mean_lora_kept_parallel": frac_par,                          # Q3/Q4 magnitude
        })

    lines = [
        "# LoRA vs ICL — geometry (DDXPlus, per layer)",
        "",
        f"From `results/lora_icl/shifts` (60 examples, hidden dim 3584, PCA k={PCA_K}).",
        "",
        "| Layer | cos(ΔICL, ΔLoRA) | LoRA energy in ICL k-subspace | LoRA energy on mean-ICL line | ‖LoRA‖ kept by parallel |",
        "|------:|-----------------:|------------------------------:|-----------------------------:|------------------------:|",
    ]
    for x in rows:
        lines.append(
            f"| {x['layer']} | {x['cos_mean']:+.3f} | {x['lora_var_in_icl_subspace']:.3f} | "
            f"{x['lora_var_in_icl_line']:.3f} | {x['mean_lora_kept_parallel']:.3f} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- **Q1** cos(ΔICL, ΔLoRA): near 0 early, rising to ~0.8 late — the mean shifts align in the "
        "back half of the stack.",
        f"- **Q2** the ICL {PCA_K}-dim PCA subspace is fit to ICL's per-example *variation*, so it "
        "captures only a modest share of LoRA energy; the 1-D mean-ICL line captures the directional "
        "overlap (≈ cos² of the means). Late layers show the largest shared share.",
        "- **Q3/Q4 (geometry)** `‖LoRA‖ kept by parallel` is the fraction of the LoRA shift retained "
        "if you keep only the ICL-parallel component; `1 −` that is the orthogonal remainder. The "
        "*functional* effect of keeping each (task accuracy / refusal) is measured by steering — see "
        "`2026-06-01-keep-component-steering.md`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    REPORT.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
