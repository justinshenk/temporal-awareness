"""Is the finetune's safety drift in the part ICL does NOT share?

On medical (DDXPlus) inputs, split the finetune (LoRA) activation shift into:
    par  = component aligned with the ICL shift   (the shared / task-adaptation direction)
    perp = the orthogonal remainder               (the finetune-only direction)
and project each onto the refusal direction r. Hypothesis: the shared part is ~r-neutral
while the finetune-only part carries the −r (toward-compliance) drift — i.e. the beneficial
task subspace is shared with ICL, and the safety-eroding component is the finetune-only remainder.

Uses only existing artifacts (no model run):
  - medical shifts  : results/lora_icl/shifts/{icl_shift,lora_shift_real}_L*.npy
  - refusal dir r   : results/safety/shifts/refusal_dir_L*.npy

Usage:
    uv run python -m scripts.safety.run_subspace_decomposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.probes.lora_icl.subspace_metrics import parallel_perp, vector_cosine

LAYERS = [0, 7, 14, 21, 28, 35, 41]
MEDICAL_SHIFTS = Path("results/lora_icl/shifts")
SAFETY_SHIFTS = Path("results/safety/shifts")
REPORT = Path("results/safety/2026-06-01-subspace-decomposition.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    def decompose(icl, lora, r, layer) -> dict:
        shared_dir = icl.mean(axis=0)                # the ICL / task-adaptation direction
        par, perp = parallel_perp(lora, shared_dir)  # split the finetune shift per example
        par_mean, perp_mean = par.mean(axis=0), perp.mean(axis=0)
        return {
            "layer": layer,
            "cos_shared_r": vector_cosine(par_mean, r),   # does shared part point along r?
            "cos_perp_r": vector_cosine(perp_mean, r),    # does finetune-only part point along r?
            "frac_shift_shared": float(
                np.linalg.norm(par_mean) / np.linalg.norm(lora.mean(axis=0))
            ),
            "cos_shared_dir_r": vector_cosine(shared_dir, r),
        }

    rows, rows_harmful = [], []
    for layer in layers:
        r = np.load(SAFETY_SHIFTS / f"refusal_dir_L{layer}.npy")
        rows.append(decompose(
            np.load(MEDICAL_SHIFTS / f"icl_shift_L{layer}.npy"),
            np.load(MEDICAL_SHIFTS / f"lora_shift_real_L{layer}.npy"), r, layer))
        rows_harmful.append(decompose(
            np.load(SAFETY_SHIFTS / f"icl_shift_L{layer}.npy"),
            np.load(SAFETY_SHIFTS / f"lora_shift_L{layer}.npy"), r, layer))

    def table(rs):
        out = [
            "| Layer | cos(shared `par`, r) | cos(finetune-only `perp`, r) | shift frac in shared | cos(shared dir, r) |",
            "|------:|---------------------:|-----------------------------:|---------------------:|-------------------:|",
        ]
        for x in rs:
            out.append(
                f"| {x['layer']} | {x['cos_shared_r']:+.3f} | {x['cos_perp_r']:+.3f} | "
                f"{x['frac_shift_shared']:.3f} | {x['cos_shared_dir_r']:+.3f} |"
            )
        return out

    def late_mean(rs, key):
        return float(np.mean([x[key] for x in rs if x["layer"] >= 21]))

    lines = [
        "# Decomposition — is finetuning's safety axis in the non-shared part?",
        "",
        "Finetune (LoRA) shift split into the ICL-aligned `par` (shared task direction) and the "
        "orthogonal `perp` (finetune-only). Magnitude-normalized cosine with the refusal direction r "
        "isolates *direction* (negative = toward compliance), robust to large shift magnitudes.",
        "",
        "## On medical (DDXPlus) inputs — where the shared component is large",
        "",
        *table(rows),
        "",
        f"Late-layer (≥21): shared `par`·r̂ = {late_mean(rows, 'cos_shared_r'):+.3f}, "
        f"finetune-only `perp`·r̂ = {late_mean(rows, 'cos_perp_r'):+.3f}, "
        f"shared fraction ≈ {late_mean(rows, 'frac_shift_shared'):.2f}, "
        f"cos(shared dir, r) = {late_mean(rows, 'cos_shared_dir_r'):+.3f}.",
        "",
        "## On harmful inputs — where safety erosion manifests",
        "",
        *table(rows_harmful),
        "",
        f"Late-layer (≥21): shared `par`·r̂ = {late_mean(rows_harmful, 'cos_shared_r'):+.3f}, "
        f"finetune-only `perp`·r̂ = {late_mean(rows_harmful, 'cos_perp_r'):+.3f}, "
        f"shared fraction ≈ {late_mean(rows_harmful, 'frac_shift_shared'):.2f}.",
        "",
        "## Reading",
        "",
        "- **Shared part is safety-neutral.** On medical inputs the shared task direction is the bulk "
        f"of the finetune shift (≈{late_mean(rows, 'frac_shift_shared'):.0%} of it late) yet is "
        f"~orthogonal to the refusal axis (cos(shared dir, r) ≈ {late_mean(rows, 'cos_shared_dir_r'):+.2f}). "
        "The part ICL reproduces does not point along refusal.",
        "- **Harm is finetune-only.** On harmful inputs the finetune shift is almost entirely the "
        f"orthogonal `perp` (shared fraction ≈ {late_mean(rows_harmful, 'frac_shift_shared'):.2f}), and "
        f"that `perp` carries the toward-compliance drift (cos(perp, r) = {late_mean(rows_harmful, 'cos_perp_r'):+.2f}).",
        "- **Verdict:** supports \"shared subspace = beneficial task adaptation\" — the ICL-shared "
        "component is safety-neutral and the compliance drift lives in the finetune-only direction. "
        "Refinement: that harmful direction is input-gated (near-zero refusal-axis content on benign "
        "medical inputs), so finetuning installs a weight change whose harm is triggered by harmful "
        "input rather than a static always-on compliance vector.",
    ]

    REPORT.write_text("\n".join(lines) + "\n")
    REPORT.with_suffix(".json").write_text(
        json.dumps({"medical": rows, "harmful": rows_harmful}, indent=2)
    )
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
