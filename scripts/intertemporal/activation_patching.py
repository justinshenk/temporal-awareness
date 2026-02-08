#!/usr/bin/env python
"""
Activation patching for intertemporal preference analysis.

Workflow:
1. For each component, call run_activation_patching() from src/experiments
2. Visualize results with heatmaps
3. Save positions.json and metadata.json

Usage:
    python scripts/intertemporal/activation_patching.py
    python scripts/intertemporal/activation_patching.py --no-components  # Skip MLP/attention
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import ModelRunner
from src.data import (
    load_pref_data_with_prompts,
    build_prompt_pairs,
    find_preference_data,
    get_preference_data_id,
)
from src.common.io import ensure_dir, save_json, get_timestamp
from src.common.token_positions import PositionsFile, PositionSpec
from src.viz import plot_layer_position_heatmap
from src.experiments import run_activation_patching
from src.common.profiler import P

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "activation_patching"

DEFAULT_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_pairs": 1,
    "threshold": 0.05,
    "n_layers_sample": 6,
    "position_step": 1,
}

# Component name mappings (defined once)
COMP_DISPLAY = {"resid_post": "Residual", "attn_out": "Attention", "mlp_out": "MLP"}
COMP_SHORT = {"resid_post": "resid", "attn_out": "attn", "mlp_out": "mlp"}


@dataclass
class Results:
    """Patching results."""
    position_sweeps: dict[str, np.ndarray] = field(default_factory=dict)
    full_sweeps: dict[str, np.ndarray] = field(default_factory=dict)
    filtered_positions: dict[str, list[int]] = field(default_factory=dict)
    token_labels: list[str] = field(default_factory=list)
    section_markers: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Script-specific utilities (not in src/)
# =============================================================================


def expand_positions(positions: list[int], max_pos: int, max_gap: int = 10) -> list[int]:
    """Expand positions to include in-between (if gap < max_gap)."""
    if len(positions) < 2:
        return positions
    result = set(positions)
    sorted_pos = sorted(positions)
    for i in range(len(sorted_pos) - 1):
        gap = sorted_pos[i + 1] - sorted_pos[i]
        if 1 < gap < max_gap:
            result.update(range(sorted_pos[i] + 1, min(sorted_pos[i + 1], max_pos)))
    return sorted(result)


def remap_markers(markers: dict[str, int], positions: list[int]) -> dict[str, int]:
    """Remap section markers to position indices."""
    result = {}
    for name, pos in markers.items():
        if pos in positions:
            result[name] = positions.index(pos)
        else:
            candidates = [i for i, p in enumerate(positions) if p <= pos]
            if candidates:
                result[name] = max(candidates)
    return result


# =============================================================================
# Visualization
# =============================================================================


def save_heatmap(matrix: np.ndarray, layers: list[int], labels: list[str],
                 markers: dict[str, int], path: Path, title: str) -> None:
    """Save heatmap with standard settings."""
    plot_layer_position_heatmap(matrix, layers, labels, path, title=title,
                                cbar_label="Recovery", vmin=0.0, vmax=1.0,
                                section_markers=markers)


def save_positions_json(full_sweep: np.ndarray, filtered_positions: list[int],
                        layers: list[int], token_labels: list[str],
                        section_markers: dict[str, int],
                        component: str, threshold: float,
                        model: str, dataset_id: str, output_dir: Path) -> None:
    """Save positions.json."""
    positions = []
    for li, layer in enumerate(layers):
        for pi, pos in enumerate(filtered_positions):
            val = float(full_sweep[li, pi])
            if not np.isnan(val) and val > threshold:
                section = None
                for sec, sec_pos in section_markers.items():
                    if pos >= sec_pos:
                        section = sec.replace("before_", "")
                positions.append(PositionSpec(position=pos, token=token_labels[pos],
                                              score=val, layer=layer, section=section))

    PositionsFile(model=model, method="activation_patching",
                  positions=sorted(positions, key=lambda p: p.score, reverse=True),
                  dataset_id=dataset_id, threshold=threshold, component=component
                  ).save(output_dir / f"positions_{COMP_SHORT[component]}.json")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preference-data", type=str)
    p.add_argument("--max-pairs", type=int, default=DEFAULT_CONFIG["max_pairs"])
    p.add_argument("--no-components", action="store_true", help="Skip MLP/attention analysis")
    p.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    p.add_argument("--n-layers-sample", type=int, default=DEFAULT_CONFIG["n_layers_sample"])
    p.add_argument("--position-step", type=int, default=DEFAULT_CONFIG["position_step"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    with P("total"):
        # Load data
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        pref_id = args.preference_data
        if not pref_id:
            recent = find_preference_data(pref_dir)
            if not recent:
                return print("No preference data found") or 1
            pref_id = get_preference_data_id(recent)

        with P("load"):
            pref_data = load_pref_data_with_prompts(pref_id, pref_dir, data_dir)
            pairs = build_prompt_pairs(pref_data, args.max_pairs, include_response=True)
            print(f"Loaded {len(pref_data.preferences)} samples, {len(pairs)} pairs")

        if not pairs:
            return print("No pairs!") or 1

        with P("model"):
            runner = ModelRunner(pref_data.model)
            print(f"Model: {runner.n_layers} layers")

        run_dir = args.output / get_timestamp()
        ensure_dir(run_dir)

        components = ["resid_post"] + ([] if args.no_components else ["attn_out", "mlp_out"])
        results = Results()

        # Run activation patching for each component via src/experiments
        for comp in components:
            print(f"\n=== {COMP_DISPLAY[comp]} ===")
            with P(f"patching_{comp}"):
                position_sweep, full_sweeps, filtered_positions, token_labels, section_markers = (
                    run_activation_patching(
                        runner=runner,
                        pref_data=pref_data,
                        max_pairs=args.max_pairs,
                        threshold=args.threshold,
                        position_sweep_component=comp,
                        full_sweep_components=[comp],
                        n_layers_sample=args.n_layers_sample,
                        position_step=args.position_step,
                    )
                )

            results.position_sweeps[comp] = position_sweep
            results.full_sweeps[comp] = full_sweeps[comp]
            results.filtered_positions[comp] = filtered_positions
            results.token_labels = token_labels
            results.section_markers = section_markers

            above = np.sum(position_sweep > args.threshold)
            print(f"  Max: {position_sweep.max():.3f}, {above} above threshold")
            print(f"  Filtered positions: {len(filtered_positions)}")

        # Compute layers used (must match n_layers_sample logic from src/)
        actual_n_layers = min(args.n_layers_sample, runner.n_layers)
        if actual_n_layers > 1:
            layers = [int(i * (runner.n_layers - 1) / (actual_n_layers - 1)) for i in range(actual_n_layers)]
        else:
            layers = [0]

        # Use resid_post filtered positions as the canonical set for visualization
        resid_positions = results.filtered_positions["resid_post"]
        expanded_positions = expand_positions(resid_positions, len(results.token_labels))
        vis_labels = [results.token_labels[i] for i in expanded_positions]
        vis_markers = remap_markers(results.section_markers, expanded_positions)

        # Save outputs
        print("\n=== Saving ===")
        for comp in components:
            short = COMP_SHORT[comp]
            filtered_positions = results.filtered_positions[comp]

            # Position sweep as broadcast matrix (using expanded positions for visualization)
            pos_vals = np.array([results.position_sweeps[comp][i] if i < len(results.position_sweeps[comp]) else 0.0
                                 for i in expanded_positions])
            broadcast = np.tile(pos_vals, (len(layers), 1))
            if comp != "resid_post":
                for pi, pos in enumerate(expanded_positions):
                    if pos >= len(results.position_sweeps[comp]) or results.position_sweeps[comp][pos] <= args.threshold:
                        broadcast[:, pi] = np.nan
            save_heatmap(broadcast, layers, vis_labels, vis_markers,
                         run_dir / f"position_sweep{'_' + short if comp != 'resid_post' else ''}.png",
                         f"Position Sweep ({COMP_DISPLAY[comp]})")

            # Full sweep heatmap (uses filtered_positions from run_activation_patching)
            full = results.full_sweeps[comp].copy()
            full_labels = [results.token_labels[i] for i in filtered_positions]
            full_markers = remap_markers(results.section_markers, filtered_positions)
            if comp != "resid_post":
                for pi, pos in enumerate(filtered_positions):
                    if pos >= len(results.position_sweeps[comp]) or results.position_sweeps[comp][pos] <= args.threshold:
                        full[:, pi] = np.nan
            save_heatmap(full, layers, full_labels, full_markers,
                         run_dir / f"heatmap_{short}.png", f"Activation Patching ({COMP_DISPLAY[comp]})")

            save_positions_json(results.full_sweeps[comp], filtered_positions,
                                layers, results.token_labels, results.section_markers,
                                comp, args.threshold, pref_data.model, pref_data.dataset_id, run_dir)

        n_significant = len(resid_positions)
        save_json({"timestamp": get_timestamp(), "model": pref_data.model,
                   "dataset_id": pref_data.dataset_id, "section_markers": results.section_markers,
                   "components": components, "threshold": args.threshold,
                   "n_significant": n_significant, "n_expanded": len(expanded_positions)},
                  run_dir / "metadata.json")

        print(f"\nSaved: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
