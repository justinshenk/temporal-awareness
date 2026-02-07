#!/usr/bin/env python
"""
Train linear probes on model activations for temporal preference analysis.

Trains TWO probes:
1. Choice probe: predicts model's short_term vs long_term choice
2. Time horizon probe: predicts prompt's time horizon (<1yr vs >1yr)

Usage:
    # Default: uses 20 samples for fast iteration
    uv run python scripts/intertemporal/train_probe.py

    # Use all samples (slower but more accurate)
    uv run python scripts/intertemporal/train_probe.py --max-samples 0

    # Specify preference data
    uv run python scripts/intertemporal/train_probe.py --preference-data abc123
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Bootstrap path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.io import ensure_dir, save_json, get_timestamp
from src.common.token_positions import build_position_labels, ResolvedPositionInfo
from src.experiments import run_probe_training
from src.profiler import P
from src.data import (
    load_pref_data_with_prompts,
    get_preference_data_id,
)
from src.models import ModelRunner
from src.probes import ProbeResult
from src.viz import plot_layer_position_heatmap

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_samples": 20,  # Small but meaningful for testing
    "layers": [],  # Empty = auto-select 5 evenly-spaced layers
    "token_positions": [
        "option_one",
        "option_two",
        {"relative_to": "end", "offset": -1},
    ],
    "test_split": 0.2,
    "random_seed": 42,
}


@dataclass
class ProbeConfig:
    """Configuration for probe training."""

    preference_data_id: Optional[str]
    model: str
    layers: list[int]
    token_positions: list
    test_split: float
    random_seed: int
    output_dir: Path


def save_probe_outputs(
    probe_type: str,
    results: list[ProbeResult],
    probes: dict,
    config: ProbeConfig,
    pref_data,
    position_info: ResolvedPositionInfo,
    resolved_layers: list[int],
    run_dir: Path,
) -> None:
    """Save outputs for a single probe type."""
    pos_labels = build_position_labels(config.token_positions, position_info)
    best = max(results, key=lambda r: r.test_accuracy)

    # JSON results
    save_json(
        {
            "probe_type": probe_type,
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "layers": resolved_layers,
            "position_labels": pos_labels,
            "position_tokens": position_info.tokens,
            "best": {
                "layer": best.layer,
                "position": best.token_position,
                "test_accuracy": best.test_accuracy,
            },
            "results": [
                {
                    "layer": r.layer,
                    "position": r.token_position,
                    "test_accuracy": r.test_accuracy,
                }
                for r in results
            ],
        },
        run_dir / f"{probe_type}_results.json",
    )

    # Heatmap with section markers mapped to keyword positions
    matrix = np.full((len(resolved_layers), len(config.token_positions)), np.nan)
    layer_idx = {l: i for i, l in enumerate(resolved_layers)}
    for r in results:
        matrix[layer_idx[r.layer], r.token_position] = r.test_accuracy

    n_train = results[0].n_train if results else 0
    n_test = results[0].n_test if results else 0

    # Map keyword positions to section markers (at RIGHT edge of bin)
    section_markers = {}
    for i, pos_spec in enumerate(config.token_positions):
        pos_name = pos_spec if isinstance(pos_spec, str) else ""
        if isinstance(pos_spec, dict):
            pos_name = pos_spec.get("relative_to", "")
        if pos_name == "option_one":
            section_markers["before_choices"] = i - 1  # Before first option
        elif pos_name == "consider":
            section_markers["before_time_horizon"] = i - 1  # Before consider
        elif pos_name == "choice_prefix":
            section_markers["before_choice_output"] = i - 1  # Before choice

    title_map = {"choice": "Choice Probe", "time_horizon": "Time Horizon Probe"}
    plot_layer_position_heatmap(
        matrix,
        resolved_layers,
        pos_labels,
        run_dir / f"{probe_type}_heatmap.png",
        title=f"{title_map.get(probe_type, probe_type)}: {pref_data.model.split('/')[-1]}",
        subtitle=f"n_train={n_train}, n_test={n_test}",
        cbar_label="Test Accuracy",
        vmin=0.5,
        vmax=1.0,
        section_markers=section_markers,
    )

    # Intervention (steering vector)
    best_probe = probes.get((probe_type, best.layer, best.token_position))
    if best_probe:
        save_json(
            {
                "type": "steering_vector",
                "source": "linear_probe",
                "probe_type": probe_type,
                "model": pref_data.model,
                "layer": best.layer,
                "position": best.token_position,
                "test_accuracy": best.test_accuracy,
                "direction": best_probe.get_steering_vector().tolist(),
                "bias": best_probe.get_bias(),
            },
            run_dir / f"{probe_type}_intervention.json",
        )

    print(
        f"  {probe_type}: Best L{best.layer} P{best.token_position} = {best.test_accuracy:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train linear probes")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--preference-data", type=str, help="Preference data ID or path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples (default from config, use 0 for all)",
    )
    args = parser.parse_args()

    with P("total"):
        # Build config from defaults + CLI overrides
        max_samples = (
            args.max_samples
            if args.max_samples is not None
            else DEFAULT_CONFIG["max_samples"]
        )
        layers = list(DEFAULT_CONFIG["layers"])
        token_positions = list(DEFAULT_CONFIG["token_positions"])
        test_split = DEFAULT_CONFIG["test_split"]
        random_seed = DEFAULT_CONFIG["random_seed"]

        output_dir = (
            Path(args.output) if args.output else PROJECT_ROOT / "out" / "probes"
        )
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        config = ProbeConfig(
            preference_data_id=getattr(args, "preference_data", None),
            model=DEFAULT_CONFIG["model"],
            layers=layers,
            token_positions=token_positions,
            test_split=test_split,
            random_seed=random_seed,
            output_dir=output_dir,
        )

        preference_dir = PROJECT_ROOT / "out" / "preference_data"
        datasets_dir = PROJECT_ROOT / "out" / "datasets"

        preference_data_id = (
            getattr(args, "preference_data", None) or config.preference_data_id
        )

        # Load data
        with P("load_data"):
            if preference_data_id:
                pref_data = load_pref_data_with_prompts(
                    preference_data_id, preference_dir, datasets_dir
                )
            else:
                recent_path = find_preference_data(preference_dir)
                if not recent_path:
                    print("Error: No preference data found.")
                    return 1
                recent_id = get_preference_data_id(recent_path)
                pref_data = load_pref_data_with_prompts(
                    recent_id, preference_dir, datasets_dir
                )
            print(
                f"Loaded: {pref_data.dataset_id} ({len(pref_data.preferences)} samples)"
            )

        print(f"Model: {pref_data.model}")

        # Load model
        with P("load_model"):
            print(f"\nLoading model: {pref_data.model}")
            runner = ModelRunner(pref_data.model)

        # Resolve layers: if empty, auto-select 5 evenly-spaced
        if not config.layers:
            n = runner.n_layers
            config.layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

        # Run probe training via shared module
        print(
            f"\nTraining probes ({len(config.layers)} layers x {len(config.token_positions)} positions)..."
        )
        with P("run_probe_training"):
            results, probes = run_probe_training(
                runner,
                pref_data,
                config.layers,
                config.token_positions,
                test_split=config.test_split,
                random_seed=config.random_seed,
                max_samples=max_samples,
            )

        if not results:
            print("Error: Probe training returned no results.")
            return 1

        meta = results["_meta"]
        resolved_layers = meta["layers"]

        # Create output directory
        ts = get_timestamp()
        run_dir = config.output_dir / ts
        ensure_dir(run_dir)

        # Save outputs for each probe type
        for probe_type in ["choice", "time_horizon"]:
            probe_results = results.get(probe_type, [])
            if not probe_results:
                print(f"  Skipping {probe_type} (no results)")
                continue

            position_info_key = (
                f"{probe_type}_position_info"
                if probe_type == "time_horizon"
                else "choice_position_info"
            )
            position_info = meta.get(position_info_key) or meta["choice_position_info"]

            save_probe_outputs(
                probe_type,
                probe_results,
                probes,
                config,
                pref_data,
                position_info,
                resolved_layers,
                run_dir,
            )

        # Save combined summary
        choice_results = results.get("choice", [])
        horizon_results = results.get("time_horizon", [])

        summary = {
            "timestamp": ts,
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "layers": resolved_layers,
            "token_positions": config.token_positions,
        }
        if choice_results:
            summary["choice_best"] = {
                "layer": max(choice_results, key=lambda r: r.test_accuracy).layer,
                "accuracy": max(r.test_accuracy for r in choice_results),
            }
        if horizon_results:
            summary["time_horizon_best"] = {
                "layer": max(horizon_results, key=lambda r: r.test_accuracy).layer,
                "accuracy": max(r.test_accuracy for r in horizon_results),
            }

        save_json(summary, run_dir / "summary.json")
        print(f"\nResults saved to {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
