#!/usr/bin/env python
"""
Train linear probes on model activations for temporal preference analysis.

Trains TWO probes:
1. Choice probe: predicts model's short_term vs long_term choice
2. Time horizon probe: predicts prompt's time horizon (<1yr vs >1yr)

Usage:
    # Default: uses 200 samples for fast iteration
    uv run python scripts/probes/train_probe_intertemporal.py

    # Use all samples (slower but more accurate)
    uv run python scripts/probes/train_probe_intertemporal.py --max-samples 0

    # Specify preference data
    uv run python scripts/probes/train_probe_intertemporal.py --preference-data abc123
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

from src.common.io import ensure_dir, load_json, parse_file_path, save_json, get_timestamp
from src.common.token_positions import build_position_labels, ResolvedPositionInfo
from src.profiler import P
from src.data import (
    load_pref_data_with_prompts,
    find_preference_data,
    get_preference_data_id,
)
from src.models import ModelRunner
from src.probes import LinearProbe, ProbeResult, prepare_samples, extract_activations
from src.viz import plot_layer_position_heatmap

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent


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
    regularization_C: float = 1.0
    n_cv_folds: int = 0


def load_config(args: argparse.Namespace) -> tuple[dict, ProbeConfig]:
    config_path = parse_file_path(
        args.config,
        default_dir_path=str(SCRIPTS_DIR / "configs"),
        default_ext=".json",
    )
    raw = load_json(config_path)
    print(f"Config: {config_path}")

    output_dir = Path(args.output or raw.get("output_dir", "out/probes"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return raw, ProbeConfig(
        preference_data_id=raw.get("preference_data_id"),
        model=raw.get("model", "Qwen/Qwen2.5-1.5B-Instruct"),
        layers=raw.get("layers", []),
        token_positions=raw.get("token_positions", []),
        test_split=raw.get("test_split", 0.2),
        random_seed=raw.get("random_seed", 42),
        output_dir=output_dir,
    )


def _train_single_probe(args):
    """Train a single probe."""
    layer, pos_idx, X_train, X_test, y_train, y_test, C, n_cv, seed = args
    probe = LinearProbe(C, random_state=seed)
    cv_mean, cv_std, train_acc = probe.train(X_train, y_train, n_cv)
    test_acc, test_prec, test_rec, test_f1 = probe.evaluate(X_test, y_test)
    result = ProbeResult(
        layer=layer, token_position=pos_idx,
        cv_accuracy_mean=cv_mean, cv_accuracy_std=cv_std, train_accuracy=train_acc,
        test_accuracy=test_acc, test_precision=test_prec, test_recall=test_rec,
        test_f1=test_f1, n_train=len(y_train), n_test=len(y_test), n_features=X_train.shape[1],
    )
    return (layer, pos_idx), result, probe


def train_probes_for_labels(
    X: dict, y: np.ndarray, config: ProbeConfig
) -> tuple[list[ProbeResult], dict]:
    """Train probes for all layer/position combinations."""
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=config.test_split,
        stratify=y, random_state=config.random_seed,
    )

    tasks = []
    for (layer, pos_idx), X_lp in sorted(X.items()):
        X_train, X_test = X_lp[train_idx], X_lp[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        tasks.append((layer, pos_idx, X_train, X_test, y_train, y_test,
                      config.regularization_C, config.n_cv_folds, config.random_seed))

    results, probes = [], {}
    for task in tasks:
        key, result, probe = _train_single_probe(task)
        results.append(result)
        probes[key] = probe

    results.sort(key=lambda r: (r.layer, r.token_position))
    return results, probes


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
    save_json({
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
            {"layer": r.layer, "position": r.token_position, "test_accuracy": r.test_accuracy}
            for r in results
        ],
    }, run_dir / f"{probe_type}_results.json")

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
        matrix, resolved_layers, pos_labels,
        run_dir / f"{probe_type}_heatmap.png",
        title=f"{title_map.get(probe_type, probe_type)}: {pref_data.model.split('/')[-1]}",
        subtitle=f"n_train={n_train}, n_test={n_test}",
        cbar_label="Test Accuracy", vmin=0.5, vmax=1.0,
        section_markers=section_markers,
    )

    # Intervention (steering vector)
    best_probe = probes.get((best.layer, best.token_position))
    if best_probe:
        save_json({
            "type": "steering_vector",
            "source": "linear_probe",
            "probe_type": probe_type,
            "model": pref_data.model,
            "layer": best.layer,
            "position": best.token_position,
            "test_accuracy": best.test_accuracy,
            "direction": best_probe.get_steering_vector().tolist(),
            "bias": best_probe.get_bias(),
        }, run_dir / f"{probe_type}_intervention.json")

    print(f"  {probe_type}: Best L{best.layer} P{best.token_position} = {best.test_accuracy:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train linear probes")
    parser.add_argument("--config", default="default", help="Config name or path")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--preference-data", type=str, help="Preference data ID or path")
    parser.add_argument("--max-samples", type=int, default=500, help="Limit samples (default=500, use 0 for all)")
    args = parser.parse_args()

    with P("total"):
        raw_config, config = load_config(args)

        preference_dir = PROJECT_ROOT / "out" / "preference_data"
        datasets_dir = PROJECT_ROOT / "out" / "datasets"

        preference_data_id = getattr(args, "preference_data", None) or config.preference_data_id

        # Load data
        with P("load_data"):
            if preference_data_id:
                pref_data = load_pref_data_with_prompts(preference_data_id, preference_dir, datasets_dir)
            else:
                recent_path = find_preference_data(preference_dir)
                if not recent_path:
                    print("Error: No preference data found.")
                    return 1
                recent_id = get_preference_data_id(recent_path)
                pref_data = load_pref_data_with_prompts(recent_id, preference_dir, datasets_dir)
            print(f"Loaded: {pref_data.dataset_id} ({len(pref_data.preferences)} samples)")

        print(f"Model: {pref_data.model}")

        # Load model
        with P("load_model"):
            print(f"\nLoading model: {pref_data.model}")
            runner = ModelRunner(pref_data.model)

        # Default layers
        if not config.layers:
            n = runner.n_layers
            config.layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

        # Default positions (including last token)
        if not config.token_positions:
            config.token_positions = [
                "situation", "task", "option_one", "option_two",
                "consider", "action", "choice_prefix",
                {"relative_to": "end", "offset": -1},  # Last token
            ]

        # Prepare samples for BOTH probe types
        with P("prepare_samples"):
            choice_samples, choice_labels = prepare_samples(
                pref_data, "choice", "choice", config.random_seed
            )
            horizon_samples, horizon_labels = prepare_samples(
                pref_data, "time_horizon", "time_horizon", config.random_seed
            )

        # Subsample if needed (use same indices for both)
        if args.max_samples > 0:
            from sklearn.model_selection import train_test_split
            n_samples = min(len(choice_samples), len(horizon_samples))
            if n_samples > args.max_samples:
                # Use choice labels for stratification (both should be similar size)
                _, choice_samples, _, choice_labels = train_test_split(
                    choice_samples, choice_labels, test_size=args.max_samples,
                    stratify=choice_labels, random_state=config.random_seed
                )
                _, horizon_samples, _, horizon_labels = train_test_split(
                    horizon_samples, horizon_labels, test_size=args.max_samples,
                    stratify=horizon_labels, random_state=config.random_seed
                )
                choice_samples, horizon_samples = list(choice_samples), list(horizon_samples)
                print(f"Subsampled to {args.max_samples} samples each")

        print(f"Choice samples: {len(choice_samples)} (0: {np.sum(choice_labels == 0)}, 1: {np.sum(choice_labels == 1)})")
        print(f"Horizon samples: {len(horizon_samples)} (0: {np.sum(horizon_labels == 0)}, 1: {np.sum(horizon_labels == 1)})")

        # Extract activations (use choice_samples, they have same prompts)
        print(f"\nExtracting activations ({len(config.layers)} layers x {len(config.token_positions)} positions)...")
        with P("extract_activations"):
            extraction = extract_activations(runner, choice_samples, config.layers, config.token_positions)

        resolved_layers = sorted(set(layer for layer, _ in extraction.X.keys()))

        # Create output directory
        ts = get_timestamp()
        run_dir = config.output_dir / ts
        ensure_dir(run_dir)

        # Train and save BOTH probes
        print("\n=== Training Choice Probe ===")
        with P("train_choice"):
            choice_results, choice_probes = train_probes_for_labels(extraction.X, choice_labels, config)
        save_probe_outputs("choice", choice_results, choice_probes, config, pref_data,
                          extraction.position_info, resolved_layers, run_dir)

        print("\n=== Training Time Horizon Probe ===")
        with P("train_horizon"):
            # Need to extract activations for horizon samples too (different sample order)
            horizon_extraction = extract_activations(runner, horizon_samples, config.layers, config.token_positions)
            horizon_results, horizon_probes = train_probes_for_labels(horizon_extraction.X, horizon_labels, config)
        save_probe_outputs("time_horizon", horizon_results, horizon_probes, config, pref_data,
                          horizon_extraction.position_info, resolved_layers, run_dir)

        # Save combined summary
        save_json({
            "timestamp": ts,
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "n_choice_samples": len(choice_samples),
            "n_horizon_samples": len(horizon_samples),
            "layers": resolved_layers,
            "token_positions": config.token_positions,
            "choice_best": {
                "layer": max(choice_results, key=lambda r: r.test_accuracy).layer,
                "accuracy": max(r.test_accuracy for r in choice_results),
            },
            "time_horizon_best": {
                "layer": max(horizon_results, key=lambda r: r.test_accuracy).layer,
                "accuracy": max(r.test_accuracy for r in horizon_results),
            },
        }, run_dir / "summary.json")

        print(f"\nResults saved to {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
