#!/usr/bin/env python
"""
Contrastive activation analysis for intertemporal preference steering.

Computes steering vectors as mean activation difference between choice classes
at positions identified by activation patching.

Usage:
    uv run python scripts/analysis/contrastive_intertemporal.py \
        --positions out/activation_patching/<timestamp>/positions_resid_post.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.io import ensure_dir, save_json, load_json, get_timestamp
from src.common.positions_schema import PositionsFile
from src.data import load_pref_data_with_prompts, find_preference_data, get_preference_data_id
from src.models import ModelRunner
from src.probes import prepare_samples
from src.profiler import P

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Defaults
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "contrastive"
DEFAULT_MAX_SAMPLES = 500
DEFAULT_TOP_N = 1
DEFAULT_SEED = 42
EPSILON = 1e-8  # Numerical stability


def extract_activations_at_positions(
    runner: ModelRunner,
    samples: list,
    layer_positions: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Extract activations at multiple (layer, position) pairs in single pass.

    Returns:
        Dict mapping (layer, position) -> array of shape (n_samples, hidden_dim)
    """
    layers = sorted(set(l for l, _ in layer_positions))
    hook_names = {l: f"blocks.{l}.hook_resid_post" for l in layers}

    def names_filter(name: str) -> bool:
        return any(h in name for h in hook_names.values())

    # Pre-allocate
    activations = {lp: [] for lp in layer_positions}
    n_samples = len(samples)

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0 or i == n_samples - 1:
            print(f"  {i + 1}/{n_samples}", end="\r")
        text = sample.prompt_text + (sample.response or "")
        with torch.no_grad():
            _, cache = runner.run_with_cache(text, names_filter=names_filter)

        for layer, position in layer_positions:
            acts = cache[hook_names[layer]]
            if isinstance(acts, torch.Tensor):
                acts = acts[0].cpu().numpy()
            pos_idx = min(position, acts.shape[0] - 1)
            activations[(layer, position)].append(acts[pos_idx])

        del cache

    print()  # Clear progress line
    return {lp: np.array(v) for lp, v in activations.items()}


def compute_contrastive_direction(acts_0: np.ndarray, acts_1: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute direction as mean(class1) - mean(class0)."""
    mean_0, mean_1 = np.mean(acts_0, axis=0), np.mean(acts_1, axis=0)
    direction = mean_1 - mean_0
    norm = np.linalg.norm(direction)

    # Projection stats
    proj_0 = np.dot(acts_0 - mean_0, direction) / (norm + EPSILON)
    proj_1 = np.dot(acts_1 - mean_1, direction) / (norm + EPSILON)

    return direction, {
        "direction_norm": float(norm),
        "n_class0": len(acts_0),
        "n_class1": len(acts_1),
        "proj_std_class0": float(np.std(proj_0)),
        "proj_std_class1": float(np.std(proj_1)),
    }


def load_positions_with_metadata(positions_path: Path) -> PositionsFile:
    """Load positions file, enriching with metadata.json if needed."""
    positions_file = PositionsFile.load(positions_path)

    metadata_path = positions_path.parent / "metadata.json"
    if metadata_path.exists() and positions_file.model == "unknown":
        metadata = load_json(metadata_path)
        positions_file.model = metadata.get("model", "unknown")
        positions_file.dataset_id = metadata.get("dataset_id")

    return positions_file


def subsample_balanced(samples: list, labels: np.ndarray, n_per_class: int, seed: int) -> tuple[list, np.ndarray]:
    """Subsample to n_per_class samples per class."""
    np.random.seed(seed)
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    if len(idx_0) > n_per_class:
        idx_0 = np.random.choice(idx_0, n_per_class, replace=False)
    if len(idx_1) > n_per_class:
        idx_1 = np.random.choice(idx_1, n_per_class, replace=False)

    selected = np.concatenate([idx_0, idx_1])
    return [samples[i] for i in selected], labels[selected]


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive activation analysis")
    parser.add_argument("--positions", type=Path, required=True, help="positions.json from activation patching")
    parser.add_argument("--preference-data", type=str, help="Preference data ID")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help=f"Max samples per class (0=all)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top positions to process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with P("total"):
        # Load positions
        if not args.positions.exists():
            print(f"Error: {args.positions} not found")
            return 1

        positions_file = load_positions_with_metadata(args.positions)
        top_positions = positions_file.get_top_n(args.top_n)
        print(f"Positions: {[(p.layer, p.position, p.token, f'{p.score:.3f}') for p in top_positions]}")

        # Load preference data
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        with P("load_data"):
            if args.preference_data:
                pref_data = load_pref_data_with_prompts(args.preference_data, pref_dir, data_dir)
            else:
                recent = find_preference_data(pref_dir)
                if not recent:
                    print("Error: No preference data found")
                    return 1
                pref_data = load_pref_data_with_prompts(get_preference_data_id(recent), pref_dir, data_dir)

        # Prepare samples
        with P("prepare_samples"):
            samples, labels = prepare_samples(pref_data, "choice", "choice", random_seed=DEFAULT_SEED)

        if args.max_samples > 0 and len(samples) > args.max_samples * 2:
            samples, labels = subsample_balanced(samples, labels, args.max_samples, DEFAULT_SEED)

        print(f"Samples: {len(samples)} (class 0: {np.sum(labels == 0)}, class 1: {np.sum(labels == 1)})")

        # Load model
        with P("load_model"):
            runner = ModelRunner(pref_data.model)

        # Extract all activations in single pass
        layer_positions = [(p.layer, p.position) for p in top_positions]
        print(f"\nExtracting activations for {len(layer_positions)} positions...")

        with P("extract_activations"):
            all_activations = extract_activations_at_positions(runner, samples, layer_positions)

        # Compute directions
        ts = get_timestamp()
        run_dir = args.output / ts
        ensure_dir(run_dir)

        interventions = []
        for pos_spec in top_positions:
            lp = (pos_spec.layer, pos_spec.position)
            acts = all_activations[lp]
            acts_0, acts_1 = acts[labels == 0], acts[labels == 1]

            with P(f"contrastive_L{lp[0]}_P{lp[1]}"):
                direction, stats = compute_contrastive_direction(acts_0, acts_1)

            print(f"L{lp[0]} P{lp[1]} ({pos_spec.token}): norm={stats['direction_norm']:.3f}")

            intervention = {
                "type": "steering_vector",
                "source": "contrastive_activation",
                "model": pref_data.model,
                "layer": pos_spec.layer,
                "position": pos_spec.position,
                "token": pos_spec.token,
                "patching_score": pos_spec.score,
                "direction": direction.tolist(),
                **stats,
            }
            interventions.append(intervention)
            save_json(intervention, run_dir / f"intervention_L{lp[0]}_P{lp[1]}.json")

        # Save outputs
        if interventions:
            save_json(interventions[0], run_dir / "intervention.json")

        save_json({
            "timestamp": ts,
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "positions_source": str(args.positions),
            "n_samples": len(samples),
            "interventions": [{"layer": i["layer"], "position": i["position"], "norm": i["direction_norm"]} for i in interventions],
        }, run_dir / "summary.json")

        print(f"\nSaved to: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
