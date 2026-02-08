#!/usr/bin/env python
"""
Contrastive activation analysis for intertemporal preference steering.

Computes steering vectors as mean activation difference between choice classes
at positions identified by activation patching.

Usage:
    uv run python scripts/intertemporal/contrastive.py \
        --positions out/activation_patching/<timestamp>/positions_resid_post.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.io import ensure_dir, save_json, load_json, get_timestamp
from src.common.token_positions import PositionsFile
from src.data import (
    load_pref_data_with_prompts,
    find_preference_data,
    get_preference_data_id,
)
from src.experiments import compute_steering_vector
from src.models import ModelRunner
from src.common.profiler import P

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_samples": 20,  # Small but meaningful for testing
    "top_n": 1,
}

# Defaults
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "contrastive"
DEFAULT_MAX_SAMPLES = 500
DEFAULT_TOP_N = 1
DEFAULT_SEED = 42


def load_positions_with_metadata(positions_path: Path) -> PositionsFile:
    """Load positions file, enriching with metadata.json if needed."""
    positions_file = PositionsFile.load(positions_path)

    metadata_path = positions_path.parent / "metadata.json"
    if metadata_path.exists() and positions_file.model == "unknown":
        metadata = load_json(metadata_path)
        positions_file.model = metadata.get("model", "unknown")
        positions_file.dataset_id = metadata.get("dataset_id")

    return positions_file


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive activation analysis")
    parser.add_argument(
        "--positions",
        type=Path,
        required=True,
        help="positions.json from activation patching",
    )
    parser.add_argument("--preference-data", type=str, help="Preference data ID")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Max samples per class (0=all)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top positions to process",
    )
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
        print(
            f"Positions: {[(p.layer, p.position, p.token, f'{p.score:.3f}') for p in top_positions]}"
        )

        # Load preference data
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        with P("load_data"):
            if args.preference_data:
                pref_data = load_pref_data_with_prompts(
                    args.preference_data, pref_dir, data_dir
                )
            else:
                recent = find_preference_data(pref_dir)
                if not recent:
                    print("Error: No preference data found")
                    return 1
                pref_data = load_pref_data_with_prompts(
                    get_preference_data_id(recent), pref_dir, data_dir
                )

        # Load model
        with P("load_model"):
            runner = ModelRunner(pref_data.model)

        # Compute steering vectors for each top position
        ts = get_timestamp()
        run_dir = args.output / ts
        ensure_dir(run_dir)

        interventions = []
        for pos_spec in top_positions:
            print(f"\nComputing steering vector for L{pos_spec.layer} P{pos_spec.position} ({pos_spec.token})...")

            with P(f"contrastive_L{pos_spec.layer}_P{pos_spec.position}"):
                direction, stats = compute_steering_vector(
                    runner, pref_data, pos_spec.layer, pos_spec.position,
                    max_samples=args.max_samples,
                )

            print(
                f"L{pos_spec.layer} P{pos_spec.position} ({pos_spec.token}): norm={stats['direction_norm']:.3f}"
            )

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
            save_json(intervention, run_dir / f"intervention_L{pos_spec.layer}_P{pos_spec.position}.json")

        # Save outputs
        if interventions:
            save_json(interventions[0], run_dir / "intervention.json")

        save_json(
            {
                "timestamp": ts,
                "model": pref_data.model,
                "dataset_id": pref_data.dataset_id,
                "positions_source": str(args.positions),
                "n_samples": stats["n_class0"] + stats["n_class1"] if interventions else 0,
                "interventions": [
                    {
                        "layer": i["layer"],
                        "position": i["position"],
                        "norm": i["direction_norm"],
                    }
                    for i in interventions
                ],
            },
            run_dir / "summary.json",
        )

        print(f"\nSaved to: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
