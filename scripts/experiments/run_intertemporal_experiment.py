#!/usr/bin/env python
"""
Run the intertemporal preference experiment.

Usage:
    # Quick test with minimal config
    uv run python scripts/experiments/run_intertemporal_experiment.py --small

    # Full pipeline
    uv run python scripts/experiments/run_intertemporal_experiment.py

    # Use existing preference data
    uv run python scripts/experiments/run_intertemporal_experiment.py --preference-data <id>

    # Skip slow steps
    uv run python scripts/experiments/run_intertemporal_experiment.py --skip-attribution --skip-steering-eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DEFAULT_DATASET_CONFIG, DEFAULT_MODEL
from src.experiments.intertemporal import ExperimentArgs, run_experiment


# Small config for quick testing (20 samples for meaningful steering vectors)
# Reward ratios balanced so model makes mixed choices (not always long-term)
SMALL_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": {
        "name": "test_minimal",
        "context": {
            "reward_unit": "dollars",
            "role": "you",
            "situation": "Choose between options.",
        },
        "options": {
            "short_term": {
                # Higher short-term rewards to make immediate choice competitive
                "reward_range": [400, 600],
                "time_range": [[1, "days"], [7, "days"]],
                "reward_steps": [2, "linear"],
                "time_steps": [2, "linear"],
            },
            "long_term": {
                # Modest long-term premium (1.5-2x) with longer delays
                "reward_range": [700, 900],
                "time_range": [[3, "months"], [12, "months"]],
                "reward_steps": [2, "linear"],
                "time_steps": [2, "linear"],
            },
        },
        "time_horizons": [None],
        "add_formatting_variations": False,
    },
    "max_samples": 20,
    "max_pairs": 1,
    "ig_steps": 3,
    "position_threshold": 0.03,
    "contrastive_max_samples": 20,
    "top_n_positions": 1,
    "steering_strengths": [-1.0, 0.0, 1.0],
    "test_prompts": ["Choose: $100 now or $300 in 3 months?"],
}

# Normal config for real experiments
NORMAL_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": DEFAULT_DATASET_CONFIG,
    "max_samples": 50,
    "max_pairs": 3,
    "ig_steps": 10,
    "position_threshold": 0.05,
    "contrastive_max_samples": 200,
    "top_n_positions": 1,
    "steering_strengths": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "test_prompts": [
        "You have two options: receive $100 today, or receive $150 in one year. Which do you prefer?",
        "Would you rather have a small reward now or a larger reward later?",
    ],
}


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use minimal config for quick testing"
    )
    parser.add_argument(
        "--preference-data", type=str,
        help="Use existing preference data instead of generating new"
    )
    parser.add_argument(
        "--skip-attribution", action="store_true",
        help="Skip attribution patching (faster)"
    )
    parser.add_argument(
        "--skip-steering-eval", action="store_true",
        help="Skip steering vector evaluation"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output directory (default: out/experiments)"
    )

    args = parser.parse_args()
    config = SMALL_CONFIG if args.small else NORMAL_CONFIG
    config_name = "SMALL (test)" if args.small else "NORMAL"

    return ExperimentArgs(
        config=config,
        config_name=config_name,
        preference_data=args.preference_data,
        skip_attribution=args.skip_attribution,
        skip_steering_eval=args.skip_steering_eval,
        output=args.output,
        project_root=PROJECT_ROOT,
    )


def main() -> int:
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
