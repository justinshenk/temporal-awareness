#!/usr/bin/env python
"""
Run the intertemporal preference experiment.

IMPORTANT DESIGN PRINCIPLES:
1. This is a thin CLI wrapper - core logic lives in src/experiments/intertemporal.py
2. Configs (SMALL_TEST_CONFIG, DEFAULT_CONFIG) are defined here, not in src/
3. NEVER use TransformerLens/NNsight/Pyvene directly - use ModelRunner via experiment module
4. No magic numbers - all config values should be explicit and documented

Usage:
    # Quick test with minimal config
    uv run python scripts/experiments/run_intertemporal_experiment.py --small

    # Full pipeline
    uv run python scripts/experiments/run_intertemporal_experiment.py

    # Use existing preference data
    uv run python scripts/experiments/run_intertemporal_experiment.py --preference-data <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.intertemporal import ExperimentArgs, run_experiment


# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Small config for quick testing (20 samples for meaningful steering vectors)
# Reward ratios balanced so model makes mixed choices (not always long-term)
SMALL_TEST_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,
    # Dataset generation config
    "dataset_config": {
        "name": "test_minimal",
        "context": {
            "reward_unit": "dollars",
            "role": "you",
            "situation": "Choose between options.",
            "labels": ["a)", "b)"],
            "method": "grid",
            "seed": 42,
        },
        "options": {
            "short_term": {
                "reward_range": [400, 600],
                "time_range": [[1, "months"], [1, "year"]],
                "reward_steps": [0, "linear"],
                "time_steps": [0, "linear"],
            },
            "long_term": {
                "reward_range": [700, 900],
                "time_range": [[10, "years"], [20, "years"]],
                "reward_steps": [0, "linear"],
                "time_steps": [0, "linear"],
            },
        },
        "time_horizons": [
            {"value": 6, "unit": "months"},
        ],  # No time horizon = no time_horizon probe
        "add_formatting_variations": False,
    },
    # Data sampling
    "max_samples": 10,  # Number of preference samples to generate
    # Activation patching config
    # Sweep parameters control granularity of position and layer search
    "max_pairs": 1,  # Number of clean/corrupted pairs for patching
    "position_threshold": 0.1,  # Threshold for filtering important positions
    "act_patch_n_layers_sample": 1,  # Number of layers to sample in sweep (evenly spaced)
    "act_patch_position_step": 5,  # Position stride for sweep (1 = every position)
    # Attribution patching config
    "ig_steps": 2,  # Integration steps for EAP-IG (higher = more accurate)
    # Steering vector config
    "contrastive_max_samples": 2,  # Samples for computing steering direction
    "top_n_positions": 1,  # Number of top positions to use
    # Steering evaluation config (uses prompts from preference data)
    "steering_strengths": [-1.0, 0.0, 1.0],  # Strengths to test
    "steering_eval_max_samples": 2,  # Number of preference prompts to evaluate
    # Probe training config
    "probe_layers": [18],  # None = auto-select 5 evenly-spaced layers
    "probe_positions": [
        {"relative_to": "end", "offset": -1},
    ],
    "probe_max_samples": 2,
}


# Default dataset config (housing scenario with time horizons)
# All configurable knobs for dataset generation:
#   - name: identifier for the dataset
#   - context: scenario setup (role, situation, reward units, labels, sampling method)
#   - options: reward/time ranges and stepping for short_term and long_term
#   - time_horizons: list of time constraints (None = no constraint)
#   - add_formatting_variations: apply label/time/number formatting variations
#   - prompt_format: name of prompt template to use (default: "default_prompt_format")
DEFAULT_DATASET_CONFIG = {
    "name": "cityhousing",
    "context": {
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing development in the city.",
        "domain": "housing",
        "method": "grid",
        },
    "options": {
    "short_term": {
        "reward_range": [1000, 4000],
        "time_range": [[2, "months"], [1, "years"]],
        "reward_steps": [3, "linear"],
        "time_steps": [3, "linear"]
    },
    "long_term": {
        "reward_range": [10000, 150000],
        "time_range": [[10, "years"], [30, "years"]],
        "reward_steps": [3, "logarithmic"],
        "time_steps": [3, "logarithmic"]
    }
    },
    "time_horizons": [
        None,
        [1, "months"],
        [6, "months"],
        [2, "years"],
        [5, "years"],
        [10, "years"],
        [30, "years"],
        [50, "years"]
    ],
    "add_formatting_variations" : True
}

# Normal config for real experiments
DEFAULT_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,
    # Dataset generation config (uses DEFAULT_DATASET_CONFIG with time horizons)
    "dataset_config": DEFAULT_DATASET_CONFIG,
    # Data sampling
    "max_samples": None,  # Number of preference samples to generate, None =  All
    # Activation patching config
    # Sweep parameters control granularity of position and layer search
    "max_pairs": 1,  # Number of clean/corrupted pairs for patching
    "position_threshold": 0.05,  # Threshold for filtering important positions
    "act_patch_n_layers_sample": 12,  # Number of layers to sample in sweep (evenly spaced)
    # Attribution patching config
    "ig_steps": 10,  # Integration steps for EAP-IG (higher = more accurate)
    # Steering vector config
    "contrastive_max_samples": 200,  # Samples for computing steering direction
    "top_n_positions": 10,  # Number of top positions to use
    # Steering evaluation config (uses prompts from preference data)
    "steering_strengths": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "steering_eval_max_samples": 10,  # Number of preference prompts to evaluate
    # Probe training config
    "probe_layers": None,  # None = auto-select 5 evenly-spaced layers
    "probe_positions": None, # None = all interesting
    "probe_max_samples": 200,
}


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--small-test", action="store_true", help="Use minimal config for quick testing"
    )
    parser.add_argument(
        "--preference-data",
        type=str,
        help="Use existing preference data instead of generating new",
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory (default: out/experiments)"
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Steps to skip: activation_patching, attribution_patching, "
        "contrastive_steering, probe_training, steering_eval",
    )

    args = parser.parse_args()
    config = dict(SMALL_TEST_CONFIG if args.small_test else DEFAULT_CONFIG)
    config_name = "SMALL (test)" if args.small_test else "NORMAL"
    if args.skip:
        config["skip"] = args.skip

    return ExperimentArgs(
        config=config,
        config_name=config_name,
        preference_data=args.preference_data,
        output=args.output,
        project_root=PROJECT_ROOT,
    )


def main() -> int:
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
