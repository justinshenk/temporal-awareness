#!/usr/bin/env python
"""
Run the intertemporal preference experiment.

IMPORTANT DESIGN PRINCIPLES:
1. This is a thin CLI wrapper - core logic lives in src/experiments/intertemporal.py
2. Configs (SMALL_CONFIG, NORMAL_CONFIG) are defined here, not in src/
3. NEVER use TransformerLens/NNsight/Pyvene directly - use ModelRunner via experiment module
4. No magic numbers - all config values should be explicit and documented

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

from src.experiments.intertemporal import ExperimentArgs, run_experiment


# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Default dataset config (housing scenario with time horizons)
# All configurable knobs for dataset generation:
#   - name: identifier for the dataset
#   - context: scenario setup (role, situation, reward units, labels, sampling method)
#   - options: reward/time ranges and stepping for short_term and long_term
#   - time_horizons: list of time constraints (None = no constraint)
#   - add_formatting_variations: apply label/time/number formatting variations
#   - prompt_format: name of prompt template to use (default: "default_prompt_format")
DEFAULT_DATASET_CONFIG = {
    "name": "standalone_default",
    "context": {
        # Scenario setup
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing development in the city.",
        # Optional context fields
        "task_in_question": "to decide between",  # Text for task description
        "reasoning_ask": "why this choice was made",  # Text for asking reasoning
        "domain": "housing",  # Domain identifier
        "extra_situation": "",  # Additional context
        # Sampling configuration
        "labels": ["a)", "b)"],  # Option labels
        "method": "grid",  # "grid" or "random" sampling
        "seed": 42,  # Random seed for reproducibility
    },
    "options": {
        "short_term": {
            "reward_range": [1000, 5000],  # [min, max] reward values
            "time_range": [[1, "months"], [6, "months"]],  # [min, max] time delays
            "reward_steps": [2, "linear"],  # [n_steps, "linear"/"logarithmic"]
            "time_steps": [2, "linear"],
        },
        "long_term": {
            "reward_range": [8000, 30000],
            "time_range": [[2, "years"], [10, "years"]],
            "reward_steps": [2, "logarithmic"],
            "time_steps": [2, "logarithmic"],
        },
    },
    # Time horizons for probes (None = no time horizon constraint)
    "time_horizons": [
        {"value": 6, "unit": "months"},
        {"value": 5, "unit": "years"},
    ],
    # Apply formatting variations (label order, time units, number spelling)
    "add_formatting_variations": True,
    # Prompt format template name
    "prompt_format": "default_prompt_format",
}


# Small config for quick testing (20 samples for meaningful steering vectors)
# Reward ratios balanced so model makes mixed choices (not always long-term)
SMALL_CONFIG = {
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
        "time_horizons": [None],  # No time horizon = no time_horizon probe
        "add_formatting_variations": False,
    },

    # Data sampling
    "max_samples": 20,  # Number of preference samples to generate

    # Activation patching config
    "max_pairs": 1,  # Number of clean/corrupted pairs for patching
    "position_threshold": 0.03,  # Threshold for filtering important positions

    # Attribution patching config
    "ig_steps": 3,  # Integration steps for EAP-IG (higher = more accurate)

    # Steering vector config
    "contrastive_max_samples": 20,  # Samples for computing steering direction
    "top_n_positions": 1,  # Number of top positions to use

    # Steering evaluation config
    "steering_strengths": [-1.0, 0.0, 1.0],  # Strengths to test
    "test_prompts": ["Choose: $100 now or $300 in 3 months?"],

    # Probe training config
    "probe_layers": None,  # None = auto-select 5 evenly-spaced layers
    "probe_positions": ["option_one", "option_two", {"relative_to": "end", "offset": -1}],
    "probe_max_samples": 20,
}

# Normal config for real experiments
NORMAL_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,

    # Dataset generation config (uses DEFAULT_DATASET_CONFIG with time horizons)
    "dataset_config": DEFAULT_DATASET_CONFIG,

    # Data sampling
    "max_samples": 50,  # Number of preference samples to generate

    # Activation patching config
    "max_pairs": 3,  # Number of clean/corrupted pairs for patching
    "position_threshold": 0.05,  # Threshold for filtering important positions

    # Attribution patching config
    "ig_steps": 10,  # Integration steps for EAP-IG (higher = more accurate)

    # Steering vector config
    "contrastive_max_samples": 200,  # Samples for computing steering direction
    "top_n_positions": 1,  # Number of top positions to use

    # Steering evaluation config
    "steering_strengths": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "test_prompts": [
        "You have two options: receive $100 today, or receive $150 in one year. Which do you prefer?",
        "Would you rather have a small reward now or a larger reward later?",
    ],

    # Probe training config
    "probe_layers": None,  # None = auto-select 5 evenly-spaced layers
    "probe_positions": [
        "option_one", "option_two", "consider",
        {"relative_to": "end", "offset": -1},
    ],
    "probe_max_samples": 200,
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
        "--skip-probes", action="store_true",
        help="Skip probe training"
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
        skip_probes=args.skip_probes,
        output=args.output,
        project_root=PROJECT_ROOT,
    )


def main() -> int:
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
