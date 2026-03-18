"""Configuration constants for intertemporal preference experiments."""

from __future__ import annotations

# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

MINIMAL_PROMPT_DATASET_CONFIG = {
    "name": "default_test",
    "context": {
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing development in the city.",
        "domain": "housing",
    },
    "options": {
        "short_term": {
            "reward_range": [1000, 2500],
            "time_range": [
                {"value": 6, "unit": "months"},
                {"value": 1, "unit": "years"},
            ],
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        },
        "long_term": {
            "reward_range": [30000, 100000],
            "time_range": [
                {"value": 10, "unit": "years"},
                {"value": 30, "unit": "years"},
            ],
            "reward_steps": [0, "logarithmic"],
            "time_steps": [0, "logarithmic"],
        },
    },
    "time_horizons": [
        {"value": 1, "unit": "months"},  # Short horizon
        {"value": 50, "unit": "years"},  # Long horizon
    ],
    "add_formatting_noise": False,
    "do_formatting_variation_grid": False,
}

# Small config for testing
MINIMAL_EXPERIMENT_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": MINIMAL_PROMPT_DATASET_CONFIG,
}

# Default prompt dataset config
DEFAULT_PROMPT_DATASET_CONFIG = {
    "name": "cityhousing",
    "context": {
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing development in the city.",
        "domain": "housing",
    },
    "options": {
        "short_term": {
            "reward_range": [1000, 4000],
            "time_range": [
                {"value": 2, "unit": "months"},
                {"value": 1, "unit": "years"},
            ],
            "reward_steps": [3, "linear"],
            "time_steps": [3, "linear"],
        },
        "long_term": {
            "reward_range": [10000, 150000],
            "time_range": [
                {"value": 10, "unit": "years"},
                {"value": 30, "unit": "years"},
            ],
            "reward_steps": [3, "logarithmic"],
            "time_steps": [3, "logarithmic"],
        },
    },
    "time_horizons": [
        {"value": 1, "unit": "months"},
        {"value": 6, "unit": "months"},
        {"value": 2, "unit": "years"},
        {"value": 5, "unit": "years"},
        {"value": 10, "unit": "years"},
        {"value": 30, "unit": "years"},
        {"value": 50, "unit": "years"},
    ],
    "add_formatting_noise": False,
}

# Normal config for real experiments
FULL_EXPERIMENT_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,
    # Dataset generation config (uses DEFAULT_PROMPT_DATASET_CONFIG with time horizons)
    "dataset_config": DEFAULT_PROMPT_DATASET_CONFIG,
}
