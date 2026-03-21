"""Default dataset configurations for intertemporal preference experiments."""

from __future__ import annotations


###########################
######### SHARED ##########
###########################

BASE_CONTEXT = {
    "reward_unit": "housing units",
    "role": "the city administration",
    "situation": "Plan for housing development in the city.",
    "domain": "housing",
}

SHORT_REWARDS = [1000, 2500]
SHORT_TIMES = [
    {"value": 6, "unit": "months"},
    {"value": 1, "unit": "years"},
]

LONG_REWARDS = [30000, 100000]
LONG_TIMES = [
    {"value": 10, "unit": "years"},
    {"value": 30, "unit": "years"},
]


###########################
######## DATASETS #########
###########################

# Simplest
NANO_PROMPT_DATASET_CONFIG = {
    "name": "nano",
    "context": BASE_CONTEXT,
    "options": {
        "short_term": {
            "reward_range": SHORT_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        },
        "long_term": {
            "reward_range": LONG_REWARDS,
            "time_range": LONG_TIMES,
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


HORIZON_SWEEP_PROMPT_DATASET_CONFIG = {
    "name": "horizon_sweep",
    "context": BASE_CONTEXT,
    "options": {
        "short_term": {
            "reward_range": SHORT_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        },
        "long_term": {
            "reward_range": LONG_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [0, "logarithmic"],
            "time_steps": [0, "logarithmic"],
        },
    },
    "time_horizons": [
        None,
        {"value": 1, "unit": "months"},
        {"value": 6, "unit": "months"},
        {"value": 2, "unit": "years"},
        {"value": 5, "unit": "years"},
        {"value": 10, "unit": "years"},
        {"value": 30, "unit": "years"},
        {"value": 50, "unit": "years"},
    ],
    "add_formatting_noise": False,
    "do_formatting_variation_grid": False,
}


REWARD_SWEEP_PROMPT_DATASET_CONFIG = {
    "name": "reward_sweep",
    "context": BASE_CONTEXT,
    "options": {
        "short_term": {
            "reward_range": SHORT_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        },
        "long_term": {
            "reward_range": LONG_REWARDS,
            "time_range": LONG_TIMES,
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

# Time Horizon Sweep
MINI_PROMPT_DATASET_CONFIG = {
    "name": "mini",
    "context": BASE_CONTEXT,
    "options": {
        "short_term": {
            "reward_range": SHORT_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        },
        "long_term": {
            "reward_range": LONG_REWARDS,
            "time_range": LONG_TIMES,
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


GRANDE_PROMPT_DATASET_CONFIG = {
    "name": "grande",
    "context": BASE_CONTEXT,
    "options": {
        "short_term": {
            "reward_range": SHORT_REWARDS,
            "time_range": SHORT_TIMES,
            "reward_steps": [3, "linear"],
            "time_steps": [3, "linear"],
        },
        "long_term": {
            "reward_range": LONG_REWARDS,
            "time_range": LONG_TIMES,
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
    "do_formatting_variation_grid": False,
}

###########################
###### DEFAULTS SET #######
###########################

MINIMAL_EXPERIMENT_DATASET_CONFIG = HORIZON_SWEEP_PROMPT_DATASET_CONFIG
# MINIMAL_EXPERIMENT_DATASET_CONFIG = NANO_PROMPT_DATASET_CONFIG

FULL_EXPERIMENT_DATASET_CONFIG = MINI_PROMPT_DATASET_CONFIG
