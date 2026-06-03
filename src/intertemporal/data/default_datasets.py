"""Default dataset configurations for intertemporal preference experiments."""

from __future__ import annotations


###########################
######### OPTIONS #########
###########################


OPTIONS_SINGLE = {
    "short_term": {
        "reward_range": [20000, 20000],
        "time_range": [
            {"value": 6, "unit": "months"},
            {"value": 6, "unit": "months"},
        ],
        "reward_steps": [0, "linear"],
        "time_steps": [0, "linear"],
    },
    "long_term": {
        "reward_range": [500000, 500000],
        "time_range": [
            {"value": 10, "unit": "years"},
            {"value": 10, "unit": "years"},
        ],
        "reward_steps": [0, "linear"],
        "time_steps": [0, "linear"],
    },
}

OPTIONS_FEW = {
    "short_term": {
        "reward_range": [10, 1000],
        "time_range": [
            {"value": 1, "unit": "months"},
            {"value": 1, "unit": "years"},
        ],
        "reward_steps": [1, "linear"],
        "time_steps": [1, "linear"],
    },
    "long_term": {
        "reward_range": [100000, 100000000],
        "time_range": [
            {"value": 10, "unit": "years"},
            {"value": 3, "unit": "decades"},
        ],
        "reward_steps": [1, "linear"],
        "time_steps": [1, "linear"],
    },
}

OPTIONS_MANY = {
    "short_term": {
        "reward_range": [10, 1000],
        "time_range": [
            {"value": 1, "unit": "months"},
            {"value": 1, "unit": "years"},
        ],
        "reward_steps": [3, "linear"],
        "time_steps": [3, "linear"],
    },
    "long_term": {
        "reward_range": [100000, 100000000],
        "time_range": [
            {"value": 10, "unit": "years"},
            {"value": 3, "unit": "decades"},
        ],
        "reward_steps": [3, "logarithmic"],
        "time_steps": [3, "logarithmic"],
    },
}


OPTIONS_GEO = {
    "short_term": {
        "reward_range": [1000, 100000],
        "time_range": [
            {"value": 1, "unit": "days"},
            {"value": 20, "unit": "years"},
        ],
        "reward_steps": [2, "logarithmic"],
        "time_steps": [5, "logarithmic"],
    },
    "long_term": {
        "reward_range": [1000, 100000],
        "time_range": [
            {"value": 1, "unit": "years"},
            {"value": 100, "unit": "years"},
        ],
        "reward_steps": [2, "logarithmic"],
        "time_steps": [5, "logarithmic"],
    },
}

###########################
######### HORIZON #########
###########################


HOR_NONE = [None]

HOR_BINARY = [
    None,
    {"value": 8, "unit": "months"},
    {"value": 15, "unit": "years"},
]

HOR_FEW = [
    None,
    {"value": 1, "unit": "years"},
    {"value": 7, "unit": "years"},
    {"value": 15, "unit": "years"},
]

HOR_COARSE_SWEEP = [
    None,
    {"value": 1, "unit": "months"},
    {"value": 6, "unit": "months"},
    {"value": 2, "unit": "years"},
    {"value": 5, "unit": "years"},
    {"value": 10, "unit": "years"},
    {"value": 30, "unit": "years"},
    {"value": 50, "unit": "years"},
]


HOR_GEO = [
    None,
    {"value": 1, "unit": "seconds"},
    {"value": 1, "unit": "hours"},
    {"value": 1, "unit": "days"},
    {"value": 1, "unit": "week"},
    {"value": 1, "unit": "months"},
    {"value": 2, "unit": "months"},
    {"value": 6, "unit": "months"},
    {"value": 1, "unit": "years"},
    {"value": 3, "unit": "years"},
    {"value": 5, "unit": "years"},
    {"value": 1, "unit": "decades"},
    {"value": 3, "unit": "decades"},
    {"value": 5, "unit": "decades"},
    {"value": 1, "unit": "centuries"},
    {"value": 2, "unit": "centuries"},
    {"value": 5, "unit": "centuries"},
]


###########################
######## CONTEXTS #########
###########################


BASE_CONTEXT = {
    "reward_unit": "dollars",
    "role": "the head of the household",
    "situation": "Plan for the future of the household based on the stated objectives and constraints.",
    "task_in_question": "choose the best investment",
    "domain": "finance",
}

###########################
######## DATASETS #########
###########################

# Simplest
NANO_CFG = {
    "name": "nano",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_BINARY,
}

MULTIFORMAT_NANO_CFG = {
    "name": "multinano",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_BINARY,
    "add_formatting_noise": True,
}


HORIZON_SWEEP_CFG = {
    "name": "horizon_sweep",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_COARSE_SWEEP,
}


SMALL_CFG = {
    "name": "small",
    "context": BASE_CONTEXT,
    "options": OPTIONS_FEW,
    "time_horizons": HOR_FEW,
}


GRANDE_CFG = {
    "name": "grande",
    "context": BASE_CONTEXT,
    "options": OPTIONS_MANY,
    "time_horizons": HOR_COARSE_SWEEP,
}


MULTILABEL_CFG = {
    "name": "multilabel",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_BINARY,
    "add_formatting_noise": False,
    "do_formatting_variation_grid": True,
}

GEOMETRY_CFG = {
    "name": "large_geometry",
    "context": BASE_CONTEXT,
    "options": OPTIONS_GEO,
    "time_horizons": HOR_GEO,
    "add_formatting_noise": False,
    "do_formatting_variation_grid": False,
    "do_context_variations": False,
    "round_time_units": True,
    "round_reward_units": True,
}

###########################
###### DEFAULTS SET #######
###########################

MINIMAL_EXPERIMENT_DATASET_CONFIG = NANO_CFG

FULL_EXPERIMENT_DATASET_CONFIG = GEOMETRY_CFG

MULTILABEL_EXPERIMENT_DATASET_CONFIG = MULTILABEL_CFG
