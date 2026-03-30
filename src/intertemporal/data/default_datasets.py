"""Default dataset configurations for intertemporal preference experiments."""

from __future__ import annotations


###########################
######### SHARED ##########
###########################


###########################
######### OPTIONS #########
###########################

BASE_CONTEXT = {
    "reward_unit": "housing units",
    "role": "the city administration",
    "situation": "Plan for housing development in the city.",
    "domain": "housing",
}

SHORT_REWARDS = [1000, 2500]
LONG_REWARDS = [30000, 100000]

LONG_TIMES = [
    {"value": 10, "unit": "years"},
    {"value": 30, "unit": "years"},
]
SHORT_TIMES = [
    {"value": 0.5, "unit": "years"},
    {"value": 1, "unit": "years"},
]

SHORT_SINGLE = {
    "reward_range": SHORT_REWARDS,
    "time_range": SHORT_TIMES,
    "reward_steps": [0, "linear"],
    "time_steps": [0, "linear"],
}
SHORT_FEW = {
    "reward_range": SHORT_REWARDS,
    "time_range": SHORT_TIMES,
    "reward_steps": [1, "linear"],
    "time_steps": [1, "linear"],
}

SHORT_MANY = {
    "reward_range": SHORT_REWARDS,
    "time_range": SHORT_TIMES,
    "reward_steps": [3, "linear"],
    "time_steps": [3, "linear"],
}

LONG_SINGLE = {
    "reward_range": LONG_REWARDS,
    "time_range": LONG_TIMES,
    "reward_steps": [0, "logarithmic"],
    "time_steps": [0, "logarithmic"],
}
LONG_FEW = {
    "reward_range": LONG_REWARDS,
    "time_range": LONG_TIMES,
    "reward_steps": [1, "linear"],
    "time_steps": [1, "linear"],
}
LONG_MANY = {
    "reward_range": LONG_REWARDS,
    "time_range": LONG_TIMES,
    "reward_steps": [3, "logarithmic"],
    "time_steps": [3, "logarithmic"],
}

OPTIONS_SINGLE = {"short_term": SHORT_SINGLE, "long_term": LONG_SINGLE}
OPTIONS_FEW = {"short_term": SHORT_FEW, "long_term": LONG_FEW}
OPTIONS_MANY = {"short_term": SHORT_MANY, "long_term": LONG_MANY}


OPTIONS_GEO = {
    "short_term": {
        "reward_range": [1000, 2500],
        "time_range": [
            {"value": 1, "unit": "days"},
            {"value": 5, "unit": "years"},
        ],
        "reward_steps": [5, "linear"],
        "time_steps": [5, "linear"],
    },
    "long_term": {
        "reward_range": [30000, 100000],
        "time_range": [
            {"value": 10, "unit": "years"},
            {"value": 70, "unit": "years"},
        ],
        "reward_steps": [5, "linear"],
        "time_steps": [5, "linear"],
    },
}

###########################
######### HORIZON #########
###########################


HOR_NONE = [None]

HOR_BINARY = [
    {"value": 1, "unit": "months"},  # Short horizon
    {"value": 50, "unit": "years"},  # Long horizon
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
    {"value": 4, "unit": "months"},
    {"value": 8, "unit": "months"},
    {"value": 1, "unit": "years"},
    {"value": 3, "unit": "years"},
    {"value": 5, "unit": "years"},
    {"value": 1, "unit": "decades"},
    {"value": 3, "unit": "decades"},
    {"value": 5, "unit": "decades"},
    {"value": 1, "unit": "centuries"},
    {"value": 2, "unit": "centuries"},
    {"value": 5, "unit": "centuries"},
    {"value": 1, "unit": "millenia"},
    {"value": 5, "unit": "millenia"},
    {"value": 10, "unit": "millenia"},
]

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

MULTINANO_CFG = {
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


REWARD_SWEEP_CFG = {
    "name": "reward_sweep",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_BINARY,
}

MINI_CFG = {
    "name": "mini",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": HOR_BINARY,
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

GEO_VIZ_CFG = {
    "name": "geo_viz",
    "context": BASE_CONTEXT,
    "options": OPTIONS_GEO,
    "time_horizons": HOR_GEO,
    "add_formatting_noise": False,
    "do_formatting_variation_grid": False,
    "do_context_variations": True,
}

###########################
###### DEFAULTS SET #######
###########################

MINIMAL_EXPERIMENT_DATASET_CONFIG = SMALL_CFG

FULL_EXPERIMENT_DATASET_CONFIG = SMALL_CFG

MULTILABEL_EXPERIMENT_DATASET_CONFIG = MULTILABEL_CFG
