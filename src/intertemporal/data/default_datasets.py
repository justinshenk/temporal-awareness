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


NONE_HOR = [None]

BINARY_HOR = [
    {"value": 1, "unit": "months"},  # Short horizon
    {"value": 50, "unit": "years"},  # Long horizon
]

FEW_HOR = [
    None,
    {"value": 1, "unit": "years"},
    {"value": 7, "unit": "years"},
    {"value": 15, "unit": "years"},
]

SWEEP_HOR = [
    None,
    {"value": 1, "unit": "months"},
    {"value": 6, "unit": "months"},
    {"value": 2, "unit": "years"},
    {"value": 5, "unit": "years"},
    {"value": 10, "unit": "years"},
    {"value": 30, "unit": "years"},
    {"value": 50, "unit": "years"},
]

###########################
######## DATASETS #########
###########################

# Simplest
NANO_CFG = {
    "name": "nano",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": BINARY_HOR,
}

MULTINANO_CFG = {
    "name": "multinano",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": BINARY_HOR,
    "add_formatting_noise": True,
}


HORIZON_SWEEP_CFG = {
    "name": "horizon_sweep",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": SWEEP_HOR,
}


REWARD_SWEEP_CFG = {
    "name": "reward_sweep",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": BINARY_HOR,
}

MINI_CFG = {
    "name": "mini",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": BINARY_HOR,
}

SMALL_CFG = {
    "name": "small",
    "context": BASE_CONTEXT,
    "options": OPTIONS_FEW,
    "time_horizons": FEW_HOR,
}


GRANDE_CFG = {
    "name": "grande",
    "context": BASE_CONTEXT,
    "options": OPTIONS_MANY,
    "time_horizons": SWEEP_HOR,
}


MULTILABEL_CFG = {
    "name": "multilabel",
    "context": BASE_CONTEXT,
    "options": OPTIONS_SINGLE,
    "time_horizons": BINARY_HOR,
    "add_formatting_noise": False,
    "do_formatting_variation_grid": True,
}

###########################
###### DEFAULTS SET #######
###########################

MINIMAL_EXPERIMENT_DATASET_CONFIG = SMALL_CFG

FULL_EXPERIMENT_DATASET_CONFIG = SMALL_CFG

MULTILABEL_EXPERIMENT_DATASET_CONFIG = MULTILABEL_CFG
