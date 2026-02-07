"""Configuration constants for intertemporal preference experiments."""

from __future__ import annotations

# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
            "time_range": [[2, "months"], [1, "years"]],
            "reward_steps": [3, "linear"],
            "time_steps": [3, "linear"],
        },
        "long_term": {
            "reward_range": [10000, 150000],
            "time_range": [[10, "years"], [30, "years"]],
            "reward_steps": [3, "logarithmic"],
            "time_steps": [3, "logarithmic"],
        },
    },
    "time_horizons": [
        None,
        [1, "months"],
        [6, "months"],
        [2, "years"],
        [5, "years"],
        [10, "years"],
        [30, "years"],
        [50, "years"],
    ],
    "add_formatting_variations": True,
}

# Normal config for real experiments
DEFAULT_EXPERIMENT_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,
    # Dataset generation config (uses DEFAULT_PROMPT_DATASET_CONFIG with time horizons)
    "dataset_config": DEFAULT_PROMPT_DATASET_CONFIG,
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
    "probe_positions": [
        "option_one",
        "option_two",
        {"relative_to": "end", "offset": -1},
    ],  # Positions that carry choice information
    "probe_max_samples": 200,
}

# Test prompt dataset config
TEST_PROMPT_DATASET_CONFIG = {
    "name": "default_test",
    "context": {
        "reward_unit": "dollars",
        "role": "you",
        "situation": "Choose between options.",
        "labels": ["a)", "b)"],
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
}


# Default query config for querying models
DEFAULT_QUERY_CONFIG = {
    "models": [DEFAULT_MODEL],
    "internals": None,
    "subsample": 1.0,
    "batch_size": 4,
    "skip_generation": True,  # Fast: infer choice from probs only
}


# Small config for testing
TEST_EXPERIMENT_CONFIG = {
    # Model
    "model": DEFAULT_MODEL,
    # Dataset generation config
    "dataset_config": TEST_PROMPT_DATASET_CONFIG,
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
