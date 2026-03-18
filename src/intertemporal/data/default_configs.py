"""Configuration constants for intertemporal preference experiments."""

from __future__ import annotations

from .default_datasets import (
    FULL_EXPERIMENT_DATASET_CONFIG,
    MINIMAL_EXPERIMENT_DATASET_CONFIG,
)


# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Smallest meaningful experiment
MINIMAL_EXPERIMENT_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": MINIMAL_EXPERIMENT_DATASET_CONFIG,
}

# Larger scale experiments
FULL_EXPERIMENT_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": FULL_EXPERIMENT_DATASET_CONFIG,
}
