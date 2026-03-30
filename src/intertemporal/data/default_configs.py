"""Configuration constants for intertemporal preference experiments."""

from __future__ import annotations

from .default_datasets import (
    FULL_EXPERIMENT_DATASET_CONFIG,
    MINIMAL_EXPERIMENT_DATASET_CONFIG,
    MULTILABEL_EXPERIMENT_DATASET_CONFIG,
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

# Multilabel dataset: do_formatting_variation_grid=True creates samples
# with both label orderings (A/B and B/A) and option orders, enabling multilabel pairs.
# For multilabel, we require:
# - different_labels: Fork 0 and Fork 1 have swapped labels (a)/b) vs b)/a))
# - same_order: short-term option appears first in both (isolates label effect from position effect)
MULTILABEL_EXPERIMENT_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": MULTILABEL_EXPERIMENT_DATASET_CONFIG,
    "pair_req": {"different_labels": True, "same_order": True},
}
