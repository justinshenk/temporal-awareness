"""Experiment configuration and results dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.positions_schema import PositionSpec
from ..data import (
    PreferenceData,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL,
)
from ..models import ModelRunner


@dataclass
class ExperimentConfig:
    """Configuration for intertemporal experiments."""

    # Data generation
    model: str = DEFAULT_MODEL
    dataset_config: dict = field(default_factory=lambda: DEFAULT_DATASET_CONFIG.copy())
    max_samples: int = 50

    # Patching
    max_pairs: int = 3
    ig_steps: int = 10
    position_threshold: float = 0.05

    # Contrastive
    contrastive_max_samples: int = 500
    top_n_positions: int = 1

    # Output
    output_dir: Optional[Path] = None


@dataclass
class ExperimentResults:
    """Results from a full experiment run."""

    pref_data: PreferenceData
    runner: ModelRunner
    output_dir: Path

    # Patching results
    position_sweep: Optional[np.ndarray] = None
    activation_patching: Optional[np.ndarray] = None
    attribution_results: Optional[dict[str, np.ndarray]] = None

    # Positions
    top_positions: list[PositionSpec] = field(default_factory=list)

    # Steering vectors
    steering_vectors: dict = field(default_factory=dict)
