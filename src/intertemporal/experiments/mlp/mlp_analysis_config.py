"""Configuration for MLP neuron analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema


@dataclass
class MLPAnalysisConfig(BaseSchema):
    """Configuration for MLP neuron analysis.

    Attributes:
        layers: Layers to analyze
        n_top_neurons: Number of top neurons to track per layer
    """

    layers: list[int] = field(default_factory=lambda: [19, 21, 24, 28, 31, 34, 35])
    n_top_neurons: int = 50
