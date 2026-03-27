"""Configuration for MLP neuron analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema


@dataclass
class MLPAnalysisConfig(BaseSchema):
    """Configuration for MLP neuron analysis.

    Attributes:
        enabled: Whether to run MLP analysis
        layers: Layers to analyze
        n_top_neurons: Number of top neurons to track per layer
        no_cache: Skip loading from cache
    """

    enabled: bool = False
    layers: list[int] = field(default_factory=lambda: [19, 21, 24, 28, 31, 34, 35])
    n_top_neurons: int = 50
    no_cache: bool = False
