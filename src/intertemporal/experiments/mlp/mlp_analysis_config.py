"""Configuration for MLP neuron analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema
from ...common.semantic_positions import DEFAULT_LAYERS, RESPONSE_POSITIONS


@dataclass
class MLPAnalysisConfig(BaseSchema):
    """Configuration for MLP neuron analysis.

    Attributes:
        layers: Layers to analyze
        n_top_neurons: Number of top neurons to track per layer
        positions: Semantic position names to analyze (from SamplePositionMapping)
        layer_position_enabled: Whether to run layer x position patching
        layer_position_positions: Position names for layer x position patching
    """

    layers: list[int] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    n_top_neurons: int = 50
    positions: list[str] = field(default_factory=lambda: list(RESPONSE_POSITIONS))

    # Layer x position patching config
    layer_position_enabled: bool = True
    layer_position_positions: list[str] = field(default_factory=lambda: list(RESPONSE_POSITIONS))
