"""Configuration for geometric (PCA) analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema


@dataclass
class GeoConfig(BaseSchema):
    """Configuration for geometric analysis on residual stream activations.

    Attributes:
        enabled: Whether to run geo analysis
        positions: Positions to analyze (semantic names or absolute)
        layers: Layers to analyze (None = all)
        n_components: Number of PCA components
        no_cache: Skip loading from cache
        geometry_pipeline: Config for full geometry pipeline (nested)
    """

    enabled: bool = False
    positions: list[str | int] | None = None
    layers: list[int] | None = None
    n_components: int = 3
    no_cache: bool = False
    geometry_pipeline: dict = field(default_factory=dict)
