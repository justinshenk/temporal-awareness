"""Configuration for difference-in-means analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema
from ...common.semantic_positions import ALL_TRAJECTORY_POSITIONS


@dataclass
class DiffMeansConfig(BaseSchema):
    """Configuration for difference-in-means analysis.

    Attributes:
        positions: Semantic position names to analyze (from SamplePositionMapping)
        critical_layer_ranges: Layer ranges to shade in plots (e.g., [(19, 24), (28, 34)])
        reference_layers: Layers to add vertical reference lines at
        annotation_layers: Layers to add text annotations at
        cosine_zoom_range: Y-axis range for zoomed cosine trajectory plot (min, max)
    """

    positions: list[str] = field(default_factory=lambda: list(ALL_TRAJECTORY_POSITIONS))

    # Visualization settings
    critical_layer_ranges: list[tuple[int, int]] = field(
        default_factory=lambda: [(19, 24), (28, 34)]
    )
    reference_layers: list[int] = field(default_factory=lambda: [19, 24])
    annotation_layers: list[int] = field(default_factory=lambda: [19, 21, 24, 31, 34])
    cosine_zoom_range: tuple[float, float] = (0.75, 1.0)
