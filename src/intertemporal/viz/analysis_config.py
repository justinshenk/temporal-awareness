"""Configuration for analysis visualizations (diffmeans, geo).

Centralizes position/layer selection so it's easy to modify.
Uses semantic position names (format_pos) that are resolved at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.semantic_positions import (
    PROMPT_LABEL_POSITIONS,
    PROMPT_CONSTRAINT_POSITIONS,
    RESPONSE_POSITIONS,
)


@dataclass
class AnalysisPositions:
    """Configurable positions for analysis plots.

    Uses semantic position names (format_pos) that are resolved via SamplePositionMapping.
    Positions are grouped by their role in the prompt structure.
    """

    # Source positions (where short-term option info appears)
    source: list[str] = field(default_factory=lambda: list(PROMPT_LABEL_POSITIONS))

    # Destination positions (where long-term/constraint info appears)
    destination: list[str] = field(default_factory=lambda: list(PROMPT_CONSTRAINT_POSITIONS))

    # Final position (where choice is made)
    final: list[str] = field(default_factory=lambda: list(RESPONSE_POSITIONS))

    @property
    def all_key_positions(self) -> list[str]:
        """All key position names."""
        return list(dict.fromkeys(self.source + self.destination + self.final))

    @property
    def for_difference_norm(self) -> list[str]:
        """Position names for difference norm plots."""
        return self.source + self.destination


@dataclass
class AnalysisLayers:
    """Configurable layers for analysis plots.

    Layers are grouped by their significance in the model.
    """

    # Early layers (embedding processing)
    early: list[int] = field(default_factory=lambda: [0, 1, 2])

    # Key transition layers (where behavior changes)
    transitions: list[int] = field(default_factory=lambda: [19, 21, 24])

    # Late layers (decision making)
    late: list[int] = field(default_factory=lambda: [31, 34])

    # Final layer
    final: int = 35

    @property
    def annotation_layers(self) -> list[int]:
        """Layers to annotate on plots."""
        return sorted(set(self.transitions + self.late))

    @property
    def representative_layers(self) -> list[int]:
        """Representative layers for grid plots (early, mid, late)."""
        return [0, 12, 24, self.final]


# Default configurations
DEFAULT_POSITIONS = AnalysisPositions()
DEFAULT_LAYERS = AnalysisLayers()


@dataclass
class DiffmeansPlotConfig:
    """Configuration for diffmeans visualization."""

    positions: AnalysisPositions = field(default_factory=AnalysisPositions)
    layers: AnalysisLayers = field(default_factory=AnalysisLayers)

    # Zoomed plots
    stability_zoom_range: tuple[float, float] = (0.75, 1.0)

    # Whether to apply final LayerNorm for logit lens
    apply_final_layernorm: bool = True

    # Plot markers
    show_layer_annotations: bool = True
    annotation_fontsize: int = 8


@dataclass
class GeoPlotConfig:
    """Configuration for geo (PCA) visualization."""

    positions: AnalysisPositions = field(default_factory=AnalysisPositions)
    layers: AnalysisLayers = field(default_factory=AnalysisLayers)

    # Number of PCs to show in variance plot
    n_pcs_variance: int = 3

    # Whether to show scree plot (variance explained cumulative)
    show_scree_plot: bool = True
