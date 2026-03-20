"""Configuration for analysis visualizations (diffmeans, geo).

Centralizes position/layer selection so it's easy to modify.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalysisPositions:
    """Configurable positions for analysis plots.

    Positions are grouped by their role in the prompt structure.
    """

    # Source positions (where short-term option info appears)
    source: list[int] = field(default_factory=lambda: [86, 87, 88])

    # Destination positions (where long-term option info appears)
    destination: list[int] = field(default_factory=lambda: [143, 144, 145])

    # Final position (where choice is made)
    final: list[int] = field(default_factory=lambda: [145, 146])

    @property
    def all_key_positions(self) -> list[int]:
        """All key positions in order."""
        return sorted(set(self.source + self.destination + self.final))

    @property
    def for_difference_norm(self) -> list[int]:
        """Positions to show in difference norm plots."""
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
