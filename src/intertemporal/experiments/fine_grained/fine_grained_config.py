"""Configuration for fine-grained patching analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FineGrainedConfig:
    """Configuration for fine-grained activation patching analysis.

    This config controls comprehensive head-level, position-level,
    path patching, and neuron-level analysis.

    Attributes:
        enabled: Whether to run fine-grained analysis
        n_layers: Number of model layers (set at runtime)
        n_heads: Number of attention heads per layer (set at runtime)

        # Head patching settings
        head_patching_enabled: Whether to run head-level patching
        head_layers: Layers to analyze for head patching (None = all)

        # Position patching settings
        position_patching_enabled: Whether to run position-level patching for top heads
        n_top_heads_for_position: Number of top heads to analyze by position
        position_range: Tuple of (start_pos, end_pos) or None for auto-detect

        # Path patching settings
        path_patching_enabled: Whether to run path patching analysis
        source_layers: Source layers for path patching (attention heads)
        dest_mlp_layers: Destination MLP layers for head-to-MLP path patching
        dest_head_layers: Destination head layers for head-to-head path patching
        n_top_source_heads: Number of top source heads to use

        # Multi-site patching settings
        multi_site_enabled: Whether to run multi-site interaction analysis
        n_components_multi_site: Number of top components for interaction analysis

        # Neuron patching settings
        neuron_patching_enabled: Whether to run neuron-level ablation
        neuron_target_layer: Layer for neuron-level analysis (e.g., L31)
        n_top_neurons: Number of top neurons to track

        # Layer-position fine heatmap settings
        layer_position_enabled: Whether to run layer x position fine patching
        layer_position_components: Components for layer-position analysis
    """

    enabled: bool = False
    n_layers: int = 0
    n_heads: int = 0

    # Head patching
    head_patching_enabled: bool = True
    head_layers: list[int] | None = None  # None = all layers

    # Position patching for top heads
    position_patching_enabled: bool = True
    n_top_heads_for_position: int = 5
    position_range: tuple[int, int] | None = None  # (start, end) or auto

    # Path patching
    path_patching_enabled: bool = True
    source_layers: list[int] = field(default_factory=lambda: [19, 21, 24])
    dest_mlp_layers: list[int] = field(default_factory=lambda: [28, 31, 34])
    dest_head_layers: list[int] = field(default_factory=lambda: [28, 29, 30, 31])
    n_top_source_heads: int = 5

    # Multi-site interaction
    multi_site_enabled: bool = True
    n_components_multi_site: int = 10

    # Neuron patching
    neuron_patching_enabled: bool = True
    neuron_target_layer: int = 31
    n_top_neurons: int = 50

    # Layer-position fine heatmap
    layer_position_enabled: bool = True
    layer_position_components: list[str] = field(
        default_factory=lambda: ["attn_out", "mlp_out"]
    )
    layer_position_layers: list[int] | None = None  # None = layers 15-35

    @classmethod
    def from_dict(cls, d: dict) -> FineGrainedConfig:
        """Create config from dict, handling defaults."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Default configuration
DEFAULT_FINE_GRAINED_CONFIG = FineGrainedConfig()
