"""Configuration for fine-grained patching analysis.

NOTE: Head attribution and position patching config are now in attn_analysis_config.py
NOTE: Neuron attribution config is now in mlp_analysis_config.py
NOTE: Layer-position patching is now in attn (for attn_out) and mlp (for mlp_out)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FineGrainedConfig:
    """Configuration for fine-grained path patching analysis.

    Attributes:
        n_layers: Number of model layers (set at runtime)
        n_heads: Number of attention heads per layer (set at runtime)

        # Path patching settings
        path_patching_enabled: Whether to run path patching analysis
        dest_mlp_layers: Destination MLP layers for head-to-MLP path patching
        dest_head_layers: Destination head layers for head-to-head path patching
        n_top_source_heads: Number of top source heads to use

        # Multi-site patching settings
        multi_site_enabled: Whether to run multi-site interaction analysis
        n_components_multi_site: Number of top components for interaction analysis
    """

    n_layers: int = 0
    n_heads: int = 0

    # Path patching
    path_patching_enabled: bool = True
    dest_mlp_layers: list[int] = field(default_factory=lambda: [28, 29, 30, 31, 34])
    dest_head_layers: list[int] = field(default_factory=lambda: [28, 29, 30, 31, 34])
    n_top_source_heads: int = 5

    # Multi-site interaction
    multi_site_enabled: bool = True
    n_components_multi_site: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> FineGrainedConfig:
        """Create config from dict, handling defaults."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
