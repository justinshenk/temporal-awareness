"""Configuration for geometry pipeline analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.semantic_positions import PROMPT_POSITIONS, RESPONSE_POSITIONS


@dataclass
class GeometryPipelineConfig:
    """Configuration for full geometry pipeline analysis.

    This config controls geometric analysis including linear probes,
    cross-position similarity, continuous time probes, etc.

    Attributes:
        enabled: Whether to run geometry pipeline analysis
        layers: Layers to analyze
        components: Model components to extract
        positions: Semantic position names to analyze
        n_pca_components: Number of PCA components
        skip_viz: Skip visualization generation
        skip_per_target_plots: Skip per-target individual plots
        run_cross_position_similarity: Run cross-position similarity analysis
        run_continuous_time_probe: Run continuous time linear probe
        seed: Random seed for reproducibility
    """

    enabled: bool = False
    layers: list[int] = field(default_factory=lambda: [0, 12, 19, 21, 24, 28, 31, 35])
    components: list[str] = field(
        default_factory=lambda: ["resid_pre", "attn_out", "mlp_out", "resid_post"]
    )
    positions: list[str] = field(
        default_factory=lambda: PROMPT_POSITIONS + RESPONSE_POSITIONS
    )
    n_pca_components: int = 10
    skip_viz: bool = False
    skip_per_target_plots: bool = True
    run_cross_position_similarity: bool = True
    run_continuous_time_probe: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> GeometryPipelineConfig:
        """Create config from dict, handling defaults."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
