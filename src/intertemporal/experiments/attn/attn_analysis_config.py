"""Configuration for attention pattern analysis.

Uses semantic position names from SamplePositionMapping instead of absolute positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema
from ...common.semantic_positions import (
    DEFAULT_LAYERS,
    PROMPT_CONSTRAINT_POSITIONS,
    PROMPT_SRC_POSITIONS,
    RESPONSE_POSITIONS,
)


@dataclass
class AttnAnalysisConfig(BaseSchema):
    """Configuration for attention pattern analysis."""

    # Layers to analyze
    layers: list[int] = field(default_factory=lambda: DEFAULT_LAYERS.copy())

    # Prompt positions (where time horizon info is encoded)
    source_positions: list[str] = field(
        default_factory=lambda: PROMPT_SRC_POSITIONS.copy()
    )

    # Response positions (where model output is generated)
    dest_positions: list[str] = field(default_factory=lambda: RESPONSE_POSITIONS.copy())

    # Whether to store full attention patterns (memory intensive)
    store_patterns: bool = True

    # Threshold for detecting dynamic attention changes
    dynamic_threshold: float = 0.1

    # --- Head Attribution Settings ---
    head_attribution_enabled: bool = True

    # --- Position Patching Settings ---
    position_patching_enabled: bool = True
    n_top_heads_for_position: int = 4
    position_patching_positions: list[str] = field(
        default_factory=lambda: list(PROMPT_CONSTRAINT_POSITIONS)
    )

    # --- Head Redundancy Analysis Settings ---
    head_redundancy_enabled: bool = True
    head_redundancy_top_n: int = 20  # Analyze top N heads by attribution

    # --- Layer-Position Patching Settings ---
    layer_position_enabled: bool = True
    layer_position_positions: list[str] = field(
        default_factory=lambda: list(RESPONSE_POSITIONS)
    )
