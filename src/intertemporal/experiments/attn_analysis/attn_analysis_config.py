"""Configuration for attention pattern analysis.

Uses semantic position names from SamplePositionMapping instead of absolute positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema

__all__ = [
    "AttnAnalysisConfig",
    "SOURCE_POSITIONS",
    "DEST_POSITIONS",
    "DEFAULT_LAYERS",
    "EARLY_LAYERS",
    "INTERMEDIATE_POSITIONS",
]


# Semantic position names (matching analyze_geometry.py)
SOURCE_POSITIONS = [
    "time_horizon",
    "post_time_horizon",
]

DEST_POSITIONS = [
    "response_choice_prefix",
    "response_choice",
    "response_reasoning_prefix",
]

# Layers for attention analysis (circuit layers)
DEFAULT_LAYERS = [19, 21, 24, 31, 34]

# Early layers for intermediate attention analysis
EARLY_LAYERS = [10, 11, 12, 13, 14, 15, 16, 17]

# Intermediate positions (where info flows from source to dest)
INTERMEDIATE_POSITIONS = [
    "action_content",
    "format_content",
    "chat_suffix",
]


@dataclass
class AttnAnalysisConfig(BaseSchema):
    """Configuration for attention pattern analysis."""

    # Layers to analyze
    layers: list[int] = field(default_factory=lambda: DEFAULT_LAYERS.copy())

    # Semantic position names for source (where attention comes FROM)
    source_positions: list[str] = field(default_factory=lambda: SOURCE_POSITIONS.copy())

    # Semantic position names for destination (where attention goes TO)
    dest_positions: list[str] = field(default_factory=lambda: DEST_POSITIONS.copy())

    # Whether to store full attention patterns (memory intensive)
    store_patterns: bool = True

    # Threshold for detecting dynamic attention changes
    dynamic_threshold: float = 0.1

    # Whether to run intermediate attention analysis
    analyze_intermediate: bool = False

    # Layers for intermediate analysis
    intermediate_layers: list[int] = field(default_factory=lambda: EARLY_LAYERS.copy())

    # Intermediate positions for info flow analysis
    intermediate_positions: list[str] = field(
        default_factory=lambda: INTERMEDIATE_POSITIONS.copy()
    )
