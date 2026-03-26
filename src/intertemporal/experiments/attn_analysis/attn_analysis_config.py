"""Configuration for attention pattern analysis.

Uses semantic position names from SamplePositionMapping instead of absolute positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema

__all__ = [
    "AttnAnalysisConfig",
    "PROMPT_POSITIONS",
    "RESPONSE_POSITIONS",
    "DEFAULT_LAYERS",
    "EARLY_LAYERS",
    "INTERMEDIATE_POSITIONS",
]


# Prompt positions (where time horizon info is encoded)
PROMPT_POSITIONS = [
    "time_horizon",
    "post_time_horizon",
]

# Response positions (where model output is generated)
RESPONSE_POSITIONS = [
    "chat_suffix",
    "response_choice_prefix",
    "response_choice",
    "response_reasoning_prefix",
    "response_reasoning",
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

    # Prompt positions (where time horizon info is encoded)
    source_positions: list[str] = field(default_factory=lambda: PROMPT_POSITIONS.copy())

    # Response positions (where model output is generated)
    dest_positions: list[str] = field(default_factory=lambda: RESPONSE_POSITIONS.copy())

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
