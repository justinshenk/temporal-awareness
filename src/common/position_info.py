"""Generic position information types.

These types are generic and can be used across different domains.
Domain-specific builders should be in their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base_schema import BaseSchema


@dataclass
class TokenPositionInfo(BaseSchema):
    """Info for a single token position.

    Attributes:
        abs_pos: Absolute position index in the sequence
        decoded_token: The decoded token string at this position
        traj_section: Either "prompt" or "response"
        format_pos: Semantic position name (e.g., "response_choice_prefix"), or None
        rel_pos: Relative position within format_pos (0-indexed), -1 if not in named position
    """

    abs_pos: int
    decoded_token: str = ""
    traj_section: str = ""  # "prompt" or "response"
    format_pos: str | None = None
    rel_pos: int = -1
