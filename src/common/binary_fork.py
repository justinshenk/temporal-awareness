"""BinaryFork: a pairwise comparison between two branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_schema import BaseSchema


@dataclass
class BinaryFork(BaseSchema):
    """A pairwise comparison between two branches at a divergence point.

    Attributes:
        next_token_ids: The two token IDs being compared (branch_a, branch_b)
        next_token_logprobs: Log-probabilities for each token
        group_idx: Which groups the two branches belong to (group_a, group_b)
    """

    next_token_ids: tuple[int, int]
    next_token_logprobs: tuple[float, float]
    group_idx: tuple[int, int] | None = None
    analysis: Any | None = None
