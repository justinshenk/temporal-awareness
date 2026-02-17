"""BranchingNode: a divergence point in a token tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_schema import BaseSchema


@dataclass
class BranchingNode(BaseSchema):
    """A node where trajectories diverge, choosing different next tokens.

    Attributes:
        next_token_ids: Token IDs chosen by each branch at this divergence point
        next_token_logprobs: Log-probabilities for each branch's chosen token
        branching_token_position: Token position in the sequence where divergence occurs
        traj_idx: Indices of trajectories that pass through this node
        vocab_logits: Full logits over vocabulary for each trajectory at this position
        forks_idx: Indices into the parent tree's forks list
    """

    next_token_ids: tuple[int, ...]
    next_token_logprobs: tuple[float, ...]
    branching_token_position: int
    traj_idx: list[int] | None = None
    vocab_logits: list[list[float]] | None = None
    forks_idx: list[int] | None = None
    analysis: Any | None = None
