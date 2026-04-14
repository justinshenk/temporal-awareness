"""BinaryFork: a pairwise comparison between two branches."""

from __future__ import annotations

from dataclasses import dataclass

from .base_schema import BaseSchema
from .analysis.metrics import ForkAnalysis


@dataclass
class BinaryFork(BaseSchema):
    """A pairwise comparison between two branches at a divergence point.

    Attributes:
        next_token_ids: The two token IDs being compared (branch_a, branch_b)
        next_token_logprobs: Log-probabilities for each token
        group_idx: Which groups the two branches belong to (group_a, group_b)
        vocab_logits: Full logits over vocabulary at this fork position (for raw logit extraction)
    """

    next_token_ids: tuple[int, int]
    next_token_logprobs: tuple[float, float]
    group_idx: tuple[int, int] | None = None
    vocab_logits: list[float] | None = None
    analysis: ForkAnalysis | None = None

    @property
    def next_token_logits(self) -> tuple[float, float] | None:
        """Raw logits for the two tokens, extracted from vocab_logits."""
        if self.vocab_logits is None:
            return None
        id_a, id_b = self.next_token_ids
        return (self.vocab_logits[id_a], self.vocab_logits[id_b])
