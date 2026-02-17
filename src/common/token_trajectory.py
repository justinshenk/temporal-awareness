"""TokenTrajectory: a sequence of tokens with logprobs and logits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .base_schema import BaseSchema


@dataclass
class TokenTrajectory(BaseSchema):
    """A sequence of tokens with associated logprobs and logits.

    All arrays have length n_sequence (full sequence length).
    The first token has logprob=0 (probability 1, since it's given).
    """

    token_ids: list[int]
    logprobs: list[float]
    logits: list[float]
    full_logits: torch.Tensor | None = None

    nodes_idx: tuple[int, ...] | None = None
    group_idx: tuple[int, ...] | None = None
    analysis: Any | None = None

    def can_have_internals(self) -> bool:
        return False

    def has_internals(self) -> bool:
        return False

    def has_internals_for(self, names_filter: callable | None = None) -> bool:
        return False

    @property
    def n_sequence(self) -> int:
        return len(self.token_ids)

    @property
    def sequence_length(self) -> int:
        return self.n_sequence

    @property
    def length(self) -> int:
        return self.n_sequence

    @property
    def n_pred(self) -> int:
        return max(0, self.n_sequence - 1)

    @property
    def predictions_length(self) -> int:
        return self.n_pred

    @property
    def pred_token_ids(self) -> list[int]:
        return self.token_ids[1:]

    @property
    def pred_logprobs(self) -> list[float]:
        return self.logprobs[1:]

    @property
    def pred_logits(self) -> list[float]:
        return self.logits[1:]

    @property
    def pred_full_logits(self) -> torch.Tensor | None:
        if self.full_logits is None:
            return None
        return self.full_logits[1:]

    @property
    def next_token_logprob_sequence(self) -> list[float]:
        return self.pred_logprobs

    @property
    def branching_points(self) -> list[int]:
        if self.nodes_idx is None:
            return []
        return list(getattr(self, "_branching_positions", []))

    def pop_heavy(self) -> None:
        self.pop_full_logits()

    def pop_full_logits(self) -> torch.Tensor | None:
        seq = self.full_logits
        self.full_logits = None
        return seq

    def to_dict(self) -> dict:
        full_logits = self.pop_full_logits()
        d = super().to_dict()
        self.full_logits = full_logits
        return d

    def get_conditional_prob(
        self, start_token_ids_pos: int, end_token_ids_pos: int
    ) -> float | None:
        if (
            start_token_ids_pos < 0
            or end_token_ids_pos > self.length
            or start_token_ids_pos >= end_token_ids_pos
        ):
            return None
        log_prob_sum = sum(self.logprobs[start_token_ids_pos:end_token_ids_pos])
        return math.exp(log_prob_sum)
