"""Fork metrics for binary choice analysis.

Provides metrics for analyzing binary forks (two competing tokens).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...base_schema import BaseSchema
from ..analysis_base import DistributionalAnalysis


@dataclass
class ForkMetrics(DistributionalAnalysis):
    """Metrics for a binary fork (two competing tokens)."""

    next_token_logprobs: tuple[float, float]  # (lp_A, lp_B) at the fork

    fork_entropy: float  # H of (p_A, p_B) normalised      — lower  = more decisive
    fork_diversity: float  # D₁ = e^H             ∈ [1, 2]   — lower  = more decisive
    fork_simpson: float  # D₂ = 1/(p_A²+p_B²)  ∈ [1, 2]   — lower  = more decisive
    fork_concentration: (
        float  # 1/D₁ = e^{-H}        ∈ [0.5, 1] — higher = more decisive
    )

    probability_ratio: float  # p_A / p_B at divergent pos       — >1 means A wins
    log_odds: float  # log(p_A / p_B)                   — >0 means A wins
    logit_diff: float  # logit_A - logit_B = lp_A - lp_B — >0 means A wins
    reciprocal_rank_a: (
        float  # 1/rank_A in binary comparison — 1.0 if A wins, 0.5 if B wins
    )

    # Raw logits (optional, populated when vocab_logits available)
    logits: tuple[float, float] | None = None  # raw logits (A, B)
    normalized_logits: tuple[float, float] | None = None  # z-score normalized


@dataclass
class ForkAnalysis(BaseSchema):
    """Analysis for a binary fork."""

    fork_idx: int
    metrics: ForkMetrics
