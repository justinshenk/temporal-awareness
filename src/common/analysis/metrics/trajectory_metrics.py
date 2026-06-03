"""Trajectory metrics for sequence analysis.

Provides metrics for analyzing token trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...base_schema import BaseSchema
from ..analysis_base import DistributionalAnalysis
from ...math import (
    empirical_cross_entropy,
    inv_perplexity,
    perplexity,
    token_ranks_from_logits,
    top_p_normalized_logprobs,
    total_logprob,
    worst_rank_position,
    worst_token_logprob,
    worst_token_position,
    worst_token_rank,
)

if TYPE_CHECKING:
    from ...token_trajectory import TokenTrajectory


@dataclass
class TopPNormalizedMetrics(DistributionalAnalysis):
    """Metrics computed with top-p normalized probabilities."""

    p: int  # number of top tokens considered
    total_logprob: float  # Σ normalized logprobs
    worst_token_logprob: float  # min(normalized logprobs)
    worst_token_position: int  # argmin(normalized logprobs) - ABSOLUTE position

    @classmethod
    def from_logits(
        cls,
        token_ids: list[int],
        full_logits,
        p: int = 100,
    ) -> TopPNormalizedMetrics | None:
        """Build normalized metrics from logits.

        Args:
            token_ids: Token IDs to compute metrics for
            full_logits: Full vocabulary logits for each position
            p: Number of top tokens to consider

        Returns:
            TopPNormalizedMetrics if any tokens fall within top-p, else None.
            The worst_token_position is relative to the input slice (0-indexed).
            Caller is responsible for adjusting to absolute position.
        """
        norm_logprobs = top_p_normalized_logprobs(token_ids, full_logits, p)

        # Filter out -inf values and track their original positions
        finite_positions = [
            i for i, lp in enumerate(norm_logprobs) if math.isfinite(lp)
        ]
        finite_logprobs = [norm_logprobs[i] for i in finite_positions]

        if not finite_logprobs:
            return None

        # Find worst position among finite values only
        worst_idx_in_finite = worst_token_position(finite_logprobs)
        actual_position = finite_positions[worst_idx_in_finite]

        return cls(
            p=p,
            total_logprob=total_logprob(finite_logprobs),
            worst_token_logprob=worst_token_logprob(finite_logprobs),
            worst_token_position=actual_position,
        )


@dataclass
class TrajectoryMetrics(DistributionalAnalysis):
    """Metrics computed over a trajectory's logprob sequence.

    Note: The raw logprobs are NOT stored here since they already exist
    on the trajectory itself (traj.logprobs). This avoids duplication.
    """

    empirical_cross_entropy: float  # H = −mean(logprobs)         — lower  = better
    inv_perplexity: float  # e^{-H} = geomean(probs)  ∈ (0, 1] — higher = better
    perplexity: float  # e^{H}  = 1/inv_ppl       ∈ [1, ∞) — lower  = better
    total_logprob: float  # Σ logprobs — length-dependent      — higher = better

    worst_token_logprob: float  # min(logprobs)                 — higher = better
    worst_token_position: int  # argmin(logprobs) — where the hardest token is

    # Rank-based metrics (only when full_logits available)
    worst_token_rank: int | None = None  # rank of worst token (1=greedy)
    worst_rank_position: int | None = None  # position of worst-ranked token

    # Top-p normalized metrics (only when full_logits available)
    top_p_normalized: TopPNormalizedMetrics | None = None

    @classmethod
    def from_logprobs(cls, logprobs: list[float]) -> TrajectoryMetrics:
        return cls(
            empirical_cross_entropy=empirical_cross_entropy(logprobs),
            inv_perplexity=inv_perplexity(logprobs),
            perplexity=perplexity(logprobs),
            total_logprob=total_logprob(logprobs),
            worst_token_logprob=worst_token_logprob(logprobs),
            worst_token_position=worst_token_position(logprobs),
        )

    @classmethod
    def from_trajectory(
        cls,
        traj: "TokenTrajectory",
        start: int = 0,
        end: int | None = None,
        rank_start: int | None = None,
        top_p: int = 100,
    ) -> TrajectoryMetrics:
        """Build metrics from a trajectory, using full_logits if available.

        Args:
            traj: The trajectory to analyze
            start: Start position for all metrics (0 = full trajectory)
            end: End position (None = end of trajectory)
            rank_start: Start position for rank-based metrics (defaults to start).
                        Only set if you need rank metrics over a different range.
            top_p: Number of top tokens for normalized metrics

        All returned position fields are ABSOLUTE (relative to full trajectory).
        """
        end_pos = end if end is not None else traj.length
        rank_start_pos = rank_start if rank_start is not None else start

        # Stage 1: Basic metrics from logprobs
        logprobs = traj.logprobs[start:end_pos]
        metrics = cls.from_logprobs(logprobs)
        # Convert worst_token_position to absolute
        metrics.worst_token_position += start

        # Stage 2: Rank-based metrics (if available)
        if traj.full_logits is not None:
            _add_rank_metrics(metrics, traj, rank_start_pos, end_pos, top_p)

        return metrics


def _add_rank_metrics(
    metrics: TrajectoryMetrics,
    traj: "TokenTrajectory",
    start: int,
    end: int,
    top_p: int,
) -> None:
    """Add rank-based metrics to existing TrajectoryMetrics.

    Modifies metrics in place, adding worst_token_rank, worst_rank_position,
    and top_p_normalized fields. All positions are reported as ABSOLUTE
    (relative to full trajectory).

    Args:
        metrics: TrajectoryMetrics to augment
        traj: Source trajectory with full_logits
        start: Start position for rank metrics
        end: End position for rank metrics
        top_p: Number of top tokens for normalized metrics
    """
    if start >= traj.full_logits.shape[0]:
        return  # No logits available for this range

    token_ids = traj.token_ids[start:end]
    logits = traj.full_logits[start:end]

    # Basic rank metrics
    ranks = token_ranks_from_logits(token_ids, logits)
    metrics.worst_token_rank = worst_token_rank(ranks)
    metrics.worst_rank_position = worst_rank_position(ranks) + start  # Absolute

    # Top-p normalized metrics
    top_p_metrics = TopPNormalizedMetrics.from_logits(token_ids, logits, top_p)
    if top_p_metrics is not None:
        top_p_metrics.worst_token_position += start  # Convert to absolute
        metrics.top_p_normalized = top_p_metrics


@dataclass
class TrajectoryAnalysis(BaseSchema):
    """Analysis for a trajectory with trunk/continuation breakdown.

    Attributes:
        traj_idx: Index of this trajectory in the tree.
        trunk_last_idx: Last index of trunk (= trunk_length - 1), or None if no trunk info.
        full_traj: Metrics over the full trajectory [0, end).
        trunk_only: Metrics over [0, trunk_length) if trunk_length > 0, else None.
        continuation_only: Metrics over [trunk_length, end) if applicable, else None.
    """

    traj_idx: int
    trunk_last_idx: int | None
    full_traj: TrajectoryMetrics
    trunk_only: TrajectoryMetrics | None
    continuation_only: TrajectoryMetrics | None

    @classmethod
    def from_trajectory(
        cls,
        traj_idx: int,
        traj: "TokenTrajectory",
        trunk_length: int | None = None,
        top_p: int = 100,
    ) -> TrajectoryAnalysis:
        """Build analysis from a trajectory with optional trunk breakdown.

        Args:
            traj_idx: Index of this trajectory.
            traj: The trajectory to analyze.
            trunk_length: Length of trunk (prompt tokens). If None, no trunk info.
            top_p: Number of top tokens for normalized metrics.

        Returns:
            TrajectoryAnalysis with full_traj always computed, and trunk_only/
            continuation_only computed when trunk_length is provided and valid.
        """
        # Always compute full trajectory metrics
        full_traj = TrajectoryMetrics.from_trajectory(traj, start=0, top_p=top_p)

        trunk_only = None
        continuation_only = None
        trunk_last_idx = None

        if trunk_length is not None and trunk_length > 0:
            trunk_last_idx = trunk_length - 1

            # Trunk metrics: [0, trunk_length)
            trunk_only = TrajectoryMetrics.from_trajectory(
                traj, start=0, end=trunk_length, top_p=top_p
            )

            # Continuation metrics: [trunk_length, end) if there's continuation
            if trunk_length < traj.length:
                continuation_only = TrajectoryMetrics.from_trajectory(
                    traj, start=trunk_length, top_p=top_p
                )

        return cls(
            traj_idx=traj_idx,
            trunk_last_idx=trunk_last_idx,
            full_traj=full_traj,
            trunk_only=trunk_only,
            continuation_only=continuation_only,
        )

    @classmethod
    def from_logprobs(cls, traj_idx: int, logprobs: list[float]) -> TrajectoryAnalysis:
        """Build analysis from raw logprobs (no trunk info)."""
        return cls(
            traj_idx=traj_idx,
            trunk_last_idx=None,
            full_traj=TrajectoryMetrics.from_logprobs(logprobs),
            trunk_only=None,
            continuation_only=None,
        )
