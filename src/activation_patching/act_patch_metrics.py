"""Metrics extracted from activation patching results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..common.base_schema import BaseSchema

if TYPE_CHECKING:
    from .act_patch_results import IntervenedChoice


@dataclass
class IntervenedChoiceMetrics(BaseSchema):
    """Metrics extracted from an IntervenedChoice for visualization/analysis.

    Provides type-safe access to metrics with sensible defaults for missing data.
    """

    # Core metrics
    recovery: float = 0.0
    flipped: bool = False

    # Logprob metrics
    logit_diff: float = 0.0
    logprob_short: float = -10.0
    logprob_long: float = -10.0

    # Probability metrics (derived from logprobs)
    prob_short: float = 0.0
    prob_long: float = 0.0

    # Rank metrics
    reciprocal_rank_short: float = 0.0
    reciprocal_rank_long: float = 0.0

    # Distribution metrics
    fork_diversity: float = 1.0  # 1.0 = one option dominates, 2.0 = balanced
    vocab_entropy: float = 0.0
    inv_perplexity: float = 0.0

    @classmethod
    def from_choice(cls, choice: IntervenedChoice | None) -> IntervenedChoiceMetrics:
        """Extract metrics from an IntervenedChoice.

        Handles missing data gracefully with safe defaults.

        Args:
            choice: IntervenedChoice to extract metrics from, or None

        Returns:
            IntervenedChoiceMetrics with extracted values or defaults
        """
        if choice is None:
            return cls()

        # Start with defaults
        metrics = cls(
            recovery=choice.recovery,
            flipped=choice.flipped,
        )

        # Get the intervened LabeledSimpleBinaryChoice
        intervened = choice.intervened
        if intervened is None:
            return metrics

        # Logprobs from _divergent_logprobs (short=0, long=1)
        try:
            lp_short, lp_long = intervened._divergent_logprobs
            metrics.logprob_short = lp_short
            metrics.logprob_long = lp_long
            metrics.prob_short = math.exp(lp_short) if lp_short > -50 else 0.0
            metrics.prob_long = math.exp(lp_long) if lp_long > -50 else 0.0
            metrics.logit_diff = lp_short - lp_long
        except Exception:
            pass

        # ForkMetrics from tree.forks[0].analysis.metrics
        try:
            tree = intervened.tree
            if tree and tree.forks and tree.forks[0].analysis:
                fork_metrics = tree.forks[0].analysis.metrics
                metrics.logit_diff = fork_metrics.logit_diff
                metrics.fork_diversity = fork_metrics.fork_diversity
                # reciprocal_rank_a is for the A token (short)
                metrics.reciprocal_rank_short = fork_metrics.reciprocal_rank_a
                # For B token, it's the complement
                metrics.reciprocal_rank_long = (
                    1.0 - fork_metrics.reciprocal_rank_a + 0.5
                )
        except Exception:
            pass

        # NodeMetrics from tree.nodes[0].analysis.metrics
        try:
            tree = intervened.tree
            if tree and tree.nodes and tree.nodes[0].analysis:
                node_metrics = tree.nodes[0].analysis.metrics
                metrics.vocab_entropy = node_metrics.vocab_entropy
        except Exception:
            pass

        # TrajectoryMetrics from chosen_traj.analysis.full_traj
        try:
            chosen = intervened.chosen_traj
            if chosen and chosen.analysis and chosen.analysis.full_traj:
                traj_metrics = chosen.analysis.full_traj
                metrics.inv_perplexity = traj_metrics.inv_perplexity
        except Exception:
            pass

        return metrics

    @classmethod
    def from_dict(cls, d: dict) -> IntervenedChoiceMetrics:
        """Create from dictionary."""
        return super().from_dict(d)
