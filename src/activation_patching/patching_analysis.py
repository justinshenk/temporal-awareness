"""Analysis classes for activation patching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.base_schema import BaseSchema
from ..common.analysis.trajectory_branch_node_analysis import ForkMetrics
from .patching_results import IntervenedChoice


@dataclass
class IntervenedChoiceAnalysis(BaseSchema):
    """Analysis of a single patching intervention on a binary choice.

    Extracts and computes metrics from the IntervenedChoice result,
    providing fork-level analysis for both original and intervened choices.

    Attributes:
        recovery: Normalized recovery metric (from IntervenedChoice)
        original_fork: ForkMetrics for the original (unpatched) choice
        intervened_fork: ForkMetrics for the intervened (patched) choice
        choice_flipped: Whether the intervention flipped the model's choice
        decoding_mismatch: Whether greedy generation mismatches choice probabilities.
            None if not verified, True if mismatch detected (possible degeneration),
            False if generation matches probability-based choice.
    """

    recovery: float
    original_fork: ForkMetrics
    intervened_fork: ForkMetrics
    choice_flipped: bool
    decoding_mismatch: bool | None = None

    @classmethod
    def from_intervened_choice(
        cls,
        ic: IntervenedChoice,
        decoding_mismatch: bool | None = None,
    ) -> IntervenedChoiceAnalysis:
        """Build analysis from an IntervenedChoice result."""
        return cls(
            recovery=ic.recovery,
            original_fork=_extract_fork_metrics(ic.original),
            intervened_fork=_extract_fork_metrics(ic.intervened),
            choice_flipped=ic.choice_flipped,
            decoding_mismatch=decoding_mismatch,
        )

    @property
    def is_valid_flip(self) -> bool:
        """True if choice flipped and no decoding mismatch detected."""
        return self.choice_flipped and self.decoding_mismatch is False

    @property
    def is_degenerate_flip(self) -> bool:
        """True if choice flipped but decoding mismatch was detected."""
        return self.choice_flipped and self.decoding_mismatch is True


@dataclass
class ActivationPatchingAnalysis(BaseSchema):
    """Aggregate analysis of activation patching results.

    Attributes:
        per_intervention: Analysis for each intervention
        mean_recovery: Mean recovery across all interventions
        flip_rate: Fraction of interventions that flipped the choice
        max_recovery: Maximum recovery achieved
        max_recovery_layer: Layer with maximum recovery (None if all layers patched together)
    """

    per_intervention: list[IntervenedChoiceAnalysis] = field(default_factory=list)

    @property
    def mean_recovery(self) -> float:
        if not self.per_intervention:
            return 0.0
        return sum(a.recovery for a in self.per_intervention) / len(self.per_intervention)

    @property
    def flip_rate(self) -> float:
        if not self.per_intervention:
            return 0.0
        return sum(1 for a in self.per_intervention if a.choice_flipped) / len(
            self.per_intervention
        )

    @property
    def max_recovery(self) -> float:
        if not self.per_intervention:
            return 0.0
        return max(a.recovery for a in self.per_intervention)

    @property
    def n_interventions(self) -> int:
        return len(self.per_intervention)

    @property
    def n_flipped(self) -> int:
        return sum(1 for a in self.per_intervention if a.choice_flipped)

    @property
    def n_valid_flips(self) -> int:
        """Count of flips that passed greedy verification."""
        return sum(1 for a in self.per_intervention if a.is_valid_flip)

    @property
    def n_degenerate_flips(self) -> int:
        """Count of flips with decoding mismatch (possible degeneration)."""
        return sum(1 for a in self.per_intervention if a.is_degenerate_flip)

    @property
    def valid_flip_rate(self) -> float:
        """Fraction of interventions that caused valid (verified) flips."""
        if not self.per_intervention:
            return 0.0
        return self.n_valid_flips / len(self.per_intervention)

    def _to_dict_hook(self, d: dict) -> dict:
        """Include computed properties in serialization."""
        d["mean_recovery"] = self.mean_recovery
        d["flip_rate"] = self.flip_rate
        d["max_recovery"] = self.max_recovery
        d["n_interventions"] = self.n_interventions
        d["n_flipped"] = self.n_flipped
        d["n_valid_flips"] = self.n_valid_flips
        d["n_degenerate_flips"] = self.n_degenerate_flips
        d["valid_flip_rate"] = self.valid_flip_rate
        return d


def _extract_fork_metrics(choice) -> ForkMetrics:
    """Extract ForkMetrics from a binary choice.

    Gets the fork metrics from the first fork in the choice's token tree.
    """
    from ..common.math import (
        log_odds,
        logprob_to_prob,
        probability_ratio,
        q_fork_concentration,
        q_fork_diversity,
        q_fork_entropy,
    )

    # Get logprobs from choice
    lp_chosen = choice.choice_logprob or 0.0
    lp_alt = choice.alternative_logprob or 0.0

    # Order as (A, B) based on choice_idx
    if choice.choice_idx == 0:
        lp_a, lp_b = lp_chosen, lp_alt
    else:
        lp_a, lp_b = lp_alt, lp_chosen

    p_a, p_b = logprob_to_prob(lp_a), logprob_to_prob(lp_b)

    return ForkMetrics(
        next_token_logprobs=(lp_a, lp_b),
        fork_entropy=q_fork_entropy(p_a, p_b, q=1.0),
        fork_diversity=q_fork_diversity(p_a, p_b, q=1.0),
        fork_simpson=q_fork_diversity(p_a, p_b, q=2.0),
        fork_concentration=q_fork_concentration(p_a, p_b, q=1.0),
        probability_ratio=probability_ratio(p_a, p_b),
        log_odds=log_odds(p_a, p_b),
        logit_diff=lp_a - lp_b,
        reciprocal_rank_a=1.0 if lp_a >= lp_b else 0.5,
    )
