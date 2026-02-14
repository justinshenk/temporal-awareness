"""Fork-level metrics for binary choice analysis.

"Fork" = a binary decision point between two alternatives (A vs B).
This is a special case of branch metrics, optimized for the common
two-choice scenario in LLM preference/choice analysis.

Input: raw scores for each candidate (p_A, p_B)
       Normalized internally so they form a distribution.

These metrics wrap entropy_diversity primitives with fork-specific
naming and a convenient (prob_a, prob_b) signature.
"""

from __future__ import annotations

import math

from .math_primitives import normalize_pair
from .entropy_diversity import (
    _EPS,
    probs_to_logprobs,
    q_diversity,
    q_concentration,
    renyi_entropy,
)


# ── Generalized fork metrics (order q) — most general ────────────────────────


def q_fork_diversity(prob_a: float, prob_b: float, q: float) -> float:
    """Effective number of options at this fork (D_q for binary).

    Wraps q_diversity from entropy_diversity.

    Args:
        prob_a, prob_b: Raw probabilities (will be normalized)
        q: Order parameter

    Returns:
        Value in [1, 2]:
            1.0 = one option dominates completely
            2.0 = perfectly balanced choice
    """
    p_a, p_b = normalize_pair(prob_a, prob_b)
    logprobs = probs_to_logprobs([p_a, p_b])
    return q_diversity(logprobs, q)


def q_fork_concentration(prob_a: float, prob_b: float, q: float) -> float:
    """Concentration at this fork (1/D_q for binary).

    Wraps q_concentration from entropy_diversity.

    Range: [0.5, 1]. Higher = more decisive.
    """
    p_a, p_b = normalize_pair(prob_a, prob_b)
    logprobs = probs_to_logprobs([p_a, p_b])
    return q_concentration(logprobs, q)


def q_fork_entropy(prob_a: float, prob_b: float, q: float) -> float:
    """Rényi entropy at this fork (H_q for binary).

    Wraps renyi_entropy from entropy_diversity.

    Range: [0, ln 2]. Lower = more decisive.
    """
    p_a, p_b = normalize_pair(prob_a, prob_b)
    logprobs = probs_to_logprobs([p_a, p_b])
    return renyi_entropy(logprobs, q)


# ── Ratio and odds metrics ───────────────────────────────────────────────────


def probability_ratio(prob_a: float, prob_b: float) -> float:
    """p_A / p_B. > 1 means A wins. ∞ if p_B = 0."""
    if prob_b < _EPS:
        return float("inf")
    return prob_a / prob_b


def log_odds(prob_a: float, prob_b: float) -> float:
    """log(p_A / p_B). > 0 means A wins. ±∞ at boundaries."""
    if prob_b < _EPS or prob_a < _EPS:
        return float("inf") if prob_a >= prob_b else float("-inf")
    return math.log(prob_a / prob_b)


# ── Decision strength metrics ────────────────────────────────────────────────


def margin(prob_a: float, prob_b: float) -> float:
    """Probability margin p_A - p_B (on normalized probs).

    Range: [-1, 1]. > 0 means A wins.
    """
    p_a, p_b = normalize_pair(prob_a, prob_b)
    return p_a - p_b


def abs_margin(prob_a: float, prob_b: float) -> float:
    """Absolute margin |p_A - p_B|.

    Range: [0, 1]. Higher = more decisive.
    """
    return abs(margin(prob_a, prob_b))


def winner(prob_a: float, prob_b: float) -> int:
    """Which option wins? 0 = A, 1 = B."""
    p_a, p_b = normalize_pair(prob_a, prob_b)
    return 0 if p_a >= p_b else 1


def winning_prob(prob_a: float, prob_b: float) -> float:
    """Probability of the winning option.

    Range: [0.5, 1]. Higher = more decisive.
    """
    p_a, p_b = normalize_pair(prob_a, prob_b)
    return max(p_a, p_b)
