"""Convenience wrappers for common parameter values.

Provides named functions for commonly-used orders:
- Diversity: richness (q=0), shannon_diversity (q=1), simpson_diversity (q=2)
- Concentration: shannon_concentration (q=1), simpson_concentration (q=2)
- Power mean: geometric_mean_prob (α=0), arithmetic_mean_prob (α=1), etc.
"""

from __future__ import annotations

from ..num_types import Num, Nums
from .diversity import q_diversity, q_concentration
from .power_mean import power_mean_from_logprobs


# ── Diversity (D_q) ───────────────────────────────────────────────────────────


def richness(logprobs: Nums) -> Num:
    """Richness D₀: count of non-zero categories.

    "How many options exist at all?" Ignores probability mass entirely.
    A uniform distribution over 10 items has richness 10, same as a
    distribution with 99% on one item and 1% spread over 9 others.
    """
    return q_diversity(logprobs, q=0.0)


def shannon_diversity(logprobs: Nums) -> Num:
    """Shannon diversity D₁ = exp(H): effective number of categories.

    "How many categories contribute meaningfully?" Balances between
    counting rare categories (like richness) and ignoring them (like Simpson).
    A distribution with 50% on one item and 50% on another has D₁ = 2.
    A distribution with 90% on one and 10% on another has D₁ ≈ 1.47.
    """
    return q_diversity(logprobs, q=1.0)


def simpson_diversity(logprobs: Nums) -> Num:
    """Simpson diversity D₂ = 1/Σpᵢ²: effective number of dominant categories.

    "If I pick two items randomly, how often are they different?"
    More sensitive to dominant categories than Shannon diversity.
    A distribution with 90% on one item has D₂ ≈ 1.22 (nearly 1 = no diversity).
    """
    return q_diversity(logprobs, q=2.0)


# ── Concentration (1/D_q) ─────────────────────────────────────────────────────


def shannon_concentration(logprobs: Nums) -> Num:
    """Shannon concentration 1/D₁ = exp(-H): how peaked is the distribution?

    "What fraction of categories capture the mass?" Range: [1/n, 1].
    A uniform distribution over 10 items has concentration 0.1.
    A distribution with all mass on one item has concentration 1.0.
    Higher = more concentrated on fewer categories.
    """
    return q_concentration(logprobs, q=1.0)


def simpson_concentration(logprobs: Nums) -> Num:
    """Simpson concentration 1/D₂ = Σpᵢ²: probability of collision.

    "If I pick two items randomly, how often are they the same?"
    Also known as the Herfindahl-Hirschman index in economics.
    Range: [1/n, 1]. Higher = more concentrated.
    """
    return q_concentration(logprobs, q=2.0)


# ── Power mean wrappers (for trajectories) ────────────────────────────────────


def geometric_mean_prob(logprobs: Nums) -> Num:
    """Geometric mean of probabilities M₀(p) = exp(mean(logprobs)).

    "What's the typical probability per position?" This is exactly
    inverse perplexity: 1/perplexity = exp(-cross_entropy).
    Range: (0, 1]. Higher = model is more confident on average.
    A sequence where every token has prob 0.5 has M₀ = 0.5.
    """
    return power_mean_from_logprobs(logprobs, alpha=0.0)


def arithmetic_mean_prob(logprobs: Nums) -> Num:
    """Arithmetic mean of probabilities M₁(p) = mean(exp(logprobs)).

    "What's the average probability?" More optimistic than geometric mean
    because high-probability tokens pull the average up.
    A sequence with probs [0.9, 0.1] has M₁ = 0.5, but M₀ = 0.30.
    """
    return power_mean_from_logprobs(logprobs, alpha=1.0)


def harmonic_mean_prob(logprobs: Nums) -> Num:
    """Harmonic mean of probabilities M₋₁(p).

    More pessimistic than geometric mean - dominated by low probabilities.
    Useful for worst-case-sensitive analysis.
    A sequence with probs [0.9, 0.1] has M₋₁ = 0.18, M₀ = 0.30, M₁ = 0.50.
    """
    return power_mean_from_logprobs(logprobs, alpha=-1.0)


def min_prob(logprobs: Nums) -> Num:
    """Minimum probability in sequence: M₋∞(p) = min(exp(logprobs)).

    The worst-case token probability. Useful for identifying
    the hardest prediction in a sequence.
    """
    return power_mean_from_logprobs(logprobs, alpha=float("-inf"))


def max_prob(logprobs: Nums) -> Num:
    """Maximum probability in sequence: M₊∞(p) = max(exp(logprobs)).

    The best-case token probability. Useful for identifying
    the easiest prediction in a sequence.
    """
    return power_mean_from_logprobs(logprobs, alpha=float("inf"))
