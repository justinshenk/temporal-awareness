"""Power mean functions.

Provides:
- power_mean: generalized mean of order α (for raw values)
- weighted_power_mean: generalized mean with custom weights (for structure-aware analysis)
- power_mean_from_logprobs: power mean of probabilities (for trajectory analysis)
"""

from __future__ import annotations

from ..num_types import Num, Nums, is_numpy, is_tensor
from .power_mean_impl import (
    _power_mean_from_logprobs_native,
    _power_mean_from_logprobs_numpy,
    _power_mean_from_logprobs_torch,
    _power_mean_native,
    _power_mean_numpy,
    _power_mean_torch,
    _weighted_power_mean_native,
    _weighted_power_mean_numpy,
    _weighted_power_mean_torch,
)


def power_mean(values: Nums, alpha: float) -> Num:
    """Generalized (power) mean of order α.

    M_α(x₁, ..., xₙ) = (Σ xᵢ^α / n)^(1/α)

    Special cases:
        α → -∞: minimum
        α = -1: harmonic mean
        α → 0:  geometric mean
        α = 1:  arithmetic mean
        α = 2:  quadratic mean (RMS)
        α → +∞: maximum

    The power mean is monotonic in α: M_α ≤ M_β for α < β.
    """
    if is_tensor(values):
        return _power_mean_torch(values, alpha)
    if is_numpy(values):
        return _power_mean_numpy(values, alpha)
    return _power_mean_native(values, alpha)


def weighted_power_mean(values: Nums, weights: Nums, alpha: float) -> Num:
    """Weighted power mean of order α.

    M_α^w(x) = (Σ wᵢ xᵢ^α)^(1/α)

    Special cases:
        α → -∞: minimum (ignoring weights)
        α → 0:  weighted geometric mean = exp(Σ wᵢ log(xᵢ))
        α = 1:  weighted arithmetic mean = Σ wᵢ xᵢ
        α → +∞: maximum (ignoring weights)

    Args:
        values: The values to average.
        weights: Probability weights (must sum to 1).
        alpha: The power mean exponent.

    Returns:
        The weighted power mean.

    Used in structure-aware analysis to compute:
        ⟨Λ_n⟩_q = (Σ πᵢ cᵢ^q)^(1/q)
    where πᵢ are escort weights and cᵢ are compliances.
    """
    if is_tensor(values):
        return _weighted_power_mean_torch(values, weights, alpha)
    if is_numpy(values):
        return _weighted_power_mean_numpy(values, weights, alpha)
    return _weighted_power_mean_native(values, weights, alpha)


def power_mean_from_logprobs(logprobs: Nums, alpha: float) -> Num:
    """Power mean M_α of probabilities, computed stably from logprobs.

    M_α(p) = (Σ pᵢ^α / n)^(1/α)

    This is the standard generalized mean applied to probabilities exp(logprobs),
    but computed in log-space for numerical stability.

    Special cases (same as power_mean, but for probabilities):
        α → -∞: min(p)  (worst-case probability)
        α = -1: harmonic mean
        α → 0:  geometric mean = exp(mean(lp)) = 1/perplexity
        α = 1:  arithmetic mean
        α → +∞: max(p)  (best-case probability)

    Relationship to diversity (D_q):
        For a probability distribution, D_q = (Σpᵢ^q)^(1/(1-q)).
        Power mean differs: M_α = (Σpᵢ^α / n)^(1/α).
        They measure different things:
        - Diversity: "effective number of categories" in a distribution
        - Power mean: "typical probability" in a sequence

    Range: (0, 1]. Higher = model more confident. Monotonic in α.
    """
    if is_tensor(logprobs):
        return _power_mean_from_logprobs_torch(logprobs, alpha)
    if is_numpy(logprobs):
        return _power_mean_from_logprobs_numpy(logprobs, alpha)
    return _power_mean_from_logprobs_native(logprobs, alpha)
