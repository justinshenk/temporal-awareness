"""Primitive operations for entropy/diversity calculations.

Provides:
- _EPS: numerical stability constant
- Conversion: probs_to_logprobs, logprobs_to_probs
- Log-sum-exp: log_sum_exp (numerically stable)
- Pointwise: surprise, rarity
"""

from __future__ import annotations

from ..num_types import Num, Nums, is_numpy, is_tensor
from .core_impl import (
    _EPS,
    _log_sum_exp_native,
    _log_sum_exp_numpy,
    _log_sum_exp_torch,
    _logprobs_to_probs_native,
    _logprobs_to_probs_numpy,
    _logprobs_to_probs_torch,
    _probs_to_logprobs_native,
    _probs_to_logprobs_numpy,
    _probs_to_logprobs_torch,
    _rarity_native,
    _rarity_numpy,
    _rarity_torch,
)

# Note: Do NOT define __all__ here - let auto_export include all public names


# ── Conversion helpers ────────────────────────────────────────────────────────


def probs_to_logprobs(probs: Nums) -> Nums:
    """Convert probabilities to log-probabilities."""
    if is_tensor(probs):
        return _probs_to_logprobs_torch(probs)
    if is_numpy(probs):
        return _probs_to_logprobs_numpy(probs)
    return _probs_to_logprobs_native(probs)


def logprobs_to_probs(logprobs: Nums) -> Nums:
    """Convert log-probabilities to probabilities."""
    if is_tensor(logprobs):
        return _logprobs_to_probs_torch(logprobs)
    if is_numpy(logprobs):
        return _logprobs_to_probs_numpy(logprobs)
    return _logprobs_to_probs_native(logprobs)


# ── Log-sum-exp ───────────────────────────────────────────────────────────────


def log_sum_exp(values: Nums) -> Num:
    """Compute log(Σ exp(xᵢ)) in a numerically stable way."""
    if is_tensor(values):
        return _log_sum_exp_torch(values)
    if is_numpy(values):
        return _log_sum_exp_numpy(values)
    return _log_sum_exp_native(values)


# ── Pointwise measures ────────────────────────────────────────────────────────


def surprise(logprob: Num) -> Num:
    """Information content (self-information) at a single position.

    s = -log p = log(1/p)

    Also known as: surprisal, Shannon information.
    Range: [0, ∞). Lower = less surprising (more expected).
    """
    return -logprob


def rarity(logprob: Num) -> Num:
    """Rarity of an outcome (inverse probability).

    r = 1/p = exp(-log p)

    Interpretation: "effective number of equiprobable alternatives."
    Range: [1, ∞). Lower = more common (model expected this).
    """
    if is_tensor(logprob):
        return _rarity_torch(logprob)
    if is_numpy(logprob):
        return _rarity_numpy(logprob)
    return _rarity_native(logprob)
