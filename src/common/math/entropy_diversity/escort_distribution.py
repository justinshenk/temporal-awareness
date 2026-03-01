"""Escort distribution (q-tilted view of abundances).

The escort distribution shows how abundances "look" through a q-lens:
    π_i^(q) = p_i^q / Σ p_j^q

Provides:
- escort_logprobs: escort distribution in log space
- escort_probs: escort distribution as probabilities
"""

from __future__ import annotations

from ..num_types import Nums, is_numpy, is_tensor
from .entropy_primitives import logprobs_to_probs
from .escort_distribution_impl import (
    _escort_logprobs_native,
    _escort_logprobs_numpy,
    _escort_logprobs_torch,
)


def escort_logprobs(logprobs: Nums, q: float) -> Nums:
    """Escort distribution: the q-tilted view of abundances (returns logprobs).

    π_i^(q) = p_i^q / Σ p_j^q

    The escort distribution shows how abundances "look" at order q:
        q → 0:  uniform over support (democratic lens)
        q = 1:  original distribution (no distortion)
        q = 2:  dominant species amplified, rare shrink
        q → ∞:  all mass on argmax (autocratic lens)
        q < 0:  rare species amplified (contrarian lens)

    The normalization constant Σ p_i^q connects to Hill numbers:
        D_q = (Σ p_i^q)^(1/(1-q))

    Returns log-probabilities for numerical stability.
    """
    if is_tensor(logprobs):
        return _escort_logprobs_torch(logprobs, q)
    if is_numpy(logprobs):
        return _escort_logprobs_numpy(logprobs, q)
    return _escort_logprobs_native(logprobs, q)


def escort_probs(logprobs: Nums, q: float) -> Nums:
    """Escort distribution as probabilities.

    Convenience wrapper: exp(escort_logprobs(logprobs, q)).
    """
    escort_lp = escort_logprobs(logprobs, q)
    return logprobs_to_probs(escort_lp)
