"""Entropy functions (Rényi family).

Provides:
- renyi_entropy: generalized entropy of order q
- shannon_entropy: special case q=1
"""

from __future__ import annotations

from ..num_types import Num, Nums, is_numpy, is_tensor
from .entropy_impl import (
    _renyi_entropy_native,
    _renyi_entropy_numpy,
    _renyi_entropy_torch,
)


def renyi_entropy(logprobs: Nums, q: float) -> Num:
    """Rényi entropy of order q (numerically stable, takes logprobs).

    H_q = (1/(1-q)) · log(Σ pᵢ^q)

    Special cases:
        q = 0:  log(S)      (Hartley entropy, log of richness)
        q = 1:  H           (Shannon entropy, via L'Hôpital)
        q = 2:  -log(Σpᵢ²)  (collision entropy)
        q → ∞:  -log(max pᵢ) (min-entropy)

    Connection to Hill numbers: D_q = exp(H_q)

    Args:
        logprobs: Log-probabilities (Sequence[float], np.ndarray, or torch.Tensor)
        q: Order parameter
    """
    if is_tensor(logprobs):
        return _renyi_entropy_torch(logprobs, q)
    if is_numpy(logprobs):
        return _renyi_entropy_numpy(logprobs, q)
    return _renyi_entropy_native(logprobs, q)


def shannon_entropy(logprobs: Nums) -> Num:
    """Shannon entropy (= renyi_entropy with q=1).

    H = -Σ pᵢ log pᵢ = -Σ exp(lpᵢ) * lpᵢ

    Takes logprobs for numerical stability.
    """
    return renyi_entropy(logprobs, q=1.0)
