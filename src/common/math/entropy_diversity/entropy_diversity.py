"""Diversity functions (Hill numbers).

Provides:
- q_diversity: Hill number D_q (effective number of categories)
- q_concentration: 1/D_q (how concentrated)
"""

from __future__ import annotations

from ..num_types import Num, Nums, is_numpy, is_tensor
from .diversity_impl import (
    _q_concentration_native,
    _q_concentration_numpy,
    _q_concentration_torch,
    _q_diversity_native,
    _q_diversity_numpy,
    _q_diversity_torch,
)


def q_diversity(logprobs: Nums, q: float) -> Num:
    """Hill number D_q: effective number of categories of order q.

    D_q = exp(H_q) where H_q is Rényi entropy.

    This is THE unified diversity measure. All standard indices are special cases:
        q → -∞: 1 / min pᵢ  (maximum rarity)
        q = 0:  richness S  (count of categories with p > 0)
        q = 1:  exp(H)      (Shannon diversity, via L'Hôpital)
        q = 2:  1 / Σpᵢ²    (Simpson diversity)
        q → +∞: 1 / max pᵢ  (Berger-Parker index)

    Range: [1, n] where n = number of categories.
    Higher = more diverse. Monotonically decreasing in q.

    Args:
        logprobs: Log-probabilities (Sequence[float], np.ndarray, or torch.Tensor)
        q: Order parameter
    """
    if is_tensor(logprobs):
        return _q_diversity_torch(logprobs, q)
    if is_numpy(logprobs):
        return _q_diversity_numpy(logprobs, q)
    return _q_diversity_native(logprobs, q)


def q_concentration(logprobs: Nums, q: float) -> Num:
    """Concentration of order q (= 1/D_q).

    The "inverse diversity" - how concentrated is the distribution?
    Range: [1/n, 1]. Higher = more concentrated.
    """
    if is_tensor(logprobs):
        return _q_concentration_torch(logprobs, q)
    if is_numpy(logprobs):
        return _q_concentration_numpy(logprobs, q)
    return _q_concentration_native(logprobs, q)
