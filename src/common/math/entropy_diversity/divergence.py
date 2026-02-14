"""Divergence functions (relative entropy).

Provides:
- kl_divergence: Kullback-Leibler divergence D_KL(p || q)
- renyi_divergence: Rényi divergence of order α

Key distinction:
- Entropy H_q(p): measures uncertainty in a single distribution
- Divergence D_α(p || q): measures "distance" between two distributions
"""

from __future__ import annotations

import numpy as np

from ..num_types import Num, Nums, is_numpy, is_tensor
from .core_impl import _EPS
from .divergence_impl import (
    _kl_divergence_native,
    _kl_divergence_numpy,
    _kl_divergence_torch,
    _renyi_divergence_native,
    _renyi_divergence_numpy,
    _renyi_divergence_torch,
)


def kl_divergence(
    p: Nums,
    q: Nums,
    normalize: bool = True,
) -> Num:
    """Kullback-Leibler divergence D_KL(p || q).

    D_KL(p || q) = Σ_i p_i log(p_i / q_i)

    Properties:
    - Non-negative: D_KL(p || q) >= 0
    - Zero iff p = q
    - Asymmetric: D_KL(p || q) != D_KL(q || p)
    - Not a metric (doesn't satisfy triangle inequality)

    Args:
        p: First distribution (the "true" distribution)
        q: Second distribution (the "model" distribution)
        normalize: If True, normalize p and q to sum to 1

    Returns:
        KL divergence value
    """
    if is_tensor(p) and is_tensor(q):
        if p.shape != q.shape:
            raise ValueError("p and q must have same shape")
        if normalize:
            p = p / p.sum().clamp(min=_EPS)
            q = q / q.sum().clamp(min=_EPS)
        return _kl_divergence_torch(p, q)

    if is_numpy(p) and is_numpy(q):
        if p.shape != q.shape:
            raise ValueError("p and q must have same shape")
        if normalize:
            p = p / np.clip(p.sum(), _EPS, None)
            q = q / np.clip(q.sum(), _EPS, None)
        return _kl_divergence_numpy(p, q)

    p_list = list(p)
    q_list = list(q)
    if len(p_list) != len(q_list):
        raise ValueError("p and q must have same length")

    if normalize:
        p_sum = sum(p_list)
        q_sum = sum(q_list)
        if p_sum > _EPS:
            p_list = [x / p_sum for x in p_list]
        if q_sum > _EPS:
            q_list = [x / q_sum for x in q_list]

    return _kl_divergence_native(p_list, q_list)


def renyi_divergence(
    p: Nums,
    q: Nums,
    alpha: float = 1.0,
    normalize: bool = True,
) -> Num:
    """Rényi divergence (relative entropy) of order α.

    D_α(p || q) = (1/(α-1)) log Σ_i p_i^α q_i^(1-α)

    Generalizes KL divergence to a family indexed by α:
        α = 0:   -log(q(support(p))) (support coverage)
        α = 0.5: twice the squared Hellinger distance
        α = 1:   KL divergence (via L'Hôpital)
        α = 2:   log of chi-squared + 1
        α → ∞:   log(max_i p_i/q_i) (max-divergence)

    Properties:
    - Non-negative for α > 0
    - Monotonically non-decreasing in α
    - D_α(p || q) = 0 iff p = q (for α > 0)

    Args:
        p: First distribution
        q: Second distribution
        alpha: Order parameter (default: 1.0 = KL divergence)
        normalize: If True, normalize p and q to sum to 1

    Returns:
        Rényi divergence value
    """
    if is_tensor(p) and is_tensor(q):
        if p.shape != q.shape:
            raise ValueError("p and q must have same shape")
        if normalize:
            p = p / p.sum().clamp(min=_EPS)
            q = q / q.sum().clamp(min=_EPS)
        return _renyi_divergence_torch(p, q, alpha)

    if is_numpy(p) and is_numpy(q):
        if p.shape != q.shape:
            raise ValueError("p and q must have same shape")
        if normalize:
            p = p / np.clip(p.sum(), _EPS, None)
            q = q / np.clip(q.sum(), _EPS, None)
        return _renyi_divergence_numpy(p, q, alpha)

    p_list = list(p)
    q_list = list(q)
    if len(p_list) != len(q_list):
        raise ValueError("p and q must have same length")

    if normalize:
        p_sum = sum(p_list)
        q_sum = sum(q_list)
        if p_sum > _EPS:
            p_list = [x / p_sum for x in p_list]
        if q_sum > _EPS:
            q_list = [x / q_sum for x in q_list]

    return _renyi_divergence_native(p_list, q_list, alpha)
