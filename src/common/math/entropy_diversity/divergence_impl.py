"""Implementation functions for divergence calculations.

Provides native/numpy/torch implementations of:
- _kl_divergence: Kullback-Leibler divergence
- _renyi_divergence: Rényi divergence
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .core_impl import _EPS


# ══════════════════════════════════════════════════════════════════════════════
# KL DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════


def _kl_divergence_native(p: Sequence[float], q: Sequence[float]) -> float:
    """KL divergence D_KL(p || q) (pure Python).

    Assumes p and q are already normalized probability distributions.
    """
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > _EPS:
            if qi < _EPS:
                return float("inf")
            kl += pi * math.log(pi / qi)
    return kl


def _kl_divergence_numpy(p: np.ndarray, q: np.ndarray) -> np.floating:
    """KL divergence D_KL(p || q) (NumPy).

    Assumes p and q are already normalized probability distributions.
    """
    # Where p > 0 and q ≈ 0, divergence is infinite
    p_positive = p > _EPS
    q_zero = q < _EPS
    if (p_positive & q_zero).any():
        return np.float64(float("inf"))

    # Compute KL only where p > 0
    mask = p_positive
    if not mask.any():
        return np.float64(0.0)

    p_safe = p[mask]
    q_safe = np.clip(q[mask], _EPS, None)
    return (p_safe * np.log(p_safe / q_safe)).sum()


def _kl_divergence_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence D_KL(p || q) (PyTorch).

    Assumes p and q are already normalized probability distributions.
    """
    # Where p > 0 and q ≈ 0, divergence is infinite
    p_positive = p > _EPS
    q_zero = q < _EPS
    if (p_positive & q_zero).any():
        return torch.tensor(float("inf"), device=p.device)

    # Compute KL only where p > 0
    mask = p_positive
    if not mask.any():
        return torch.tensor(0.0, device=p.device)

    p_safe = p[mask]
    q_safe = q[mask].clamp(min=_EPS)
    return (p_safe * (p_safe / q_safe).log()).sum()


# ══════════════════════════════════════════════════════════════════════════════
# RÉNYI DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════


def _renyi_divergence_native(
    p: Sequence[float], q: Sequence[float], alpha: float
) -> float:
    """Rényi divergence of order α (pure Python).

    D_α(p || q) = (1/(α-1)) log Σ_i p_i^α q_i^(1-α)

    Assumes p and q are already normalized probability distributions.
    """
    # α = 1: KL divergence (via L'Hôpital's rule)
    if abs(alpha - 1.0) < _EPS:
        return _kl_divergence_native(p, q)

    # α = 0: -log Σ_i q_i for i where p_i > 0 (support coverage)
    if abs(alpha) < _EPS:
        support_q = sum(qi for pi, qi in zip(p, q) if pi > _EPS)
        return -math.log(support_q) if support_q > _EPS else float("inf")

    # α → ∞: max log ratio
    if alpha == float("inf"):
        max_ratio = 0.0
        for pi, qi in zip(p, q):
            if pi > _EPS:
                if qi < _EPS:
                    return float("inf")
                max_ratio = max(max_ratio, pi / qi)
        return math.log(max_ratio) if max_ratio > 0 else float("-inf")

    # General case: (1/(α-1)) log Σ p^α q^(1-α)
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > _EPS:
            if qi < _EPS:
                # p_i > 0 but q_i = 0: infinite divergence
                return float("inf")
            total += (pi**alpha) * (qi ** (1 - alpha))

    if total < _EPS:
        return float("inf")
    return math.log(total) / (alpha - 1)


def _renyi_divergence_numpy(
    p: np.ndarray, q: np.ndarray, alpha: float
) -> np.floating:
    """Rényi divergence of order α (NumPy).

    Assumes p and q are already normalized probability distributions.
    """
    # α = 1: KL divergence
    if abs(alpha - 1.0) < _EPS:
        return _kl_divergence_numpy(p, q)

    # α = 0: support coverage
    if abs(alpha) < _EPS:
        support_mask = p > _EPS
        support_q = q[support_mask].sum()
        if support_q < _EPS:
            return np.float64(float("inf"))
        return -np.log(support_q)

    # α → ∞: max log ratio
    if alpha == float("inf"):
        p_positive = p > _EPS
        q_zero = q < _EPS
        if (p_positive & q_zero).any():
            return np.float64(float("inf"))
        mask = p_positive
        if not mask.any():
            return np.float64(float("-inf"))
        ratio = p[mask] / np.clip(q[mask], _EPS, None)
        return np.log(ratio.max())

    # General case
    p_positive = p > _EPS
    q_zero = q < _EPS
    if (p_positive & q_zero).any():
        return np.float64(float("inf"))

    # Compute Σ p^α q^(1-α) only where p > 0
    mask = p_positive
    if not mask.any():
        return np.float64(float("inf"))

    p_safe = p[mask]
    q_safe = np.clip(q[mask], _EPS, None)
    total = (np.power(p_safe, alpha) * np.power(q_safe, 1 - alpha)).sum()

    if total < _EPS:
        return np.float64(float("inf"))
    return np.log(total) / (alpha - 1)


def _renyi_divergence_torch(
    p: torch.Tensor, q: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Rényi divergence of order α (PyTorch).

    Assumes p and q are already normalized probability distributions.
    """
    # α = 1: KL divergence
    if abs(alpha - 1.0) < _EPS:
        return _kl_divergence_torch(p, q)

    # α = 0: support coverage
    if abs(alpha) < _EPS:
        support_mask = p > _EPS
        support_q = q[support_mask].sum()
        if support_q < _EPS:
            return torch.tensor(float("inf"), device=p.device)
        return -support_q.log()

    # α → ∞: max log ratio
    if alpha == float("inf"):
        p_positive = p > _EPS
        q_zero = q < _EPS
        if (p_positive & q_zero).any():
            return torch.tensor(float("inf"), device=p.device)
        mask = p_positive
        if not mask.any():
            return torch.tensor(float("-inf"), device=p.device)
        ratio = p[mask] / q[mask].clamp(min=_EPS)
        return ratio.max().log()

    # General case
    p_positive = p > _EPS
    q_zero = q < _EPS
    if (p_positive & q_zero).any():
        return torch.tensor(float("inf"), device=p.device)

    # Compute Σ p^α q^(1-α) only where p > 0
    mask = p_positive
    if not mask.any():
        return torch.tensor(float("inf"), device=p.device)

    p_safe = p[mask]
    q_safe = q[mask].clamp(min=_EPS)
    total = (p_safe.pow(alpha) * q_safe.pow(1 - alpha)).sum()

    if total < _EPS:
        return torch.tensor(float("inf"), device=p.device)
    return total.log() / (alpha - 1)
