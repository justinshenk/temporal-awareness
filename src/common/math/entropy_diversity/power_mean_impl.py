"""Implementation functions for power mean calculations.

Provides native/numpy/torch implementations of:
- _power_mean: generalized mean of order α
- _weighted_power_mean: generalized mean with custom weights
- _power_mean_from_logprobs: power mean of probabilities
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp as scipy_logsumexp

from .core_impl import _EPS, _log_sum_exp_native


# ── Power mean (raw values) ───────────────────────────────────────────────────


def _power_mean_native(values: Sequence[float], alpha: float) -> float:
    """Generalized (power) mean of order α (pure Python).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if not values:
        return 0.0

    n = len(values)

    # Limiting cases
    if alpha == float("-inf"):
        return min(values)
    if alpha == float("inf"):
        return max(values)

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        active = list(values)
    else:
        active = [v for v in values if v > _EPS]
    if not active:
        return 0.0

    if abs(alpha) < _EPS:
        # Geometric mean: exp(mean(log(x)))
        return math.exp(sum(math.log(v) for v in active) / n)

    # General case
    powered = sum(v**alpha for v in active)
    return (powered / n) ** (1.0 / alpha)


def _power_mean_numpy(values: np.ndarray, alpha: float) -> np.floating:
    """Generalized (power) mean of order α (NumPy).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if values.size == 0:
        return np.float64(0.0)

    n = values.size

    # Limiting cases
    if alpha == float("-inf"):
        return values.min()
    if alpha == float("inf"):
        return values.max()

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        active = values
    else:
        active = values[values > _EPS]
    if active.size == 0:
        return np.float64(0.0)

    if abs(alpha) < _EPS:
        # Geometric mean: exp(mean(log(x)))
        return np.exp(np.log(active).sum() / n)

    # General case
    powered = np.power(active, alpha).sum()
    return np.power(powered / n, 1.0 / alpha)


def _power_mean_torch(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Generalized (power) mean of order α (PyTorch).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)

    n = values.numel()

    # Limiting cases
    if alpha == float("-inf"):
        return values.min()
    if alpha == float("inf"):
        return values.max()

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        active = values
    else:
        active = values[values > _EPS]
    if active.numel() == 0:
        return torch.tensor(0.0, device=values.device)

    if abs(alpha) < _EPS:
        # Geometric mean: exp(mean(log(x)))
        return (active.log().sum() / n).exp()

    # General case
    powered = (active**alpha).sum()
    return (powered / n) ** (1.0 / alpha)


# ── Weighted power mean ──────────────────────────────────────────────────────


def _weighted_power_mean_native(
    values: Sequence[float], weights: Sequence[float], alpha: float
) -> float:
    """Weighted power mean of order α (pure Python).

    M_α^w(x) = (Σ wᵢ xᵢ^α)^(1/α)

    Assumes weights are already normalized (sum to 1).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")
    if not values:
        return 0.0

    # For min/max, include all values with positive weight (including 0s)
    if alpha == float("-inf"):
        weighted_values = [v for v, w in zip(values, weights) if w > _EPS]
        if not weighted_values:
            return 0.0
        return min(weighted_values)
    if alpha == float("inf"):
        weighted_values = [v for v, w in zip(values, weights) if w > _EPS]
        if not weighted_values:
            return 0.0
        return max(weighted_values)

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        active = [(v, w) for v, w in zip(values, weights) if w > _EPS]
    else:
        active = [(v, w) for v, w in zip(values, weights) if w > _EPS and v > _EPS]
    if not active:
        return 0.0

    # Renormalize weights after filtering
    total_w = sum(w for _, w in active)
    if total_w < _EPS:
        return 0.0
    active = [(v, w / total_w) for v, w in active]

    if abs(alpha) < _EPS:
        # Weighted geometric mean: exp(Σ wᵢ log(xᵢ))
        log_sum = sum(w * math.log(v) for v, w in active)
        return math.exp(log_sum)

    # General case: (Σ wᵢ xᵢ^α)^(1/α)
    powered = sum(w * (v**alpha) for v, w in active)
    return powered ** (1.0 / alpha) if powered > _EPS else 0.0


def _weighted_power_mean_numpy(
    values: np.ndarray, weights: np.ndarray, alpha: float
) -> np.floating:
    """Weighted power mean of order α (NumPy).

    Assumes weights are already normalized (sum to 1).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if values.shape != weights.shape:
        raise ValueError("values and weights must have same shape")
    if values.size == 0:
        return np.float64(0.0)

    # For min/max, include all values with positive weight (including 0s)
    if alpha == float("-inf"):
        weighted_mask = weights > _EPS
        if not weighted_mask.any():
            return np.float64(0.0)
        return values[weighted_mask].min()
    if alpha == float("inf"):
        weighted_mask = weights > _EPS
        if not weighted_mask.any():
            return np.float64(0.0)
        return values[weighted_mask].max()

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        mask = weights > _EPS
    else:
        mask = (weights > _EPS) & (values > _EPS)
    if not mask.any():
        return np.float64(0.0)

    active_v = values[mask]
    active_w = weights[mask]

    # Renormalize weights after filtering
    total_w = active_w.sum()
    if total_w < _EPS:
        return np.float64(0.0)
    active_w = active_w / total_w

    if abs(alpha) < _EPS:
        # Weighted geometric mean
        return np.exp((active_w * np.log(active_v)).sum())

    # General case
    powered = (active_w * np.power(active_v, alpha)).sum()
    return np.power(powered, 1.0 / alpha) if powered > _EPS else np.float64(0.0)


def _weighted_power_mean_torch(
    values: torch.Tensor, weights: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Weighted power mean of order α (PyTorch).

    Assumes weights are already normalized (sum to 1).

    For min/max (α = ±∞), values of 0 are included.
    For α > 0, values of 0 are included (0^α = 0 is valid).
    For α ≤ 0, values of 0 are excluded to avoid log(0) or division by 0.
    """
    if values.shape != weights.shape:
        raise ValueError("values and weights must have same shape")
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)

    # For min/max, include all values with positive weight (including 0s)
    if alpha == float("-inf"):
        weighted_mask = weights > _EPS
        if not weighted_mask.any():
            return torch.tensor(0.0, device=values.device)
        return values[weighted_mask].min()
    if alpha == float("inf"):
        weighted_mask = weights > _EPS
        if not weighted_mask.any():
            return torch.tensor(0.0, device=values.device)
        return values[weighted_mask].max()

    # For α > 0, include zero values (0^α = 0 is valid)
    # For α ≤ 0, exclude zero values to avoid log(0) or division by 0
    if alpha > _EPS:
        mask = weights > _EPS
    else:
        mask = (weights > _EPS) & (values > _EPS)
    if not mask.any():
        return torch.tensor(0.0, device=values.device)

    active_v = values[mask]
    active_w = weights[mask]

    # Renormalize weights after filtering
    total_w = active_w.sum()
    if total_w < _EPS:
        return torch.tensor(0.0, device=values.device)
    active_w = active_w / total_w

    if abs(alpha) < _EPS:
        # Weighted geometric mean
        return (active_w * active_v.log()).sum().exp()

    # General case
    powered = (active_w * active_v.pow(alpha)).sum()
    return (
        powered.pow(1.0 / alpha)
        if powered > _EPS
        else torch.tensor(0.0, device=values.device)
    )


# ── Power mean from logprobs (for trajectories) ───────────────────────────────


def _power_mean_from_logprobs_native(logprobs: Sequence[float], alpha: float) -> float:
    """Power mean M_α of probabilities, computed from logprobs (pure Python)."""
    if not logprobs:
        return 0.0

    n = len(logprobs)
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return 0.0

    # Limiting cases
    if alpha == float("-inf"):
        return math.exp(min(finite_lps))  # min(p)
    if alpha == float("inf"):
        return math.exp(max(finite_lps))  # max(p)

    # α = 0: geometric mean = exp(mean(logprobs))
    if abs(alpha) < _EPS:
        return math.exp(sum(finite_lps) / n)

    # General case: M_α = (Σ pᵢ^α / n)^(1/α)
    # = exp((1/α) · (logsumexp(α·lp) - log(n)))
    log_sum = _log_sum_exp_native([alpha * lp for lp in finite_lps])
    return math.exp((log_sum - math.log(n)) / alpha)


def _power_mean_from_logprobs_numpy(logprobs: np.ndarray, alpha: float) -> np.floating:
    """Power mean M_α of probabilities, computed from logprobs (NumPy)."""
    if logprobs.size == 0:
        return np.float64(0.0)

    n = logprobs.size
    finite_mask = np.isfinite(logprobs)
    if not finite_mask.any():
        return np.float64(0.0)

    finite_lps = logprobs[finite_mask]

    # Limiting cases
    if alpha == float("-inf"):
        return np.exp(finite_lps.min())  # min(p)
    if alpha == float("inf"):
        return np.exp(finite_lps.max())  # max(p)

    # α = 0: geometric mean = exp(mean(logprobs))
    if abs(alpha) < _EPS:
        return np.exp(finite_lps.sum() / n)

    # General case: M_α = (Σ pᵢ^α / n)^(1/α)
    log_sum = scipy_logsumexp(alpha * finite_lps)
    return np.exp((log_sum - np.log(n)) / alpha)


def _power_mean_from_logprobs_torch(
    logprobs: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Power mean M_α of probabilities, computed from logprobs (PyTorch)."""
    if logprobs.numel() == 0:
        return torch.tensor(0.0, device=logprobs.device)

    n = logprobs.numel()
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(0.0, device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # Limiting cases
    if alpha == float("-inf"):
        return finite_lps.min().exp()  # min(p)
    if alpha == float("inf"):
        return finite_lps.max().exp()  # max(p)

    # α = 0: geometric mean = exp(mean(logprobs))
    if abs(alpha) < _EPS:
        return (finite_lps.sum() / n).exp()

    # General case: M_α = (Σ pᵢ^α / n)^(1/α)
    log_sum = torch.logsumexp(alpha * finite_lps, dim=-1)
    return ((log_sum - math.log(n)) / alpha).exp()
