"""Implementation functions for escort distribution calculations.

Provides native/numpy/torch implementations of:
- _escort_logprobs: escort distribution in log space
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp as scipy_logsumexp

from .core_impl import _EPS, _log_sum_exp_native


def _escort_logprobs_native(logprobs: Sequence[float], q: float) -> list[float]:
    """Escort distribution in log space (pure Python).

    log π_i^(q) = q·lp_i - logsumexp(q·lp)

    Special cases:
        q → -∞: all mass on argmin (antimode/rarest)
        q = 0:  uniform over support
        q = 1:  original distribution
        q → +∞: all mass on argmax (mode)
    """
    if not logprobs:
        return []

    # q → +∞: all mass on argmax (the mode)
    if q == float("inf"):
        finite_lps = [(i, lp) for i, lp in enumerate(logprobs) if math.isfinite(lp)]
        if not finite_lps:
            return [float("-inf")] * len(logprobs)
        max_lp = max(lp for _, lp in finite_lps)
        max_indices = [i for i, lp in finite_lps if lp == max_lp]
        log_uniform = -math.log(len(max_indices))
        return [log_uniform if i in max_indices else float("-inf") for i in range(len(logprobs))]

    # q → -∞: all mass on argmin (the antimode/rarest)
    if q == float("-inf"):
        finite_lps = [(i, lp) for i, lp in enumerate(logprobs) if math.isfinite(lp)]
        if not finite_lps:
            return [float("-inf")] * len(logprobs)
        min_lp = min(lp for _, lp in finite_lps)
        min_indices = [i for i, lp in finite_lps if lp == min_lp]
        log_uniform = -math.log(len(min_indices))
        return [log_uniform if i in min_indices else float("-inf") for i in range(len(logprobs))]

    # q=0: uniform over support
    if q == 0:
        finite_count = sum(1 for lp in logprobs if math.isfinite(lp))
        if finite_count == 0:
            return [float("-inf")] * len(logprobs)
        log_uniform = -math.log(finite_count)
        return [log_uniform if math.isfinite(lp) else float("-inf") for lp in logprobs]

    # q=1: identity (original distribution)
    if abs(q - 1.0) < _EPS:
        return list(logprobs)

    # General case
    scaled = [q * lp for lp in logprobs]
    log_norm = _log_sum_exp_native(scaled)
    return [s - log_norm for s in scaled]


def _escort_logprobs_numpy(logprobs: np.ndarray, q: float) -> np.ndarray:
    """Escort distribution in log space (NumPy).

    Special cases:
        q → -∞: all mass on argmin (antimode/rarest)
        q = 0:  uniform over support
        q = 1:  original distribution
        q → +∞: all mass on argmax (mode)
    """
    if logprobs.size == 0:
        return logprobs

    # q → +∞: all mass on argmax (the mode)
    if q == float("inf"):
        finite_mask = np.isfinite(logprobs)
        if not finite_mask.any():
            return np.full_like(logprobs, float("-inf"))
        finite_lps = np.where(finite_mask, logprobs, float("-inf"))
        max_lp = finite_lps.max()
        is_max = finite_mask & (logprobs == max_lp)
        n_max = is_max.sum()
        result = np.full_like(logprobs, float("-inf"))
        result[is_max] = -np.log(n_max)
        return result

    # q → -∞: all mass on argmin (the antimode/rarest)
    if q == float("-inf"):
        finite_mask = np.isfinite(logprobs)
        if not finite_mask.any():
            return np.full_like(logprobs, float("-inf"))
        finite_lps = np.where(finite_mask, logprobs, float("inf"))
        min_lp = finite_lps.min()
        is_min = finite_mask & (logprobs == min_lp)
        n_min = is_min.sum()
        result = np.full_like(logprobs, float("-inf"))
        result[is_min] = -np.log(n_min)
        return result

    # q=0: uniform over support
    if q == 0:
        finite_mask = np.isfinite(logprobs)
        finite_count = finite_mask.sum()
        if finite_count == 0:
            return np.full_like(logprobs, float("-inf"))
        log_uniform = -np.log(finite_count)
        result = np.full_like(logprobs, float("-inf"))
        result[finite_mask] = log_uniform
        return result

    # q=1: identity (original distribution)
    if abs(q - 1.0) < _EPS:
        return logprobs.copy()

    # General case
    scaled = q * logprobs
    log_norm = scipy_logsumexp(scaled)
    return scaled - log_norm


def _escort_logprobs_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Escort distribution in log space (PyTorch).

    Special cases:
        q → -∞: all mass on argmin (antimode/rarest)
        q = 0:  uniform over support
        q = 1:  original distribution
        q → +∞: all mass on argmax (mode)
    """
    if logprobs.numel() == 0:
        return logprobs

    # q → +∞: all mass on argmax (the mode)
    if q == float("inf"):
        finite_mask = torch.isfinite(logprobs)
        if not finite_mask.any():
            return torch.full_like(logprobs, float("-inf"))
        finite_lps = torch.where(finite_mask, logprobs, torch.tensor(float("-inf")))
        max_lp = finite_lps.max()
        is_max = finite_mask & (logprobs == max_lp)
        n_max = is_max.sum()
        result = torch.full_like(logprobs, float("-inf"))
        result[is_max] = -torch.log(n_max.float())
        return result

    # q → -∞: all mass on argmin (the antimode/rarest)
    if q == float("-inf"):
        finite_mask = torch.isfinite(logprobs)
        if not finite_mask.any():
            return torch.full_like(logprobs, float("-inf"))
        finite_lps = torch.where(finite_mask, logprobs, torch.tensor(float("inf")))
        min_lp = finite_lps.min()
        is_min = finite_mask & (logprobs == min_lp)
        n_min = is_min.sum()
        result = torch.full_like(logprobs, float("-inf"))
        result[is_min] = -torch.log(n_min.float())
        return result

    # q=0: uniform over support
    if q == 0:
        finite_mask = torch.isfinite(logprobs)
        finite_count = finite_mask.sum()
        if finite_count == 0:
            return torch.full_like(logprobs, float("-inf"))
        log_uniform = -torch.log(finite_count.float())
        result = torch.full_like(logprobs, float("-inf"))
        result[finite_mask] = log_uniform
        return result

    # q=1: identity (original distribution)
    if abs(q - 1.0) < _EPS:
        return logprobs.clone()

    # General case
    scaled = q * logprobs
    log_norm = torch.logsumexp(scaled, dim=-1, keepdim=True)
    return scaled - log_norm
