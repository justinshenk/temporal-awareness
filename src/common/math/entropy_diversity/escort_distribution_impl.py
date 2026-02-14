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
    """
    if not logprobs:
        return []

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
    """Escort distribution in log space (NumPy)."""
    if logprobs.size == 0:
        return logprobs

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
    """Escort distribution in log space (PyTorch)."""
    if logprobs.numel() == 0:
        return logprobs

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
