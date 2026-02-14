"""Implementation functions for entropy calculations.

Provides native/numpy/torch implementations of:
- _renyi_entropy: Rényi entropy of order q
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp as scipy_logsumexp

from .core_impl import _EPS, _log_sum_exp_native


def _renyi_entropy_native(logprobs: Sequence[float], q: float) -> float:
    """Rényi entropy of order q (pure Python, takes logprobs)."""
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return float("inf")

    # q = 0: Hartley entropy = log(count of non-zero)
    if q == 0:
        return math.log(len(finite_lps))

    # q = 1: Shannon entropy H = -Σ pᵢ log pᵢ = -Σ exp(lp) * lp
    if abs(q - 1.0) < _EPS:
        return -sum(math.exp(lp) * lp for lp in finite_lps)

    # q → ∞: min-entropy = -log(max p) = -max(lp)
    if q == float("inf"):
        return -max(finite_lps)

    # q → -∞: max-entropy = -log(min p) = -min(lp)
    if q == float("-inf"):
        return -min(finite_lps)

    # General case: use log-sum-exp for stability
    # H_q = (1/(1-q)) * log(Σ pᵢ^q) = (1/(1-q)) * log(Σ exp(q * lp))
    log_sum = _log_sum_exp_native([q * lp for lp in finite_lps])
    return log_sum / (1.0 - q)


def _renyi_entropy_numpy(logprobs: np.ndarray, q: float) -> np.floating:
    """Rényi entropy of order q (NumPy, takes logprobs)."""
    finite_mask = np.isfinite(logprobs)
    if not finite_mask.any():
        return np.float64(float("inf"))

    finite_lps = logprobs[finite_mask]

    # q = 0: Hartley entropy = log(count of non-zero)
    if q == 0:
        return np.log(len(finite_lps))

    # q = 1: Shannon entropy H = -Σ pᵢ log pᵢ = -Σ exp(lp) * lp
    if abs(q - 1.0) < _EPS:
        probs = np.exp(finite_lps)
        return -(probs * finite_lps).sum()

    # q → ∞: min-entropy = -log(max p) = -max(lp)
    if q == float("inf"):
        return -finite_lps.max()

    # q → -∞: max-entropy = -log(min p) = -min(lp)
    if q == float("-inf"):
        return -finite_lps.min()

    # General case: use log-sum-exp for stability
    log_sum = scipy_logsumexp(q * finite_lps)
    return log_sum / (1.0 - q)


def _renyi_entropy_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Rényi entropy of order q (PyTorch, takes logprobs)."""
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(float("inf"), device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # q = 0: Hartley entropy = log(count of non-zero)
    if q == 0:
        return torch.log(
            torch.tensor(
                finite_lps.numel(), dtype=logprobs.dtype, device=logprobs.device
            )
        )

    # q = 1: Shannon entropy H = -Σ pᵢ log pᵢ = -Σ exp(lp) * lp
    if abs(q - 1.0) < _EPS:
        probs = finite_lps.exp()
        return -(probs * finite_lps).sum()

    # q → ∞: min-entropy = -log(max p) = -max(lp)
    if q == float("inf"):
        return -finite_lps.max()

    # q → -∞: max-entropy = -log(min p) = -min(lp)
    if q == float("-inf"):
        return -finite_lps.min()

    # General case: use log-sum-exp for stability
    log_sum = torch.logsumexp(q * finite_lps, dim=-1)
    return log_sum / (1.0 - q)
