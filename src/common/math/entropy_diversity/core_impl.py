"""Implementation functions for core entropy/diversity primitives.

Provides native/numpy/torch implementations of:
- log_sum_exp: numerically stable log-sum-exp
- probs_to_logprobs / logprobs_to_probs: conversion helpers
- rarity: inverse probability
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp as scipy_logsumexp

_EPS = 1e-12


# ── Conversion helpers ────────────────────────────────────────────────────────


def _probs_to_logprobs_native(probs: Sequence[float]) -> list[float]:
    """Convert probabilities to log-probabilities (pure Python)."""
    return [math.log(p) if p > _EPS else float("-inf") for p in probs]


def _probs_to_logprobs_numpy(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities to log-probabilities (NumPy)."""
    return np.log(np.clip(probs, _EPS, None))


def _probs_to_logprobs_torch(probs: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to log-probabilities (PyTorch)."""
    return torch.log(probs.clamp(min=_EPS))


def _logprobs_to_probs_native(logprobs: Sequence[float]) -> list[float]:
    """Convert log-probabilities to probabilities (pure Python)."""
    return [math.exp(lp) if math.isfinite(lp) else 0.0 for lp in logprobs]


def _logprobs_to_probs_numpy(logprobs: np.ndarray) -> np.ndarray:
    """Convert log-probabilities to probabilities (NumPy)."""
    return np.exp(logprobs)


def _logprobs_to_probs_torch(logprobs: torch.Tensor) -> torch.Tensor:
    """Convert log-probabilities to probabilities (PyTorch)."""
    return logprobs.exp()


# ── Log-sum-exp ───────────────────────────────────────────────────────────────


def _log_sum_exp_native(values: Sequence[float]) -> float:
    """Compute log(Σ exp(xᵢ)) in a numerically stable way (pure Python)."""
    if not values:
        return float("-inf")
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("-inf")
    max_val = max(finite)
    return max_val + math.log(sum(math.exp(v - max_val) for v in finite))


def _log_sum_exp_numpy(values: np.ndarray) -> np.floating:
    """Compute log(Σ exp(xᵢ)) using scipy.special.logsumexp."""
    return scipy_logsumexp(values)


def _log_sum_exp_torch(values: torch.Tensor) -> torch.Tensor:
    """Compute log(Σ exp(xᵢ)) using torch.logsumexp."""
    return torch.logsumexp(values, dim=-1)


# ── Pointwise measures ────────────────────────────────────────────────────────


def _rarity_native(logprob: float) -> float:
    """Rarity of an outcome (pure Python)."""
    return math.exp(-logprob)


def _rarity_numpy(logprob: np.floating) -> np.floating:
    """Rarity of an outcome (NumPy)."""
    return np.exp(-logprob)


def _rarity_torch(logprob: torch.Tensor) -> torch.Tensor:
    """Rarity of an outcome (PyTorch)."""
    return (-logprob).exp()
