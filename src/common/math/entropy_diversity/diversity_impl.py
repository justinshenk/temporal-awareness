"""Implementation functions for diversity calculations.

Provides native/numpy/torch implementations of:
- _q_diversity: Hill number D_q
- _q_concentration: 1/D_q
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .core_impl import _EPS
from .entropy_impl import (
    _renyi_entropy_native,
    _renyi_entropy_numpy,
    _renyi_entropy_torch,
)


# ── q-Diversity (Hill numbers) ────────────────────────────────────────────────


def _q_diversity_native(logprobs: Sequence[float], q: float) -> float:
    """Hill number D_q (pure Python)."""
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return 0.0

    # q = 0: richness = count of non-zero
    if q == 0:
        return float(len(finite_lps))

    # D_q = exp(H_q) for all other cases
    H_q = _renyi_entropy_native(finite_lps, q)
    return math.exp(H_q) if math.isfinite(H_q) else float("inf")


def _q_diversity_numpy(logprobs: np.ndarray, q: float) -> np.floating:
    """Hill number D_q (NumPy)."""
    finite_mask = np.isfinite(logprobs)
    if not finite_mask.any():
        return np.float64(0.0)

    finite_lps = logprobs[finite_mask]

    # q = 0: richness = count of non-zero
    if q == 0:
        return np.float64(len(finite_lps))

    # D_q = exp(H_q) for all other cases
    H_q = _renyi_entropy_numpy(finite_lps, q)
    return np.exp(H_q)


def _q_diversity_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Hill number D_q (PyTorch)."""
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(0.0, device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # q = 0: richness = count of non-zero
    if q == 0:
        return torch.tensor(
            finite_lps.numel(), dtype=logprobs.dtype, device=logprobs.device
        )

    # D_q = exp(H_q) for all other cases
    H_q = _renyi_entropy_torch(finite_lps, q)
    return H_q.exp()


# ── q-Concentration (1/D_q) ───────────────────────────────────────────────────


def _q_concentration_native(logprobs: Sequence[float], q: float) -> float:
    """Concentration of order q (pure Python)."""
    d = _q_diversity_native(logprobs, q)
    return 1.0 / d if d > _EPS else float("inf")


def _q_concentration_numpy(logprobs: np.ndarray, q: float) -> np.floating:
    """Concentration of order q (NumPy)."""
    d = _q_diversity_numpy(logprobs, q)
    return 1.0 / d if d > _EPS else np.float64(float("inf"))


def _q_concentration_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Concentration of order q (PyTorch)."""
    d = _q_diversity_torch(logprobs, q)
    return 1.0 / d
