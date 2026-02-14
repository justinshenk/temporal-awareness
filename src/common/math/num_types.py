"""Numeric type aliases for math module.

Provides union types for functions that accept both Python scalars/sequences,
NumPy arrays, and PyTorch tensors, with automatic dispatch to optimized implementations.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

# ── Scalar types ────────────────────────────────────────────────────────────────

Num = Union[float, np.floating, torch.Tensor]
"""A single numeric value: float | np.floating | torch.Tensor (0-d or scalar)."""

# ── Sequence types ──────────────────────────────────────────────────────────────

Nums = Union[Sequence[float], np.ndarray, torch.Tensor]
"""A sequence of numeric values: Sequence[float] | np.ndarray | torch.Tensor (1-d)."""

# ── Type guards ─────────────────────────────────────────────────────────────────


def is_tensor(x: Num | Nums) -> bool:
    """Check if x is a torch.Tensor."""
    return isinstance(x, torch.Tensor)


def is_numpy(x: Num | Nums) -> bool:
    """Check if x is a numpy array or numpy scalar."""
    return isinstance(x, (np.ndarray, np.floating))
