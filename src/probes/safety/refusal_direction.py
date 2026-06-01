"""Refusal direction (Arditi et al. 2024) and signed projection of shifts onto it.

The refusal direction at a layer is the difference of mean last-token residuals
between harmful and harmless prompts on the base model:

    r = mean(resid | harmful) - mean(resid | harmless)

Adding +r pushes toward refusal; a shift with a *negative* projection onto the
unit refusal direction has moved the model toward compliance (less refusal).
"""

from __future__ import annotations

import numpy as np


def refusal_direction(harmful_resids: np.ndarray, harmless_resids: np.ndarray) -> np.ndarray:
    """Unit refusal direction from harmful/harmless last-token residual sets.

    Args:
        harmful_resids: ``(n_h, d)`` last-token residuals on harmful prompts.
        harmless_resids: ``(n_s, d)`` last-token residuals on harmless prompts.

    Returns:
        ``(d,)`` unit vector pointing from harmless toward harmful (= +refusal).
    """
    h = np.asarray(harmful_resids, dtype=np.float64)
    s = np.asarray(harmless_resids, dtype=np.float64)
    if h.ndim != 2 or s.ndim != 2:
        raise ValueError("resid sets must be 2-D (n, d)")
    if h.shape[1] != s.shape[1]:
        raise ValueError(f"dim mismatch: harmful {h.shape[1]} vs harmless {s.shape[1]}")
    diff = h.mean(axis=0) - s.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm == 0.0:
        raise ValueError("refusal direction is zero (harmful and harmless means coincide)")
    return diff / norm


def project_onto(shifts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Signed projection of each shift vector onto ``direction``.

    Args:
        shifts: ``(n, d)`` shift set.
        direction: ``(d,)`` vector (need not be unit; it is unit-normalized here).

    Returns:
        ``(n,)`` signed projections (negative ⇒ moved toward compliance / away from refusal).
    """
    X = np.asarray(shifts, dtype=np.float64)
    r = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(r)
    if norm == 0.0:
        raise ValueError("cannot project onto a zero direction")
    return X @ (r / norm)
