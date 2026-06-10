"""Geometry of a predicted residual shift vs the true shift.

Why does adding the fitted ridge map ``δ_pred = a·Wᵀ`` (steering) recover nothing where adding the
true ``δ_true = lora_resid − base_resid`` (the lockstep oracle) recovers the capability? These
metrics quantify, over acting-site tokens, how well the predicted shift matches the true one:

  * ``mean_cosine`` — directional alignment (does the predicted shift even point the right way?);
  * ``median_norm_ratio`` — magnitude ``‖δ_pred‖ / ‖δ_true‖``;
  * ``r2`` — multi-output variance of the true shift explained by the prediction.

Near-zero cosine / R² is the estimator failure made concrete: the best linear map cannot reach the
true shift, so adding ``W·a`` never moves the base model onto the LoRA's manifold.
"""

from __future__ import annotations

import numpy as np


def _rows(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (n_tokens, dim) array, got shape {a.shape}")
    return a


def mean_row_cosine(pred, true, eps: float = 1e-12) -> float:
    """Mean per-token cosine between predicted and true shift (0 where either row is ~zero)."""
    P, T = _rows(pred), _rows(true)
    pn, tn = np.linalg.norm(P, axis=1), np.linalg.norm(T, axis=1)
    cos = (P * T).sum(1) / np.maximum(pn * tn, eps)
    cos[(pn < eps) | (tn < eps)] = 0.0
    return float(cos.mean())


def median_norm_ratio(pred, true, eps: float = 1e-12) -> float:
    """Median ``‖δ_pred‖ / ‖δ_true‖`` over tokens with non-degenerate true shift."""
    P, T = _rows(pred), _rows(true)
    pn, tn = np.linalg.norm(P, axis=1), np.linalg.norm(T, axis=1)
    mask = tn > eps
    return float(np.median(pn[mask] / tn[mask])) if mask.any() else 0.0


def r2_multioutput(pred, true) -> float:
    """1 − SS_res/SS_tot over all coordinates (variance of the true shift explained)."""
    P, T = _rows(pred), _rows(true)
    ss_res = float(((T - P) ** 2).sum())
    ss_tot = float(((T - T.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def shift_geometry(pred, true) -> dict:
    """Bundle the three metrics plus the token count."""
    return {
        "mean_cosine": mean_row_cosine(pred, true),
        "median_norm_ratio": median_norm_ratio(pred, true),
        "r2": r2_multioutput(pred, true),
        "n_tokens": int(_rows(true).shape[0]),
    }
