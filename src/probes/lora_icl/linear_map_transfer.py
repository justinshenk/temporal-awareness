"""Cross-model linear activation maps for LoRA capability transfer.

Fits per-layer ridge maps from donor-model residual space to recipient-model residual space on
paired final-token states of a shared prompt corpus, and applies them to LoRA *shift* vectors.
A shift is a difference of states, so ``map_shift`` is the pure linear part (the fitted means
cancel); ``map_state`` re-attaches them for mapping absolute states. Held-out R² is reported so
a downstream transfer null is attributable (bad map vs. no shared structure).

All algebra is model-free and unit-tested on CPU; the model forwards live in
``scripts/lora_icl/run_lora_map_transfer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearMap:
    """A fitted donor→recipient map: ``y ≈ (x − mean_x) @ weights + mean_y``."""

    weights: np.ndarray   # (d_donor, d_recipient)
    mean_x: np.ndarray
    mean_y: np.ndarray
    r2_holdout: float

    def map_shift(self, delta: np.ndarray) -> np.ndarray:
        """Map a difference-of-states vector; means cancel out of a difference."""
        return np.asarray(delta, dtype=np.float64) @ self.weights

    def map_state(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_x) @ self.weights + self.mean_y

    def to_arrays(self) -> dict:
        return {"weights": self.weights, "mean_x": self.mean_x, "mean_y": self.mean_y,
                "r2_holdout": self.r2_holdout}

    @classmethod
    def from_arrays(cls, weights, mean_x, mean_y, r2_holdout) -> "LinearMap":
        return cls(np.asarray(weights), np.asarray(mean_x), np.asarray(mean_y),
                   float(r2_holdout))


def fit_linear_map(source: np.ndarray, target: np.ndarray, lam: float,
                   holdout_frac: float = 0.2, seed: int = 42) -> LinearMap:
    """Ridge-fit ``target ≈ source @ W`` on centered data, scored on a held-out split.

    ``lam`` is the ridge penalty; the held-out R² is
    ``1 − ‖Y − Ŷ‖²_F / ‖Y − Ȳ‖²_F`` over the held-out rows, with the means taken from the
    training rows only (the held-out rows never touch the fit).
    """
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"paired row matrices required, got {x.shape} and {y.shape}")
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 paired rows, got {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, int(round(n * holdout_frac)))
    hold, train = perm[:n_hold], perm[n_hold:]

    mean_x = x[train].mean(0)
    mean_y = y[train].mean(0)
    xc = x[train] - mean_x
    yc = y[train] - mean_y
    d = xc.shape[1]
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(d), xc.T @ yc)

    pred = (x[hold] - mean_x) @ w + mean_y
    ss_res = float(((y[hold] - pred) ** 2).sum())
    ss_tot = float(((y[hold] - mean_y) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return LinearMap(weights=w, mean_x=mean_x, mean_y=mean_y, r2_holdout=r2)


def norm_matched_random(vector: np.ndarray, seed: int) -> np.ndarray:
    """A random direction carrying exactly the input vector's norm (the dose control)."""
    v = np.asarray(vector, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r = rng.normal(size=v.shape)
    return r * (np.linalg.norm(v) / np.linalg.norm(r))
