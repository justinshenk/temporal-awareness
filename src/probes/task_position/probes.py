"""Probe training and inference for task-position targets."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


class RidgeProbe:
    """Ridge regression probe on residual-stream activations.

    Wraps sklearn's Ridge with save/load, a direction accessor, and R² scoring.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("probe not fit; call .fit(X, y) first")
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))

    def direction(self) -> np.ndarray:
        """Return the learned linear direction (coefficient vector)."""
        if self._model is None:
            raise RuntimeError("probe not fit; call .fit(X, y) first")
        return self._model.coef_.copy()

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"alpha": self.alpha, "model": self._model}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        with open(path, "rb") as f:
            data = pickle.load(f)
        probe = cls(alpha=data["alpha"])
        probe._model = data["model"]
        return probe
