"""CAST-style conditional steering gate: a logistic probe on base activations.

The contrast with the ridge map is the whole point of the frontier. A fixed steering
vector (CAA / Arditi) shifts *every* input toward refusal, so raising harmful-refusal
also raises benign over-refusal. CAST (Conditional Activation Steering, Lee et al. 2024)
adds an explicit gate: a linear probe on the prompt's activation decides whether to steer
at all, so benign prompts are (ideally) left untouched.

The ridge map ``W·a`` claims to get that conditionality *for free* — ``W·a ≈ 0`` on benign
by construction — with no separate classifier. This gate is the honest comparator: a plain
logistic regression on the base last-token residual at one gate layer, ``True`` ⇒ steer.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticGate:
    """Logistic probe over base last-token residuals: predict harmful (⇒ steer) vs benign.

    ``fit`` standardizes features then fits L2 logistic regression on stacked
    harmful (label 1) and benign (label 0) residuals. ``predict`` returns a boolean
    steer-mask (``True`` ⇒ the prompt is harmful, apply the steering vector).
    """

    def __init__(self, C: float = 1.0, seed: int = 0):
        self._clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        self._scaler = StandardScaler()
        self._fitted = False
        self.train_accuracy: float | None = None

    def fit(self, harmful_resids: np.ndarray, benign_resids: np.ndarray) -> "LogisticGate":
        h = np.asarray(harmful_resids, dtype=np.float64)
        b = np.asarray(benign_resids, dtype=np.float64)
        if h.ndim != 2 or b.ndim != 2:
            raise ValueError("residual sets must be 2-D (n, d)")
        if h.shape[0] == 0 or b.shape[0] == 0:
            raise ValueError("need both harmful and benign examples to fit the gate")
        if h.shape[1] != b.shape[1]:
            raise ValueError(f"dim mismatch: harmful {h.shape[1]} vs benign {b.shape[1]}")
        X = np.vstack([h, b])
        y = np.concatenate([np.ones(len(h)), np.zeros(len(b))])
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, y)
        self._fitted = True
        self.train_accuracy = float(self._clf.score(Xs, y))
        return self

    def predict(self, resids: np.ndarray) -> np.ndarray:
        """Boolean steer-mask: ``True`` where the probe predicts harmful."""
        if not self._fitted:
            raise RuntimeError("gate is not fitted; call fit() first")
        X = np.asarray(resids, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("residuals must be 2-D (n, d)")
        return self._clf.predict(self._scaler.transform(X)).astype(bool)
