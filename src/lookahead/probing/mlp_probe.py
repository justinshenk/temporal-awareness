"""MLP probe variant (H&L robustness check, Block 3A).

Hewitt & Liang (2019) warned that higher-capacity probes can fit
control-task labels they "shouldn't" be able to. We re-run the
staircase with a small 2-layer MLP to confirm that:

  (a) on code: the null result holds (MLP probe does NOT beat the
      max-earlier baseline either) — the diagnostic is robust to
      probe class.
  (b) on rhyme/qa-neutral: the positive gap holds (MLP probe also
      shows that target-position info isn't recoverable from earlier
      positions) — planning evidence is not linearity-dependent.

Interface is sklearn-compatible so it slots into the existing
`train_commitment_probes()` function with a one-line swap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPProbe(BaseEstimator, ClassifierMixin):
    """Small MLP with sklearn-style .fit() / .predict() / .predict_proba().

    Inherits from BaseEstimator + ClassifierMixin for full sklearn compatibility
    (required for cross_val_score, __sklearn_tags__, get_params, set_params).

    Designed to be a controlled comparison to LogisticRegression. Not a
    deep network — just enough capacity to test nonlinear separability
    without overfitting to noise.

    Defaults match common probing-literature choices:
        hidden_dim = max(64, d_input / 2)
        2 layers (input → hidden → output)
        ReLU activation
        Dropout 0.1
        Adam, lr=1e-3, weight_decay=1e-4
        Up to 200 epochs, early-stopping on validation accuracy
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        max_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.model_: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # ----- sklearn-style API -----

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        h = self.hidden_dim if self.hidden_dim is not None else max(64, n_features // 2)

        self.model_ = nn.Sequential(
            nn.Linear(n_features, h),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(h, n_classes),
        ).to(self.device)

        # Small held-out split for early stopping (10% of fit data)
        n = X.shape[0]
        idx_perm = np.random.RandomState(self.random_state).permutation(n)
        n_val = max(5, n // 10)
        val_idx = idx_perm[:n_val]
        tr_idx = idx_perm[n_val:]
        X_tr, y_tr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y_idx[tr_idx]).long()
        X_va, y_va = torch.from_numpy(X[val_idx]), torch.from_numpy(y_idx[val_idx]).long()
        X_tr, y_tr = X_tr.to(self.device), y_tr.to(self.device)
        X_va, y_va = X_va.to(self.device), y_va.to(self.device)

        optim = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_acc = -1.0
        best_state = None
        patience = 0

        for epoch in range(self.max_epochs):
            # Mini-batch training
            self.model_.train()
            perm = torch.randperm(X_tr.shape[0], device=self.device)
            for s in range(0, X_tr.shape[0], self.batch_size):
                bi = perm[s : s + self.batch_size]
                logits = self.model_(X_tr[bi])
                loss = F.cross_entropy(logits, y_tr[bi])
                optim.zero_grad()
                loss.backward()
                optim.step()

            # Validation
            self.model_.eval()
            with torch.no_grad():
                preds = self.model_(X_va).argmax(dim=-1)
                val_acc = (preds == y_va).float().mean().item()

            if val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPProbe.fit() must be called before predict_proba().")
        self.model_.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            logits = self.model_(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())


def make_probe(probe_type: str = "linear", **kwargs):
    """Factory: returns either a LogisticRegression or an MLPProbe.

    Pass to `train_commitment_probes()` via the `probe_factory` hook.
    """
    if probe_type == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 2000),
            solver=kwargs.get("solver", "lbfgs"),
            random_state=kwargs.get("random_state", 42),
        )
    elif probe_type == "mlp":
        return MLPProbe(
            hidden_dim=kwargs.get("hidden_dim"),
            max_epochs=kwargs.get("max_epochs", 200),
            lr=kwargs.get("lr", 1e-3),
            random_state=kwargs.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown probe_type {probe_type!r}; use 'linear' or 'mlp'.")


__all__ = ["MLPProbe", "make_probe"]
