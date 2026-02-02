"""Linear probe for binary classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class ProbeResult:
    """Result from training a linear probe."""

    layer: int
    token_position: int  # Index in token_positions config
    cv_accuracy_mean: float
    cv_accuracy_std: float
    train_accuracy: float
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0


class LinearProbe:
    """Linear probe for binary classification on activations.

    Uses L2-regularized logistic regression. The probe's weight vector
    (normalized) can be used as a steering vector for intervention.
    """

    def __init__(
        self,
        regularization_C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """Initialize probe.

        Args:
            regularization_C: Inverse of regularization strength (higher = less reg)
            max_iter: Maximum iterations for solver
            random_state: Random seed for reproducibility
        """
        self.regularization_C = regularization_C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None
        self._is_fitted = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_cv_folds: int = 5,
    ) -> tuple[float, float, float]:
        """Train the probe with cross-validation.

        Args:
            X: Activation features, shape (n_samples, n_features)
            y: Binary labels, shape (n_samples,)
            n_cv_folds: Number of cross-validation folds

        Returns:
            Tuple of (cv_accuracy_mean, cv_accuracy_std, train_accuracy)
        """
        self.model = LogisticRegression(
            C=self.regularization_C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="liblinear",  # Faster for small datasets
        )

        # Cross-validation (skip if n_cv_folds <= 1)
        if n_cv_folds > 1:
            cv_scores = cross_val_score(
                self.model, X, y, cv=n_cv_folds, scoring="accuracy", n_jobs=1
            )
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        else:
            cv_mean, cv_std = 0.0, 0.0

        # Fit on full training data
        self.model.fit(X, y)
        self._is_fitted = True

        train_accuracy = self.model.score(X, y)

        return cv_mean, cv_std, train_accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Activation features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Activation features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, 2)
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Evaluate probe on test data.

        Args:
            X: Test features, shape (n_samples, n_features)
            y: Test labels, shape (n_samples,)

        Returns:
            Tuple of (accuracy, precision, recall, f1)
        """
        y_pred = self.predict(X)
        return (
            accuracy_score(y, y_pred),
            precision_score(y, y_pred, zero_division=0),
            recall_score(y, y_pred, zero_division=0),
            f1_score(y, y_pred, zero_division=0),
        )

    def get_steering_vector(self, normalize: bool = True) -> np.ndarray:
        """Get the probe's weight vector for use as a steering direction.

        The weight vector points from class 0 to class 1. For temporal
        preference probes trained with short_term=0, long_term=1, this
        points toward long-term preference.

        Args:
            normalize: Whether to L2-normalize the vector

        Returns:
            Weight vector, shape (n_features,)
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call train() first.")

        weights = self.model.coef_[0]  # Shape: (n_features,)

        if normalize:
            norm = np.linalg.norm(weights)
            if norm > 0:
                weights = weights / norm

        return weights

    def get_bias(self) -> float:
        """Get the probe's bias term."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call train() first.")
        return self.model.intercept_[0]

    @property
    def is_fitted(self) -> bool:
        """Whether the probe has been trained."""
        return self._is_fitted
