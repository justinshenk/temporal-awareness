"""Unit tests for the CAST-style logistic gate (model-free, toy residuals)."""

from __future__ import annotations

import numpy as np
import pytest

from src.probes.safety.conditional_gate import LogisticGate


def _toy_clusters(seed: int = 0, n: int = 40, d: int = 8, sep: float = 6.0):
    """Two linearly separable Gaussian clusters in d-space (harmful shifted by +sep)."""
    rng = np.random.default_rng(seed)
    harmful = rng.standard_normal((n, d)) + sep
    benign = rng.standard_normal((n, d)) - sep
    return harmful, benign


def test_fit_separates_toy_clusters():
    harmful, benign = _toy_clusters()
    gate = LogisticGate().fit(harmful, benign)
    assert gate.train_accuracy == pytest.approx(1.0)


def test_predict_returns_correct_mask():
    harmful, benign = _toy_clusters()
    gate = LogisticGate().fit(harmful, benign)
    # harmful rows -> True (steer), benign rows -> False (leave alone)
    assert gate.predict(harmful).all()
    assert not gate.predict(benign).any()


def test_predict_single_row_is_bool_array():
    harmful, benign = _toy_clusters()
    gate = LogisticGate().fit(harmful, benign)
    mask = gate.predict(harmful[:1])
    assert mask.shape == (1,)
    assert mask.dtype == bool
    assert bool(mask[0]) is True


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogisticGate().predict(np.zeros((1, 8)))


def test_degenerate_single_class_raises():
    harmful, _ = _toy_clusters()
    with pytest.raises(ValueError):
        LogisticGate().fit(harmful, np.empty((0, harmful.shape[1])))


def test_dim_mismatch_raises():
    harmful, benign = _toy_clusters(d=8)
    with pytest.raises(ValueError):
        LogisticGate().fit(harmful, benign[:, :4])
