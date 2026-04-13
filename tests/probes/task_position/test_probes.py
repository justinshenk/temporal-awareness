"""Tests for ridge probes on synthetic data."""

import numpy as np
import pytest

from src.probes.task_position.probes import RidgeProbe


def test_ridge_recovers_linear_signal():
    """A linear probe should recover a linear signal from noisy features."""
    rng = np.random.default_rng(0)
    d = 16
    n = 500
    true_w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ true_w + 0.05 * rng.normal(size=n)

    probe = RidgeProbe(alpha=1.0)
    probe.fit(X[:400], y[:400])
    r2 = probe.score(X[400:], y[400:])
    assert r2 > 0.95, f"expected R² > 0.95, got {r2}"


def test_ridge_direction_is_dim_d():
    probe = RidgeProbe()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 8))
    y = rng.normal(size=100)
    probe.fit(X, y)
    direction = probe.direction()
    assert direction.shape == (8,)


def test_ridge_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 8))
    y = rng.normal(size=100)

    probe = RidgeProbe(alpha=0.5)
    probe.fit(X, y)
    path = tmp_path / "probe.pkl"
    probe.save(path)

    loaded = RidgeProbe.load(path)
    assert np.allclose(probe.predict(X), loaded.predict(X))
    assert loaded.alpha == 0.5


def test_ridge_raises_if_not_fit():
    probe = RidgeProbe()
    with pytest.raises(RuntimeError, match="not fit"):
        probe.predict(np.zeros((1, 8)))
