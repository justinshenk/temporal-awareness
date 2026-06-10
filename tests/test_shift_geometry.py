"""Tests for predicted-vs-true residual-shift geometry (pure numpy, no model)."""

from __future__ import annotations

import numpy as np

from src.probes.attribution.shift_geometry import (
    mean_row_cosine,
    median_norm_ratio,
    r2_multioutput,
    shift_geometry,
)


def _rng():
    return np.random.default_rng(0)


def test_identity_is_perfect():
    T = _rng().standard_normal((50, 8))
    assert mean_row_cosine(T, T) > 0.999
    assert abs(median_norm_ratio(T, T) - 1.0) < 1e-9
    assert r2_multioutput(T, T) > 0.999


def test_scaled_prediction_keeps_direction_loses_magnitude():
    T = _rng().standard_normal((40, 6))
    P = 2.0 * T
    assert mean_row_cosine(P, T) > 0.999          # direction perfect
    assert abs(median_norm_ratio(P, T) - 2.0) < 1e-9  # twice too big
    assert r2_multioutput(P, T) < 0.9             # magnitude error penalized


def test_orthogonal_prediction_is_zero_cosine():
    n = 30
    T = np.zeros((n, 4)); T[:, 0] = 1.0   # all along e0
    P = np.zeros((n, 4)); P[:, 1] = 1.0   # all along e1
    assert abs(mean_row_cosine(P, T)) < 1e-9
    assert r2_multioutput(P, T) <= 0.0    # no variance explained


def test_zero_prediction():
    T = _rng().standard_normal((20, 5))
    P = np.zeros_like(T)
    assert mean_row_cosine(P, T) == 0.0
    assert median_norm_ratio(P, T) == 0.0
    assert r2_multioutput(P, T) < 1.0


def test_shift_geometry_bundle_keys_and_count():
    T = _rng().standard_normal((12, 3))
    P = _rng().standard_normal((12, 3))
    g = shift_geometry(P, T)
    assert set(g) == {"mean_cosine", "median_norm_ratio", "r2", "n_tokens"}
    assert g["n_tokens"] == 12
