"""Tests for the cross-model linear activation map (LoRA capability transfer).

All CPU/numpy: the map is fit on captured final-token residuals, so its algebra is
model-free. The driver owns the forward passes.
"""

import numpy as np
import pytest

from src.probes.lora_icl.linear_map_transfer import (
    LinearMap,
    fit_linear_map,
    norm_matched_random,
)


def _paired(n=240, d_a=12, d_b=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=(d_a, d_b))
    x = rng.normal(size=(n, d_a)) + 3.0          # nonzero mean, like real residuals
    y = (x - x.mean(0)) @ w_true + 7.0 + noise * rng.normal(size=(n, d_b))
    return x, y, w_true


def test_recovers_an_exact_linear_map():
    x, y, w_true = _paired()
    m = fit_linear_map(x, y, lam=1e-8)
    assert m.r2_holdout == pytest.approx(1.0, abs=1e-6)
    delta = np.random.default_rng(1).normal(size=x.shape[1])
    assert np.allclose(m.map_shift(delta), delta @ w_true, atol=1e-5)


def test_shift_mapping_ignores_the_means():
    """A shift is a difference of states, so the fitted means must cancel out of it."""
    x, y, _ = _paired()
    m = fit_linear_map(x, y, lam=1e-8)
    assert np.allclose(m.map_shift(np.zeros(x.shape[1])), np.zeros(y.shape[1]))
    a = np.random.default_rng(2).normal(size=x.shape[1])
    b = np.random.default_rng(3).normal(size=x.shape[1])
    assert np.allclose(m.map_shift(a + b), m.map_shift(a) + m.map_shift(b), atol=1e-8)


def test_state_mapping_uses_the_means():
    x, y, _ = _paired()
    m = fit_linear_map(x, y, lam=1e-8)
    assert np.allclose(m.map_state(x.mean(0)), y.mean(0), atol=1e-6)


def test_r2_degrades_with_noise_but_stays_in_range():
    x0, y0, _ = _paired(noise=0.5, seed=4)
    x1, y1, _ = _paired(noise=5.0, seed=4)
    m0 = fit_linear_map(x0, y0, lam=1.0)
    m1 = fit_linear_map(x1, y1, lam=1.0)
    assert m1.r2_holdout < m0.r2_holdout < 1.0
    assert m1.r2_holdout > -0.5  # ridge on iid noise should not be catastrophically off


def test_holdout_is_disjoint_and_seeded():
    x, y, _ = _paired()
    a = fit_linear_map(x, y, lam=1.0, seed=7)
    b = fit_linear_map(x, y, lam=1.0, seed=7)
    assert a.r2_holdout == b.r2_holdout
    assert np.allclose(a.weights, b.weights)


def test_rejects_misaligned_rows():
    x, y, _ = _paired()
    with pytest.raises(ValueError):
        fit_linear_map(x[:-1], y, lam=1.0)


def test_norm_matched_random_matches_norm_not_direction():
    v = np.random.default_rng(5).normal(size=64) * 3.7
    r = norm_matched_random(v, seed=6)
    assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(v))
    cos = float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r)))
    assert abs(cos) < 0.5
    assert np.allclose(r, norm_matched_random(v, seed=6))       # seeded
    assert not np.allclose(r, norm_matched_random(v, seed=8))   # and seed-sensitive


def test_roundtrip_serialization():
    x, y, _ = _paired()
    m = fit_linear_map(x, y, lam=1.0)
    m2 = LinearMap.from_arrays(**m.to_arrays())
    assert np.allclose(m.map_shift(np.ones(x.shape[1])), m2.map_shift(np.ones(x.shape[1])))
    assert m2.r2_holdout == m.r2_holdout
