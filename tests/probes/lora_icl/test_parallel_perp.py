"""Tests for parallel/perpendicular decomposition of shift vectors."""

import numpy as np
import pytest

from src.probes.lora_icl.subspace_metrics import parallel_perp


def test_decomposition_reconstructs_and_is_orthogonal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    d = rng.normal(size=8)
    par, perp = parallel_perp(X, d)
    assert np.allclose(par + perp, X)
    # every perp row orthogonal to d
    assert np.allclose(perp @ d, 0.0, atol=1e-9)
    # every par row parallel to d (cross-component with any orthogonal vector is captured by par only)
    u = d / np.linalg.norm(d)
    assert np.allclose(par, np.outer(par @ u, u))


def test_vector_aligned_with_direction_has_zero_perp():
    d = np.array([0.0, 3.0, 0.0])
    par, perp = parallel_perp(np.array([0.0, 5.0, 0.0]), d)
    assert np.allclose(perp, 0.0)
    assert np.allclose(par, np.array([[0.0, 5.0, 0.0]]))


def test_vector_orthogonal_to_direction_has_zero_par():
    d = np.array([1.0, 0.0])
    par, perp = parallel_perp(np.array([0.0, 7.0]), d)
    assert np.allclose(par, 0.0)
    assert np.allclose(perp, np.array([[0.0, 7.0]]))


def test_zero_direction_raises():
    with pytest.raises(ValueError, match="zero direction"):
        parallel_perp(np.ones((2, 3)), np.zeros(3))
