"""Tests for fraction-of-energy-in-subspace."""

import numpy as np
import pytest

from src.probes.lora_icl.subspace_metrics import fraction_in_subspace, pca_components


def test_full_space_captures_all():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    basis = np.eye(5)  # full space
    assert fraction_in_subspace(X, basis) == pytest.approx(1.0)


def test_orthogonal_subspace_captures_none():
    X = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # all along e0
    basis = np.array([[0.0], [1.0], [0.0]])           # spans e1
    assert fraction_in_subspace(X, basis) == pytest.approx(0.0)


def test_half_energy():
    X = np.array([[3.0, 4.0]])  # squared norm 25; e0 component 9
    basis = np.array([[1.0], [0.0]])
    assert fraction_in_subspace(X, basis) == pytest.approx(9.0 / 25.0)


def test_planted_subspace_captures_most():
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=(200, 2))
    plane = np.zeros((2, 8))
    plane[0, 0] = 1.0
    plane[1, 1] = 1.0
    X = coeffs @ plane + 0.01 * rng.normal(size=(200, 8))
    basis = pca_components(X, k=2)
    assert fraction_in_subspace(X, basis) > 0.99


def test_zero_vectors_raise():
    with pytest.raises(ValueError, match="zero total norm"):
        fraction_in_subspace(np.zeros((3, 4)), np.eye(4)[:, :2])
