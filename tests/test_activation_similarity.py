"""Tests for the representation-similarity primitives (CPU, no network)."""

from __future__ import annotations

import torch

from src.probes.attribution.activation_similarity import linear_cka, subspace_overlap


def _orthonormal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q


# --- linear CKA --------------------------------------------------------------


def test_cka_identity_is_one():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 16, generator=g)
    assert abs(linear_cka(X, X) - 1.0) < 1e-9


def test_cka_invariant_to_orthogonal_rotation_and_scale():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(300, 12, generator=g, dtype=torch.float64)
    Q = _orthonormal(12, seed=2)
    Y = 3.7 * (X @ Q)                        # rotate + isotropic scale
    assert abs(linear_cka(X, Y) - 1.0) < 1e-9


def test_cka_independent_is_low():
    g = torch.Generator().manual_seed(3)
    X = torch.randn(500, 32, generator=g, dtype=torch.float64)
    Y = torch.randn(500, 32, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    assert linear_cka(X, Y) < 0.15          # independent Gaussians: near-zero similarity


def test_cka_symmetric():
    g = torch.Generator().manual_seed(5)
    X = torch.randn(150, 10, generator=g, dtype=torch.float64)
    Y = torch.randn(150, 10, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-12


# --- subspace overlap --------------------------------------------------------


def test_subspace_overlap_identical_is_one():
    g = torch.Generator().manual_seed(7)
    A = torch.randn(400, 20, generator=g, dtype=torch.float64)
    assert abs(subspace_overlap(A, A, k=5) - 1.0) < 1e-9


def test_subspace_overlap_orthogonal_is_zero():
    # A varies only in dims 0..3, B only in dims 4..7 → top-4 subspaces are orthogonal
    g = torch.Generator().manual_seed(8)
    n, d = 300, 10
    A = torch.zeros(n, d, dtype=torch.float64)
    B = torch.zeros(n, d, dtype=torch.float64)
    A[:, 0:4] = torch.randn(n, 4, generator=g, dtype=torch.float64)
    B[:, 4:8] = torch.randn(n, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    assert subspace_overlap(A, B, k=4) < 1e-6


def test_subspace_overlap_in_unit_range():
    g = torch.Generator().manual_seed(10)
    A = torch.randn(200, 16, generator=g, dtype=torch.float64)
    B = torch.randn(200, 16, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    ov = subspace_overlap(A, B, k=8)
    assert 0.0 <= ov <= 1.0
