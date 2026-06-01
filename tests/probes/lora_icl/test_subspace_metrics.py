"""Tests for subspace-comparison metrics (LoRA vs ICL activation shifts).

Test-forward: these pin the numerical contract of the cosine / principal-angle
metrics before any model is involved. All cases use synthetic vectors so they
run on CPU with no model download.
"""

import numpy as np
import pytest

from src.probes.lora_icl.subspace_metrics import (
    LayerSubspaceResult,
    mean_direction_cosine,
    pca_components,
    principal_angles,
    random_cosine_null_std,
    subspace_overlap,
    vector_cosine,
)


# ---------------------------------------------------------------- vector cosine
def test_cosine_parallel_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert vector_cosine(a, 2.5 * a) == pytest.approx(1.0)


def test_cosine_antiparallel_is_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    assert vector_cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_orthogonal_is_zero():
    assert vector_cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_zero_vector_raises():
    with pytest.raises(ValueError, match="zero"):
        vector_cosine(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_cosine_random_highdim_near_zero():
    rng = np.random.default_rng(0)
    d = 4096
    cosines = [
        vector_cosine(rng.normal(size=d), rng.normal(size=d)) for _ in range(200)
    ]
    # E[cos] ~ 0, std ~ 1/sqrt(d); the mean of 200 draws should be tiny.
    assert abs(np.mean(cosines)) < 5 * random_cosine_null_std(d)


def test_random_cosine_null_std():
    assert random_cosine_null_std(100) == pytest.approx(0.1)


# ------------------------------------------------------- mean-direction cosine
def test_mean_direction_cosine_matches_manual():
    U = np.array([[1.0, 0.0], [3.0, 0.0]])  # mean -> (2, 0)
    V = np.array([[0.0, 5.0], [0.0, 1.0]])  # mean -> (0, 3)
    assert mean_direction_cosine(U, V) == pytest.approx(0.0)


def test_mean_direction_cosine_aligned():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(10, 32))
    assert mean_direction_cosine(base, 2.0 * base) == pytest.approx(1.0)


# ------------------------------------------------------------- pca components
def test_pca_components_shape_and_orthonormal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 16))
    comps = pca_components(X, k=4)
    assert comps.shape == (16, 4)
    # columns orthonormal
    gram = comps.T @ comps
    assert np.allclose(gram, np.eye(4), atol=1e-8)


def test_pca_components_recovers_planted_plane():
    # Data lives in the span of e0, e1 -> top-2 PCs should span that plane.
    rng = np.random.default_rng(3)
    coeffs = rng.normal(size=(200, 2))
    basis = np.zeros((2, 8))
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    X = coeffs @ basis
    comps = pca_components(X, k=2)
    plane = np.zeros((8, 2))
    plane[0, 0] = 1.0
    plane[1, 1] = 1.0
    angles = principal_angles(comps, plane)
    assert np.allclose(angles, 0.0, atol=1e-6)


# ------------------------------------------------------------ principal angles
def test_principal_angles_identical_subspace_zero():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(20, 3))
    angles = principal_angles(A, A.copy())
    # arccos loses precision for singular values near 1, so 0 angles land ~1e-8.
    assert np.allclose(angles, 0.0, atol=1e-6)


def test_principal_angles_orthogonal_subspace_ninety():
    d = 10
    A = np.eye(d)[:, :3]
    B = np.eye(d)[:, 3:6]
    angles = principal_angles(A, B)
    assert np.allclose(angles, np.pi / 2, atol=1e-8)


def test_principal_angles_invariant_to_within_subspace_rotation():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(12, 3))
    # Rotate A's basis within its own span via a random 3x3 rotation.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A_rot = A @ q
    B = rng.normal(size=(12, 3))
    assert np.allclose(
        np.sort(principal_angles(A, B)),
        np.sort(principal_angles(A_rot, B)),
        atol=1e-6,
    )


def test_principal_angles_partial_overlap():
    d = 8
    # Subspaces share e0; differ on the second basis vector.
    A = np.stack([np.eye(d)[0], np.eye(d)[1]], axis=1)
    B = np.stack([np.eye(d)[0], np.eye(d)[2]], axis=1)
    angles = np.sort(principal_angles(A, B))
    # One shared direction (angle 0), one orthogonal (angle 90).
    assert angles[0] == pytest.approx(0.0, abs=1e-8)
    assert angles[1] == pytest.approx(np.pi / 2, abs=1e-8)


# ------------------------------------------------------------ subspace overlap
def test_subspace_overlap_identical_is_one():
    rng = np.random.default_rng(6)
    U = rng.normal(size=(100, 32))
    assert subspace_overlap(U, U.copy(), k=4) == pytest.approx(1.0, abs=1e-8)


def test_subspace_overlap_orthogonal_is_zero():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(100, 2))
    b = rng.normal(size=(100, 2))
    U = np.concatenate([a, np.zeros((100, 2))], axis=1)  # spans dims 0,1
    V = np.concatenate([np.zeros((100, 2)), b], axis=1)  # spans dims 2,3
    assert subspace_overlap(U, V, k=2) == pytest.approx(0.0, abs=1e-8)


# ----------------------------------------------------------- result dataclass
def test_layer_result_roundtrips():
    res = LayerSubspaceResult(
        layer=10,
        mean_cosine=0.42,
        principal_angles_deg=[3.1, 12.0, 44.0],
        subspace_overlap=0.81,
        n_examples=18,
        hidden_dim=3584,
    )
    restored = LayerSubspaceResult.from_dict(res.to_dict())
    assert restored.layer == 10
    assert restored.mean_cosine == pytest.approx(0.42)
    assert restored.principal_angles_deg == [3.1, 12.0, 44.0]
