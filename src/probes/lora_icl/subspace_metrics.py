"""Metrics for comparing two activation-shift sets as directions / subspaces.

Used to test whether the LoRA-induced activation shift and the ICL-induced
activation shift occupy the same region of the residual stream.

Conventions
-----------
A *shift set* is an array of shape ``(n_examples, hidden_dim)``: one residual
shift vector per held-out example, at a fixed layer and token site. From a shift
set we derive:
  - a single *mean direction* (the average shift), compared via cosine, and
  - a *subspace* (top-k PCA directions of the per-example shifts), compared via
    principal angles.

All functions are pure NumPy and model-free so they unit-test on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.base_schema import BaseSchema


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Raises on a zero vector."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cannot compute cosine of a zero vector")
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def random_cosine_null_std(hidden_dim: int) -> float:
    """Std of the cosine between two independent isotropic vectors in R^d.

    The expected cosine is 0; its standard deviation is ``1/sqrt(d)``. This is
    the reference scale for "is the observed alignment above chance?".
    """
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return 1.0 / np.sqrt(hidden_dim)


def mean_direction_cosine(u_shifts: np.ndarray, v_shifts: np.ndarray) -> float:
    """Cosine between the mean shift of each set.

    Args:
        u_shifts: ``(n, d)`` shift set (e.g. ICL shifts).
        v_shifts: ``(m, d)`` shift set (e.g. LoRA shifts). ``n`` need not equal ``m``.
    """
    u = np.asarray(u_shifts, dtype=np.float64)
    v = np.asarray(v_shifts, dtype=np.float64)
    return vector_cosine(u.mean(axis=0), v.mean(axis=0))


def pca_components(shifts: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` principal directions of a shift set, as orthonormal columns.

    The shifts are mean-centered across examples before the SVD, so this
    captures the directions of *variation* in the shift, not the mean shift.

    Returns:
        ``(d, k)`` array with orthonormal columns.
    """
    X = np.asarray(shifts, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"shifts must be 2-D (n, d); got shape {X.shape}")
    n, d = X.shape
    if k < 1 or k > min(n, d):
        raise ValueError(f"k={k} must be in [1, min(n, d)={min(n, d)}]")
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:k].T


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces.

    Each argument is a ``(d, k)`` matrix whose columns span a subspace (the
    columns need not be orthonormal). Returns ``min(k_a, k_b)`` angles in
    ``[0, pi/2]``: 0 means a shared direction, ``pi/2`` means orthogonal.
    """
    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("bases must be 2-D (d, k) matrices")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"bases must share the ambient dimension; got {A.shape[0]} vs {B.shape[0]}"
        )
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.sort(np.arccos(sv))


def subspace_overlap(u_shifts: np.ndarray, v_shifts: np.ndarray, k: int) -> float:
    """Mean cosine of the principal angles between the two top-``k`` PCA subspaces.

    1.0 means the ``k``-dim subspaces of variation coincide; 0.0 means they are
    fully orthogonal. This is the standard Grassmann-style overlap score.
    """
    a = pca_components(u_shifts, k)
    b = pca_components(v_shifts, k)
    angles = principal_angles(a, b)
    return float(np.mean(np.cos(angles)))


def fraction_in_subspace(vectors: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of the total squared norm of ``vectors`` lying in ``basis``'s span.

    Args:
        vectors: ``(n, d)`` rows whose energy is being measured.
        basis: ``(d, k)`` matrix with orthonormal columns spanning the subspace.

    Returns:
        ``sum ||proj_basis(v)||^2 / sum ||v||^2`` in ``[0, 1]`` — an R²-like share of
        ``vectors``' energy captured by the subspace.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    B = np.asarray(basis, dtype=np.float64)
    total = float((X * X).sum())
    if total == 0.0:
        raise ValueError("vectors have zero total norm")
    coords = X @ B                  # (n, k) coordinates in the subspace
    captured = float((coords * coords).sum())
    return captured / total


def parallel_perp(vectors: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each row of ``vectors`` into components parallel / perpendicular to ``direction``.

    Args:
        vectors: ``(n, d)`` (a single ``(d,)`` vector is accepted and treated as one row).
        direction: ``(d,)`` reference direction (unit-normalized internally).

    Returns:
        ``(par, perp)`` each shaped like ``vectors``; ``par + perp == vectors`` and every row of
        ``perp`` is orthogonal to ``direction``.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    u = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(u)
    if norm == 0.0:
        raise ValueError("cannot decompose along a zero direction")
    u = u / norm
    coeff = X @ u
    par = np.outer(coeff, u)
    return par, X - par


@dataclass
class LayerSubspaceResult(BaseSchema):
    """Per-layer comparison of LoRA vs ICL activation shifts."""

    layer: int = 0
    mean_cosine: float = 0.0
    principal_angles_deg: list[float] = field(default_factory=list)
    subspace_overlap: float = 0.0
    n_examples: int = 0
    hidden_dim: int = 0
