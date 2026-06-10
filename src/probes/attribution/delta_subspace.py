"""PCA bands of the residual shift δ, for locating capability-critical directions in variance-space.

The uncentered second moment ``Σ = (1/n) Σ δ_t δ_tᵀ`` has top eigenvectors along the high-energy
directions of the shift (the consistent format/style component) and a low-energy tail (problem-
specific variation). Banding δ by these eigenvectors and oracle-injecting one band at a time shows
which band carries the recovery — testing whether the computation signal is low-variance (the reason
an MSE-fit feed-forward map, which chases variance, misses it).
"""

from __future__ import annotations

import torch


def pca_bands(D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose the uncentered second moment of rows of ``D`` ``(n, d)``.

    Returns ``(V, eigvals)`` with eigenvectors as **descending**-eigenvalue columns ``(d, d)`` and
    the matching eigenvalues ``(d,)``. ``V[:, :k]`` is the top-k (highest-energy) band; ``V[:, k:]``
    the tail.
    """
    D = D.to(torch.float64)
    sigma = (D.T @ D) / D.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(sigma)   # ascending
    order = torch.argsort(eigvals, descending=True)
    return eigvecs[:, order], eigvals[order]


def energy_fraction(eigvals: torch.Tensor, k: int) -> float:
    """Fraction of total shift energy captured by the top-k band."""
    total = float(eigvals.sum())
    return float(eigvals[:k].sum()) / total if total > 0 else 0.0
