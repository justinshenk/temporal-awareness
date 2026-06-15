"""Representation-similarity primitives for comparing two adaptation methods' activation edits.

Used to ask whether LoRA and LoReFT make the *same* change to commonsense activations. Two views:

  * **linear CKA** (`linear_cka`) — global similarity of two (n, d) activation sets, invariant to
    orthogonal rotation and isotropic scaling, so it answers "do the two steered representations live
    in the same geometry" regardless of basis;
  * **PCA-subspace overlap** (`subspace_overlap`) — normalized projection overlap between the top-k
    high-energy subspaces of two edit sets δ (built on :func:`delta_subspace.pca_bands`), i.e. how
    much of one method's dominant edit directions the other's span captures.

Per-token directional alignment of the edits is covered by
:func:`shift_geometry.mean_row_cosine` (reused by the driver), so it is not duplicated here.
"""

from __future__ import annotations

import torch

from src.probes.attribution.delta_subspace import pca_bands


def _centered_f64(X: torch.Tensor) -> torch.Tensor:
    X = X.to(torch.float64)
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between paired activation sets ``X``/``Y`` (both ``(n, d)``), in [0, 1].

    ``CKA = ‖Yᵀ X‖_F² / (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F)`` on column-centered ``X``/``Y`` — the standard
    HSIC-based linear CKA. 1.0 iff the two centered sets are related by an orthogonal transform and
    isotropic scale; invariant to those, so it compares geometry, not raw coordinates.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    Xc, Yc = _centered_f64(X), _centered_f64(Y)
    hsic_xy = float((Yc.T @ Xc).pow(2).sum())
    norm_xx = float((Xc.T @ Xc).pow(2).sum()) ** 0.5
    norm_yy = float((Yc.T @ Yc).pow(2).sum()) ** 0.5
    denom = norm_xx * norm_yy
    return hsic_xy / denom if denom > 0 else 0.0


def subspace_overlap(A: torch.Tensor, B: torch.Tensor, k: int) -> float:
    """Normalized top-k PCA-subspace overlap between two edit sets ``A``/``B`` (rows are δ), in [0, 1].

    Takes the top-k high-energy eigenvectors of each set's uncentered second moment
    (:func:`delta_subspace.pca_bands`) and returns ``‖Pₐᵀ P_b‖_F² / k`` — the mean of the squared
    cosines of the principal angles between the two k-subspaces. 1.0 iff the subspaces coincide,
    0.0 iff orthogonal.
    """
    if not 1 <= k <= min(A.shape[1], B.shape[1]):
        raise ValueError(f"k={k} out of range for dims {A.shape[1]}, {B.shape[1]}")
    Va, _ = pca_bands(A)
    Vb, _ = pca_bands(B)
    Pa, Pb = Va[:, :k], Vb[:, :k]
    return float((Pa.T @ Pb).pow(2).sum() / k)
