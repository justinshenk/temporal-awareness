"""Tests for the cross-model alignment + LoReFT transplant operator (CPU, no network, fast).

The transplant maps a recipient residual row into the donor's space with an orthogonal Procrustes
map ``A``, applies the donor's own LoReFT edit there, and projects the edit back with ``Aᵀ``. These
tests pin the crux properties in miniature: Procrustes recovers a known random rotation+translation
and yields an orthonormal ``A``; the streaming cross-covariance stats are chunk-invariant; and
``TransplantEdit`` is shape-correct and the identity when the donor edit is zero.
"""

from __future__ import annotations

import torch

from src.probes.attribution.crossmodel_align import (
    CrossCovStats,
    TransplantEdit,
    fit_procrustes,
    procrustes_residual,
)
from src.probes.attribution.loreft_intervention import LoReFTIntervention


def make_rotated_pair(n: int, d: int, seed: int = 0):
    """Recipient rows ``X_R`` and donor rows ``X_D = (X_R - μ_R) Qᵀ + μ_D`` for random orthogonal Q.

    With this construction ``h_D - μ_D = Q (h_R - μ_R)``, so the recipient→donor Procrustes map is
    exactly ``Q``.
    """
    g = torch.Generator().manual_seed(seed)
    X_R = torch.randn(n, d, generator=g, dtype=torch.float64)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(a)                          # random orthogonal
    if torch.det(Q) < 0:                               # make it a proper rotation
        Q[:, 0] = -Q[:, 0]
    mu_R = X_R.mean(0)
    mu_D = torch.randn(d, generator=g, dtype=torch.float64)
    X_D = (X_R - mu_R) @ Q.T + mu_D
    return X_R, X_D, Q, mu_R, mu_D


def test_procrustes_recovers_rotation_and_translation():
    """fit_procrustes returns A≈Q, mean_R≈E[X_R], mean_D≈E[X_D] for a known rotation+translation."""
    d, n = 12, 5000
    X_R, X_D, Q, mu_R, mu_D = make_rotated_pair(n, d, seed=1)
    A, mean_R, mean_D = fit_procrustes((X_R, X_D))
    assert torch.allclose(A.double(), Q, atol=1e-4), (A - Q).abs().max()
    assert torch.allclose(mean_R.double(), mu_R, atol=1e-4)
    assert torch.allclose(mean_D.double(), mu_D, atol=1e-4)


def test_procrustes_A_is_orthonormal():
    """A is orthonormal: AᵀA ≈ I."""
    d, n = 12, 3000
    X_R, X_D, *_ = make_rotated_pair(n, d, seed=2)
    A, _, _ = fit_procrustes((X_R, X_D))
    eye = torch.eye(d, dtype=A.dtype)
    assert torch.allclose(A.T @ A, eye, atol=1e-4), (A.T @ A - eye).abs().max()


def test_procrustes_residual_zero_for_exact_rotation():
    """A perfect rotation+translation gives a ~0 relative Procrustes residual; noise raises it."""
    d, n = 12, 4000
    X_R, X_D, *_ = make_rotated_pair(n, d, seed=5)
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    state = stats.state()
    A, mean_R, mean_D = fit_procrustes(state)
    assert procrustes_residual(state, A, mean_R, mean_D) < 1e-3

    g = torch.Generator().manual_seed(99)
    noisy = CrossCovStats(d)
    noisy.update(X_R, X_D + 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64))
    nstate = noisy.state()
    An, mRn, mDn = fit_procrustes(nstate)
    assert procrustes_residual(nstate, An, mRn, mDn) > 1e-2


def test_crosscov_chunked_equals_oneshot():
    """Streaming cross-cov stats from two chunks equal one-shot accumulation."""
    d, n = 12, 600
    X_R, X_D, *_ = make_rotated_pair(n, d, seed=3)

    one = CrossCovStats(d)
    one.update(X_R, X_D)

    two = CrossCovStats(d)
    two.update(X_R[:250], X_D[:250])
    two.update(X_R[250:], X_D[250:])

    sa, sb = one.state(), two.state()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.allclose(sa[k].double(), sb[k].double(), atol=1e-6, rtol=1e-6), k


def test_fit_procrustes_from_stats_equals_from_rows():
    """Passing a stats dict yields the same fit as passing raw rows."""
    d, n = 12, 2000
    X_R, X_D, *_ = make_rotated_pair(n, d, seed=4)
    A_rows, mR_rows, mD_rows = fit_procrustes((X_R, X_D))
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    A_stats, mR_stats, mD_stats = fit_procrustes(stats.state())
    assert torch.allclose(A_rows, A_stats, atol=1e-5)
    assert torch.allclose(mR_rows, mR_stats, atol=1e-5)
    assert torch.allclose(mD_rows, mD_stats, atol=1e-5)


def test_transplant_edit_shape_and_zero_donor_edit_is_identity():
    """Output shape (b, d); a zero donor edit makes TransplantEdit the identity on h_R."""
    d, r, b = 16, 4, 7
    donor = LoReFTIntervention(d, r)
    # Zero the source map so source(h)=0 and the LoReFT edit is δ = (0 − hR)Rᵀ ... not zero.
    # To force δ_D = 0 exactly we need source(h) = hR for all h: set source weight=R.T, bias=0.
    with torch.no_grad():
        R = donor.subspace()                           # (d, r)
        donor.source.weight.copy_(R.T)
        donor.source.bias.zero_()
    h_R = torch.randn(b, d)
    A = torch.eye(d)                                   # any orthonormal A works for the identity check
    edit = TransplantEdit(donor, A, mean_R=torch.randn(d), mean_D=torch.randn(d))
    out = edit(h_R)
    assert out.shape == (b, d)
    assert torch.allclose(out, h_R, atol=1e-5), (out - h_R).abs().max()


def test_transplant_edit_matches_manual_operator():
    """TransplantEdit reproduces ĥ_D=μ_D+A(h_R−μ_R); δ=LoReFT(ĥ_D)−ĥ_D; h_R+Aᵀδ."""
    d, r, b = 16, 4, 5
    donor = LoReFTIntervention(d, r)
    g = torch.Generator().manual_seed(7)
    a = torch.randn(d, d, generator=g)
    A, _ = torch.linalg.qr(a)
    mean_R = torch.randn(d, generator=g)
    mean_D = torch.randn(d, generator=g)
    h_R = torch.randn(b, d, generator=g)

    h_D = mean_D + (h_R - mean_R) @ A.T
    delta = donor(h_D) - h_D
    expected = h_R + delta @ A                          # Aᵀ applied on the right is @ A
    edit = TransplantEdit(donor, A, mean_R, mean_D)
    assert torch.allclose(edit(h_R), expected, atol=1e-5), (edit(h_R) - expected).abs().max()
