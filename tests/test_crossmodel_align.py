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
    AffineTransplantEdit,
    CrossCovStats,
    TransplantEdit,
    affine_residual,
    fit_affine_bridge,
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


def make_affine_pair(n: int, d: int, seed: int = 0, cond: float = 4.0):
    """Recipient rows ``X_R`` and donor rows ``X_D = (X_R - μ_R) M + μ_D`` for a NON-orthogonal M.

    ``M = Q1 diag(s) Q2`` with singular values log-spaced in ``[1/cond, 1] * cond**0.5`` — a shear no
    rotation can represent, so the affine bridge fits it exactly while Procrustes cannot.
    """
    g = torch.Generator().manual_seed(seed)
    X_R = torch.randn(n, d, generator=g, dtype=torch.float64)
    q1, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    s = torch.logspace(-0.5, 0.5, d, base=cond, dtype=torch.float64)
    M = q1 @ torch.diag(s) @ q2
    mu_R = X_R.mean(0)
    mu_D = torch.randn(d, generator=g, dtype=torch.float64)
    X_D = (X_R - mu_R) @ M + mu_D
    return X_R, X_D, M, mu_R, mu_D


def test_affine_bridge_recovers_shear_where_procrustes_cannot():
    """fit_affine_bridge recovers a known non-orthogonal map (residual ≈ 0); Procrustes cannot."""
    d, n = 12, 5000
    X_R, X_D, M, _, mu_D = make_affine_pair(n, d, seed=11)
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    state = stats.state()

    W_F, W_G, mean_R, mean_D = fit_affine_bridge(state, lam=1e-10)
    assert torch.allclose(W_F.double(), M, atol=1e-4), (W_F.double() - M).abs().max()
    assert torch.allclose(mean_D.double(), mu_D, atol=1e-4)
    assert affine_residual(state, W_F, mean_R, mean_D) < 1e-3

    A, mR, mD = fit_procrustes(state)
    assert procrustes_residual(state, A, mR, mD) > 0.05  # a rotation cannot represent the shear


def test_affine_bridge_nests_procrustes_for_pure_rotation():
    """On rotation+translation data the affine bridge reduces to the Procrustes solution."""
    d, n = 12, 5000
    X_R, X_D, Q, *_ = make_rotated_pair(n, d, seed=12)
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    state = stats.state()
    W_F, _, mean_R, mean_D = fit_affine_bridge(state, lam=1e-10)
    A, _, _ = fit_procrustes(state)
    assert torch.allclose(W_F, A.T, atol=1e-3), (W_F - A.T).abs().max()  # rows act on the right
    res_aff = affine_residual(state, W_F, mean_R, mean_D)
    res_pro = procrustes_residual(state, A, mean_R, mean_D)
    assert abs(res_aff - res_pro) < 1e-4


def test_affine_backward_inverts_forward():
    """For an invertible true map, the two ridge directions compose to identity: W_F @ W_G ≈ I."""
    d, n = 12, 8000
    X_R, X_D, *_ = make_affine_pair(n, d, seed=13, cond=3.0)
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    W_F, W_G, _, _ = fit_affine_bridge(stats.state(), lam=1e-10)
    eye = torch.eye(d, dtype=W_F.dtype)
    assert torch.allclose(W_F @ W_G, eye, atol=1e-3), (W_F @ W_G - eye).abs().max()


def test_affine_rank_truncation_orders_residual():
    """Truncating W_F to lower rank can only raise the fit residual; full rank matches untruncated."""
    d, n = 12, 5000
    X_R, X_D, *_ = make_affine_pair(n, d, seed=14)
    stats = CrossCovStats(d)
    stats.update(X_R, X_D)
    state = stats.state()
    res = []
    for rank in (2, 6, d):
        W_F, _, mean_R, mean_D = fit_affine_bridge(state, lam=1e-10, rank=rank)
        res.append(affine_residual(state, W_F, mean_R, mean_D))
    assert res[0] >= res[1] >= res[2]
    W_full, _, mean_R, mean_D = fit_affine_bridge(state, lam=1e-10)
    assert abs(res[2] - affine_residual(state, W_full, mean_R, mean_D)) < 1e-6


def test_affine_residual_heldout_matches_bruteforce():
    """affine_residual on a held-out stats object equals the brute-force row computation.

    The maps and means come from TRAIN rows; the residual is evaluated on VAL rows centered by the
    train means — exactly the held-out protocol the experiment script uses.
    """
    d, n = 12, 4000
    X_R, X_D, *_ = make_affine_pair(n, d, seed=15)
    g = torch.Generator().manual_seed(16)
    X_D = X_D + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64)  # imperfect fit
    train = CrossCovStats(d)
    train.update(X_R[:3000], X_D[:3000])
    val = CrossCovStats(d)
    val.update(X_R[3000:], X_D[3000:])
    W_F, _, mean_R, mean_D = fit_affine_bridge(train.state(), lam=1e-6)

    got = affine_residual(val.state(), W_F, mean_R, mean_D)
    pred = (X_R[3000:] - mean_R.double()) @ W_F.double()
    target = X_D[3000:] - mean_D.double()
    want = float(torch.linalg.norm(target - pred) / torch.linalg.norm(target))
    assert abs(got - want) < 1e-6, (got, want)


def test_affine_transplant_edit_zero_donor_edit_is_identity():
    """A zero donor edit makes AffineTransplantEdit the identity on h_R, for any W_F/W_G."""
    d, r, b = 16, 4, 7
    donor = LoReFTIntervention(d, r)
    with torch.no_grad():
        R = donor.subspace()
        donor.source.weight.copy_(R.T)
        donor.source.bias.zero_()
    g = torch.Generator().manual_seed(17)
    W_F = torch.randn(d, d, generator=g)
    W_G = torch.randn(d, d, generator=g)
    h_R = torch.randn(b, d, generator=g)
    edit = AffineTransplantEdit(donor, W_F, W_G, mean_R=torch.randn(d, generator=g),
                                mean_D=torch.randn(d, generator=g))
    out = edit(h_R)
    assert out.shape == (b, d)
    assert torch.allclose(out, h_R, atol=1e-5), (out - h_R).abs().max()


def test_affine_transplant_edit_matches_manual_operator():
    """AffineTransplantEdit reproduces ĥ_D=μ_D+(h_R−μ_R)W_F; δ=LoReFT(ĥ_D)−ĥ_D; h_R+δW_G."""
    d, r, b = 16, 4, 5
    donor = LoReFTIntervention(d, r)
    g = torch.Generator().manual_seed(18)
    W_F = torch.randn(d, d, generator=g)
    W_G = torch.randn(d, d, generator=g)
    mean_R = torch.randn(d, generator=g)
    mean_D = torch.randn(d, generator=g)
    h_R = torch.randn(b, d, generator=g)

    h_D = mean_D + (h_R - mean_R) @ W_F
    delta = donor(h_D) - h_D
    expected = h_R + delta @ W_G
    edit = AffineTransplantEdit(donor, W_F, W_G, mean_R, mean_D)
    assert torch.allclose(edit(h_R), expected, atol=1e-5), (edit(h_R) - expected).abs().max()


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
