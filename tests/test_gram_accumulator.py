"""TDD anchors for the streamed primal-ridge GramAccumulator (the heart of the experiment).

All tests run on CPU in float64 and are exact/deterministic. They implement the four
anchors specified for the attribution math: (a) streaming sums vs brute force,
(b) closed-form RSS vs brute-force Σ‖Wa−δ‖² for d>n and d<n, (c) two-form RSS
agreement, (d) the λ→0 (OLS) and λ→∞ (R²→0) limits — plus held-out scoring and plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.probes.attribution.gram_accumulator import GramAccumulator

DEV = "cpu"
DT = torch.float64


def make_synthetic(d: int, n: int, seed: int = 0, noise: float = 0.1):
    """Random A (n,d) and D = A Wtrueᵀ + noise (n,d), float64 tensors."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, generator=g, dtype=DT)
    Wtrue = torch.randn(d, d, generator=g, dtype=DT) * 0.3
    D = A @ Wtrue.T + noise * torch.randn(n, d, generator=g, dtype=DT)
    return A, D


def feed_in_blocks(acc, A, D, block_sizes):
    i = 0
    for b in block_sizes:
        acc.update(A[i:i + b], D[i:i + b])
        i += b


# --- Anchor (a): streamed s, G, P vs brute force -----------------------------------

def test_s_streaming_matches_bruteforce():
    A, D = make_synthetic(d=8, n=37, seed=1)
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    feed_in_blocks(acc, A, D, [10, 7, 1, 19])
    assert torch.allclose(acc.s, (D * D).sum().to(DT), rtol=1e-12, atol=1e-12)


def test_G_and_P_streaming_match_bruteforce():
    A, D = make_synthetic(d=8, n=40, seed=2)
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    feed_in_blocks(acc, A, D, [13, 13, 14])
    assert torch.allclose(acc.G, A.T @ A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(acc.P, A.T @ D, rtol=1e-12, atol=1e-12)


def test_Gd_streaming_matches_bruteforce():
    A, D = make_synthetic(d=8, n=40, seed=7)
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    feed_in_blocks(acc, A, D, [10, 20, 10])
    assert torch.allclose(acc.Gd, D.T @ D, rtol=1e-12, atol=1e-12)
    assert torch.allclose(acc.s, torch.trace(D.T @ D), rtol=1e-12)  # s = tr(Gd)


def test_delta_survival_full_basis_is_one_and_partial_matches_bruteforce():
    A, D = make_synthetic(d=10, n=50, seed=8)
    acc = GramAccumulator(10, device=DEV, dtype=DT)
    acc.update(A, D)
    # full orthonormal basis captures all δ energy
    full = torch.eye(10, dtype=DT)
    assert abs(acc.delta_survival(full) - 1.0) < 1e-10
    # partial basis: tr(Vᵀ Gd V)/s equals Σ‖VVᵀδ‖²/Σ‖δ‖²
    V = acc.manifold_basis(k=3, which="base")
    proj = (D @ V) @ V.T
    expected = float((proj * proj).sum() / (D * D).sum())
    assert abs(acc.delta_survival(V) - expected) < 1e-9


def test_manifold_basis_lora_matches_lora_gram_eig():
    A, D = make_synthetic(d=8, n=40, seed=9)
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    acc.update(A, D)
    Alora = A + D
    V = acc.manifold_basis(k=4, which="lora")
    _ev, evec = torch.linalg.eigh(Alora.T @ Alora)
    assert torch.allclose(V.abs(), evec[:, -4:].abs(), atol=1e-8)  # same span (sign-free)


def test_blockwise_equals_single_update():
    A, D = make_synthetic(d=6, n=30, seed=3)
    one = GramAccumulator(6, device=DEV, dtype=DT)
    one.update(A, D)
    many = GramAccumulator(6, device=DEV, dtype=DT)
    feed_in_blocks(many, A, D, [5, 5, 5, 5, 5, 5])
    assert torch.allclose(one.G, many.G) and torch.allclose(one.P, many.P)
    assert torch.allclose(one.s, many.s) and one.n_tokens == many.n_tokens == 30


# --- Anchor (b): closed-form RSS vs brute force, both regimes -----------------------

def _bruteforce_rss(W, A, D):
    resid = A @ W.T - D
    return float((resid * resid).sum())


@pytest.mark.parametrize("d,n", [(12, 5), (5, 40)])  # d>n and d<n
def test_rss_matches_bruteforce(d, n):
    A, D = make_synthetic(d=d, n=n, seed=4)
    acc = GramAccumulator(d, device=DEV, dtype=DT)
    acc.update(A, D)
    fit = acc.solve(lam=1.3)
    assert fit.rss == pytest.approx(_bruteforce_rss(fit.W, A, D), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("d,n", [(12, 5), (5, 40)])
def test_W_equals_direct_ridge_solution(d, n):
    A, D = make_synthetic(d=d, n=n, seed=5)
    acc = GramAccumulator(d, device=DEV, dtype=DT)
    acc.update(A, D)
    lam = 0.7
    G = (A.T @ A).numpy()
    P = (A.T @ D).numpy()
    Wt_direct = np.linalg.solve(G + lam * np.eye(d), P)   # (G+λI)⁻¹ P = Wᵀ
    W_direct = torch.tensor(Wt_direct.T)
    assert torch.allclose(acc.solve(lam).W, W_direct, rtol=1e-8, atol=1e-8)


# --- Anchor (c): two-form RSS agreement (the transpose-convention guard) ------------

@pytest.mark.parametrize("d,n", [(12, 5), (5, 40)])
@pytest.mark.parametrize("lam", [0.1, 1.0, 10.0, 100.0])
def test_rss_two_form_agreement(d, n, lam):
    A, D = make_synthetic(d=d, n=n, seed=6)
    acc = GramAccumulator(d, device=DEV, dtype=DT)
    acc.update(A, D)
    fit = acc.solve(lam)
    tol = 1e-8 * max(fit.s, 1.0)
    assert fit.crosscheck_abs_err < tol
    assert fit.rss == pytest.approx(fit.rss_crosscheck, abs=tol)


# --- Anchor (d): limits ------------------------------------------------------------

def test_lambda_to_zero_gives_ols_r2():
    # d < n so G = AᵀA is invertible; tiny λ ⇒ R² → tr(Pᵀ G⁻¹ P)/s.
    A, D = make_synthetic(d=5, n=60, seed=7, noise=0.3)
    acc = GramAccumulator(5, device=DEV, dtype=DT)
    acc.update(A, D)
    fit = acc.solve(lam=1e-10)
    G = A.T @ A
    P = A.T @ D
    ols_frac = float((P * torch.linalg.solve(G, P)).sum()) / float(acc.s)
    assert fit.r2_insample == pytest.approx(ols_frac, rel=1e-6, abs=1e-6)


def test_lambda_to_infinity_gives_zero():
    A, D = make_synthetic(d=8, n=30, seed=8)
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    acc.update(A, D)
    fit = acc.solve(lam=1e12)
    assert fit.r2_insample == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(fit.W, torch.zeros_like(fit.W), atol=1e-6)


# --- Held-out scoring --------------------------------------------------------------

def test_score_heldout_matches_bruteforce():
    A_tr, D_tr = make_synthetic(d=10, n=50, seed=9)
    A_te, D_te = make_synthetic(d=10, n=23, seed=10)
    tr = GramAccumulator(10, device=DEV, dtype=DT)
    tr.update(A_tr, D_tr)
    W = tr.solve(lam=2.0).W
    te = GramAccumulator(10, device=DEV, dtype=DT)
    te.update(A_te, D_te)
    score = te.score(W)
    assert score.rss_te == pytest.approx(_bruteforce_rss(W, A_te, D_te), rel=1e-9, abs=1e-9)


def test_heldout_r2_peaks_at_interior_lambda():
    # Same generative law for train and test ⇒ held-out R² peaks at an interior λ*.
    d, n = 20, 200
    A_tr, D_tr = make_synthetic(d=d, n=n, seed=11, noise=1.0)
    A_te, D_te = make_synthetic(d=d, n=n, seed=12, noise=1.0)
    tr = GramAccumulator(d, device=DEV, dtype=DT); tr.update(A_tr, D_tr)
    te = GramAccumulator(d, device=DEV, dtype=DT); te.update(A_te, D_te)
    lambdas = np.logspace(-3, 5, 25)
    r2 = [te.score(tr.solve(float(l)).W).r2_te for l in lambdas]
    best = int(np.argmax(r2))
    assert 0 < best < len(lambdas) - 1


# --- Plumbing ----------------------------------------------------------------------

def test_state_dict_roundtrip():
    A, D = make_synthetic(d=7, n=20, seed=13)
    acc = GramAccumulator(7, device=DEV, dtype=DT, layer=21)
    acc.update(A, D)
    back = GramAccumulator.from_state_dict(acc.state_dict(), device=DEV)
    assert back.layer == 21 and back.n_tokens == 20 and back.dim == 7
    assert torch.allclose(back.G, acc.G) and torch.allclose(back.P, acc.P)
    assert torch.allclose(back.s, acc.s)


def test_merge_equals_concatenation():
    A1, D1 = make_synthetic(d=6, n=15, seed=14)
    A2, D2 = make_synthetic(d=6, n=21, seed=15)
    merged = GramAccumulator(6, device=DEV, dtype=DT); merged.update(A1, D1)
    other = GramAccumulator(6, device=DEV, dtype=DT); other.update(A2, D2)
    merged.merge(other)
    concat = GramAccumulator(6, device=DEV, dtype=DT)
    concat.update(torch.cat([A1, A2]), torch.cat([D1, D2]))
    assert torch.allclose(merged.G, concat.G) and torch.allclose(merged.P, concat.P)
    assert torch.allclose(merged.s, concat.s) and merged.n_tokens == 36


def test_update_shape_mismatch_raises():
    acc = GramAccumulator(8, device=DEV, dtype=DT)
    with pytest.raises(ValueError):
        acc.update(torch.randn(4, 8, dtype=DT), torch.randn(3, 8, dtype=DT))
    with pytest.raises(ValueError):
        acc.update(torch.randn(4, 7, dtype=DT), torch.randn(4, 7, dtype=DT))
