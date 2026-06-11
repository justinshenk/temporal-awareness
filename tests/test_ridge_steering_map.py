"""Tests for the ridge steering map (CPU, no network, fast).

Distilling a trained LoReFT edit into a rank-r conditional affine map reduces to a ridge fit of the
delta ``δ = h_steered − h_base`` from ``h_base``, truncated to rank r. These tests pin the crux
properties in miniature: an exactly-rank-r synthetic map is recovered (δ̂≈δ, RidgeEdit≈h+δ); fitting
below the true rank leaves strictly more residual than fitting at it; the streaming sufficient
statistics are accumulation-order/chunk invariant; and ``RidgeEdit`` is shape-correct and the
identity when its weights are zero.
"""

from __future__ import annotations

import torch

from src.probes.attribution.ridge_steering_map import (
    PairStats,
    RidgeEdit,
    fit_residual,
    fit_ridge_rank,
)


def make_rank_r_problem(n: int, d: int, r: int, seed: int = 0):
    """Synthetic pairs with ``δ = (h − mx) @ B_true + c``, ``B_true`` exactly rank r."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g, dtype=torch.float32)
    left = torch.randn(d, r, generator=g, dtype=torch.float32)
    right = torch.randn(r, d, generator=g, dtype=torch.float32)
    b_true = left @ right                              # exactly rank r (r < d)
    mx = h.mean(0)
    c = torch.randn(d, generator=g, dtype=torch.float32)
    delta = (h - mx) @ b_true + c
    return h, delta


def test_recovers_exact_rank_r_map():
    """At the true rank with tiny ridge, δ̂≈δ and RidgeEdit(h)≈h+δ."""
    d, r, n = 16, 4, 4000
    h, delta = make_rank_r_problem(n, d, r, seed=1)
    stats = PairStats(d)
    stats.update(h, h + delta)
    P, Q, mean_x, mean_d = fit_ridge_rank(stats.state(), rank=r, ridge_lambda=1e-6)

    delta_hat = ((h - mean_x) @ P) @ Q + mean_d
    rel = (delta_hat - delta).norm() / delta.norm()
    assert rel < 1e-3, rel

    edit = RidgeEdit(P, Q, mean_x, mean_d)
    assert torch.allclose(edit(h), h + delta, atol=1e-2, rtol=1e-3)


def test_lower_rank_has_strictly_higher_residual():
    """Fitting below the true rank leaves strictly more residual than fitting at it."""
    d, r, n = 16, 6, 4000
    h, delta = make_rank_r_problem(n, d, r, seed=2)
    stats = PairStats(d)
    stats.update(h, h + delta)
    state = stats.state()

    def residual(rank: int) -> float:
        P, Q, mx, md = fit_ridge_rank(state, rank=rank, ridge_lambda=1e-6)
        delta_hat = ((h - mx) @ P) @ Q + md
        return float((delta_hat - delta).norm() / delta.norm())

    assert residual(r - 2) > residual(r) + 1e-4


def test_pairstats_chunked_equals_oneshot():
    """Sufficient stats from two chunks equal one-shot accumulation."""
    d, n = 16, 500
    h, delta = make_rank_r_problem(n, d, 4, seed=3)
    steered = h + delta

    one = PairStats(d)
    one.update(h, steered)

    two = PairStats(d)
    two.update(h[:200], steered[:200])
    two.update(h[200:], steered[200:])

    sa, sb = one.state(), two.state()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.allclose(sa[k].float(), sb[k].float(), atol=1e-3, rtol=1e-4), k


def test_fit_residual_matches_bruteforce():
    """The stats-only relative residual equals the direct ‖δ−δ̂‖/‖δ‖ over the rows."""
    d, r, n = 16, 5, 3000
    h, delta = make_rank_r_problem(n, d, r, seed=4)
    stats = PairStats(d)
    stats.update(h, h + delta)
    state = stats.state()
    for rank in (2, 4, 5):
        P, Q, mx, md = fit_ridge_rank(state, rank=rank, ridge_lambda=1e-3)
        delta_hat = ((h - mx) @ P) @ Q + md
        brute = float((delta_hat - delta).norm() / delta.norm())
        assert abs(fit_residual(state, P, Q, mx, md) - brute) < 1e-3, (rank, brute)


def test_ridge_edit_shape_and_zero_is_identity():
    """Output shape (b, d); zero P/Q/mean_d makes RidgeEdit the identity."""
    d, r, b = 16, 4, 7
    h = torch.randn(b, d)
    edit = RidgeEdit(torch.zeros(d, r), torch.zeros(r, d), torch.randn(d), torch.zeros(d))
    out = edit(h)
    assert out.shape == (b, d)
    assert torch.allclose(out, h, atol=1e-6)
