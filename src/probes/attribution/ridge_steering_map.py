"""Distil a trained LoReFT edit into a ridge-fit rank-r conditional affine map ("ridge steering map").

LoReFT learns a per-position edit ``h ↦ h'`` whose *value* is set by task loss; this module asks a
simpler question: how well can that edit be reproduced by a single conditional affine map of the
residual stream? For each layer we collect paired rows — ``h_base`` (clean forward) and ``h_steered``
(the LoReFT post-edit value) — over the f7+l7 prompt positions, and fit the delta
``δ = h_steered − h_base`` from ``h_base`` by ridge regression, truncated to rank r:

    δ̂(h) = (h − mean_x) @ B,   B ≈ U[:, :r] · S[:r] · Vh[:r]  (P=U S, Q=Vh)

Applied at inference as ``h' = h + δ̂(h)``, :class:`RidgeEdit` is a drop-in for
:class:`LoReFTIntervention` in :class:`PositionEditHook`. Memory stays bounded: only per-layer
``d×d`` sufficient statistics are accumulated (:class:`PairStats`), never raw rows. Only the
fit/edit geometry lives here (unit-testable on tiny d); the model forwards that gather the pairs live
in ``scripts/attribution/fit_ridge_steering.py``.
"""

from __future__ import annotations

import torch
from torch import nn


class PairStats:
    """Streaming sufficient statistics for one layer's ``(h_base, δ)`` ridge fit.

    Accumulates ``n``, the sums ``sx, sd``, the ``d×d`` cross-products ``xtx, xtd`` and the scalar
    total δ-energy ``ssd = Σ‖δ‖²`` in float64, so :func:`fit_ridge_rank` can recover the centered
    covariances and :func:`fit_residual` the relative fit error without ever storing raw rows.
    Chunked updates are exactly equivalent to a one-shot update (the accumulators are additive).
    """

    def __init__(self, d: int, device: torch.device | str = "cpu"):
        self.d = d
        self.n = 0
        self.sx = torch.zeros(d, dtype=torch.float64, device=device)
        self.sd = torch.zeros(d, dtype=torch.float64, device=device)
        self.xtx = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.xtd = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.ssd = 0.0

    def update(self, h_base: torch.Tensor, h_steered: torch.Tensor) -> None:
        x = h_base.detach().to(self.xtx.device, torch.float64)
        delta = (h_steered.detach() - h_base.detach()).to(self.xtx.device, torch.float64)
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sd += delta.sum(0)
        self.xtx += x.T @ x
        self.xtd += x.T @ delta
        self.ssd += float((delta * delta).sum())

    def state(self) -> dict:
        """Sufficient statistics on CPU, so downstream fits are device-independent."""
        return {"n": torch.tensor(float(self.n), dtype=torch.float64),
                "sx": self.sx.cpu(), "sd": self.sd.cpu(),
                "xtx": self.xtx.cpu(), "xtd": self.xtd.cpu(),
                "ssd": torch.tensor(self.ssd, dtype=torch.float64)}


def fit_ridge_rank(stats: dict, rank: int, ridge_lambda: float):
    """Rank-``rank`` ridge map of ``δ`` on ``h_base`` from sufficient statistics ``stats``.

    Solves the centered ridge system ``B = (cov_xx + λI)⁻¹ cov_xd`` in float64, then truncates ``B``
    to ``rank`` via SVD and factors it as ``P=U S``, ``Q=Vh`` so that ``δ̂ = (h − mean_x) @ P @ Q +
    mean_d``. Returns ``P:(d,rank)``, ``Q:(rank,d)``, ``mean_x:(d,)``, ``mean_d:(d,)`` in float32.
    """
    n = float(stats["n"])
    mean_x = stats["sx"] / n
    mean_d = stats["sd"] / n
    cov_xx = stats["xtx"] / n - torch.outer(mean_x, mean_x)
    cov_xd = stats["xtd"] / n - torch.outer(mean_x, mean_d)
    d = mean_x.shape[0]
    ridged = cov_xx + ridge_lambda * torch.eye(d, dtype=torch.float64)
    b_full = torch.linalg.solve(ridged, cov_xd)
    u, s, vh = torch.linalg.svd(b_full, full_matrices=False)
    P = (u[:, :rank] * s[:rank]).to(torch.float32)
    Q = vh[:rank, :].to(torch.float32)
    return P, Q, mean_x.to(torch.float32), mean_d.to(torch.float32)


def fit_residual(stats: dict, P: torch.Tensor, Q: torch.Tensor, mean_x: torch.Tensor,
                 mean_d: torch.Tensor) -> float:
    """Relative ``‖δ − δ̂‖ / ‖δ‖`` over the fit set, computed from sufficient statistics alone.

    With ``δc = δ − mean_d``, ``xc = h − mean_x`` and ``B̂ = P @ Q``, the per-row prediction is
    ``δ̂ = xc @ B̂ + mean_d``, so ``E‖δ − δ̂‖² = E‖δc‖² − 2 tr(B̂ᵀ C_xd) + tr(B̂ᵀ C_xx B̂)`` with
    ``E‖δc‖² = ssd/n − ‖mean_d‖²``. The denominator ``‖δ‖`` uses the raw δ-energy ``ssd/n``.
    """
    n = float(stats["n"])
    mx = mean_x.to(torch.float64)
    md = mean_d.to(torch.float64)
    bhat = P.to(torch.float64) @ Q.to(torch.float64)
    cov_xx = stats["xtx"] / n - torch.outer(mx, mx)
    cov_xd = stats["xtd"] / n - torch.outer(mx, md)
    raw_dd = float(stats["ssd"]) / n                              # E‖δ‖²
    centered_dd = raw_dd - float(md @ md)                         # E‖δ − mean_d‖²
    cross = float((bhat * cov_xd).sum())                          # tr(B̂ᵀ C_xd)
    fit_energy = float((bhat.T @ cov_xx * bhat.T).sum())          # tr(B̂ᵀ C_xx B̂)
    resid = max(centered_dd - 2.0 * cross + fit_energy, 0.0)
    return (resid / raw_dd) ** 0.5 if raw_dd > 0 else 0.0


class RidgeEdit(nn.Module):
    """Conditional affine edit ``h ↦ h + (((h − mean_x) @ P) @ Q + mean_d)``.

    A drop-in for :class:`LoReFTIntervention` in :class:`PositionEditHook`: ``forward`` returns the
    absolute post-edit residual. Zero ``P``/``Q``/``mean_d`` make it the identity.
    """

    def __init__(self, P: torch.Tensor, Q: torch.Tensor, mean_x: torch.Tensor,
                 mean_d: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)
        self.register_buffer("Q", Q)
        self.register_buffer("mean_x", mean_x)
        self.register_buffer("mean_d", mean_d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (((h - self.mean_x) @ self.P) @ self.Q + self.mean_d)
