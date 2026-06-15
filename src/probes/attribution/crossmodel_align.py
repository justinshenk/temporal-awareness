"""Cross-model LoReFT transplant: align two models' residual streams and carry an edit across.

A donor model owns a *trained* LoReFT edit ``δ_D``; a recipient model never saw it. Because both
models share a tokenizer, the same prompt produces row-corresponding residual streams, so we can fit
a per-layer linear bridge between their spaces and apply the donor's edit to the recipient.

The bridge is an **orthogonal Procrustes** map recipient→donor. From centered paired rows
``X_Rᶜ = X_R − μ_R`` and ``X_Dᶜ = X_D − μ_D`` we take ``A = U Vᵀ`` where ``U Σ Vᵀ = SVD(X_Dᶜᵀ X_Rᶜ)``,
so ``A (h_R − μ_R) ≈ (h_D − μ_D)`` with ``A`` orthonormal (inverse ``Aᵀ``). Only per-layer ``d×d``
sufficient statistics are accumulated (:class:`CrossCovStats`), never raw rows — like
:class:`~src.probes.attribution.ridge_steering_map.PairStats`.

The transplant operator (:class:`TransplantEdit`, a drop-in for
:class:`~src.probes.attribution.loreft_intervention.PositionEditHook`, exactly like ``RidgeEdit``)
maps a recipient row into the donor's space, applies the donor's own LoReFT edit there, and projects
the edit back::

    ĥ_D = μ_D + A (h_R − μ_R);   δ_D = LoReFT_L(ĥ_D) − ĥ_D;   h_R' = h_R + Aᵀ δ_D

The **affine bridge** relaxes the orthogonality constraint: :func:`fit_affine_bridge` ridge-fits a
general forward map ``W_F`` (``ĥ_Dᶜ = h_Rᶜ W_F``) and a reverse map ``W_G`` (``ĥ_Rᶜ = h_Dᶜ W_G``)
from the same sufficient statistics, and :class:`AffineTransplantEdit` applies the donor edit
through them (deltas are tangent vectors — they transform by the linear part ``W_G``, not the affine
map). Procrustes is the special case ``W_F = Aᵀ, W_G = A``. Comparing :func:`affine_residual` with
:func:`procrustes_residual` on the same stats asks whether a misalignment is an orthogonality
artifact or intrinsic to the paired rows.

When the donor edit is zero the operators are the identity. Only fit/edit geometry lives here
(unit-testable on tiny ``d``); the model forwards that gather the pairs live in
``scripts/attribution/transplant_loreft.py``. Rows are stored as ``(n, d)`` matrices, so the linear
maps act on the right (``A (h−μ)`` becomes ``(h−μ) @ Aᵀ`` and ``Aᵀ δ`` becomes ``δ @ A``).
"""

from __future__ import annotations

import torch
from torch import nn


class CrossCovStats:
    """Streaming sufficient statistics for one layer's recipient↔donor bridge fits.

    Accumulates ``n``, the row sums ``sr, sd``, the ``d×d`` cross-product ``c = X_Dᵀ X_R`` and the
    two grams ``gr = X_Rᵀ X_R``, ``gd = X_Dᵀ X_D`` in float64, so :func:`fit_procrustes` and
    :func:`fit_affine_bridge` can recover the centered (cross-)covariances and means — and the
    residual functions the fit errors — without ever storing raw rows. The squared norms ``ssr``,
    ``ssd`` are the gram traces. Chunked updates are exactly equivalent to a one-shot update (the
    accumulators are additive).
    """

    def __init__(self, d: int, device: torch.device | str = "cpu"):
        self.d = d
        self.n = 0
        self.sr = torch.zeros(d, dtype=torch.float64, device=device)
        self.sd = torch.zeros(d, dtype=torch.float64, device=device)
        self.c = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.gr = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.gd = torch.zeros(d, d, dtype=torch.float64, device=device)

    def update(self, rows_recipient: torch.Tensor, rows_donor: torch.Tensor) -> None:
        xr = rows_recipient.detach().to(self.c.device, torch.float64)
        xd = rows_donor.detach().to(self.c.device, torch.float64)
        if xr.shape != xd.shape:
            raise ValueError(f"paired rows must match: {tuple(xr.shape)} vs {tuple(xd.shape)}")
        self.n += xr.shape[0]
        self.sr += xr.sum(0)
        self.sd += xd.sum(0)
        self.c += xd.T @ xr
        self.gr += xr.T @ xr
        self.gd += xd.T @ xd

    def state(self) -> dict:
        """Sufficient statistics on CPU, so downstream fits are device-independent."""
        return {"n": torch.tensor(float(self.n), dtype=torch.float64),
                "sr": self.sr.cpu(), "sd": self.sd.cpu(), "c": self.c.cpu(),
                "gr": self.gr.cpu(), "gd": self.gd.cpu(),
                "ssr": self.gr.trace().cpu(),                       # Σ‖h_R‖²  (for the residuals)
                "ssd": self.gd.trace().cpu()}                       # Σ‖h_D‖²


def _stats_from_rows(rows_recipient: torch.Tensor, rows_donor: torch.Tensor) -> dict:
    stats = CrossCovStats(rows_recipient.shape[1])
    stats.update(rows_recipient, rows_donor)
    return stats.state()


def fit_procrustes(stats_or_rows):
    """Recipient→donor orthogonal Procrustes map from sufficient stats or a ``(X_R, X_D)`` pair.

    Accepts either a :meth:`CrossCovStats.state` dict or a tuple of raw row tensors. With centered
    cross-covariance ``M = c/n − μ_D μ_Rᵀ`` and ``U Σ Vᵀ = SVD(M)``, returns ``A = U Vᵀ``
    (orthonormal, ``A (h_R − μ_R) ≈ h_D − μ_D``) plus ``mean_R`` and ``mean_D``, all float32.
    """
    stats = _stats_from_rows(*stats_or_rows) if isinstance(stats_or_rows, tuple) else stats_or_rows
    n = float(stats["n"])
    mean_r = stats["sr"] / n
    mean_d = stats["sd"] / n
    cross = stats["c"] / n - torch.outer(mean_d, mean_r)            # E[(h_D−μ_D)(h_R−μ_R)ᵀ]
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    A = u @ vh
    return A.to(torch.float32), mean_r.to(torch.float32), mean_d.to(torch.float32)


def procrustes_residual(stats: dict, A: torch.Tensor, mean_R: torch.Tensor,
                        mean_D: torch.Tensor) -> float:
    """Relative ``‖X_Dᶜ − A X_Rᶜ‖ / ‖X_Dᶜ‖`` over the fit set, from sufficient stats alone.

    With ``M = c/n − μ_D μ_Rᵀ`` and orthonormal ``A``, the centered Procrustes error is
    ``E‖(h_D−μ_D) − A(h_R−μ_R)‖² = E‖h_D−μ_D‖² + E‖h_R−μ_R‖² − 2 tr(Aᵀ M)``. The per-model centered
    energies come from the accumulated squared norms: ``E‖h_R−μ_R‖² = ssr/n − ‖μ_R‖²`` (likewise for
    D). The denominator ``‖X_Dᶜ‖`` is the donor centered energy. A value near 0 means ``A`` aligns the
    two streams well; near 1 means the rotation cannot reconcile them.
    """
    n = float(stats["n"])
    mean_r = mean_R.to(torch.float64)
    mean_d = mean_D.to(torch.float64)
    cross = stats["c"] / n - torch.outer(mean_d, mean_r)
    align = float((A.to(torch.float64) * cross).sum())             # tr(Aᵀ M)
    var_r = float(stats["ssr"]) / n - float(mean_r @ mean_r)       # E‖h_R−μ_R‖²
    var_d = float(stats["ssd"]) / n - float(mean_d @ mean_d)       # E‖h_D−μ_D‖²
    resid = max(var_d + var_r - 2.0 * align, 0.0)
    return (resid / var_d) ** 0.5 if var_d > 0 else 0.0


def _truncate_rank(W: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(W, full_matrices=False)
    return u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]


def fit_affine_bridge(stats: dict, lam: float, rank: int | None = None):
    """Ridge-fit both directions of a general affine bridge from sufficient stats.

    Solves the centered normal equations ``(G_R + λ̃I) W_F = M̃ᵀ`` and ``(G_D + λ̃I) W_G = M̃`` with
    ``G_R/G_D`` the centered grams, ``M̃ = c/n − μ_D μ_Rᵀ`` the centered cross-covariance and ``λ̃ =
    lam · mean(diag(G))`` (``lam`` is *relative* to the per-side activation scale). ``rank`` SVD-
    truncates both maps (``None`` = full rank). Returns ``W_F`` (``ĥ_Dᶜ = h_Rᶜ W_F``), ``W_G``
    (``ĥ_Rᶜ = h_Dᶜ W_G``), ``mean_R``, ``mean_D``, all float32.
    """
    n = float(stats["n"])
    mean_r = stats["sr"] / n
    mean_d = stats["sd"] / n
    cross = stats["c"] / n - torch.outer(mean_d, mean_r)            # M̃ = E[(h_D−μ_D)(h_R−μ_R)ᵀ]
    cov_r = stats["gr"] / n - torch.outer(mean_r, mean_r)
    cov_d = stats["gd"] / n - torch.outer(mean_d, mean_d)
    d = mean_r.shape[0]
    eye = torch.eye(d, dtype=torch.float64, device=mean_r.device)
    W_F = torch.linalg.solve(cov_r + lam * cov_r.diagonal().mean() * eye, cross.T)
    W_G = torch.linalg.solve(cov_d + lam * cov_d.diagonal().mean() * eye, cross)
    if rank is not None and rank < d:
        W_F = _truncate_rank(W_F, rank)
        W_G = _truncate_rank(W_G, rank)
    return (W_F.to(torch.float32), W_G.to(torch.float32),
            mean_r.to(torch.float32), mean_d.to(torch.float32))


def affine_residual(stats: dict, W_F: torch.Tensor, mean_R: torch.Tensor,
                    mean_D: torch.Tensor) -> float:
    """Relative ``‖X_Dᶜ − X_Rᶜ W_F‖ / ‖X_Dᶜ‖`` over the stats' rows, centered by the *given* means.

    Unlike :func:`procrustes_residual` this stays exact when ``mean_R``/``mean_D`` come from a
    different (training) set than ``stats`` — the held-out protocol: fit on train stats, score on
    val stats. With ``μ`` the given means and ``K = E[h_Rᶜᵀ h_Dᶜ]``, ``S_R = E[h_Rᶜᵀ h_Rᶜ]``
    (both expanded about ``μ``), the error is ``E‖h_Dᶜ‖² − 2 tr(W_Fᵀ K) + tr(W_Fᵀ S_R W_F)``.
    """
    n = float(stats["n"])
    mu_r = mean_R.to(torch.float64)
    mu_d = mean_D.to(torch.float64)
    w = W_F.to(torch.float64)
    er = stats["sr"] / n                                           # E[h_R]
    ed = stats["sd"] / n
    # E‖h_D − μ_D‖² and the val-rows second moments expanded about the (train) means:
    var_d = float(stats["ssd"]) / n - 2.0 * float(mu_d @ ed) + float(mu_d @ mu_d)
    S_R = (stats["gr"] / n - torch.outer(er, mu_r) - torch.outer(mu_r, er)
           + torch.outer(mu_r, mu_r))                              # E[h_Rᶜᵀ h_Rᶜ]
    K = (stats["c"].T / n - torch.outer(er, mu_d) - torch.outer(mu_r, ed)
         + torch.outer(mu_r, mu_d))                                # E[h_Rᶜᵀ h_Dᶜ]
    cross = float((w * K).sum())                                   # tr(W_Fᵀ K)
    fit_energy = float((w * (S_R @ w)).sum())                      # tr(W_Fᵀ S_R W_F)
    resid = max(var_d - 2.0 * cross + fit_energy, 0.0)
    return (resid / var_d) ** 0.5 if var_d > 0 else 0.0


class TransplantEdit(nn.Module):
    """Recipient-space transplant of a donor LoReFT edit through a Procrustes bridge.

    Wraps a donor :class:`LoReFTIntervention`. Buffers the orthonormal map ``A`` (recipient→donor)
    and the means ``mean_R``/``mean_D``. ``forward(h_R)`` lifts ``h_R`` into donor space, takes the
    donor's own edit there, and projects it back, returning the post-edit recipient row::

        ĥ_D = mean_D + (h_R − mean_R) @ Aᵀ
        δ_D = donor(ĥ_D) − ĥ_D
        return h_R + δ_D @ A

    Drop-in for :class:`PositionEditHook`. A zero donor edit makes it the identity.
    """

    def __init__(self, donor: nn.Module, A: torch.Tensor, mean_R: torch.Tensor,
                 mean_D: torch.Tensor):
        super().__init__()
        self.donor = donor
        self.register_buffer("A", A)
        self.register_buffer("mean_R", mean_R)
        self.register_buffer("mean_D", mean_D)

    def forward(self, h_R: torch.Tensor) -> torch.Tensor:
        h_D = self.mean_D + (h_R - self.mean_R) @ self.A.T
        delta_D = self.donor(h_D) - h_D
        return h_R + delta_D @ self.A


class AffineTransplantEdit(nn.Module):
    """Recipient-space transplant of a donor LoReFT edit through a general affine bridge.

    Like :class:`TransplantEdit` but with independent ridge-fit forward/backward linear maps
    instead of a single rotation::

        ĥ_D = mean_D + (h_R − mean_R) @ W_F
        δ_D = donor(ĥ_D) − ĥ_D
        return h_R + δ_D @ W_G

    Drop-in for :class:`PositionEditHook`. A zero donor edit makes it the identity.
    """

    def __init__(self, donor: nn.Module, W_F: torch.Tensor, W_G: torch.Tensor,
                 mean_R: torch.Tensor, mean_D: torch.Tensor):
        super().__init__()
        self.donor = donor
        self.register_buffer("W_F", W_F)
        self.register_buffer("W_G", W_G)
        self.register_buffer("mean_R", mean_R)
        self.register_buffer("mean_D", mean_D)

    def forward(self, h_R: torch.Tensor) -> torch.Tensor:
        h_D = self.mean_D + (h_R - self.mean_R) @ self.W_F
        delta_D = self.donor(h_D) - h_D
        return h_R + delta_D @ self.W_G
