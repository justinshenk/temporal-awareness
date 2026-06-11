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

When the donor edit is zero the operator is the identity. Only fit/edit geometry lives here
(unit-testable on tiny ``d``); the model forwards that gather the pairs live in
``scripts/attribution/transplant_loreft.py``. Rows are stored as ``(n, d)`` matrices, so the linear
maps act on the right (``A (h−μ)`` becomes ``(h−μ) @ Aᵀ`` and ``Aᵀ δ`` becomes ``δ @ A``).
"""

from __future__ import annotations

import torch
from torch import nn


class CrossCovStats:
    """Streaming sufficient statistics for one layer's recipient→donor Procrustes fit.

    Accumulates ``n``, the row sums ``sr, sd`` and the ``d×d`` cross-product ``c = X_Dᵀ X_R`` in
    float64, so :func:`fit_procrustes` can recover the centered cross-covariance and means without
    ever storing raw rows. Chunked updates are exactly equivalent to a one-shot update (the
    accumulators are additive).
    """

    def __init__(self, d: int, device: torch.device | str = "cpu"):
        self.d = d
        self.n = 0
        self.sr = torch.zeros(d, dtype=torch.float64, device=device)
        self.sd = torch.zeros(d, dtype=torch.float64, device=device)
        self.c = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.ssr = 0.0
        self.ssd = 0.0

    def update(self, rows_recipient: torch.Tensor, rows_donor: torch.Tensor) -> None:
        xr = rows_recipient.detach().to(self.c.device, torch.float64)
        xd = rows_donor.detach().to(self.c.device, torch.float64)
        if xr.shape != xd.shape:
            raise ValueError(f"paired rows must match: {tuple(xr.shape)} vs {tuple(xd.shape)}")
        self.n += xr.shape[0]
        self.sr += xr.sum(0)
        self.sd += xd.sum(0)
        self.c += xd.T @ xr
        self.ssr += float((xr * xr).sum())                         # Σ‖h_R‖²  (for the residual)
        self.ssd += float((xd * xd).sum())                         # Σ‖h_D‖²

    def state(self) -> dict:
        """Sufficient statistics on CPU, so downstream fits are device-independent."""
        return {"n": torch.tensor(float(self.n), dtype=torch.float64),
                "sr": self.sr.cpu(), "sd": self.sd.cpu(), "c": self.c.cpu(),
                "ssr": torch.tensor(self.ssr, dtype=torch.float64),
                "ssd": torch.tensor(self.ssd, dtype=torch.float64)}


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
