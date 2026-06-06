"""Streamed primal-ridge accumulators for ``W·a ≈ δ`` over a token stream.

Holds the three sufficient statistics of a multivariate ridge regression of the
per-token LoRA shift ``δ_t`` on the base activation ``a_t``::

    G = Σ a_t a_tᵀ   (d, d)     P = Σ a_t δ_tᵀ   (d, d)     s = Σ ‖δ_t‖²   (scalar)

with rows = tokens (``A``, ``D`` are ``(n_tok, d)``). The primal ``d×d`` form is used
because the token count ``N`` (~10k CoT tokens) far exceeds ``d`` (4096); the dual
``N×N`` form would be infeasible. The map is never materialized via an explicit
inverse — every quantity comes from a Cholesky solve on the PD matrix ``G + λI``.

Closed forms (rows-as-tokens convention, see module tests for the brute-force checks)::

    S   = (G + λI)⁻¹                     Wᵀ = S P          W = Pᵀ S
    RSS = s − 2·tr(WP) + tr(WGWᵀ)        R² = 1 − RSS/s
    RSS = s − tr(PᵀSP) − λ·tr(PᵀS²P)     (algebraic cross-check — must match)

Limits emerge naturally: λ→0 ⇒ R² → tr(Pᵀ G⁻¹ P)/s (OLS explained fraction);
λ→∞ ⇒ W → 0, R² → 0.
"""

from __future__ import annotations

import torch

from src.probes.attribution.ridge_fit import HeldoutScore, RidgeFit


class GramAccumulator:
    """Streaming ``G``, ``P``, ``s`` for one layer's primal ridge fit.

    All state is kept in ``dtype`` (default float64) on ``device``. Feed blocks of
    aligned activations/shifts with :meth:`update`; close the form with :meth:`solve`
    (train) and :meth:`score` (held-out).
    """

    def __init__(self, dim: int, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float64, layer: int = -1):
        self.dim = dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.layer = layer
        self.G = torch.zeros(dim, dim, dtype=dtype, device=self.device)
        self.P = torch.zeros(dim, dim, dtype=dtype, device=self.device)
        self.s = torch.zeros((), dtype=dtype, device=self.device)
        self.n_tokens = 0

    def update(self, A_block: torch.Tensor, D_block: torch.Tensor) -> None:
        """Accumulate one block. ``A_block``/``D_block`` are ``(n_tok, d)`` aligned token-wise."""
        if A_block.shape != D_block.shape:
            raise ValueError(f"A_block {tuple(A_block.shape)} != D_block {tuple(D_block.shape)}")
        if A_block.ndim != 2 or A_block.shape[1] != self.dim:
            raise ValueError(f"expected (n_tok, {self.dim}), got {tuple(A_block.shape)}")
        A = A_block.to(self.device, self.dtype)
        D = D_block.to(self.device, self.dtype)
        self.G += A.T @ A
        self.P += A.T @ D
        self.s += (D * D).sum()
        self.n_tokens += A.shape[0]

    def _chol(self, lam: float) -> torch.Tensor:
        eye = torch.eye(self.dim, dtype=self.dtype, device=self.device)
        return torch.linalg.cholesky(self.G + lam * eye)

    def solve(self, lam: float) -> RidgeFit:
        """Closed-form fit at ``lam``: returns ``W`` plus in-sample R² and the cross-check gap."""
        L = self._chol(lam)
        Wt = torch.cholesky_solve(self.P, L)        # S P  = Wᵀ              (d, d)
        S2P = torch.cholesky_solve(Wt, L)           # S (S P) = S² P         (d, d)
        s = float(self.s)

        # standard form: RSS = s − 2 tr(WP) + tr(WGWᵀ),  with Wᵀ = Wt
        tr_WP = float((self.P * Wt).sum())          # tr(Pᵀ S P) = tr(WP)
        tr_WGWt = float(((self.G @ Wt) * Wt).sum())  # tr(Wᵀᵀ G Wᵀ) = tr(WGWᵀ)
        rss = s - 2.0 * tr_WP + tr_WGWt

        # independent cross-check: RSS = s − tr(PᵀSP) − λ tr(PᵀS²P)
        tr_PSP = float((self.P * Wt).sum())
        tr_PS2P = float((self.P * S2P).sum())
        rss_cross = s - tr_PSP - lam * tr_PS2P

        r2 = 1.0 - rss / s if s > 0 else 0.0
        return RidgeFit(
            layer=self.layer, lam=float(lam), n_tokens=self.n_tokens, s=s,
            rss=rss, rss_crosscheck=rss_cross, crosscheck_abs_err=abs(rss - rss_cross),
            r2_insample=r2, W=Wt.T.contiguous(),
        )

    def score(self, W: torch.Tensor) -> HeldoutScore:
        """Score a (train-fit) dense ``W`` (d, d) on these held-out accumulators."""
        Wd = W.to(self.device, self.dtype)
        s = float(self.s)
        tr_WP = float((Wd * self.P.T).sum())             # tr(W P)
        tr_WGWt = float(((Wd @ self.G) * Wd).sum())      # tr(W G Wᵀ)
        rss_te = s - 2.0 * tr_WP + tr_WGWt
        r2_te = 1.0 - rss_te / s if s > 0 else 0.0
        return HeldoutScore(layer=self.layer, lam=0.0, n_tokens=self.n_tokens,
                            s_te=s, rss_te=rss_te, r2_te=r2_te)

    def merge(self, other: "GramAccumulator") -> None:
        """Add another accumulator's statistics in place (for sharded collection)."""
        if other.dim != self.dim:
            raise ValueError(f"dim mismatch {other.dim} != {self.dim}")
        self.G += other.G.to(self.device, self.dtype)
        self.P += other.P.to(self.device, self.dtype)
        self.s += other.s.to(self.device, self.dtype)
        self.n_tokens += other.n_tokens

    def state_dict(self) -> dict:
        return {"dim": self.dim, "layer": self.layer, "n_tokens": self.n_tokens,
                "G": self.G.cpu(), "P": self.P.cpu(), "s": self.s.cpu()}

    @classmethod
    def from_state_dict(cls, d: dict, device: str | torch.device = "cuda") -> "GramAccumulator":
        acc = cls(dim=d["dim"], device=device, dtype=d["G"].dtype, layer=d.get("layer", -1))
        acc.G = d["G"].to(acc.device, acc.dtype)
        acc.P = d["P"].to(acc.device, acc.dtype)
        acc.s = d["s"].to(acc.device, acc.dtype)
        acc.n_tokens = int(d["n_tokens"])
        return acc
