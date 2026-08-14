"""Result dataclasses for the primal per-token ridge attribution.

Lightweight, serializable records produced by :class:`GramAccumulator`. The dense
``W`` map is carried on :class:`RidgeFit` for downstream steering but is excluded
from ``to_dict``/``get_id`` automatically (``BaseSchema`` skips tensors).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.common.base_schema import BaseSchema


@dataclass
class RidgeFit(BaseSchema):
    """Closed-form primal ridge fit at one layer and one ``lambda``.

    ``rss``/``r2_insample`` are the standard form ``s - 2 tr(WP) + tr(WGWᵀ)``;
    ``rss_crosscheck`` is the independent two-form value ``s - tr(PᵀSP) - λ tr(PᵀS²P)``.
    They must agree to machine precision — ``crosscheck_abs_err`` is their gap and is
    the first TDD assertion / runtime invariant (catches a Pᵀ transpose mistake).
    """

    layer: int = -1
    lam: float = 0.0
    n_tokens: int = 0
    s: float = 0.0
    rss: float = 0.0
    rss_crosscheck: float = 0.0
    crosscheck_abs_err: float = 0.0
    r2_insample: float = 0.0
    W: torch.Tensor | None = None


@dataclass
class HeldoutScore(BaseSchema):
    """Held-out fit of a train ``W`` scored on disjoint accumulators (the verdict licensor)."""

    layer: int = -1
    lam: float = 0.0
    n_tokens: int = 0
    s_te: float = 0.0
    rss_te: float = 0.0
    r2_te: float = 0.0


@dataclass
class LayerSweep(BaseSchema):
    """A full lambda sweep at one layer: in-sample + held-out R², plus the chosen lambda*."""

    layer: int = -1
    lambdas: list[float] = field(default_factory=list)
    r2_insample: list[float] = field(default_factory=list)
    r2_te: list[float] = field(default_factory=list)
    crosscheck_abs_err: list[float] = field(default_factory=list)
    lam_star: float = 0.0
    r2_te_star: float = 0.0
    # The constant baseline: the share of δ's energy the single best FIXED vector already explains.
    # `r2_te` divides by the uncentred Σ‖δ‖², so wherever δ points mostly one way a map scores high
    # without capturing anything input-conditional — and how constant δ is happens to be the exact
    # property the register/procedure contrast is about, so cross-task `r2_te` comparisons are
    # confounded by it. `r2_te_centred = (r2_te − r2_const)/(1 − r2_const)` is that confound removed;
    # it needs no refit. None on trees collected before first moments were recorded.
    r2_const_te: float | None = None
    r2_te_centred: float | None = None
