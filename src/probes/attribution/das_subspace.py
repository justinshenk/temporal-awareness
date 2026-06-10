"""Distributed Alignment Search for the δ-subspace: pick the injected directions by task loss.

PCA bands select the injected δ-subspace by *variance* (eigenvectors of δδᵀ); that objective is
blind to the capability (the high-variance head recovers nothing). DAS instead learns an orthonormal
basis ``R`` (d×r) by gradient descent on a *behavioral* loss — make the base model, injected with
``a + Π_R(δ_true)`` at one layer, reproduce the LoRA's per-step greedy next-token decisions. The
injection is byte-identical to the PCA-band oracle (:func:`projected_injection` with ``V = R``), so a
DAS-R-vs-PCA-top-r comparison at matched rank isolates a single variable: how the subspace is chosen.

Only the geometry/loss glue lives here (unit-testable without PEFT); the model forwards that build the
cache and run the training/eval loop live in the GSM8K script.
"""

from __future__ import annotations

import torch
from torch import nn


class OrthoSubspace(nn.Module):
    """A learnable orthonormal ``d×r`` basis ``R`` (columns orthonormal via reduced QR).

    The raw parameter is unconstrained; :meth:`forward` returns its reduced-QR factor, so ``R`` is
    orthonormal at every step regardless of the optimizer (gradients flow through the QR). At ``r=d``
    the span is all of ``ℝ^d`` and the injection collapses to the full oracle.
    """

    def __init__(self, d: int, r: int, seed: int = 0):
        super().__init__()
        if not 0 < r <= d:
            raise ValueError(f"need 0 < r <= d, got r={r}, d={d}")
        g = torch.Generator().manual_seed(seed)
        self.raw = nn.Parameter(torch.randn(d, r, generator=g))

    def forward(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.raw, mode="reduced")
        return q


def inject_value(a: torch.Tensor, delta: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Injected residual ``a + Π_R(δ)`` — keep base ``a`` plus the R-subspace part of the true shift.

    Differentiable in ``R`` (so DAS can train it). ``a``/``delta`` are ``(n, d)`` cached residuals;
    ``R`` is ``(d, r)`` with orthonormal columns. At ``span(R)=ℝ^d`` this returns ``a + delta`` (the
    full oracle residual). Mirror of :func:`lockstep_oracle.projected_injection` written against the
    cached ``delta`` directly so the cache need not store ``lora_resid``.
    """
    return a + (delta @ R) @ R.T


def subspace_lm_loss(logits: torch.Tensor, target_ids: torch.Tensor, plen: int) -> torch.Tensor:
    """Cross-entropy of the injected forward against the LoRA's greedy targets over the CoT span.

    ``logits`` is ``(1, seq, vocab)`` from the base+injection forward; ``logits[0, t]`` predicts the
    token after position ``t``. ``target_ids`` is ``(seq,)`` with ``target_ids[t] = argmax`` of the
    LoRA's logits at ``t`` (its greedy next token given the *base* context). We supervise the
    positions that predict CoT tokens: ``t in [plen-1, seq-2]`` against ``target_ids[t+1]``.
    """
    seq = logits.shape[1]
    if not 1 <= plen <= seq:
        raise ValueError(f"need 1 <= plen <= seq, got plen={plen}, seq={seq}")
    pred = logits[0, plen - 1:seq - 1]          # predicts tokens at positions [plen, seq-1]
    tgt = target_ids[plen:seq]                   # the LoRA greedy tokens at those positions
    return nn.functional.cross_entropy(pred, tgt)
