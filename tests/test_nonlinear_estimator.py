"""Tests for the nonlinear δ estimator (CPU, synthetic data, no model)."""

from __future__ import annotations

import torch
from torch import nn

from src.probes.attribution.nonlinear_estimator import (
    DeltaMLP,
    NonlinearSteerHook,
    fit_delta_mlp,
)


def _teacher(d, h, seed=0):
    """A fixed random 2-layer teacher MLP — the target the student must learn (and linear can't)."""
    g = torch.Generator().manual_seed(seed)
    M1 = torch.randn(d, h, generator=g) * (1.0 / d ** 0.5)
    M2 = torch.randn(h, d, generator=g) * (1.0 / h ** 0.5)
    return M1, M2


def _samples(teacher, n, seed):
    """Fresh inputs through the SAME teacher (train/val must share the mapping to generalize)."""
    M1, M2 = teacher
    a = torch.randn(n, M1.shape[0], generator=torch.Generator().manual_seed(seed))
    return a, nn.functional.gelu(a @ M1) @ M2


def test_delta_mlp_shapes():
    mlp = DeltaMLP(8, 16)
    out = mlp(torch.randn(5, 8))
    assert out.shape == (5, 8)


def test_fit_beats_linear_least_squares_on_nonlinear_target():
    d = 12
    teacher = _teacher(d, h=32, seed=0)
    a_tr, d_tr = _samples(teacher, 2000, seed=1)
    a_val, d_val = _samples(teacher, 500, seed=2)
    # linear least-squares baseline cosine on val
    W = torch.linalg.lstsq(a_tr, d_tr).solution  # (d, d), maps a→δ as a@W
    pv_lin = a_val @ W
    lin_cos = float(((pv_lin * d_val).sum(1) /
                     (pv_lin.norm(dim=1) * d_val.norm(dim=1) + 1e-8)).mean())
    mlp, metrics = fit_delta_mlp(a_tr, d_tr, a_val, d_val, hidden=64, epochs=60,
                                 batch=256, dropout=0.0, device="cpu", seed=0)
    assert metrics["val_cosine"] > lin_cos + 0.1  # nonlinear clearly better
    assert metrics["val_cosine"] > 0.5


# --- NonlinearSteerHook ------------------------------------------------------


class _IdentityLayer(nn.Module):
    def forward(self, x):
        return (x,)


class _StubModel(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_IdentityLayer() for _ in range(n_layers)])
        self.model = inner

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def test_hook_adds_alpha_f_of_hs():
    torch.manual_seed(0)
    d, alpha = 6, 0.5
    model = _StubModel(1, d)
    mlp = DeltaMLP(d, 8, dropout=0.0)
    x = torch.randn(1, 4, d)
    expected = x + alpha * mlp(x)
    hook = NonlinearSteerHook(model, mlp, layer=0, alpha=alpha)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, expected, atol=1e-5)


def test_hook_removed_restores():
    d = 6
    model = _StubModel(1, d)
    mlp = DeltaMLP(d, 8, dropout=0.0)
    x = torch.randn(1, 4, d)
    hook = NonlinearSteerHook(model, mlp, layer=0)
    hook.remove()
    assert torch.allclose(model(x), x, atol=1e-6)
