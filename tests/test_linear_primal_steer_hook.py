"""Tests for LinearPrimalSteerHook math (CPU, tiny Llama-layout stub, no network).

The hook registers on ``model.model.layers[li]`` (Llama layout). We use a minimal stub
whose layers are identity (returning a tuple like a real decoder block), so the hooked
output equals exactly ``input + α·(input @ Wᵀ)`` and we can assert it elementwise.
"""

from __future__ import annotations

import torch
from torch import nn

from src.probes.safety.steering_hook import LinearPrimalSteerHook


class _IdentityLayer(nn.Module):
    def forward(self, x):
        return (x,)  # decoder blocks return a tuple; hook reads output[0]


class _StubModel(nn.Module):
    def __init__(self, n_layers: int, d: int):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_IdentityLayer() for _ in range(n_layers)])
        self.model = inner

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def test_primal_hook_adds_alpha_W_hs_every_position():
    torch.manual_seed(0)
    d, alpha = 4, 0.5
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    x = torch.randn(1, 6, d)
    hook = LinearPrimalSteerHook(model, {0: W}, alpha)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, x + alpha * (x @ W.T), atol=1e-5)


def test_primal_hook_zero_W_is_noop():
    d = 4
    model = _StubModel(1, d)
    x = torch.randn(1, 5, d)
    hook = LinearPrimalSteerHook(model, {0: torch.zeros(d, d)}, 1.0)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, x, atol=1e-6)


def test_primal_hook_removed_restores():
    d = 4
    model = _StubModel(1, d)
    x = torch.randn(1, 5, d)
    hook = LinearPrimalSteerHook(model, {0: torch.randn(d, d)}, 1.0)
    hook.remove()
    assert torch.allclose(model(x), x, atol=1e-6)


def test_primal_hook_steers_single_decode_step():
    # all-position mode must steer a (1,1,d) cached decode step (not skip it)
    d, alpha = 4, 1.0
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    x = torch.randn(1, 1, d)
    hook = LinearPrimalSteerHook(model, {0: W}, alpha)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, x + alpha * (x @ W.T), atol=1e-5)


def test_primal_hook_last_token_skips_decode_step():
    d = 4
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    x = torch.randn(1, 1, d)
    hook = LinearPrimalSteerHook(model, {0: W}, 1.0, last_token=True)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, x, atol=1e-6)  # cached decode step skipped
