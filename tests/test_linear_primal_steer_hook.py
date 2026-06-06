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


def test_primal_hook_norm_preserve_keeps_per_position_norm():
    # norm_preserve must rescale each steered position back to its original residual norm
    torch.manual_seed(0)
    d, alpha = 8, 1.0
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    x = torch.randn(1, 5, d)
    hook = LinearPrimalSteerHook(model, {0: W}, alpha, norm_preserve=True)
    out = model(x)
    hook.remove()
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # direction must still have moved (not a no-op)
    assert not torch.allclose(out, x, atol=1e-3)


def test_primal_hook_projection_restricts_delta_to_subspace():
    # the injected delta (out - input) must lie in span(V); base residual is left intact
    torch.manual_seed(0)
    d, k, alpha = 8, 3, 1.0
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    V, _ = torch.linalg.qr(torch.randn(d, k))  # orthonormal (d, k)
    x = torch.randn(1, 5, d)
    hook = LinearPrimalSteerHook(model, {0: W}, alpha, project_bases={0: V})
    out = model(x)
    hook.remove()
    delta = out - x
    assert torch.allclose(delta, (delta @ V) @ V.T, atol=1e-4)   # delta in span(V)
    assert torch.allclose(delta, (alpha * (x @ W.T) @ V) @ V.T, atol=1e-4)  # exactly Π_V(α Wᵀx)
    assert not torch.allclose(out, x, atol=1e-3)


def test_primal_hook_prefill_only_steers_prompt_skips_decode():
    torch.manual_seed(0)
    d, alpha = 4, 1.0
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    # prompt pass (seq>1): all positions steered
    prompt = torch.randn(1, 6, d)
    hook = LinearPrimalSteerHook(model, {0: W}, alpha, prefill_only=True)
    out_prompt = model(prompt)
    assert torch.allclose(out_prompt, prompt + alpha * (prompt @ W.T), atol=1e-5)
    # decode step (seq==1): skipped (un-reinjected)
    step = torch.randn(1, 1, d)
    out_step = model(step)
    hook.remove()
    assert torch.allclose(out_step, step, atol=1e-6)


def test_primal_hook_last_token_skips_decode_step():
    d = 4
    model = _StubModel(1, d)
    W = torch.randn(d, d)
    x = torch.randn(1, 1, d)
    hook = LinearPrimalSteerHook(model, {0: W}, 1.0, last_token=True)
    out = model(x)
    hook.remove()
    assert torch.allclose(out, x, atol=1e-6)  # cached decode step skipped
