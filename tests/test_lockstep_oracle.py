"""Tests for the lockstep dual-forward residual oracle (CPU stub, no network).

The oracle overwrites a base forward's residual at chosen layers with a LoRA forward's
residual, in lockstep over decode steps. Two crux properties are asserted in miniature:

  * overwriting **every** layer is degenerate — the base forward's logits become exactly the
    LoRA forward's logits (the positive control = "just run LoRA"), and lockstep decoding then
    reproduces LoRA's greedy tokens token-for-token;
  * overwriting a **single** layer leaves base layers above it computing with base weights, so
    it is *not* degenerate.

The stub has a Llama-style ``model.layers`` ModuleList so the real ``PerTokenResidualCapture``
and ``OverwriteResidualHook`` register on it unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    blended_injection,
    lockstep_generate,
)
from src.probes.extraction import PerTokenResidualCapture


class _AddBias(nn.Module):
    """Decoder-block stand-in: adds a per-mode, per-layer bias and returns a tuple."""

    def __init__(self, idx: int, owner: "_StubLM"):
        super().__init__()
        self.idx, self.owner = idx, owner

    def forward(self, x):
        bias = (self.owner.lora_bias if self.owner.mode == "lora" else self.owner.base_bias)[self.idx]
        return (x + bias,)


class _StubLM(nn.Module):
    """Tiny deterministic LM: embed → N additive-bias blocks → linear head.

    ``mode`` toggles base vs lora weights (distinct per-layer biases), the miniature of an
    adapter being on/off on a shared base.
    """

    def __init__(self, n_layers: int, d: int, vocab: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, d)
        self.embed.weight.data = torch.randn(vocab, d, generator=g)
        inner = nn.Module()
        inner.layers = nn.ModuleList([_AddBias(i, self) for i in range(n_layers)])
        self.model = inner
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight.data = torch.randn(vocab, d, generator=g)
        self.base_bias = torch.randn(n_layers, d, generator=g)
        self.lora_bias = torch.randn(n_layers, d, generator=g)
        self.mode = "base"

    def set_mode(self, m: str) -> None:
        self.mode = m

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return self.head(h)


def _greedy(stub: _StubLM, ids: torch.Tensor, mode: str, max_new: int) -> torch.Tensor:
    """Plain greedy decode in a single mode — the reference trajectory."""
    stub.set_mode(mode)
    S = ids
    with torch.no_grad():
        for _ in range(max_new):
            nxt = int(stub(S)[0, -1].argmax())
            S = torch.cat([S, torch.tensor([[nxt]])], dim=1)
    return S


def _capture_fn(stub, cap, layers):
    def capture_residuals(S):
        cap.clear()
        stub.set_mode("lora")
        with cap.capturing(), torch.no_grad():
            stub(S)
        return {li: cap.captured[li] for li in layers}
    return capture_residuals


def _base_logits_fn(stub, inject):
    def base_logits(S):
        stub.set_mode("base")
        with inject.injecting(), torch.no_grad():
            return stub(S)
    return base_logits


# --- OverwriteResidualHook ---------------------------------------------------


def test_overwrite_replaces_layer_output_all_positions():
    stub = _StubLM(2, 4, 7)
    inject = OverwriteResidualHook(stub, [0])
    cap = PerTokenResidualCapture(stub, [0])  # registered after inject → sees overwritten output
    repl = torch.randn(3, 4)
    inject.set_values({0: repl})
    with inject.injecting(), cap.capturing(), torch.no_grad():
        stub(torch.tensor([[1, 2, 3]]))
    cap.remove()
    inject.remove()
    assert torch.allclose(cap.captured[0], repl, atol=1e-5)


def test_overwrite_disabled_is_noop():
    stub = _StubLM(2, 4, 7)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        ref = stub(ids).clone()
    inject = OverwriteResidualHook(stub, [0])
    inject.set_values({0: torch.randn(3, 4)})  # values set but injecting() never entered
    with torch.no_grad():
        out = stub(ids)
    inject.remove()
    assert torch.allclose(out, ref, atol=1e-6)


def test_overwrite_removed_restores():
    stub = _StubLM(2, 4, 7)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        ref = stub(ids).clone()
    inject = OverwriteResidualHook(stub, [0])
    inject.remove()
    with torch.no_grad():
        assert torch.allclose(stub(ids), ref, atol=1e-6)


# --- degeneracy: all-layers == LoRA, single-layer != LoRA --------------------


def _lora_residuals(stub, layers):
    stub.set_mode("lora")
    cap = PerTokenResidualCapture(stub, layers)
    with cap.capturing(), torch.no_grad():
        logits = stub(torch.tensor([[1, 2, 3]]))
    res = {li: cap.captured[li].clone() for li in layers}
    cap.remove()
    return logits, res


def test_all_layers_overwrite_makes_base_logits_equal_lora_logits():
    stub = _StubLM(3, 4, 7)
    layers = range(3)
    lora_logits, lora_res = _lora_residuals(stub, layers)
    stub.set_mode("base")
    inject = OverwriteResidualHook(stub, layers)
    inject.set_values(lora_res)
    with inject.injecting(), torch.no_grad():
        base_logits = stub(torch.tensor([[1, 2, 3]]))
    inject.remove()
    assert torch.allclose(base_logits, lora_logits, atol=1e-5)


def test_single_layer_overwrite_is_not_degenerate():
    stub = _StubLM(3, 4, 7)
    layers = range(3)
    lora_logits, lora_res = _lora_residuals(stub, layers)
    stub.set_mode("base")
    inject = OverwriteResidualHook(stub, layers)
    inject.set_values({1: lora_res[1]})  # only layer 1; layer 2 runs with base weights
    with inject.injecting(), torch.no_grad():
        base_logits = stub(torch.tensor([[1, 2, 3]]))
    inject.remove()
    assert not torch.allclose(base_logits, lora_logits, atol=1e-4)


# --- blended_injection (fidelity interpolation) ------------------------------


def test_blend_t0_is_steering_t1_is_oracle():
    torch.manual_seed(0)
    d = 5
    a = torch.randn(4, d)
    lora_resid = torch.randn(4, d)
    W = torch.randn(d, d)
    assert torch.allclose(blended_injection(a, lora_resid, W, 0.0), a + a @ W.T, atol=1e-5)
    assert torch.allclose(blended_injection(a, lora_resid, W, 1.0), lora_resid, atol=1e-5)


# --- lockstep_generate decode loop -------------------------------------------


def test_lockstep_all_layers_reproduces_lora_greedy():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])
    ref = _greedy(stub, ids, "lora", max_new=6)
    cap = PerTokenResidualCapture(stub, range(3))
    inject = OverwriteResidualHook(stub, range(3))
    out = lockstep_generate(
        ids, _capture_fn(stub, cap, range(3)), _base_logits_fn(stub, inject), inject,
        inject_layers=range(3), max_new=6, eos_id=None, device="cpu",
    )
    cap.remove()
    inject.remove()
    assert torch.equal(out, ref)


def test_lockstep_single_layer_runs_and_has_expected_length():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])
    cap = PerTokenResidualCapture(stub, [1])
    inject = OverwriteResidualHook(stub, [1])
    out = lockstep_generate(
        ids, _capture_fn(stub, cap, [1]), _base_logits_fn(stub, inject), inject,
        inject_layers=[1], max_new=5, eos_id=None, device="cpu",
    )
    cap.remove()
    inject.remove()
    assert out.shape == (1, 2 + 5)


def test_lockstep_stops_on_eos():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])
    # eos = the very first token base-pass would emit → must stop after exactly one token
    cap = PerTokenResidualCapture(stub, range(3))
    inject = OverwriteResidualHook(stub, range(3))
    first = lockstep_generate(
        ids, _capture_fn(stub, cap, range(3)), _base_logits_fn(stub, inject), inject,
        inject_layers=range(3), max_new=1, eos_id=None, device="cpu",
    )[0, -1].item()
    out = lockstep_generate(
        ids, _capture_fn(stub, cap, range(3)), _base_logits_fn(stub, inject), inject,
        inject_layers=range(3), max_new=6, eos_id=first, device="cpu",
    )
    cap.remove()
    inject.remove()
    assert out.shape == (1, 3) and out[0, -1].item() == first
