"""Tests for temporal gating of the lockstep oracle (CPU stub, no network).

Phase 1 of the temporal decomposition (tasks/temporal_oracle_decomposition.md). Two kinds of
property are asserted:

  * **gate predicates** — periodic / structural (post-'=' result span) / step-boundary — fire on
    the expected decode steps, and ``result_only`` / ``planning_only`` are exact complements;
  * **the gated decode loop** — with ``always`` it reproduces ``lockstep_generate`` token-for-token
    (and patches every step); with ``never`` it reproduces plain base greedy and never consults the
    LoRA capture; the returned patched-mask matches the gate re-applied offline (accounting).

The model stub mirrors ``tests/test_lockstep_oracle.py`` (Llama-style ``model.layers`` so the real
``OverwriteResidualHook`` / ``PerTokenResidualCapture`` register unchanged).
"""

from __future__ import annotations

import torch
from torch import nn

from src.probes.attribution.lockstep_oracle import OverwriteResidualHook, lockstep_generate
from src.probes.attribution.temporal_gate import (
    always,
    gated_lockstep_generate,
    in_result_span,
    never,
    periodic,
    planning_only,
    result_only,
    step_boundary,
)
from src.probes.extraction import PerTokenResidualCapture


# --- gate predicates (no model) ----------------------------------------------


def test_periodic_fires_on_multiples():
    g = periodic(3)
    assert [g([], s) for s in range(7)] == [True, False, False, True, False, False, True]


def test_periodic_one_is_always():
    g = periodic(1)
    assert all(g([], s) for s in range(10))


def test_in_result_span_opens_on_equals_closes_on_nondigit():
    assert in_result_span(["="]) is True
    assert in_result_span(["=", " "]) is True            # space keeps the span open
    assert in_result_span(["=", " ", "7"]) is True        # result digit keeps it open
    assert in_result_span(["=", " ", "7", "\n"]) is False  # non-digit closes it
    assert in_result_span(["3", " ", "+"]) is False        # never opened


def test_result_and_planning_partition():
    """Every step is patched by exactly one of result_only / planning_only (exact complement)."""
    stream = ["3", " ", "+", " ", "4", " ", "=", " ", "7", "\n",
              "5", " ", "*", " ", "2", " ", "=", " ", "1", "0"]
    for t in range(len(stream)):
        prefix = stream[:t]
        assert result_only(prefix, t) != planning_only(prefix, t)


def test_result_only_patches_result_digit_not_the_equals():
    stream = ["3", " ", "+", " ", "4", " ", "=", " ", "7", "\n"]
    # generating the '=' token (index 6): prefix has no '=' yet → planning, not result
    assert result_only(stream[:6], 6) is False
    # generating the result digit '7' (index 8): prefix ends inside the open span → result
    assert result_only(stream[:8], 8) is True


def test_step_boundary_fires_at_start_and_after_newline():
    assert step_boundary([], 0) is True
    assert step_boundary(["7", "\n"], 2) is True          # previous token contained a newline
    assert step_boundary(["7"], 1) is False


# --- model stub for the decode-loop equivalences -----------------------------


class _AddBias(nn.Module):
    def __init__(self, idx: int, owner: "_StubLM"):
        super().__init__()
        self.idx, self.owner = idx, owner

    def forward(self, x):
        bias = (self.owner.lora_bias if self.owner.mode == "lora" else self.owner.base_bias)[self.idx]
        return (x + bias,)


class _StubLM(nn.Module):
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


def test_gated_always_equals_lockstep_generate():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])

    cap1 = PerTokenResidualCapture(stub, range(3))
    inj1 = OverwriteResidualHook(stub, range(3))
    ref = lockstep_generate(ids, _capture_fn(stub, cap1, range(3)), _base_logits_fn(stub, inj1),
                            inj1, range(3), max_new=6, eos_id=None, device="cpu")
    cap1.remove()
    inj1.remove()

    cap2 = PerTokenResidualCapture(stub, range(3))
    inj2 = OverwriteResidualHook(stub, range(3))
    out, mask = gated_lockstep_generate(ids, _capture_fn(stub, cap2, range(3)),
                                        _base_logits_fn(stub, inj2), inj2, range(3),
                                        gate=always, decode_token=str, max_new=6,
                                        eos_id=None, device="cpu")
    cap2.remove()
    inj2.remove()
    assert torch.equal(out, ref)
    assert all(mask) and len(mask) == 6


def test_gated_never_equals_base_greedy_and_skips_capture():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])

    stub.set_mode("base")
    S = ids
    with torch.no_grad():
        for _ in range(6):
            nxt = int(stub(S)[0, -1].argmax())
            S = torch.cat([S, torch.tensor([[nxt]])], dim=1)

    cap = PerTokenResidualCapture(stub, range(3))
    inj = OverwriteResidualHook(stub, range(3))
    calls = {"n": 0}

    def counting_capture(X):
        calls["n"] += 1
        return _capture_fn(stub, cap, range(3))(X)

    out, mask = gated_lockstep_generate(ids, counting_capture, _base_logits_fn(stub, inj), inj,
                                        range(3), gate=never, decode_token=str, max_new=6,
                                        eos_id=None, device="cpu")
    cap.remove()
    inj.remove()
    assert torch.equal(out, S)              # never patching == plain base greedy
    assert not any(mask)
    assert calls["n"] == 0                  # capture forward skipped on pass-through steps


def test_mask_matches_offline_gate_application():
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])
    cap = PerTokenResidualCapture(stub, [1])
    inj = OverwriteResidualHook(stub, [1])
    gate = periodic(2)
    out, mask = gated_lockstep_generate(ids, _capture_fn(stub, cap, [1]), _base_logits_fn(stub, inj),
                                        inj, [1], gate=gate, decode_token=str, max_new=5,
                                        eos_id=None, device="cpu")
    cap.remove()
    inj.remove()
    gen = out[0, ids.shape[1]:].tolist()
    decoded = [str(t) for t in gen]
    expected = [gate(decoded[:i], i) for i in range(len(gen))]
    assert mask == expected
    assert len(mask) == 5
