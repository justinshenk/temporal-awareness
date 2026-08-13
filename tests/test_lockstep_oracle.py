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

import pytest

from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    blended_injection,
    control_injection,
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


# --- control_injection (empirical floors for the oracle) ---------------------
#
# On a procedure a broken injection scores ~0: it cannot produce the right integer by accident.
# On a k-way register task it scores ~chance, so a partial recovery cannot be told from a garbled
# one without a floor. These two controls supply it — same δ magnitude, destroyed content.


def test_mean_delta_replaces_every_position_with_the_average_shift():
    a = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lora_resid = torch.tensor([[1.0, 3.0], [3.0, 3.0], [5.0, 6.0]])   # δ column means: 3.0, 4.0
    out = control_injection(a, lora_resid, "mean_delta")
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]).expand(3, 2))


def test_mean_delta_is_a_noop_when_the_shift_is_already_constant():
    """The control only destroys *per-token* structure; a genuinely pointwise δ survives it."""
    a = torch.randn(4, 3)
    delta = torch.tensor([[1.0, -2.0, 0.5]]).expand(4, 3)
    out = control_injection(a, a + delta, "mean_delta")
    assert torch.allclose(out, a + delta, atol=1e-6)


def test_shuffle_positions_permutes_the_shift_across_tokens():
    a = torch.zeros(4, 2)
    delta = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    out = control_injection(a, a + delta, "shuffle_positions",
                            generator=torch.Generator().manual_seed(0))
    assert sorted(out[:, 0].tolist()) == [1.0, 2.0, 3.0, 4.0]   # same multiset of shifts...
    assert not torch.allclose(out, delta)                        # ...in a different order


def test_shuffle_positions_is_seeded():
    a, delta = torch.zeros(8, 2), torch.arange(16.0).reshape(8, 2)
    def run(seed):
        return control_injection(a, a + delta, "shuffle_positions",
                                 generator=torch.Generator().manual_seed(seed))
    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1))


def test_controls_preserve_shape_and_dtype():
    a, lora_resid = torch.randn(5, 3), torch.randn(5, 3)
    for mode in ("mean_delta", "shuffle_positions"):
        out = control_injection(a, lora_resid, mode, generator=torch.Generator().manual_seed(0))
        assert out.shape == a.shape and out.dtype == a.dtype


def test_control_alpha_scales_the_shift_and_defaults_to_one():
    """alpha=1.0 must reproduce the committed floor runs exactly — it is a regression guard."""
    a = torch.zeros(3, 2)
    delta = torch.tensor([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    full = control_injection(a, a + delta, "mean_delta")
    half = control_injection(a, a + delta, "mean_delta", alpha=0.5)
    assert torch.allclose(full, delta)
    assert torch.allclose(half, delta * 0.5)


def test_control_alpha_zero_is_the_unpatched_base():
    """The alpha sweep's left endpoint must be exactly 'no intervention', not 'small intervention'."""
    a = torch.randn(4, 3)
    lora_resid = torch.randn(4, 3)
    for mode in ("mean_delta", "shuffle_positions"):
        out = control_injection(a, lora_resid, mode, generator=torch.Generator().manual_seed(0),
                                alpha=0.0)
        assert torch.allclose(out, a, atol=1e-6)


def test_control_alpha_applies_to_shuffle_too():
    a = torch.zeros(4, 2)
    delta = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    out = control_injection(a, a + delta, "shuffle_positions",
                            generator=torch.Generator().manual_seed(0), alpha=0.5)
    assert sorted(out[:, 0].tolist()) == [0.5, 1.0, 1.5, 2.0]


def test_mean_delta_uses_generated_positions_only():
    """The statistic must come from generated rows; prompt rows outnumber them ~15:1 and their
    diverse directions cancel the mean down to a third of its per-token norm (measured: 30 -> 11)."""
    a = torch.zeros(5, 2)
    delta = torch.cat([torch.tensor([[9.0, 0.0], [-9.0, 0.0], [9.0, 0.0]]),   # prompt: cancels
                       torch.tensor([[0.0, 4.0], [0.0, 6.0]])])              # generated: mean (0,5)
    out = control_injection(a, a + delta, "mean_delta", prompt_len=3)
    assert torch.allclose(out, torch.tensor([[0.0, 5.0]]).expand(5, 2))


def test_shuffle_permutes_only_generated_rows():
    """Prompt<->prompt swaps are nearly harmless, so a shuffle over all rows barely intervenes."""
    a = torch.zeros(5, 2)
    delta = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [7.0, 7.0], [8.0, 8.0]])
    out = control_injection(a, a + delta, "shuffle_positions",
                            generator=torch.Generator().manual_seed(0), prompt_len=3)
    assert torch.allclose(out[:3], delta[:3])                      # prompt rows untouched
    assert sorted(out[3:, 0].tolist()) == [7.0, 8.0]               # generated rows permuted


def test_random_matched_keeps_per_token_norms_but_not_directions():
    """The floor the oracle's 0.990 actually needs: same magnitudes, random directions."""
    a = torch.zeros(6, 8)
    delta = torch.randn(6, 8) * 3.0
    out = control_injection(a, a + delta, "random_matched",
                            generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out.norm(dim=1), delta.norm(dim=1), atol=1e-4)   # norms matched
    cos = torch.nn.functional.cosine_similarity(out, delta, dim=1).abs()
    assert (cos < 0.95).all()                                              # directions destroyed


def test_random_matched_is_seeded():
    a, delta = torch.zeros(4, 6), torch.randn(4, 6)
    def run(seed):
        return control_injection(a, a + delta, "random_matched",
                                 generator=torch.Generator().manual_seed(seed))
    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1))


def test_a_control_actually_perturbs_the_decode():
    """The test today's bug demanded: assert the intervention INTERVENES.

    Every prior control test asserts tensor algebra, and all of them pass on a control that never
    reaches the model. This one drives the real decode loop and requires the tokens to move.
    """
    stub = _StubLM(3, 4, 7)
    ids = torch.tensor([[1, 2]])
    def decode(control, alpha):
        cap = PerTokenResidualCapture(stub, [1])
        inject = OverwriteResidualHook(stub, [1])
        def capture_residuals(S):
            cap.clear()
            stub.set_mode("lora")
            with cap.capturing(), torch.no_grad():
                stub(S)
            lora_resid = {1: cap.captured[1]}
            if control is None:
                return lora_resid
            cap.clear()
            stub.set_mode("base")
            with cap.capturing(), torch.no_grad():
                stub(S)
            return {1: control_injection(cap.captured[1], lora_resid[1], control,
                                         torch.Generator().manual_seed(0), alpha)}
        out = lockstep_generate(ids, capture_residuals, _base_logits_fn(stub, inject), inject,
                                [1], max_new=6, eos_id=None, device="cpu")
        cap.remove(); inject.remove()
        return out
    plain = _greedy(stub, ids, "base", max_new=6)
    assert not torch.equal(decode("random_matched", 8.0), plain), "control did not perturb the decode"


def test_control_injection_rejects_an_unknown_mode():
    a = torch.randn(2, 2)
    with pytest.raises(ValueError, match="unknown control"):
        control_injection(a, a, "nonsense")


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
