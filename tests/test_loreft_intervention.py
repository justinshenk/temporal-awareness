"""Tests for the LoReFT intervention glue (CPU, no network).

Asserts the crux properties in miniature: the edit is the LoReFT formula ``h + (Wh + b − hR)Rᵀ``
(fixed point when the learned source equals the projection, full rewrite at full rank, shift
confined to span(R)); exactly R+W+b train and R stays orthonormal through optimizer steps; the
f7+l7 position arithmetic handles short prompts and left-padding offsets; the hook edits only the
selected positions of the selected layers and passes decode steps through untouched; and a tiny
frozen Llama's LM loss decreases when only the interventions train.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.attribution.loreft_intervention import (
    LoReFTIntervention,
    PositionEditHook,
    intervention_locations,
    response_labels,
)


def tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
                      max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def test_fixed_point_when_source_equals_projection():
    """If source(h) == hR exactly, the edit must be the identity."""
    torch.manual_seed(0)
    iv = LoReFTIntervention(d=6, r=3, seed=1)
    R = iv.subspace()
    with torch.no_grad():
        iv.source.weight.copy_(R.T)
        iv.source.bias.zero_()
    h = torch.randn(5, 6)
    assert torch.allclose(iv(h), h, atol=1e-5)


def test_full_rank_rewrites_to_source():
    """At r=d the base contribution cancels: output == source(h) @ Rᵀ."""
    torch.manual_seed(0)
    iv = LoReFTIntervention(d=6, r=6, seed=2)
    h = torch.randn(4, 6)
    R = iv.subspace()
    expected = iv.source(h) @ R.T
    assert torch.allclose(iv(h), expected, atol=1e-5)


def test_shift_lies_in_span_R():
    """The edit h' − h must live entirely in span(R): re-projecting it is a no-op."""
    torch.manual_seed(0)
    iv = LoReFTIntervention(d=8, r=2, seed=3)
    h = torch.randn(5, 8)
    shift = iv(h) - h
    R = iv.subspace()
    assert torch.allclose(shift @ R @ R.T, shift, atol=1e-5)


def test_param_count_matches_formula():
    """Trainable params = d·r (raw R) + r·d + r (linear source) — the LoReFT budget."""
    iv = LoReFTIntervention(d=16, r=4, seed=0)
    n = sum(p.numel() for p in iv.parameters() if p.requires_grad)
    assert n == 16 * 4 + 4 * 16 + 4
    iv_paper = LoReFTIntervention(d=4096, r=8, seed=0)
    n_paper = sum(p.numel() for p in iv_paper.parameters() if p.requires_grad)
    assert n_paper == 65_544                       # per layer at the paper's commonsense config


def test_gradients_flow_and_R_stays_orthonormal():
    torch.manual_seed(0)
    iv = LoReFTIntervention(d=8, r=3, seed=4)
    opt = torch.optim.Adam(iv.parameters(), lr=0.05)
    h = torch.randn(6, 8)
    for _ in range(5):
        opt.zero_grad()
        loss = iv(h).pow(2).sum()
        loss.backward()
        assert iv.subspace.raw.grad is not None
        assert iv.source.weight.grad is not None
        assert iv.source.bias.grad is not None
        opt.step()
    R = iv.subspace()
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)


def test_intervention_locations_f7_l7():
    # long prompt: disjoint first-7 + last-7
    assert intervention_locations(20, 7, 7) == list(range(7)) + list(range(13, 20))
    # short prompt: overlap collapses to the unique union
    assert intervention_locations(10, 7, 7) == list(range(10))
    # tiny prompt: every position, once
    assert intervention_locations(3, 7, 7) == [0, 1, 2]
    # left-padding offset shifts everything
    assert intervention_locations(20, 7, 7, offset=5) == [p + 5 for p in
                                                          list(range(7)) + list(range(13, 20))]


def test_intervention_locations_rejects_bad_plen():
    with pytest.raises(ValueError):
        intervention_locations(0, 7, 7)


def test_response_labels_masks_prompt_only():
    """Labels are −100 strictly before each sample's response start, untouched after (incl. eos)."""
    ids = torch.tensor([[9, 9, 5, 6, 7, 2],        # 2 pads+prompt, response starts at 2
                        [9, 9, 9, 9, 8, 2]])       # response starts at 4
    labels = response_labels(ids, torch.tensor([2, 4]))
    assert labels[0, :2].eq(-100).all() and labels[0, 2:].eq(torch.tensor([5, 6, 7, 2])).all()
    assert labels[1, :4].eq(-100).all() and labels[1, 4:].eq(torch.tensor([8, 2])).all()
    assert ids[0, 0] == 9                          # input ids untouched (labels are a copy)


def test_hook_edits_only_selected_positions():
    """A +10 edit at layer 0, positions {0,1}: layer-0 output differs by exactly 10 there, 0 elsewhere."""

    class AddTen(nn.Module):
        def forward(self, h):
            return h + 10.0

    model = tiny_llama()
    hook = PositionEditHook(model, {0: AddTen()})
    captured = {}

    def capture_fn(module, inputs, output):
        captured["hs"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    cap_handle = model.model.layers[0].register_forward_hook(capture_fn)
    ids = torch.randint(0, 64, (2, 6), generator=torch.Generator().manual_seed(0))
    mask = torch.ones_like(ids)

    with torch.no_grad():
        model(ids, attention_mask=mask)
    base_hs = captured["hs"]

    hook.set_locations(torch.tensor([[0, 1], [0, 1]]))
    with hook.injecting(), torch.no_grad():
        model(ids, attention_mask=mask)
    edited_hs = captured["hs"]

    diff = edited_hs - base_hs
    assert torch.allclose(diff[:, :2], torch.full_like(diff[:, :2], 10.0), atol=1e-4)
    assert torch.allclose(diff[:, 2:], torch.zeros_like(diff[:, 2:]), atol=1e-6)

    cap_handle.remove()
    hook.remove()


def test_hook_disabled_and_decode_steps_pass_through():
    model = tiny_llama()
    ids = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = model(ids, attention_mask=mask).logits

    hook = PositionEditHook(model, {0: LoReFTIntervention(d=16, r=4, seed=5)})
    hook.set_locations(torch.tensor([[0, 1, 4, 5]]))

    # outside injecting(): identical to no hook at all
    with torch.no_grad():
        off = model(ids, attention_mask=mask).logits
    assert torch.equal(off, ref)

    # seq shorter than max location +1 (a decode step on KV cache): guard skips the edit
    with hook.injecting(), torch.no_grad():
        one_ref = model(ids[:, :1], attention_mask=mask[:, :1]).logits
    hook.enabled = False
    with torch.no_grad():
        one_off = model(ids[:, :1], attention_mask=mask[:, :1]).logits
    assert torch.equal(one_ref, one_off)

    # full-length prefill DOES change the logits
    with hook.injecting(), torch.no_grad():
        on = model(ids, attention_mask=mask).logits
    assert not torch.allclose(on, ref, atol=1e-6)
    hook.remove()


def test_tiny_frozen_llama_loss_decreases_with_intervention_training():
    """End-to-end: frozen tiny Llama + per-layer LoReFT at f2+l2 prompt positions, loss drops."""
    model = tiny_llama()
    for p in model.parameters():
        p.requires_grad_(False)

    interventions = {li: LoReFTIntervention(d=16, r=4, seed=10 + li) for li in range(2)}
    hook = PositionEditHook(model, interventions)
    params = [p for iv in interventions.values() for p in iv.parameters()]
    opt = torch.optim.Adam(params, lr=5e-3)

    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 64, (4, 10), generator=g)
    mask = torch.ones_like(ids)
    plen = 6
    locs = torch.tensor([intervention_locations(plen, 2, 2)] * 4)
    labels = response_labels(ids, torch.full((4,), plen))
    hook.set_locations(locs)

    def step() -> float:
        with hook.injecting():
            out = model(ids, attention_mask=mask, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        return float(out.loss.detach())

    first = step()
    for _ in range(29):
        last = step()
    assert last < first
    hook.remove()
