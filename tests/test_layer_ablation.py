"""Tests for identity-ablation of decoder layers (CPU, no network).

Crux properties: an ablated layer is a verified no-op (its output residual equals its input residual);
ablating every layer collapses the stack so the final pre-norm hidden equals the post-embedding hidden;
the active set is mutable for a cumulative sweep; and batch>1 is rejected like the sibling hooks.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.attribution.layer_ablation import IdentityAblationHook
from src.probes.extraction import PerTokenResidualCapture


def tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                      num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
                      max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def test_ablated_layer_is_a_noop():
    """With layer li ablated, its captured output residual equals the residual entering it.

    The ablation hook is registered BEFORE the same-layer capture so the capture observes the
    ablated (returned) output — PyTorch propagates a hook's return value only to later hooks.
    """
    model = tiny_llama()
    li = 2
    ablate = IdentityAblationHook(model, [li])
    ablate.set_layers([li])
    cap_in = PerTokenResidualCapture(model, [li - 1])     # residual that flows INTO layer li
    cap_out = PerTokenResidualCapture(model, [li])        # residual leaving layer li (ablated)
    ids = torch.tensor([[3, 9, 1, 40, 7]])
    with torch.no_grad(), cap_in.capturing(), cap_out.capturing(), ablate.ablating():
        model(ids, use_cache=False)
    assert torch.allclose(cap_out.captured[li], cap_in.captured[li - 1], atol=1e-5)
    for c in (cap_in, cap_out, ablate):
        c.remove()


def test_disabled_hook_passes_through():
    """Outside the ablating() context the layer runs normally (output != input)."""
    model = tiny_llama()
    li = 2
    cap_in = PerTokenResidualCapture(model, [li - 1])
    cap_out = PerTokenResidualCapture(model, [li])
    ablate = IdentityAblationHook(model, [li])
    ablate.set_layers([li])
    ids = torch.tensor([[3, 9, 1, 40, 7]])
    with torch.no_grad(), cap_in.capturing(), cap_out.capturing():
        model(ids, use_cache=False)                       # ablating() NOT entered
    assert not torch.allclose(cap_out.captured[li], cap_in.captured[li - 1], atol=1e-3)
    for c in (cap_in, cap_out, ablate):
        c.remove()


def test_ablating_all_layers_equals_post_embedding():
    """Ablating every layer makes the final layer's output residual equal the token embeddings."""
    model = tiny_llama()
    layers = list(range(model.config.num_hidden_layers))
    last = layers[-1]
    ablate = IdentityAblationHook(model, layers)
    ablate.set_layers(layers)
    cap_out = PerTokenResidualCapture(model, [last])      # registered after ablate → sees ablated out
    ids = torch.tensor([[3, 9, 1, 40, 7]])
    with torch.no_grad():
        embed = model.model.embed_tokens(ids)[0].float()
        with cap_out.capturing(), ablate.ablating():
            model(ids, use_cache=False)
    assert torch.allclose(cap_out.captured[last], embed, atol=1e-4), \
        (cap_out.captured[last] - embed).abs().max()
    cap_out.remove()
    ablate.remove()


def test_set_layers_mutates_active_set():
    """A cumulative sweep: only the currently-active layers are no-ops."""
    model = tiny_llama()
    ablate = IdentityAblationHook(model, [2, 3])           # before captures → same-layer captures see ablation
    cap = {li: PerTokenResidualCapture(model, [li]) for li in (2, 3)}
    cap_in = {li: PerTokenResidualCapture(model, [li - 1]) for li in (2, 3)}
    ids = torch.tensor([[3, 9, 1, 40, 7]])

    ablate.set_layers([3])                                 # only layer 3 ablated
    with torch.no_grad(), ablate.ablating(), \
            cap[2].capturing(), cap[3].capturing(), cap_in[2].capturing(), cap_in[3].capturing():
        model(ids, use_cache=False)
    assert torch.allclose(cap[3].captured[3], cap_in[3].captured[2], atol=1e-5)       # 3 is no-op
    assert not torch.allclose(cap[2].captured[2], cap_in[2].captured[1], atol=1e-3)   # 2 runs
    for c in list(cap.values()) + list(cap_in.values()):
        c.remove()
    ablate.remove()


def test_batch_gt_one_rejected():
    """The hook asserts batch=1, matching OverwriteResidualHook / PerTokenResidualCapture."""
    model = tiny_llama()
    ablate = IdentityAblationHook(model, [1])
    ablate.set_layers([1])
    ids = torch.tensor([[3, 9, 1], [4, 2, 5]])
    with pytest.raises(AssertionError):
        with torch.no_grad(), ablate.ablating():
            model(ids, use_cache=False)
    ablate.remove()
