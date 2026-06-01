"""Offline test that additive steering adds the vector to the layer's residual."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.steering_hook import AdditionSteeringHook


def _tiny_model():
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def _resid_at(model, layer, ids):
    cap = PerTokenResidualCapture(model, layers=[layer])
    with cap.capturing(), torch.no_grad():
        model(ids, use_cache=False)
    out = cap.captured[layer].clone()
    cap.remove()
    return out


def test_steering_adds_vector_to_layer_output():
    model = _tiny_model()
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    base = _resid_at(model, 0, ids)

    v = torch.full((32,), 0.5)
    hook = AdditionSteeringHook(model, {0: v})
    steered = _resid_at(model, 0, ids)
    hook.remove()

    # layer 0's own output should be exactly the baseline plus v
    assert torch.allclose(steered, base + 0.5, atol=1e-4)


def test_disabled_is_noop():
    model = _tiny_model()
    ids = torch.tensor([[2, 4, 6]])
    base = _resid_at(model, 1, ids)
    hook = AdditionSteeringHook(model, {1: torch.ones(32)})
    hook.enabled = False
    same = _resid_at(model, 1, ids)
    hook.remove()
    assert torch.allclose(base, same)
