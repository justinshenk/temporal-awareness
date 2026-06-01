"""Offline test that directional ablation removes the target direction from residuals."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.ablation_hook import DirectionalAblationHook


def _tiny_model():
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_ablation_zeroes_component_along_direction():
    model = _tiny_model()
    torch.manual_seed(1)
    direction = torch.randn(32)

    ablate = DirectionalAblationHook(model, direction)        # registered first → runs first
    capture = PerTokenResidualCapture(model, layers=[0, 1])   # reads the ablated output
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    with capture.capturing(), torch.no_grad():
        model(ids, use_cache=False)

    u = (direction / direction.norm())
    for layer in (0, 1):
        resid = capture.captured[layer]            # (seq, 32), already ablated
        comp = (resid @ u).abs().max().item()
        assert comp < 1e-4, f"layer {layer} residual still has direction component {comp}"

    capture.remove()
    ablate.remove()
    assert ablate._hooks == []


def test_disabled_hook_is_noop():
    model = _tiny_model()
    ablate = DirectionalAblationHook(model, torch.randn(32))
    ablate.enabled = False
    capture = PerTokenResidualCapture(model, layers=[1])
    ids = torch.tensor([[2, 4, 6]])
    with capture.capturing(), torch.no_grad():
        model(ids, use_cache=False)
    # with ablation disabled, the residual retains its (nonzero) content
    assert capture.captured[1].abs().sum().item() > 0
    capture.remove()
    ablate.remove()
