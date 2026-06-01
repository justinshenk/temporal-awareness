"""Offline smoke test for the residual-capture + PEFT plumbing.

Uses a tiny random Llama model (no download, no gated weights) to verify the load-
bearing assumption in extract_shifts.py: hooks registered on the base decoder
layers still fire when the model is wrapped with a LoRA adapter, so base and LoRA
residuals are captured at the same sites.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.shift_extraction import last_token_residual


def _tiny_model():
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_capture_fires_on_base_and_peft():
    model = _tiny_model()
    capture = PerTokenResidualCapture(model, layers=[0, 1])
    ids = torch.tensor([[1, 5, 9, 3, 7]])

    capture.clear()
    with capture.capturing(), torch.no_grad():
        model(ids, use_cache=False)
    base_site = last_token_residual(capture.captured)
    assert set(base_site) == {0, 1}
    assert base_site[0].shape == (32,)

    # Wrap with LoRA; the same decoder-layer modules carry the hooks.
    lora = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                          task_type="CAUSAL_LM")
    )
    capture.clear()
    with capture.capturing(), torch.no_grad():
        lora(ids, use_cache=False)
    lora_site = last_token_residual(capture.captured)
    assert set(lora_site) == {0, 1}  # hooks fired during the PEFT forward
    assert lora_site[1].shape == (32,)

    capture.remove()
    assert capture._hooks == []
