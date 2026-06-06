"""End-to-end plumbing smoke for the attribution pipeline on a tiny in-memory Llama.

No network / no GPU: builds a small LlamaForCausalLM (real ``model.model.layers`` layout)
and runs capture → assemble_blocks → GramAccumulator → solve → score → primal steer hook →
generate. Validates that every piece connects and the two-form cross-check stays tiny.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_attribution_pipeline_smoke():
    from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice
    from src.probes.attribution.gram_accumulator import GramAccumulator
    from src.probes.extraction import PerTokenResidualCapture
    from src.probes.safety.steering_hook import LinearPrimalSteerHook

    model = _tiny_llama()
    d, layers = 32, [0, 1]
    capture = PerTokenResidualCapture(model, layers)
    ids = torch.randint(0, 128, (1, 12))

    # "base" capture
    capture.clear()
    with capture.capturing(), torch.no_grad():
        model(ids, use_cache=False)
    base_cap = {l: t.clone() for l, t in capture.captured.items()}

    # "lora" capture = base + a known primal shift (gives a real, fittable δ)
    Wtrue = {l: 0.1 * torch.randn(d, d) for l in layers}
    shift = LinearPrimalSteerHook(model, Wtrue, alpha=1.0)
    capture.clear()
    with capture.capturing(), torch.no_grad():
        model(ids, use_cache=False)
    lora_cap = {l: t.clone() for l, t in capture.captured.items()}
    shift.remove()
    capture.remove()

    sl = cot_token_slice(prompt_len=4, full_len=12)
    accs = {l: GramAccumulator(d, device="cpu") for l in layers}
    for l in layers:
        a, delta = assemble_blocks(base_cap, lora_cap, l, sl)
        assert a.shape == (8, d) and delta.shape == (8, d)
        accs[l].update(a, delta)

    fit = accs[0].solve(lam=1.0)
    assert fit.W.shape == (d, d)
    assert fit.crosscheck_abs_err < 1e-6 * max(fit.s, 1.0)
    score = accs[1].score(fit.W)
    assert isinstance(score.r2_te, float)

    # steering hook installs and the model still generates
    hook = LinearPrimalSteerHook(model, {0: fit.W.to(torch.float32)}, alpha=0.5)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=3, do_sample=False)
    hook.remove()
    assert out.shape[1] == ids.shape[1] + 3
