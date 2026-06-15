"""Tests for the logit lens (CPU, no network).

The load-bearing property: the lens of the *final* decoder layer's residual equals the model's real
logits — so per-layer ranks are comparable to the model's actual decision. Plus rank/logit extraction
against a hand-set logits vector, and that the target rank is a valid vocabulary index.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.probes.attribution.logit_lens import LogitLens
from src.probes.extraction import PerTokenResidualCapture


def tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
                      max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def test_lens_of_final_layer_equals_model_logits():
    """Projecting the last decoder layer's residual through norm+lm_head reproduces the model logits."""
    model = tiny_llama()
    last = model.config.num_hidden_layers - 1
    lens = LogitLens(model)
    capture = PerTokenResidualCapture(model, [last])
    ids = torch.tensor([[3, 9, 1, 40, 7]])
    with torch.no_grad(), capture.capturing():
        real = model(ids, use_cache=False).logits
    lens_logits = lens.project(capture.captured[last])
    capture.remove()
    assert torch.allclose(lens_logits, real[0], atol=1e-4), (lens_logits - real[0]).abs().max()


def test_lens_final_layer_argmax_matches_next_token():
    """The lens argmax at the last position equals the token the model would greedily emit."""
    model = tiny_llama()
    last = model.config.num_hidden_layers - 1
    lens = LogitLens(model)
    capture = PerTokenResidualCapture(model, [last])
    ids = torch.tensor([[5, 2, 18, 33]])
    with torch.no_grad(), capture.capturing():
        real = model(ids, use_cache=False).logits
    emitted = int(real[0, -1].argmax())
    rank, _ = lens.rank_of(capture.captured[last], emitted, row=-1)
    capture.remove()
    assert rank == 0


def test_rank_of_matches_hand_set_logits():
    """rank_of returns the count of tokens strictly outranking the target, and its logit."""

    class StubHead:
        def __init__(self, logits):
            self.weight = torch.zeros(logits.shape[-1], 1)
            self._logits = logits

        def __call__(self, h):
            return self._logits

    lens = LogitLens.__new__(LogitLens)
    lens.norm = lambda h: h
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])      # token 1 is top, then 2, 3, 0
    lens.lm_head = StubHead(logits)
    dummy = torch.zeros(1, 1)
    assert lens.rank_of(dummy, token_id=1) == (0, 5.0)
    assert lens.rank_of(dummy, token_id=2) == (1, 3.0)
    assert lens.rank_of(dummy, token_id=0) == (3, 1.0)


def test_rank_in_vocab_range():
    """A target token's rank is always a valid 0-based vocabulary index."""
    model = tiny_llama()
    last = model.config.num_hidden_layers - 1
    lens = LogitLens(model)
    capture = PerTokenResidualCapture(model, [last])
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad(), capture.capturing():
        model(ids, use_cache=False)
    rank, _ = lens.rank_of(capture.captured[last], token_id=50, row=-1)
    capture.remove()
    assert 0 <= rank < model.config.vocab_size
