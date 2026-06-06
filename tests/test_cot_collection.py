"""Tests for CoT-slice / block-assembly glue (CPU, fast)."""

from __future__ import annotations

import pytest
import torch

from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice


def test_cot_token_slice():
    assert cot_token_slice(10, 25) == slice(10, 25)
    assert cot_token_slice(0, 5) == slice(0, 5)


def test_cot_token_slice_invalid_raises():
    with pytest.raises(ValueError):
        cot_token_slice(30, 10)
    with pytest.raises(ValueError):
        cot_token_slice(-1, 10)


def test_assemble_blocks_values_dtype_shape():
    d = 4
    base = {7: torch.arange(5 * d, dtype=torch.float32).reshape(5, d)}
    lora = {7: base[7] + 2.0}
    sl = cot_token_slice(prompt_len=2, full_len=5)  # CoT = positions 2,3,4
    a, delta = assemble_blocks(base, lora, 7, sl)
    assert a.shape == (3, d) and delta.shape == (3, d)
    assert a.dtype == torch.float64 and delta.dtype == torch.float64
    assert torch.allclose(a, base[7][2:5].double())
    assert torch.allclose(delta, torch.full((3, d), 2.0, dtype=torch.float64))


def test_assemble_blocks_missing_layer_raises():
    base = {7: torch.zeros(5, 4)}
    lora = {7: torch.zeros(5, 4)}
    with pytest.raises(ValueError):
        assemble_blocks(base, lora, 21, slice(0, 5))


def test_assemble_blocks_shape_mismatch_raises():
    base = {7: torch.zeros(5, 4)}
    lora = {7: torch.zeros(6, 4)}
    with pytest.raises(ValueError):
        assemble_blocks(base, lora, 7, slice(0, 5))
