"""Tests for model-free shift-set assembly."""

import numpy as np
import pytest

from src.probes.lora_icl.shift_extraction import (
    last_token_residual,
    stack_shift_set,
)


def test_last_token_residual_picks_final_row():
    capture = {0: np.arange(6.0).reshape(3, 2), 5: np.ones((4, 2))}
    out = last_token_residual(capture)
    assert np.array_equal(out[0], np.array([4.0, 5.0]))
    assert np.array_equal(out[5], np.array([1.0, 1.0]))


def test_last_token_residual_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        last_token_residual({})


def test_last_token_residual_rejects_non_2d():
    with pytest.raises(ValueError, match="seq, d"):
        last_token_residual({0: np.ones(4)})


def test_stack_shift_set_computes_variant_minus_reference():
    ref = [{10: np.array([1.0, 1.0])}, {10: np.array([2.0, 0.0])}]
    var = [{10: np.array([3.0, 1.0])}, {10: np.array([2.0, 5.0])}]
    shifts = stack_shift_set(ref, var, layer=10)
    assert shifts.shape == (2, 2)
    assert np.array_equal(shifts, np.array([[2.0, 0.0], [0.0, 5.0]]))


def test_stack_shift_set_identical_is_zero():
    ref = [{0: np.array([1.0, 2.0, 3.0])}]
    shifts = stack_shift_set(ref, [dict(ref[0])], layer=0)
    assert np.allclose(shifts, 0.0)


def test_stack_shift_set_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        stack_shift_set([{0: np.zeros(2)}], [], layer=0)


def test_stack_shift_set_missing_layer_raises():
    with pytest.raises(ValueError, match="missing layer"):
        stack_shift_set([{0: np.zeros(2)}], [{1: np.zeros(2)}], layer=0)
