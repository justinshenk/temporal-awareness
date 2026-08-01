"""Layer selections must be valid for whatever model is actually being run."""

from __future__ import annotations

import pytest

from src.intertemporal.common.model_layers import (
    REFERENCE_N_LAYERS,
    scale_layers,
    validate_layers,
)
from src.intertemporal.common.semantic_positions import DEFAULT_LAYERS
from src.intertemporal.geometry.geometry_utils import LAYERS

QWEN3_4B = 36
LLAMA31_8B = 32

HARDCODED = [("geometry LAYERS", LAYERS), ("semantic DEFAULT_LAYERS", DEFAULT_LAYERS)]


@pytest.mark.parametrize("name,layers", HARDCODED)
def test_hardcoded_lists_are_valid_for_the_reference_model(name, layers):
    validate_layers(layers, QWEN3_4B)


@pytest.mark.parametrize("name,layers", HARDCODED)
def test_hardcoded_lists_are_invalid_for_llama_without_scaling(name, layers):
    """This is the blocker: 34 and 35 do not exist on a 32-layer model."""
    with pytest.raises(ValueError, match="out of range"):
        validate_layers(layers, LLAMA31_8B)


@pytest.mark.parametrize("name,layers", HARDCODED)
def test_scaling_makes_them_valid_for_llama(name, layers):
    validate_layers(scale_layers(layers, LLAMA31_8B), LLAMA31_8B)


def test_scaling_is_identity_at_the_reference_depth():
    assert scale_layers(LAYERS, QWEN3_4B) == sorted(set(LAYERS))


def test_endpoints_are_preserved():
    scaled = scale_layers([0, 35], LLAMA31_8B)
    assert scaled[0] == 0
    assert scaled[-1] == LLAMA31_8B - 1


def test_fractional_depth_is_approximately_preserved():
    for layer in (12, 18, 24, 31):
        (mapped,) = scale_layers([layer], LLAMA31_8B)
        ref_depth = layer / (REFERENCE_N_LAYERS - 1)
        got_depth = mapped / (LLAMA31_8B - 1)
        assert abs(ref_depth - got_depth) < 0.02


def test_ordering_and_uniqueness_hold():
    scaled = scale_layers(LAYERS, LLAMA31_8B)
    assert scaled == sorted(scaled)
    assert len(scaled) == len(set(scaled))


def test_rejects_nonsense_depths():
    with pytest.raises(ValueError):
        scale_layers([0], 0)
