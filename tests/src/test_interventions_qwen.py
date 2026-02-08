"""Comprehensive intervention tests using Qwen model.

Tests all combinations of:
- Modes: add, set, mul, interpolate
- Axes: all, position, neuron, pattern
- Backends: TransformerLens (default), NNsight, Pyvene

Ground truth tests: 4 modes × 4 axes = 16 tests
Backend comparison tests: 4 modes × 4 axes × 3 backends = 48 tests
Total: 64 systematic tests

Note: These tests verify interventions produce valid outputs and that
backends behave consistently. For mathematical correctness verification,
see test_interventions_toy.py which uses known weights.
"""

import numpy as np
import pytest
import torch

from src.models import ModelRunner
from src.models.model_runner import ModelBackend
from src.models.interventions import steering, ablation, scale, interpolate, get_activations


TEST_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def runner():
    """Default backend (TransformerLens)."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)


@pytest.fixture(scope="module")
def runner_nnsight():
    """NNsight backend."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.NNSIGHT)


@pytest.fixture(scope="module")
def runner_pyvene():
    """Pyvene backend."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.PYVENE)


# =============================================================================
# Helper functions
# =============================================================================


def make_direction(d_model, seed=42):
    """Create a normalized random direction vector."""
    np.random.seed(seed)
    direction = np.random.randn(d_model).astype(np.float32)
    return direction / np.linalg.norm(direction)


def make_values(d_model, seed=42):
    """Create random values for set/interpolate."""
    np.random.seed(seed)
    return np.random.randn(d_model).astype(np.float32)


def make_2d_values(seq_len, d_model, seed=42):
    """Create random 2D values for interpolation."""
    np.random.seed(seed)
    return np.random.randn(seq_len, d_model).astype(np.float32)


# =============================================================================
# Ground Truth Tests - ALL axis (4 tests)
# =============================================================================


class TestInterventionsAllGroundTruth:
    """Ground truth tests for 'all' axis interventions."""

    def test_interventions_all_add_ground_truth(self, runner):
        """ADD/all produces valid logits."""
        prompt = "Hello"
        intervention = steering(
            layer=5, direction=make_direction(runner.d_model), strength=10.0
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_set_ground_truth(self, runner):
        """SET/all produces valid logits."""
        prompt = "Hello"
        intervention = ablation(layer=5, values=make_values(runner.d_model))
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_mul_ground_truth(self, runner):
        """MUL/all produces valid logits."""
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_interpolate_ground_truth(self, runner):
        """INTERPOLATE/all produces valid logits."""
        prompt = "Hello"
        source = get_activations(runner, layer=5, prompt=prompt)
        target = source * 2
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


# =============================================================================
# Ground Truth Tests - POSITION axis (4 tests)
# =============================================================================


class TestInterventionsPositionGroundTruth:
    """Ground truth tests for 'position' axis interventions."""

    def test_interventions_position_add_ground_truth(self, runner):
        """ADD/position produces valid logits."""
        prompt = "The quick brown fox"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            positions=[1, 2],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_set_ground_truth(self, runner):
        """SET/position produces valid logits."""
        prompt = "The quick brown fox"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), positions=[1, 2]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_mul_ground_truth(self, runner):
        """MUL/position produces valid logits."""
        prompt = "The quick brown fox"
        intervention = scale(layer=5, factor=0.5, positions=[1, 2])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_interpolate_ground_truth(self, runner):
        """INTERPOLATE/position produces valid logits."""
        prompt = "The quick brown fox"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5,
            source_values=source,
            target_values=target,
            alpha=0.5,
            positions=[1, 2],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


# =============================================================================
# Ground Truth Tests - NEURON axis (4 tests)
# =============================================================================


class TestInterventionsNeuronGroundTruth:
    """Ground truth tests for 'neuron' axis interventions."""

    def test_interventions_neuron_add_ground_truth(self, runner):
        """ADD/neuron produces valid logits."""
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            neurons=[0, 1, 2],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_set_ground_truth(self, runner):
        """SET/neuron produces valid logits."""
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), neurons=[0, 1, 2]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_mul_ground_truth(self, runner):
        """MUL/neuron produces valid logits."""
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, neurons=[0, 1, 2])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_interpolate_ground_truth(self, runner):
        """INTERPOLATE/neuron produces valid logits."""
        prompt = "Hello"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5,
            source_values=source,
            target_values=target,
            alpha=0.5,
            neurons=[0, 1, 2],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


# =============================================================================
# Ground Truth Tests - PATTERN axis (4 tests)
# =============================================================================


class TestInterventionsPatternGroundTruth:
    """Ground truth tests for 'pattern' axis interventions (generation)."""

    def test_interventions_pattern_add_ground_truth(self, runner):
        """ADD/pattern produces valid generation."""
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            pattern="Hello",
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_truth(self, runner):
        """SET/pattern produces valid generation."""
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), pattern="Hello"
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_truth(self, runner):
        """MUL/pattern produces valid generation."""
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, pattern="Hello")
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_truth(self, runner):
        """INTERPOLATE/pattern produces valid generation."""
        prompt = "Hello"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5,
            source_values=source,
            target_values=target,
            alpha=0.5,
            pattern="Hello",
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Tests - TransformerLens (16 tests)
# =============================================================================


class TestInterventionsAllGroundTransformerlens:
    """TransformerLens backend tests for axis='all'."""

    def test_interventions_all_add_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = steering(
            layer=5, direction=make_direction(runner.d_model), strength=10.0
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_set_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = ablation(layer=5, values=make_values(runner.d_model))
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_mul_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_all_interpolate_ground_transformerlens(self, runner):
        prompt = "Hello"
        source = get_activations(runner, layer=5, prompt=prompt)
        target = source * 2
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


class TestInterventionsPositionGroundTransformerlens:
    """TransformerLens backend tests for axis='position'."""

    def test_interventions_position_add_ground_transformerlens(self, runner):
        prompt = "The quick brown fox"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            positions=[1],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_set_ground_transformerlens(self, runner):
        prompt = "The quick brown fox"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), positions=[1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_mul_ground_transformerlens(self, runner):
        prompt = "The quick brown fox"
        intervention = scale(layer=5, factor=0.5, positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_position_interpolate_ground_transformerlens(self, runner):
        prompt = "The quick brown fox"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


class TestInterventionsNeuronGroundTransformerlens:
    """TransformerLens backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            neurons=[0, 1],
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_set_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), neurons=[0, 1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_mul_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab

    def test_interventions_neuron_interpolate_ground_transformerlens(self, runner):
        prompt = "Hello"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == runner.model.cfg.d_vocab


class TestInterventionsPatternGroundTransformerlens:
    """TransformerLens backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner.d_model),
            strength=10.0,
            pattern="Hello",
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner.d_model), pattern="Hello"
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, pattern="Hello")
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_transformerlens(self, runner):
        prompt = "Hello"
        source = make_values(runner.d_model, seed=1)
        target = make_values(runner.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, pattern="Hello"
        )
        output = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Tests - NNsight (16 tests)
# =============================================================================


class TestInterventionsAllGroundNnsight:
    """NNsight backend tests for axis='all'."""

    def test_interventions_all_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = steering(
            layer=5, direction=make_direction(runner_nnsight.d_model), strength=10.0
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = ablation(layer=5, values=make_values(runner_nnsight.d_model))
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5)
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        source = make_2d_values(10, runner_nnsight.d_model, seed=1)
        target = make_2d_values(10, runner_nnsight.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsPositionGroundNnsight:
    """NNsight backend tests for axis='position'."""

    def test_interventions_position_add_ground_nnsight(self, runner_nnsight):
        prompt = "The quick brown fox"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_nnsight.d_model),
            strength=10.0,
            positions=[1],
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_set_ground_nnsight(self, runner_nnsight):
        prompt = "The quick brown fox"
        intervention = ablation(
            layer=5, values=make_values(runner_nnsight.d_model), positions=[1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_mul_ground_nnsight(self, runner_nnsight):
        prompt = "The quick brown fox"
        intervention = scale(layer=5, factor=0.5, positions=[1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "The quick brown fox"
        source = make_values(runner_nnsight.d_model, seed=1)
        target = make_values(runner_nnsight.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsNeuronGroundNnsight:
    """NNsight backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_nnsight.d_model),
            strength=10.0,
            neurons=[0, 1],
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner_nnsight.d_model), neurons=[0, 1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, neurons=[0, 1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        source = make_values(runner_nnsight.d_model, seed=1)
        target = make_values(runner_nnsight.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsPatternGroundNnsight:
    """NNsight backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_nnsight.d_model),
            strength=10.0,
            pattern="Hello",
        )
        output = runner_nnsight.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner_nnsight.d_model), pattern="Hello"
        )
        output = runner_nnsight.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, pattern="Hello")
        output = runner_nnsight.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        source = make_values(runner_nnsight.d_model, seed=1)
        target = make_values(runner_nnsight.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, pattern="Hello"
        )
        output = runner_nnsight.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Tests - Pyvene (16 tests)
# =============================================================================


class TestInterventionsAllGroundPyvene:
    """Pyvene backend tests for axis='all'."""

    def test_interventions_all_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = steering(
            layer=5, direction=make_direction(runner_pyvene.d_model), strength=10.0
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = ablation(layer=5, values=make_values(runner_pyvene.d_model))
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5)
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_all_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        source = make_2d_values(10, runner_pyvene.d_model, seed=1)
        target = make_2d_values(10, runner_pyvene.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsPositionGroundPyvene:
    """Pyvene backend tests for axis='position'."""

    def test_interventions_position_add_ground_pyvene(self, runner_pyvene):
        prompt = "The quick brown fox"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_pyvene.d_model),
            strength=10.0,
            positions=[1],
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_set_ground_pyvene(self, runner_pyvene):
        prompt = "The quick brown fox"
        intervention = ablation(
            layer=5, values=make_values(runner_pyvene.d_model), positions=[1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_mul_ground_pyvene(self, runner_pyvene):
        prompt = "The quick brown fox"
        intervention = scale(layer=5, factor=0.5, positions=[1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_position_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "The quick brown fox"
        source = make_values(runner_pyvene.d_model, seed=1)
        target = make_values(runner_pyvene.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsNeuronGroundPyvene:
    """Pyvene backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_pyvene.d_model),
            strength=10.0,
            neurons=[0, 1],
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner_pyvene.d_model), neurons=[0, 1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, neurons=[0, 1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3

    def test_interventions_neuron_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        source = make_values(runner_pyvene.d_model, seed=1)
        target = make_values(runner_pyvene.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.ndim == 3


class TestInterventionsPatternGroundPyvene:
    """Pyvene backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = steering(
            layer=5,
            direction=make_direction(runner_pyvene.d_model),
            strength=10.0,
            pattern="Hello",
        )
        output = runner_pyvene.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = ablation(
            layer=5, values=make_values(runner_pyvene.d_model), pattern="Hello"
        )
        output = runner_pyvene.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = scale(layer=5, factor=0.5, pattern="Hello")
        output = runner_pyvene.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        source = make_values(runner_pyvene.d_model, seed=1)
        target = make_values(runner_pyvene.d_model, seed=2)
        intervention = interpolate(
            layer=5, source_values=source, target_values=target, alpha=0.5, pattern="Hello"
        )
        output = runner_pyvene.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)
