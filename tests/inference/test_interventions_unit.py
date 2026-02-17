"""Unit tests for intervention data classes and helper functions.

Tests the Intervention, InterventionTarget classes and helper functions (steering, ablation, etc.)
These tests do NOT require a model - they test the intervention creation logic only.

For testing intervention EFFECTS with models, see test_interventions_qwen.py
which tests all backends with a real model and verifies consistency.
"""

import numpy as np
import pytest
import torch

from src.inference.interventions import (
    Intervention,
    InterventionTarget,
    create_intervention_hook,
    steering,
    ablation,
    patch,
    scale,
    interpolate,
)


# Test constants
D_MODEL = 64


def make_direction(d_model: int = D_MODEL) -> np.ndarray:
    """Unit direction vector."""
    vec = np.ones(d_model, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# =============================================================================
# InterventionTarget Data Class Tests
# =============================================================================


class TestInterventionTarget:
    """Test InterventionTarget class for specifying intervention locations."""

    def test_all(self):
        t = InterventionTarget.all()
        assert t.axis == "all"

    def test_at_positions_single(self):
        t = InterventionTarget.at_positions(3)
        assert t.axis == "position"
        assert t.positions == [3]

    def test_at_positions_multiple(self):
        t = InterventionTarget.at_positions([1, 3, 5])
        assert t.positions == [1, 3, 5]

    def test_at_neurons(self):
        t = InterventionTarget.at_neurons([0, 2])
        assert t.axis == "neuron"
        assert t.neurons == [0, 2]

    def test_on_pattern(self):
        t = InterventionTarget.on_pattern("I select:")
        assert t.axis == "pattern"
        assert t.pattern == "I select:"

    def test_validation_position_requires_positions(self):
        with pytest.raises(ValueError):
            InterventionTarget(axis="position")

    def test_validation_neuron_requires_neurons(self):
        with pytest.raises(ValueError):
            InterventionTarget(axis="neuron")

    def test_validation_pattern_requires_pattern(self):
        with pytest.raises(ValueError):
            InterventionTarget(axis="pattern")


# =============================================================================
# Intervention Properties Tests
# =============================================================================


class TestInterventionProperties:
    """Test Intervention class properties."""

    def test_hook_name_default_component(self):
        intervention = steering(layer=5, direction=make_direction())
        assert intervention.hook_name == "blocks.5.hook_resid_post"

    def test_hook_name_attn_out(self):
        intervention = steering(layer=5, direction=make_direction(), component="attn_out")
        assert intervention.hook_name == "blocks.5.hook_attn_out"

    def test_hook_name_mlp_out(self):
        intervention = steering(layer=5, direction=make_direction(), component="mlp_out")
        assert intervention.hook_name == "blocks.5.hook_mlp_out"

    def test_scaled_values(self):
        direction = make_direction()
        strength = 10.0
        intervention = steering(layer=0, direction=direction, strength=strength)
        expected = direction * strength
        np.testing.assert_allclose(intervention.scaled_values, expected, rtol=1e-5)

    def test_mode_add(self):
        intervention = steering(layer=0, direction=make_direction())
        assert intervention.mode == "add"

    def test_mode_set(self):
        intervention = ablation(layer=0)
        assert intervention.mode == "set"

    def test_mode_mul(self):
        intervention = scale(layer=0, factor=2.0)
        assert intervention.mode == "mul"

    def test_mode_interpolate(self):
        source = np.zeros((3, D_MODEL), dtype=np.float32)
        target = np.ones((3, D_MODEL), dtype=np.float32)
        intervention = interpolate(layer=0, source_values=source, target_values=target, alpha=0.5)
        assert intervention.mode == "interpolate"


# =============================================================================
# Helper Function Tests - steering()
# =============================================================================


class TestSteering:
    """Test steering() helper function."""

    def test_creates_add_mode(self):
        i = steering(layer=5, direction=make_direction())
        assert i.mode == "add"

    def test_normalizes_direction_by_default(self):
        direction = np.array([3.0, 4.0])  # length 5
        i = steering(layer=5, direction=direction)
        norm = np.linalg.norm(i.values)
        assert abs(norm - 1.0) < 1e-5

    def test_no_normalize_flag(self):
        direction = np.array([3.0, 4.0])
        i = steering(layer=5, direction=direction, normalize=False)
        np.testing.assert_array_almost_equal(i.values, direction)

    def test_strength_scaling(self):
        direction = make_direction(4)
        i = steering(layer=5, direction=direction, strength=10.0)
        expected = direction * 10.0
        np.testing.assert_allclose(i.scaled_values, expected, rtol=1e-5)

    def test_position_targeting(self):
        i = steering(layer=5, direction=make_direction(), positions=[1, 3])
        assert i.target.axis == "position"
        assert i.target.positions == [1, 3]

    def test_neuron_targeting(self):
        i = steering(layer=5, direction=make_direction(), neurons=[0, 2])
        assert i.target.axis == "neuron"
        assert i.target.neurons == [0, 2]

    def test_pattern_targeting(self):
        i = steering(layer=5, direction=make_direction(), pattern="test")
        assert i.target.axis == "pattern"
        assert i.target.pattern == "test"

    def test_component_selection(self):
        i = steering(layer=5, direction=make_direction(), component="attn_out")
        assert i.component == "attn_out"
        assert i.hook_name == "blocks.5.hook_attn_out"


# =============================================================================
# Helper Function Tests - ablation()
# =============================================================================


class TestAblation:
    """Test ablation() helper function."""

    def test_creates_set_mode(self):
        i = ablation(layer=3)
        assert i.mode == "set"

    def test_defaults_to_zero(self):
        i = ablation(layer=3)
        assert i.values[0] == 0

    def test_accepts_scalar_value(self):
        i = ablation(layer=3, values=5.0)
        assert i.values[0] == 5.0

    def test_accepts_array_value(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        i = ablation(layer=3, values=values)
        np.testing.assert_array_equal(i.values, values)

    def test_position_targeting(self):
        i = ablation(layer=3, positions=[2])
        assert i.target.axis == "position"
        assert i.target.positions == [2]


# =============================================================================
# Helper Function Tests - patch()
# =============================================================================


class TestPatch:
    """Test patch() helper function."""

    def test_creates_set_mode(self):
        values = np.random.randn(10, 768)
        i = patch(layer=4, values=values)
        assert i.mode == "set"

    def test_preserves_shape(self):
        values = np.random.randn(10, 768)
        i = patch(layer=4, values=values)
        assert i.values.shape == (10, 768)

    def test_position_targeting(self):
        values = np.random.randn(768)
        i = patch(layer=4, values=values, positions=[0])
        assert i.target.axis == "position"
        assert i.target.positions == [0]


# =============================================================================
# Helper Function Tests - scale()
# =============================================================================


class TestScale:
    """Test scale() helper function."""

    def test_creates_mul_mode(self):
        i = scale(layer=5, factor=0.5)
        assert i.mode == "mul"

    def test_stores_factor(self):
        i = scale(layer=5, factor=0.5)
        assert i.values[0] == 0.5

    def test_position_targeting(self):
        i = scale(layer=5, factor=2.0, positions=[1, 2])
        assert i.target.axis == "position"
        assert i.target.positions == [1, 2]


# =============================================================================
# Helper Function Tests - interpolate()
# =============================================================================


class TestInterpolate:
    """Test interpolate() helper function."""

    def test_creates_interpolate_mode(self):
        source = np.zeros((3, 4))
        target = np.ones((3, 4))
        i = interpolate(layer=5, source_values=source, target_values=target, alpha=0.5)
        assert i.mode == "interpolate"

    def test_stores_alpha(self):
        source = np.zeros((3, 4))
        target = np.ones((3, 4))
        i = interpolate(layer=5, source_values=source, target_values=target, alpha=0.7)
        assert i.alpha == 0.7

    def test_stores_source_as_values(self):
        source = np.zeros((3, 4))
        target = np.ones((3, 4))
        i = interpolate(layer=5, source_values=source, target_values=target, alpha=0.5)
        np.testing.assert_array_equal(i.values, source)

    def test_stores_target_values(self):
        source = np.zeros((3, 4))
        target = np.ones((3, 4))
        i = interpolate(layer=5, source_values=source, target_values=target, alpha=0.5)
        np.testing.assert_array_equal(i.target_values, target)

    def test_requires_target_values(self):
        with pytest.raises(ValueError, match="target_values"):
            Intervention(
                layer=0,
                mode="interpolate",
                values=np.zeros(D_MODEL),
                target_values=None,
            )


# =============================================================================
# create_intervention_hook Tests - Hook Logic
# =============================================================================


class TestCreateInterventionHook:
    """Test create_intervention_hook produces correct hook functions."""

    def test_add_mode_hook(self):
        """ADD mode adds direction to activation."""
        direction = make_direction(D_MODEL)
        intervention = steering(layer=0, direction=direction, strength=2.0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook_fn(activation)

        expected = direction * 2.0
        for pos in range(3):
            np.testing.assert_allclose(result[0, pos].numpy(), expected, rtol=1e-5)

    def test_set_mode_hook_zero(self):
        """SET mode with zero sets activation to zero."""
        intervention = ablation(layer=0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.ones(1, 3, D_MODEL)
        result = hook_fn(activation)

        assert torch.allclose(result, torch.zeros_like(result))

    def test_set_mode_hook_values(self):
        """SET mode with values sets activation to those values."""
        values = np.array([1.0, 2.0, 3.0, 4.0] * (D_MODEL // 4), dtype=np.float32)
        intervention = ablation(layer=0, values=values)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook_fn(activation)

        expected = torch.tensor(values)
        for pos in range(3):
            assert torch.allclose(result[0, pos], expected)

    def test_mul_mode_hook(self):
        """MUL mode multiplies activation by factor."""
        intervention = scale(layer=0, factor=2.0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.ones(1, 3, D_MODEL)
        result = hook_fn(activation)

        assert torch.allclose(result, activation * 2)

    def test_mul_mode_zero(self):
        """MUL mode with factor=0 zeros activation."""
        intervention = scale(layer=0, factor=0.0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.ones(1, 3, D_MODEL)
        result = hook_fn(activation)

        assert result.sum() == 0

    def test_position_targeting(self):
        """Position-targeted intervention only affects specified positions."""
        direction = make_direction(D_MODEL)
        intervention = steering(layer=0, direction=direction, strength=10.0, positions=[1])
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 5, D_MODEL)
        result = hook_fn(activation.clone())

        # Position 1 should be modified
        assert result[0, 1].sum() > 0
        # Other positions should remain zero
        assert result[0, 0].sum() == 0
        assert result[0, 2].sum() == 0
        assert result[0, 3].sum() == 0
        assert result[0, 4].sum() == 0

    def test_multiple_positions(self):
        """Multiple position targets all affected."""
        direction = make_direction(D_MODEL)
        intervention = steering(layer=0, direction=direction, strength=10.0, positions=[0, 2, 4])
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 5, D_MODEL)
        result = hook_fn(activation)

        assert result[0, 0].sum() > 0
        assert result[0, 2].sum() > 0
        assert result[0, 4].sum() > 0
        assert result[0, 1].sum() == 0
        assert result[0, 3].sum() == 0

    def test_out_of_bounds_position_skipped(self):
        """Out-of-bounds positions are silently skipped."""
        intervention = steering(layer=0, direction=make_direction(), strength=10.0, positions=[100])
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.ones(1, 3, D_MODEL)
        result = hook_fn(activation)
        # Should be unchanged
        assert torch.allclose(result, activation)

    def test_interpolate_alpha_0(self):
        """Interpolate with alpha=0 returns source values."""
        source = np.ones((3, D_MODEL), dtype=np.float32)
        target = np.ones((3, D_MODEL), dtype=np.float32) * 10

        intervention = interpolate(layer=0, source_values=source, target_values=target, alpha=0.0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook_fn(activation)

        expected = torch.tensor(source)
        assert torch.allclose(result[0], expected)

    def test_interpolate_alpha_1(self):
        """Interpolate with alpha=1 returns target values."""
        source = np.ones((3, D_MODEL), dtype=np.float32)
        target = np.ones((3, D_MODEL), dtype=np.float32) * 10

        intervention = interpolate(layer=0, source_values=source, target_values=target, alpha=1.0)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook_fn(activation)

        expected = torch.tensor(target)
        assert torch.allclose(result[0], expected)

    def test_interpolate_alpha_midpoint(self):
        """Interpolate with alpha=0.5 returns midpoint."""
        source = np.zeros((3, D_MODEL), dtype=np.float32)
        target = np.ones((3, D_MODEL), dtype=np.float32) * 10

        intervention = interpolate(layer=0, source_values=source, target_values=target, alpha=0.5)
        hook_fn, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook_fn(activation)

        expected = torch.ones(3, D_MODEL) * 5
        assert torch.allclose(result[0], expected)
