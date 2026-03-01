"""Numerical correctness tests for intervention modes.

These tests verify that each intervention mode (add, set, mul, interpolate)
produces NUMERICALLY CORRECT outputs, not just correct shapes.

Each mode has specific mathematical semantics:
- ADD: act_out = act_in + values
- SET: act_out = values (replaces activation completely)
- MUL: act_out = act_in * values
- INTERPOLATE: act_out = source + alpha * (target - source)

Tests cover:
1. Basic numerical correctness for each mode
2. Edge cases (zeros, negative values, large magnitudes)
3. Different axis targeting (all, position, neuron)
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
    scale,
    interpolate,
)


# =============================================================================
# Test Constants
# =============================================================================

D_MODEL = 4  # Small for easy manual verification
TOLERANCE = 1e-6


# =============================================================================
# ADD Mode Numerical Tests
# =============================================================================


class TestAddModeNumerical:
    """Test ADD mode: result = activation + values"""

    def test_add_basic_numerical(self):
        """ADD mode with known inputs produces exact expected output."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1, 1, 4]
        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[11.0, 22.0, 33.0, 44.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE), (
            f"Expected {expected}, got {result}"
        )

    def test_add_negative_values(self):
        """ADD mode with negative values subtracts from activation."""
        act = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
        values = np.array([-5.0, -10.0, -15.0, -20.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_add_zero_values(self):
        """ADD mode with zero values leaves activation unchanged."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        assert torch.allclose(result, act, atol=TOLERANCE)

    def test_add_large_magnitude_values(self):
        """ADD mode with large values adds correctly."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[1001.0, 2002.0, 3003.0, 4004.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_add_small_magnitude_values(self):
        """ADD mode with very small values adds correctly."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[1.001, 2.002, 3.003, 4.004]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_add_multi_position_all_axis(self):
        """ADD mode applies to all positions when axis='all'."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])  # [1, 3, 4]
        values = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[11.0, 12.0, 13.0, 14.0],
                                  [15.0, 16.0, 17.0, 18.0],
                                  [19.0, 20.0, 21.0, 22.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_add_with_strength_scaling(self):
        """ADD mode respects strength parameter (values are pre-scaled)."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        # Use steering helper which applies strength
        direction = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        intervention = steering(layer=0, direction=direction, strength=5.0, normalize=False)
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # scaled_values = direction * strength = [5, 5, 5, 5]
        expected = torch.tensor([[[6.0, 7.0, 8.0, 9.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)


# =============================================================================
# SET Mode Numerical Tests
# =============================================================================


class TestSetModeNumerical:
    """Test SET mode: result = values (replaces activation)"""

    def test_set_replaces_activation_completely(self):
        """SET mode completely replaces activation values."""
        act = torch.tensor([[[100.0, 200.0, 300.0, 400.0]]])  # Original values
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)
        # Verify original values do NOT appear
        assert not torch.any(result == 100.0)
        assert not torch.any(result == 200.0)

    def test_set_to_zeros(self):
        """SET mode can zero out activations."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.zeros(1, 1, 4)
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_set_to_negative_values(self):
        """SET mode can set negative values."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_set_multi_position(self):
        """SET mode applies to all positions with 1D values."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])  # [1, 3, 4]
        values = np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.full((1, 3, 4), 99.0)
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_set_with_2d_values(self):
        """SET mode with 2D values sets each position independently."""
        act = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]]])  # [1, 3, 4]
        values = np.array([[1.0, 1.0, 1.0, 1.0],
                           [2.0, 2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0, 3.0]], dtype=np.float32)  # [3, 4]

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[1.0, 1.0, 1.0, 1.0],
                                  [2.0, 2.0, 2.0, 2.0],
                                  [3.0, 3.0, 3.0, 3.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_set_ablation_helper(self):
        """ablation() helper creates correct SET intervention."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        # Default ablation zeros out
        intervention = ablation(layer=0)
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Should be zeros (or close to it due to scalar expansion)
        assert result.sum().abs() < TOLERANCE


# =============================================================================
# MUL Mode Numerical Tests
# =============================================================================


class TestMulModeNumerical:
    """Test MUL mode: result = activation * values"""

    def test_mul_basic_scaling(self):
        """MUL mode multiplies activation by scalar factor."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([2.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_zero_factor(self):
        """MUL mode with factor=0 zeros out activation."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([0.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.zeros(1, 1, 4)
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_negative_factor(self):
        """MUL mode with negative factor inverts sign."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([-1.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_fractional_factor(self):
        """MUL mode with factor < 1 reduces magnitude."""
        act = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
        values = np.array([0.5], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_factor_one_unchanged(self):
        """MUL mode with factor=1 leaves activation unchanged."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([1.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        assert torch.allclose(result, act, atol=TOLERANCE)

    def test_mul_per_element_scaling(self):
        """MUL mode with vector values scales each element differently."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[1.0, 4.0, 9.0, 16.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_multi_position(self):
        """MUL mode applies to all positions."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0]]])  # [1, 2, 4]
        values = np.array([2.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[2.0, 4.0, 6.0, 8.0],
                                  [10.0, 12.0, 14.0, 16.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_mul_scale_helper(self):
        """scale() helper creates correct MUL intervention."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        intervention = scale(layer=0, factor=3.0)
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[3.0, 6.0, 9.0, 12.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)


# =============================================================================
# INTERPOLATE Mode Numerical Tests
# =============================================================================


class TestInterpolateModeNumerical:
    """Test INTERPOLATE mode: result = source + alpha * (target - source)"""

    def test_interpolate_alpha_zero_gives_source(self):
        """INTERPOLATE with alpha=0 returns source values."""
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.0,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.ones(1, 1, 4) * 999  # Shouldn't matter for interpolate
        result = hook(act)

        expected = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE), (
            f"alpha=0 should give source. Expected {expected}, got {result}"
        )

    def test_interpolate_alpha_one_gives_target(self):
        """INTERPOLATE with alpha=1 returns target values."""
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=1.0,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.ones(1, 1, 4) * 999
        result = hook(act)

        expected = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE), (
            f"alpha=1 should give target. Expected {expected}, got {result}"
        )

    def test_interpolate_alpha_half_gives_midpoint(self):
        """INTERPOLATE with alpha=0.5 returns midpoint."""
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.5,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.ones(1, 1, 4) * 999
        result = hook(act)

        # midpoint = source + 0.5 * (target - source) = 0 + 0.5 * 10 = 5
        expected = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE), (
            f"alpha=0.5 should give midpoint. Expected {expected}, got {result}"
        )

    def test_interpolate_alpha_quarter(self):
        """INTERPOLATE with alpha=0.25 returns 1/4 towards target."""
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.25,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 1, 4)
        result = hook(act)

        # result = 0 + 0.25 * (100 - 0) = 25
        expected = torch.tensor([[[25.0, 50.0, 75.0, 100.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_interpolate_with_nonzero_source(self):
        """INTERPOLATE works correctly when source is nonzero."""
        source = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        target = np.array([[110.0, 120.0, 130.0, 140.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.5,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 1, 4)
        result = hook(act)

        # result = source + 0.5 * (target - source)
        # = [10,20,30,40] + 0.5 * [100,100,100,100]
        # = [10,20,30,40] + [50,50,50,50] = [60, 70, 80, 90]
        expected = torch.tensor([[[60.0, 70.0, 80.0, 90.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_interpolate_negative_source_positive_target(self):
        """INTERPOLATE works when interpolating from negative to positive."""
        source = np.array([[-10.0, -20.0, -30.0, -40.0]], dtype=np.float32)
        target = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.5,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 1, 4)
        result = hook(act)

        # midpoint between -x and +x is 0
        expected = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_interpolate_multiple_positions(self):
        """INTERPOLATE works with multiple sequence positions."""
        source = np.array([[0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # [3, 4]
        target = np.array([[10.0, 10.0, 10.0, 10.0],
                           [20.0, 20.0, 20.0, 20.0],
                           [30.0, 30.0, 30.0, 30.0]], dtype=np.float32)  # [3, 4]

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.5,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 3, 4)
        result = hook(act)

        expected = torch.tensor([[[5.0, 5.0, 5.0, 5.0],
                                  [10.0, 10.0, 10.0, 10.0],
                                  [15.0, 15.0, 15.0, 15.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_interpolate_helper_function(self):
        """interpolate() helper creates correct intervention."""
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        intervention = interpolate(
            layer=0,
            source_values=source,
            target_values=target,
            alpha=0.3,
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 1, 4)
        result = hook(act)

        # result = 0 + 0.3 * 100 = 30
        expected = torch.tensor([[[30.0, 30.0, 30.0, 30.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_interpolate_formula_verification(self):
        """Verify exact interpolation formula: source + alpha * (target - source)."""
        # Use arbitrary values to verify formula
        source = np.array([[3.0, 7.0, 11.0, 13.0]], dtype=np.float32)
        target = np.array([[23.0, 37.0, 51.0, 73.0]], dtype=np.float32)
        alpha = 0.35

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=alpha,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")

        act = torch.zeros(1, 1, 4)
        result = hook(act)

        # Manual calculation
        expected_values = source + alpha * (target - source)
        expected = torch.tensor(expected_values).unsqueeze(0)
        assert torch.allclose(result, expected, atol=TOLERANCE), (
            f"Formula mismatch. Expected {expected}, got {result}"
        )


# =============================================================================
# Position Axis Targeting Tests
# =============================================================================


class TestPositionAxisNumerical:
    """Test axis='position' targeting for all modes."""

    def test_add_position_only_affects_target_position(self):
        """ADD with position targeting only modifies specified position."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])  # [1, 3, 4]
        values = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.at_positions([1]),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Position 0 and 2 unchanged
        assert torch.allclose(result[0, 0], act[0, 0], atol=TOLERANCE)
        assert torch.allclose(result[0, 2], act[0, 2], atol=TOLERANCE)
        # Position 1 modified: [5,6,7,8] + [10,10,10,10] = [15,16,17,18]
        expected_pos1 = torch.tensor([15.0, 16.0, 17.0, 18.0])
        assert torch.allclose(result[0, 1], expected_pos1, atol=TOLERANCE)

    def test_set_position_only_affects_target_position(self):
        """SET with position targeting only replaces specified position."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])
        values = np.array([[99.0, 99.0, 99.0, 99.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="set",
            values=values,
            target=InterventionTarget.at_positions([0]),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Position 0 replaced
        expected_pos0 = torch.tensor([99.0, 99.0, 99.0, 99.0])
        assert torch.allclose(result[0, 0], expected_pos0, atol=TOLERANCE)
        # Positions 1 and 2 unchanged
        assert torch.allclose(result[0, 1], act[0, 1], atol=TOLERANCE)
        assert torch.allclose(result[0, 2], act[0, 2], atol=TOLERANCE)

    def test_mul_position_only_affects_target_position(self):
        """MUL with position targeting only scales specified position."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])
        values = np.array([[2.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.at_positions([2]),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Positions 0 and 1 unchanged
        assert torch.allclose(result[0, 0], act[0, 0], atol=TOLERANCE)
        assert torch.allclose(result[0, 1], act[0, 1], atol=TOLERANCE)
        # Position 2 scaled: [9,10,11,12] * 2 = [18,20,22,24]
        expected_pos2 = torch.tensor([18.0, 20.0, 22.0, 24.0])
        assert torch.allclose(result[0, 2], expected_pos2, atol=TOLERANCE)

    def test_interpolate_position_only_affects_target_position(self):
        """INTERPOLATE with position targeting only interpolates specified position."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]])
        source = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="interpolate",
            values=source,
            target_values=target,
            alpha=0.5,
            target=InterventionTarget.at_positions([1]),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Positions 0 and 2 unchanged
        assert torch.allclose(result[0, 0], act[0, 0], atol=TOLERANCE)
        assert torch.allclose(result[0, 2], act[0, 2], atol=TOLERANCE)
        # Position 1 interpolated: 0 + 0.5 * 100 = 50
        expected_pos1 = torch.tensor([50.0, 50.0, 50.0, 50.0])
        assert torch.allclose(result[0, 1], expected_pos1, atol=TOLERANCE)

    def test_multiple_positions_add(self):
        """ADD with multiple position targets.

        Note: When using 2D values with multiple positions, the LAST row of
        values is used for all targeted positions (current implementation).
        """
        act = torch.tensor([[[1.0, 1.0, 1.0, 1.0],
                             [2.0, 2.0, 2.0, 2.0],
                             [3.0, 3.0, 3.0, 3.0],
                             [4.0, 4.0, 4.0, 4.0]]])  # [1, 4, 4]
        # Use 1D values for clarity - same value applied to all positions
        values = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.at_positions([0, 2]),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Position 0: [1,1,1,1] + [10,10,10,10] = [11,11,11,11]
        expected_pos0 = torch.tensor([11.0, 11.0, 11.0, 11.0])
        assert torch.allclose(result[0, 0], expected_pos0, atol=TOLERANCE)
        # Position 1: unchanged
        assert torch.allclose(result[0, 1], act[0, 1], atol=TOLERANCE)
        # Position 2: [3,3,3,3] + [10,10,10,10] = [13,13,13,13]
        expected_pos2 = torch.tensor([13.0, 13.0, 13.0, 13.0])
        assert torch.allclose(result[0, 2], expected_pos2, atol=TOLERANCE)
        # Position 3: unchanged
        assert torch.allclose(result[0, 3], act[0, 3], atol=TOLERANCE)


# =============================================================================
# Batch Dimension Tests
# =============================================================================


class TestBatchDimensionNumerical:
    """Test that interventions work correctly with batch size > 1."""

    def test_add_batch_applies_to_all_samples(self):
        """ADD mode applies same values to all batch samples."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]],
                            [[10.0, 20.0, 30.0, 40.0]]])  # [2, 1, 4]
        values = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Batch 0: [1,2,3,4] + [5,5,5,5] = [6,7,8,9]
        expected_b0 = torch.tensor([[6.0, 7.0, 8.0, 9.0]])
        assert torch.allclose(result[0], expected_b0, atol=TOLERANCE)
        # Batch 1: [10,20,30,40] + [5,5,5,5] = [15,25,35,45]
        expected_b1 = torch.tensor([[15.0, 25.0, 35.0, 45.0]])
        assert torch.allclose(result[1], expected_b1, atol=TOLERANCE)

    def test_mul_batch_applies_to_all_samples(self):
        """MUL mode applies same factor to all batch samples."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]],
                            [[10.0, 20.0, 30.0, 40.0]]])  # [2, 1, 4]
        values = np.array([2.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="mul",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Batch 0: [1,2,3,4] * 2 = [2,4,6,8]
        expected_b0 = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
        assert torch.allclose(result[0], expected_b0, atol=TOLERANCE)
        # Batch 1: [10,20,30,40] * 2 = [20,40,60,80]
        expected_b1 = torch.tensor([[20.0, 40.0, 60.0, 80.0]])
        assert torch.allclose(result[1], expected_b1, atol=TOLERANCE)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCasesNumerical:
    """Test edge cases and boundary conditions."""

    def test_very_large_values(self):
        """Operations work with very large values."""
        act = torch.tensor([[[1e6, 2e6, 3e6, 4e6]]])
        values = np.array([1e6, 1e6, 1e6, 1e6], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[2e6, 3e6, 4e6, 5e6]]])
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_very_small_values(self):
        """Operations work with very small values."""
        act = torch.tensor([[[1e-6, 2e-6, 3e-6, 4e-6]]])
        values = np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[2e-6, 3e-6, 4e-6, 5e-6]]])
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_mixed_sign_values(self):
        """Operations work with mixed positive/negative values."""
        act = torch.tensor([[[-1.0, 2.0, -3.0, 4.0]]])
        values = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)

    def test_out_of_bounds_position_skipped(self):
        """Position targets beyond sequence length are safely skipped."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0]]])  # [1, 2, 4]
        values = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.at_positions([100]),  # Out of bounds
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        # Should be unchanged since position 100 doesn't exist
        assert torch.allclose(result, act, atol=TOLERANCE)

    def test_single_element_sequence(self):
        """Operations work with single position sequences."""
        act = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1, 1, 4]
        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        intervention = Intervention(
            layer=0,
            mode="add",
            values=values,
            target=InterventionTarget.all(),
        )
        hook, _ = create_intervention_hook(intervention, torch.float32, "cpu")
        result = hook(act.clone())

        expected = torch.tensor([[[11.0, 22.0, 33.0, 44.0]]])
        assert torch.allclose(result, expected, atol=TOLERANCE)
