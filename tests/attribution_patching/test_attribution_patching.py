"""Tests for attribution patching module."""

import pytest
import numpy as np
import torch

from src.attribution_patching import (
    AttributionMetric,
    AttributionTarget,
    AttributionScore,
    AttributionPatchingResult,
    AggregatedAttributionResult,
    attribute_simple,
)
from src.attribution_patching.attribution_algorithms import (
    build_position_arrays,
    compute_attribution_vectorized,
)


class TestAttributionMetric:
    """Tests for AttributionMetric class."""

    def test_metric_creation(self):
        """Test basic metric creation."""
        metric = AttributionMetric(
            target_token_ids=(100, 200),
            clean_logit_diff=1.5,
            corrupted_logit_diff=-0.5,
        )
        assert metric.target_token_ids == (100, 200)
        assert metric.clean_logit_diff == 1.5
        assert metric.corrupted_logit_diff == -0.5

    def test_metric_diff(self):
        """Test diff property computation."""
        metric = AttributionMetric(
            target_token_ids=(100, 200),
            clean_logit_diff=2.0,
            corrupted_logit_diff=-1.0,
        )
        assert metric.diff == 3.0

    def test_compute_raw(self):
        """Test compute_raw returns differentiable tensor."""
        metric = AttributionMetric(target_token_ids=(5, 10))
        # Create dummy logits [batch, seq, vocab]
        logits = torch.randn(1, 10, 100, requires_grad=True)
        result = metric.compute_raw(logits)

        assert result.requires_grad
        assert result.ndim == 0  # Scalar


class TestAttributionTarget:
    """Tests for AttributionTarget class."""

    def test_default_target(self):
        """Test default target configuration."""
        target = AttributionTarget()
        assert target.position_mode == "all"
        assert target.layers == "all"
        assert "standard" in target.methods
        assert "eap" in target.methods

    def test_standard_only(self):
        """Test standard_only factory."""
        target = AttributionTarget.standard_only()
        assert target.methods == ["standard"]

    def test_with_ig(self):
        """Test with_ig factory."""
        target = AttributionTarget.with_ig(steps=20)
        assert "eap_ig" in target.methods
        assert target.ig_steps == 20

    def test_resolve_layers_all(self):
        """Test layer resolution with 'all'."""
        target = AttributionTarget(layers="all")
        available = [0, 1, 2, 3, 4]
        resolved = target.resolve_layers(available)
        assert resolved == available

    def test_resolve_layers_specific(self):
        """Test layer resolution with specific layers."""
        target = AttributionTarget(layers=[1, 3])
        available = [0, 1, 2, 3, 4]
        resolved = target.resolve_layers(available)
        assert resolved == [1, 3]


class TestAttributionScore:
    """Tests for AttributionScore class."""

    def test_score_creation(self):
        """Test basic score creation."""
        score = AttributionScore(
            layer=5,
            position=10,
            score=0.75,
            component="resid_post",
        )
        assert score.layer == 5
        assert score.position == 10
        assert score.score == 0.75

    def test_score_sorting(self):
        """Test scores sort by absolute value."""
        scores = [
            AttributionScore(layer=0, position=0, score=0.5),
            AttributionScore(layer=0, position=1, score=-0.9),
            AttributionScore(layer=0, position=2, score=0.3),
        ]
        sorted_scores = sorted(scores)
        # Should be sorted by abs(score) descending
        assert sorted_scores[0].score == -0.9
        assert sorted_scores[1].score == 0.5
        assert sorted_scores[2].score == 0.3


class TestAttributionPatchingResult:
    """Tests for AttributionPatchingResult class."""

    def test_result_creation(self):
        """Test basic result creation."""
        scores = np.random.randn(10, 20)
        result = AttributionPatchingResult(
            scores=scores,
            layers=list(range(10)),
            component="resid_post",
            method="standard",
        )
        assert result.n_layers == 10
        assert result.n_positions == 20

    def test_get_top_scores(self):
        """Test getting top scores."""
        scores = np.zeros((3, 5))
        scores[1, 2] = 10.0
        scores[2, 4] = -8.0
        scores[0, 0] = 5.0

        result = AttributionPatchingResult(
            scores=scores,
            layers=[0, 1, 2],
            component="resid_post",
            method="standard",
        )

        top = result.get_top_scores(3)
        assert len(top) == 3
        assert top[0].layer == 1
        assert top[0].position == 2
        assert top[0].score == 10.0

    def test_get_top_targets(self):
        """Test conversion to activation patching targets."""
        scores = np.zeros((3, 5))
        scores[1, 2] = 10.0

        result = AttributionPatchingResult(
            scores=scores,
            layers=[0, 1, 2],
            component="resid_post",
            method="standard",
        )

        targets = result.get_top_targets_for_activation_patching(1)
        assert len(targets) == 1
        assert targets[0].token_positions == [2]
        assert targets[0].layers == [1]


class TestAggregatedAttributionResult:
    """Tests for AggregatedAttributionResult class."""

    def test_empty_result(self):
        """Test empty result."""
        result = AggregatedAttributionResult()
        assert len(result.results) == 0
        assert result.n_pairs == 1

    def test_aggregate_single(self):
        """Test aggregating single result."""
        scores = np.random.randn(5, 10)
        inner = AttributionPatchingResult(
            scores=scores,
            layers=[0, 1, 2, 3, 4],
            method="standard",
        )
        result = AggregatedAttributionResult(results={"test": inner})

        aggregated = AggregatedAttributionResult.aggregate([result])
        assert "test" in aggregated.results

    def test_get_consensus_target(self):
        """Test consensus target finding."""
        # Create results where position 5, layer 2 appears in multiple methods
        scores1 = np.zeros((5, 10))
        scores1[2, 5] = 10.0
        scores2 = np.zeros((5, 10))
        scores2[2, 5] = 8.0

        result = AggregatedAttributionResult(
            results={
                "method1": AttributionPatchingResult(scores=scores1, layers=list(range(5)), method="standard"),
                "method2": AttributionPatchingResult(scores=scores2, layers=list(range(5)), method="eap"),
            }
        )

        target = result.get_consensus_target(n=1, min_methods=2)
        assert target is not None
        assert target.layers == [2]
        assert target.token_positions == [5]
        assert target.position_mode == "explicit"

    def test_get_layer_target(self):
        """Test layer-based target finding."""
        # Create results where layer 2 has high scores across methods
        scores1 = np.zeros((5, 10))
        scores1[2, 3] = 10.0
        scores1[2, 7] = 8.0
        scores2 = np.zeros((5, 10))
        scores2[2, 4] = 9.0
        scores2[3, 1] = 7.0

        result = AggregatedAttributionResult(
            results={
                "method1": AttributionPatchingResult(scores=scores1, layers=list(range(5)), method="standard"),
                "method2": AttributionPatchingResult(scores=scores2, layers=list(range(5)), method="eap"),
            }
        )

        target = result.get_layer_target(n_layers=2, min_methods=1)
        assert target is not None
        assert 2 in target.layers  # Layer 2 should be included (highest scores)
        assert target.position_mode == "all"  # Should patch all positions
        assert target.token_positions is None  # No specific positions

    def test_get_union_target(self):
        """Test union target finding."""
        # Create results with different top positions per method
        scores1 = np.zeros((5, 10))
        scores1[2, 3] = 10.0  # Layer 2, pos 3
        scores1[1, 5] = 8.0   # Layer 1, pos 5
        scores2 = np.zeros((5, 10))
        scores2[2, 4] = 9.0   # Layer 2, pos 4
        scores2[3, 3] = 7.0   # Layer 3, pos 3

        result = AggregatedAttributionResult(
            results={
                "method1": AttributionPatchingResult(scores=scores1, layers=list(range(5)), method="standard"),
                "method2": AttributionPatchingResult(scores=scores2, layers=list(range(5)), method="eap"),
            }
        )

        target = result.get_union_target(n=2, min_methods=1)
        assert target is not None
        assert target.position_mode == "explicit"
        # Should include positions 3, 4, 5 (union of top 2 from each method)
        assert 3 in target.token_positions
        assert 4 in target.token_positions or 5 in target.token_positions

    def test_get_recommended_target_modes(self):
        """Test get_recommended_target with different modes."""
        scores = np.zeros((5, 10))
        scores[2, 5] = 10.0

        result = AggregatedAttributionResult(
            results={
                "method1": AttributionPatchingResult(scores=scores, layers=list(range(5)), method="standard"),
            }
        )

        # Layer mode
        layer_target = result.get_recommended_target(n=2, mode="layer")
        assert layer_target is not None
        assert layer_target.position_mode == "all"

        # Union mode
        union_target = result.get_recommended_target(n=2, mode="union")
        assert union_target is not None
        assert union_target.position_mode == "explicit"

        # Consensus mode
        consensus_target = result.get_recommended_target(n=2, mode="consensus")
        # With only one method, consensus with min_methods=2 returns None
        assert consensus_target is None


class TestCoreFunctions:
    """Tests for core attribution functions."""

    def test_build_position_arrays_identity(self):
        """Test position arrays with identity mapping."""
        pos_mapping = {i: i for i in range(10)}
        clean_pos, corr_pos, valid = build_position_arrays(pos_mapping, 10, 10)

        assert len(clean_pos) == 10
        np.testing.assert_array_equal(clean_pos, corr_pos)
        assert all(valid)

    def test_build_position_arrays_offset(self):
        """Test position arrays with offset mapping."""
        pos_mapping = {i: i + 2 for i in range(5)}
        clean_pos, corr_pos, valid = build_position_arrays(pos_mapping, 5, 10)

        np.testing.assert_array_equal(corr_pos, clean_pos + 2)
        assert all(valid)

    def test_build_position_arrays_out_of_bounds(self):
        """Test position arrays with out of bounds mapping."""
        pos_mapping = {0: 0, 1: 5, 2: 10}  # position 2 maps to 10, out of bounds for corr_len=8
        clean_pos, corr_pos, valid = build_position_arrays(pos_mapping, 3, 8)

        assert valid[0] == True
        assert valid[1] == True
        assert valid[2] == False  # 10 >= 8, out of bounds

    def test_compute_attribution_vectorized(self):
        """Test vectorized attribution computation."""
        # Create test tensors
        clean_act = torch.randn(1, 5, 10)
        corr_act = torch.randn(1, 5, 10)
        grad = torch.ones(1, 5, 10)

        clean_pos = np.arange(5)
        corr_pos = np.arange(5)
        valid = np.ones(5, dtype=bool)

        scores = compute_attribution_vectorized(
            clean_act, corr_act, grad, clean_pos, corr_pos, valid
        )

        assert scores.shape == (5,)
        # Attribution = (clean - corr) * grad, summed along features
        expected = (clean_act[0] - corr_act[0]).sum(dim=-1).numpy()
        np.testing.assert_array_almost_equal(scores, expected, decimal=5)
