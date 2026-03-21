"""Tests for attribution patching module."""

import pytest
import numpy as np
import torch

from src.attribution_patching import (
    AttributionMetric,
    AttributionSettings,
    AttributionScore,
    AttributionPatchingResult,
    AttributionSummary,
)
from src.common.token_positions import build_position_arrays
from src.attribution_patching.vectorized import compute_attribution_vectorized


class TestAttributionMetric:
    """Tests for AttributionMetric class."""

    def test_metric_creation(self):
        metric = AttributionMetric(
            target_token_ids=(100, 200),
            clean_logit_diff=1.5,
            corrupted_logit_diff=-0.5,
        )
        assert metric.target_token_ids == (100, 200)
        assert metric.clean_logit_diff == 1.5
        assert metric.corrupted_logit_diff == -0.5

    def test_metric_diff(self):
        metric = AttributionMetric(
            target_token_ids=(100, 200),
            clean_logit_diff=2.0,
            corrupted_logit_diff=-1.0,
        )
        assert metric.diff == 3.0

    def test_compute_raw(self):
        metric = AttributionMetric(target_token_ids=(5, 10))
        logits = torch.randn(1, 10, 100, requires_grad=True)
        result = metric.compute_raw(logits)
        assert result.requires_grad
        assert result.ndim == 0


class TestAttributionSettings:
    """Tests for AttributionSettings class."""

    def test_default_settings(self):
        settings = AttributionSettings()
        assert "standard" in settings.methods
        assert "eap" in settings.methods

    def test_standard_only(self):
        settings = AttributionSettings.standard_only()
        assert settings.methods == ["standard"]

    def test_with_ig(self):
        settings = AttributionSettings.with_ig(steps=20)
        assert "eap_ig" in settings.methods
        assert settings.ig_steps == 20


class TestAttributionScore:
    """Tests for AttributionScore class."""

    def test_score_creation(self):
        score = AttributionScore(layer=5, position=10, score=0.75, component="resid_post")
        assert score.layer == 5
        assert score.position == 10
        assert score.score == 0.75

    def test_score_sorting(self):
        scores = [
            AttributionScore(layer=0, position=0, score=0.5),
            AttributionScore(layer=0, position=1, score=-0.9),
            AttributionScore(layer=0, position=2, score=0.3),
        ]
        sorted_scores = sorted(scores)
        assert sorted_scores[0].score == -0.9
        assert sorted_scores[1].score == 0.5
        assert sorted_scores[2].score == 0.3


class TestAttributionPatchingResult:
    """Tests for AttributionPatchingResult class."""

    def test_result_creation(self):
        scores = np.random.randn(10, 20)
        result = AttributionPatchingResult(
            scores=scores, layers=list(range(10)), component="resid_post", method="standard"
        )
        assert result.n_layers == 10
        assert result.n_positions == 20

    def test_get_top_scores(self):
        scores = np.zeros((3, 5))
        scores[1, 2] = 10.0
        scores[2, 4] = -8.0
        scores[0, 0] = 5.0

        result = AttributionPatchingResult(
            scores=scores, layers=[0, 1, 2], component="resid_post", method="standard"
        )

        top = result.get_top_scores(3)
        assert len(top) == 3
        assert top[0].layer == 1
        assert top[0].position == 2
        assert top[0].score == 10.0

    def test_get_top_targets(self):
        scores = np.zeros((3, 5))
        scores[1, 2] = 10.0

        result = AttributionPatchingResult(
            scores=scores, layers=[0, 1, 2], component="resid_post", method="standard"
        )

        targets = result.get_top_targets(1)
        assert len(targets) == 1
        assert 2 in targets[0].positions
        assert 1 in targets[0].layers


class TestAttributionSummary:
    """Tests for AttributionSummary class."""

    def test_empty_result(self):
        result = AttributionSummary()
        assert len(result.results) == 0
        assert result.n_pairs == 1

    def test_aggregate_single(self):
        scores = np.random.randn(5, 10)
        inner = AttributionPatchingResult(scores=scores, layers=[0, 1, 2, 3, 4], method="standard")
        result = AttributionSummary(results={"test": inner})
        aggregated = AttributionSummary.aggregate([result])
        assert "test" in aggregated.results

    def test_get_position_target(self):
        scores1 = np.zeros((5, 10))
        scores1[2, 5] = 10.0
        scores2 = np.zeros((5, 10))
        scores2[2, 5] = 8.0

        result = AttributionSummary(
            results={
                "method1": AttributionPatchingResult(scores=scores1, layers=list(range(5)), method="standard"),
                "method2": AttributionPatchingResult(scores=scores2, layers=list(range(5)), method="eap"),
            }
        )

        target = result.get_position_target(n=1, min_methods=2)
        assert target is not None
        assert 2 in target.layers
        assert 5 in target.positions

    def test_get_layer_target(self):
        scores1 = np.zeros((5, 10))
        scores1[2, 3] = 10.0
        scores1[2, 7] = 8.0
        scores2 = np.zeros((5, 10))
        scores2[2, 4] = 9.0
        scores2[3, 1] = 7.0

        result = AttributionSummary(
            results={
                "method1": AttributionPatchingResult(scores=scores1, layers=list(range(5)), method="standard"),
                "method2": AttributionPatchingResult(scores=scores2, layers=list(range(5)), method="eap"),
            }
        )

        target = result.get_layer_target(n_layers=2, min_methods=1)
        assert target is not None
        assert 2 in target.layers

    def test_get_target(self):
        scores = np.zeros((5, 10))
        scores[2, 5] = 10.0

        result = AttributionSummary(
            results={
                "method1": AttributionPatchingResult(scores=scores, layers=list(range(5)), method="standard"),
            }
        )

        layer_target = result.get_target(n=2, mode="layer")
        assert layer_target is not None

        position_target = result.get_target(n=2, mode="position")
        assert position_target is not None


class TestCoreFunctions:
    """Tests for core attribution functions."""

    def test_build_position_arrays_identity(self):
        pos_mapping = {i: i for i in range(10)}
        src_pos, dst_pos, valid = build_position_arrays(pos_mapping, 10, 10)
        assert len(src_pos) == 10
        np.testing.assert_array_equal(src_pos, dst_pos)
        assert all(valid)

    def test_build_position_arrays_offset(self):
        pos_mapping = {i: i + 2 for i in range(5)}
        src_pos, dst_pos, valid = build_position_arrays(pos_mapping, 5, 10)
        np.testing.assert_array_equal(dst_pos, src_pos + 2)
        assert all(valid)

    def test_build_position_arrays_out_of_bounds(self):
        pos_mapping = {0: 0, 1: 5, 2: 10}
        src_pos, dst_pos, valid = build_position_arrays(pos_mapping, 3, 8)
        assert valid[0] == True
        assert valid[1] == True
        assert valid[2] == False

    def test_compute_attribution_vectorized(self):
        clean_act = torch.randn(1, 5, 10)
        corr_act = torch.randn(1, 5, 10)
        grad = torch.ones(1, 5, 10)

        clean_pos = np.arange(5)
        corr_pos = np.arange(5)
        valid = np.ones(5, dtype=bool)

        scores = compute_attribution_vectorized(clean_act, corr_act, grad, clean_pos, corr_pos, valid)
        assert scores.shape == (5,)
        expected = (clean_act[0] - corr_act[0]).sum(dim=-1).numpy()
        np.testing.assert_array_almost_equal(scores, expected, decimal=5)
