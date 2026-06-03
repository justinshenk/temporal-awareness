"""Tests for batched intervention methods in ModelRunner.

Tests:
1. compute_trajectories_batch_with_intervention - batched forward with intervention
2. compute_trajectories_batch_with_intervention_and_cache - batched forward with intervention + cache
3. multilabel_choose - BinaryChoiceRunner method that uses the batched methods

These tests verify that batched methods produce equivalent results to sequential calls.
"""

import numpy as np
import pytest
import torch

from src.inference import ModelRunner
from src.inference.model_runner import ModelBackend
from src.inference.interventions import steering
from src.binary_choice.binary_choice_runner import BinaryChoiceRunner


TEST_MODEL = "gpt2"  # Small model for fast tests


@pytest.fixture(scope="module")
def runner():
    """TransformerLens runner for testing."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)


@pytest.fixture(scope="module")
def choice_runner():
    """BinaryChoiceRunner for multilabel_choose tests."""
    return BinaryChoiceRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)


def make_direction(d_model, seed=42):
    """Create a normalized random direction vector."""
    np.random.seed(seed)
    direction = np.random.randn(d_model).astype(np.float32)
    return direction / np.linalg.norm(direction)


# =============================================================================
# compute_trajectories_batch_with_intervention Tests
# =============================================================================


class TestBatchWithIntervention:
    """Tests for compute_trajectories_batch_with_intervention."""

    def test_batch_with_intervention_produces_trajectories(self, runner):
        """Batched intervention method produces valid trajectories."""
        prompts = ["Hello world", "The quick brown fox"]
        token_ids_batch = [runner.encode_ids(p) for p in prompts]

        intervention = steering(
            layer=3,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )

        trajs = runner.compute_trajectories_batch_with_intervention(
            token_ids_batch, intervention
        )

        assert len(trajs) == 2
        for traj in trajs:
            assert traj.token_ids is not None
            assert traj.logprobs is not None

    def test_batch_with_intervention_matches_sequential(self, runner):
        """Batched results match sequential intervention calls."""
        prompts = ["Hello", "World"]
        token_ids_batch = [runner.encode_ids(p) for p in prompts]

        intervention = steering(
            layer=3,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )

        # Batched call
        batch_trajs = runner.compute_trajectories_batch_with_intervention(
            token_ids_batch, intervention
        )

        # Sequential calls
        seq_trajs = [
            runner.compute_trajectory_with_intervention(ids, intervention)
            for ids in token_ids_batch
        ]

        # Compare results
        for batch_traj, seq_traj in zip(batch_trajs, seq_trajs):
            # Token IDs should match
            assert batch_traj.token_ids == seq_traj.token_ids
            # Logprobs should be very close
            torch.testing.assert_close(
                torch.tensor(batch_traj.logprobs),
                torch.tensor(seq_traj.logprobs),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_batch_with_no_intervention(self, runner):
        """Batched method works with intervention=None."""
        prompts = ["Test one", "Test two"]
        token_ids_batch = [runner.encode_ids(p) for p in prompts]

        trajs = runner.compute_trajectories_batch_with_intervention(
            token_ids_batch, intervention=None
        )

        assert len(trajs) == 2

    def test_batch_with_empty_batch(self, runner):
        """Batched method handles empty batch correctly."""
        trajs = runner.compute_trajectories_batch_with_intervention(
            [], intervention=None
        )
        assert trajs == []

    def test_batch_with_single_item(self, runner):
        """Batched method handles single item batch correctly."""
        token_ids_batch = [runner.encode_ids("Hello")]

        intervention = steering(
            layer=3,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )

        trajs = runner.compute_trajectories_batch_with_intervention(
            token_ids_batch, intervention
        )

        assert len(trajs) == 1
        assert trajs[0].token_ids == token_ids_batch[0]


# =============================================================================
# compute_trajectories_batch_with_intervention_and_cache Tests
# =============================================================================


class TestBatchWithInterventionAndCache:
    """Tests for compute_trajectories_batch_with_intervention_and_cache."""

    def test_batch_with_cache_produces_internals(self, runner):
        """Batched intervention+cache method produces trajectories with internals."""
        prompts = ["Hello world", "The quick brown fox"]
        token_ids_batch = [runner.encode_ids(p) for p in prompts]
        layer = 3

        intervention = steering(
            layer=layer,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )

        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        trajs = runner.compute_trajectories_batch_with_intervention_and_cache(
            token_ids_batch, intervention, names_filter
        )

        assert len(trajs) == 2
        for i, traj in enumerate(trajs):
            assert traj.token_ids is not None
            assert traj.logprobs is not None
            # Check internals are attached
            assert traj.internals is not None
            assert f"blocks.{layer}.hook_resid_post" in traj.internals
            # Check shape - should be [1, seq_len, d_model] to match sequential API
            cached_act = traj.internals[f"blocks.{layer}.hook_resid_post"]
            expected_seq_len = len(token_ids_batch[i])
            assert cached_act.shape == (1, expected_seq_len, runner.d_model)

    def test_batch_with_cache_matches_sequential(self, runner):
        """Batched intervention+cache results match sequential calls."""
        prompts = ["Hello", "World"]
        token_ids_batch = [runner.encode_ids(p) for p in prompts]
        layer = 3

        intervention = steering(
            layer=layer,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )

        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        # Batched call
        batch_trajs = runner.compute_trajectories_batch_with_intervention_and_cache(
            token_ids_batch, intervention, names_filter
        )

        # Sequential calls
        seq_trajs = [
            runner.compute_trajectory_with_intervention_and_cache(
                ids, intervention, names_filter
            )
            for ids in token_ids_batch
        ]

        # Compare results
        for batch_traj, seq_traj in zip(batch_trajs, seq_trajs):
            # Token IDs should match
            assert batch_traj.token_ids == seq_traj.token_ids
            # Logprobs should be very close
            torch.testing.assert_close(
                torch.tensor(batch_traj.logprobs),
                torch.tensor(seq_traj.logprobs),
                rtol=1e-4,
                atol=1e-4,
            )
            # Internals should match
            for name in batch_traj.internals:
                torch.testing.assert_close(
                    batch_traj.internals[name],
                    seq_traj.internals[name],
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_batch_with_cache_empty_batch(self, runner):
        """Batched intervention+cache handles empty batch correctly."""
        layer = 3
        intervention = steering(
            layer=layer,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        trajs = runner.compute_trajectories_batch_with_intervention_and_cache(
            [], intervention, names_filter
        )
        assert trajs == []

    def test_batch_with_cache_single_item(self, runner):
        """Batched intervention+cache handles single item batch correctly."""
        token_ids_batch = [runner.encode_ids("Hello")]
        layer = 3

        intervention = steering(
            layer=layer,
            direction=make_direction(runner.d_model),
            strength=10.0,
        )
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        trajs = runner.compute_trajectories_batch_with_intervention_and_cache(
            token_ids_batch, intervention, names_filter
        )

        assert len(trajs) == 1
        assert trajs[0].token_ids == token_ids_batch[0]
        assert f"blocks.{layer}.hook_resid_post" in trajs[0].internals


# =============================================================================
# multilabel_choose Tests
# =============================================================================


class TestMultilabelChoose:
    """Tests for BinaryChoiceRunner.multilabel_choose."""

    def test_multilabel_choose_basic(self, choice_runner):
        """multilabel_choose produces valid results."""
        prompt = "Choose: a) apple or b) banana? I pick:"
        choice_prefix = " "
        labels = [("a)", "b)")]

        result = choice_runner.multilabel_choose(prompt, choice_prefix, labels)

        assert result is not None
        assert result.tree is not None
        assert result.label_pairs == (("a)", "b)"),)

    def test_multilabel_choose_with_intervention(self, choice_runner):
        """multilabel_choose with intervention produces valid results."""
        prompt = "Choose: a) apple or b) banana? I pick:"
        choice_prefix = " "
        labels = [("a)", "b)")]

        intervention = steering(
            layer=3,
            direction=make_direction(choice_runner.d_model),
            strength=10.0,
        )

        result = choice_runner.multilabel_choose(
            prompt, choice_prefix, labels, intervention=intervention
        )

        assert result is not None
        assert result.tree is not None

    def test_multilabel_choose_with_cache(self, choice_runner):
        """multilabel_choose with cache produces valid results with internals."""
        prompt = "Choose: a) apple or b) banana? I pick:"
        choice_prefix = " "
        labels = [("a)", "b)")]
        layer = 3

        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        result = choice_runner.multilabel_choose(
            prompt, choice_prefix, labels, with_cache=True, names_filter=names_filter
        )

        assert result is not None
        assert result.tree is not None
        # Check that trajectories have internals
        for traj in result.tree.trajs:
            assert traj.internals is not None
            assert f"blocks.{layer}.hook_resid_post" in traj.internals

    def test_multilabel_choose_with_intervention_and_cache(self, choice_runner):
        """multilabel_choose with both intervention and cache works."""
        prompt = "Choose: a) apple or b) banana? I pick:"
        choice_prefix = " "
        labels = [("a)", "b)")]
        layer = 3

        intervention = steering(
            layer=layer,
            direction=make_direction(choice_runner.d_model),
            strength=10.0,
        )

        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        result = choice_runner.multilabel_choose(
            prompt,
            choice_prefix,
            labels,
            intervention=intervention,
            with_cache=True,
            names_filter=names_filter,
        )

        assert result is not None
        assert result.tree is not None
        # Check that trajectories have internals
        for traj in result.tree.trajs:
            assert traj.internals is not None
            assert f"blocks.{layer}.hook_resid_post" in traj.internals

    def test_multilabel_choose_multiple_labels(self, choice_runner):
        """multilabel_choose with multiple label pairs works."""
        prompt = "Pick one: I pick:"
        choice_prefix = " "
        labels = [("a)", "b)"), ("1)", "2)"), ("x)", "y)")]

        intervention = steering(
            layer=3,
            direction=make_direction(choice_runner.d_model),
            strength=10.0,
        )

        result = choice_runner.multilabel_choose(
            prompt, choice_prefix, labels, intervention=intervention
        )

        assert result is not None
        assert result.tree is not None
        assert len(result.label_pairs) == 3
        # Should have 6 trajectories (2 per label pair)
        assert len(result.tree.trajs) == 6
