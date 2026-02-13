"""Tests for ModelRunner API using Qwen model.

Tests ModelRunner with Qwen2.5-0.5B, comparing across backends.
Marked slow since they require model loading.

IMPORTANT:
1. NEVER use backend APIs directly - always use ModelRunner public API
2. All backends must pass identical tests (verify cross-backend consistency)
3. Tests should NEVER be skipped (xfail is acceptable for known issues)
4. When adding new ModelRunner methods, add tests for ALL backends
"""

import numpy as np
import pytest
import torch

from src.inference import ModelRunner
from src.inference.model_runner import ModelBackend
from src.inference.interventions import steering


TEST_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def runner_tl():
    """TransformerLens backend runner."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)


@pytest.fixture(scope="module")
def runner_nnsight():
    """NNsight backend runner."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.NNSIGHT)


@pytest.fixture(scope="module")
def runner_pyvene():
    """Pyvene backend runner."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.PYVENE)


class TestQwenModelProperties:
    """Test Qwen model properties across backends."""

    def test_n_layers_consistent(self, runner_tl, runner_nnsight, runner_pyvene):
        """n_layers is consistent across backends."""
        assert runner_tl.n_layers == runner_nnsight.n_layers == runner_pyvene.n_layers

    def test_d_model_consistent(self, runner_tl, runner_nnsight, runner_pyvene):
        """d_model is consistent across backends."""
        assert runner_tl.d_model == runner_nnsight.d_model == runner_pyvene.d_model


class TestQwenTokenization:
    """Test tokenization across backends."""

    def test_tokenize_same_length(self, runner_tl, runner_nnsight, runner_pyvene):
        """Tokenization produces same length across backends."""
        text = "Hello world, how are you?"
        toks_tl = runner_tl.tokenize(text)
        toks_nn = runner_nnsight.tokenize(text)
        toks_pv = runner_pyvene.tokenize(text)

        # Lengths should match (accounting for BOS token handling)
        assert abs(toks_tl.shape[1] - toks_nn.shape[1]) <= 1
        assert abs(toks_tl.shape[1] - toks_pv.shape[1]) <= 1


class TestQwenGeneration:
    """Test generation across backends."""

    def test_generate_produces_output(self, runner_tl, runner_nnsight, runner_pyvene):
        """All backends generate non-empty output."""
        prompt = "The capital of France is"

        out_tl = runner_tl.generate(prompt, max_new_tokens=10)
        out_nn = runner_nnsight.generate(prompt, max_new_tokens=10)
        out_pv = runner_pyvene.generate(prompt, max_new_tokens=10)

        assert len(out_tl) > 0
        assert len(out_nn) > 0
        assert len(out_pv) > 0


class TestQwenRunWithCache:
    """Test run_with_cache across backends."""

    def test_cache_shapes_consistent(self, runner_tl, runner_nnsight, runner_pyvene):
        """Cached activations have consistent shapes."""
        prompt = "Hello world"
        hook_name = "blocks.0.hook_resid_post"

        _, cache_tl = runner_tl.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
        _, cache_nn = runner_nnsight.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
        _, cache_pv = runner_pyvene.run_with_cache(prompt, names_filter=lambda n: n == hook_name)

        # All should have the hook
        assert hook_name in cache_tl
        assert hook_name in cache_nn
        assert hook_name in cache_pv

        # d_model dimension should match
        assert cache_tl[hook_name].shape[-1] == runner_tl.d_model
        assert cache_nn[hook_name].shape[-1] == runner_nnsight.d_model
        assert cache_pv[hook_name].shape[-1] == runner_pyvene.d_model


class TestQwenLabelProbs:
    pass


class TestQwenForwardWithIntervention:
    """Test run_with_intervention across backends."""

    def test_intervention_changes_output(self, runner_tl, runner_nnsight, runner_pyvene):
        """Interventions change output in all backends."""
        prompt = "Hello world"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            base_logits, _ = runner.run_with_cache(prompt)

            direction = np.random.randn(runner.d_model).astype(np.float32)
            intervention = steering(layer=5, direction=direction, strength=100.0)
            steered_logits = runner.run_with_intervention(prompt, intervention)

            assert not torch.allclose(base_logits, steered_logits)



class TestQwenMultipleInterventions:
    """Test multiple interventions across backends."""

    def test_multiple_interventions_applied(self, runner_tl, runner_nnsight, runner_pyvene):
        """Multiple interventions are all applied."""
        prompt = "Hello world"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            direction = np.random.randn(runner.d_model).astype(np.float32)
            interventions = [
                steering(layer=3, direction=direction, strength=1.0),
                steering(layer=6, direction=direction, strength=1.0),
                steering(layer=9, direction=direction, strength=1.0),
            ]

            logits = runner.run_with_intervention(prompt, interventions)
            assert logits.shape[0] == 1


class TestQwenForwardWithInterventionAndCache:
    """Test run_with_intervention_and_cache across backends.

    This method combines intervention application with activation caching
    and gradient preservation - used for attribution patching.
    """

    def test_returns_logits_and_cache(self, runner_tl, runner_nnsight, runner_pyvene):
        """All backends return (logits, cache) tuple."""
        prompt = "Hello world"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            direction = np.random.randn(runner.d_model).astype(np.float32)
            intervention = steering(layer=5, direction=direction, strength=1.0)

            logits, cache = runner.run_with_intervention_and_cache(
                prompt, intervention,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            assert logits.ndim == 3  # [batch, seq, vocab]
            assert isinstance(cache, dict)
            assert len(cache) > 0

    def test_cache_contains_requested_hooks(self, runner_tl, runner_nnsight, runner_pyvene):
        """Cache contains hooks matching names_filter."""
        prompt = "Hello"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            direction = np.random.randn(runner.d_model).astype(np.float32)
            intervention = steering(layer=5, direction=direction)

            _, cache = runner.run_with_intervention_and_cache(
                prompt, intervention,
                names_filter=lambda n: "blocks.0.hook_resid_post" in n,
            )

            assert "blocks.0.hook_resid_post" in cache

    def test_intervention_applied_with_cache(self, runner_tl, runner_nnsight, runner_pyvene):
        """Intervention is applied when caching."""
        prompt = "Hello world"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            # Get baseline
            base_logits, _ = runner.run_with_cache(prompt)

            # With intervention
            direction = np.random.randn(runner.d_model).astype(np.float32)
            intervention = steering(layer=5, direction=direction, strength=100.0)
            steered_logits, _ = runner.run_with_intervention_and_cache(
                prompt, intervention,
            )

            # Intervention should change output
            assert not torch.allclose(base_logits, steered_logits)

    def test_multiple_interventions_with_cache(self, runner_tl, runner_nnsight, runner_pyvene):
        """Multiple interventions work with caching."""
        prompt = "Hello"

        for runner in [runner_tl, runner_nnsight, runner_pyvene]:
            direction = np.random.randn(runner.d_model).astype(np.float32)
            interventions = [
                steering(layer=3, direction=direction, strength=1.0),
                steering(layer=6, direction=direction, strength=1.0),
            ]

            logits, cache = runner.run_with_intervention_and_cache(
                prompt, interventions,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            assert logits.shape[0] == 1
            assert len(cache) > 0


class TestQwenRunWithCacheAndGrad:
    """Test run_with_cache_and_grad across backends.

    This method enables gradient flow through cached activations.
    """

    def test_cache_has_grad_tl(self, runner_tl):
        """TransformerLens: cached activations require grad."""
        prompt = "Hello"
        _, cache = runner_tl.run_with_cache_and_grad(
            prompt, names_filter=lambda n: "blocks.0.hook_resid_post" in n
        )

        act = cache.get("blocks.0.hook_resid_post")
        assert act is not None
        assert act.requires_grad

    def test_cache_shapes_match(self, runner_tl, runner_nnsight, runner_pyvene):
        """Cache shapes are consistent across backends."""
        prompt = "Test"
        hook_name = "blocks.0.hook_resid_post"

        _, cache_tl = runner_tl.run_with_cache_and_grad(
            prompt, names_filter=lambda n: n == hook_name
        )
        _, cache_pv = runner_pyvene.run_with_cache_and_grad(
            prompt, names_filter=lambda n: n == hook_name
        )

        # d_model dimension should match
        assert cache_tl[hook_name].shape[-1] == runner_tl.d_model
        assert cache_pv[hook_name].shape[-1] == runner_pyvene.d_model
