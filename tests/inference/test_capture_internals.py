"""Tests for activation capture functionality (run_with_cache).

Tests the existing run_with_cache API that is used throughout the codebase
for capturing internal activations during forward passes.

Tests all backends: TransformerLens (default), NNsight, Pyvene
"""

from collections.abc import Mapping

import numpy as np
import pytest
import torch

from src.inference import ModelRunner
from src.inference.model_runner import ModelBackend
from src.inference.interventions import get_activations, compute_mean_activations


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
# Basic run_with_cache Tests
# =============================================================================


class TestRunWithCacheBasic:
    """Basic tests for run_with_cache functionality."""

    def test_returns_logits_and_cache(self, runner):
        """run_with_cache returns (logits, cache) tuple."""
        prompt = "Hello world"
        logits, cache = runner.run_with_cache(prompt)

        assert isinstance(logits, torch.Tensor)
        # Cache can be dict or dict-like (e.g., ActivationCache from TransformerLens)
        assert isinstance(cache, Mapping) or hasattr(cache, "__getitem__")
        assert logits.ndim == 3  # [batch, seq, vocab]

    def test_cache_contains_activations(self, runner):
        """Cache contains activation tensors for requested hooks."""
        prompt = "Hello world"
        layer = 0
        hook_name = f"blocks.{layer}.hook_resid_post"

        logits, cache = runner.run_with_cache(
            prompt, names_filter=lambda n: n == hook_name
        )

        assert hook_name in cache
        assert isinstance(cache[hook_name], torch.Tensor)

    def test_activation_shape(self, runner):
        """Cached activations have correct shape [batch, seq, d_model]."""
        prompt = "Hello world"
        layer = 5
        hook_name = f"blocks.{layer}.hook_resid_post"

        logits, cache = runner.run_with_cache(
            prompt, names_filter=lambda n: n == hook_name
        )

        acts = cache[hook_name]
        assert acts.ndim == 3
        assert acts.shape[0] == 1  # batch size
        assert acts.shape[2] == runner.d_model

    def test_names_filter_limits_cache(self, runner):
        """names_filter limits which hooks are captured."""
        prompt = "Hello"
        target_hook = "blocks.0.hook_resid_post"
        other_hook = "blocks.1.hook_resid_post"

        _, cache = runner.run_with_cache(
            prompt, names_filter=lambda n: n == target_hook
        )

        assert target_hook in cache
        assert other_hook not in cache

    def test_no_filter_captures_all(self, runner):
        """Without filter, captures all standard hooks."""
        prompt = "Hi"
        _, cache = runner.run_with_cache(prompt, names_filter=None)

        # Should have hooks for all layers
        n_layers = runner.n_layers
        for i in range(n_layers):
            assert f"blocks.{i}.hook_resid_post" in cache


# =============================================================================
# Backend Comparison Tests
# =============================================================================


class TestRunWithCacheBackends:
    """Test run_with_cache produces consistent results across backends."""

    def test_nnsight_returns_logits_and_cache(self, runner_nnsight):
        """NNsight backend returns (logits, cache) tuple."""
        prompt = "Hello world"
        logits, cache = runner_nnsight.run_with_cache(prompt)

        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, Mapping) or hasattr(cache, "__getitem__")

    def test_pyvene_returns_logits_and_cache(self, runner_pyvene):
        """Pyvene backend returns (logits, cache) tuple."""
        prompt = "Hello world"
        logits, cache = runner_pyvene.run_with_cache(prompt)

        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, Mapping) or hasattr(cache, "__getitem__")

    def test_transformerlens_pyvene_same_activation_shape(self, runner, runner_pyvene):
        """TransformerLens and Pyvene backends produce activations with same shape.

        With prepend_bos=False (default), both backends now tokenize consistently,
        so shapes should match directly without any slicing.
        """
        prompt = "Hello"
        layer = 5
        hook_name = f"blocks.{layer}.hook_resid_post"
        names_filter = lambda n: n == hook_name

        # Both backends now use prepend_bos=False by default
        _, cache_tl = runner.run_with_cache(prompt, names_filter=names_filter)
        _, cache_pv = runner_pyvene.run_with_cache(prompt, names_filter=names_filter)

        act_tl = cache_tl[hook_name]
        act_pv = cache_pv[hook_name]

        assert act_tl.shape == act_pv.shape

    def test_transformerlens_pyvene_similar_activations(self, runner, runner_pyvene):
        """TransformerLens and Pyvene backends produce similar activation values.

        With prepend_bos=False (default), both backends now tokenize consistently,
        producing identical token sequences. This means activations should match.
        """
        prompt = "Hello"
        layer = 5
        hook_name = f"blocks.{layer}.hook_resid_post"
        names_filter = lambda n: n == hook_name

        # Both backends now use prepend_bos=False by default
        _, cache_tl = runner.run_with_cache(prompt, names_filter=names_filter)
        _, cache_pv = runner_pyvene.run_with_cache(prompt, names_filter=names_filter)

        act_tl = cache_tl[hook_name].float()
        act_pv = cache_pv[hook_name].float()

        # Backends should produce very similar activations for identical inputs
        assert torch.allclose(act_tl, act_pv, atol=1e-2, rtol=1e-2)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetActivations:
    """Test get_activations helper function."""

    def test_returns_numpy_array(self, runner):
        """get_activations returns numpy array."""
        acts = get_activations(runner, layer=5, prompt="Hello")
        assert isinstance(acts, np.ndarray)

    def test_shape_is_2d(self, runner):
        """get_activations returns [seq_len, d_model] shape."""
        acts = get_activations(runner, layer=5, prompt="Hello")
        assert acts.ndim == 2
        assert acts.shape[1] == runner.d_model

    def test_different_prompts_different_activations(self, runner):
        """Different prompts produce different activations."""
        acts1 = get_activations(runner, layer=5, prompt="Hello")
        acts2 = get_activations(runner, layer=5, prompt="Goodbye")

        # Compare mean activations (different prompts may have different seq lengths)
        mean1 = acts1.mean(axis=0)
        mean2 = acts2.mean(axis=0)
        assert not np.allclose(mean1, mean2)


class TestComputeMeanActivations:
    """Test compute_mean_activations helper function."""

    def test_returns_numpy_array(self, runner):
        """compute_mean_activations returns numpy array."""
        means = compute_mean_activations(runner, layer=5, prompts=["Hello", "World"])
        assert isinstance(means, np.ndarray)

    def test_shape_is_1d(self, runner):
        """compute_mean_activations returns [d_model] shape."""
        means = compute_mean_activations(runner, layer=5, prompts=["Hello", "World"])
        assert means.ndim == 1
        assert means.shape[0] == runner.d_model

    def test_single_prompt_works(self, runner):
        """compute_mean_activations works with single prompt string."""
        means = compute_mean_activations(runner, layer=5, prompts="Hello")
        assert isinstance(means, np.ndarray)
        assert means.shape[0] == runner.d_model


# =============================================================================
# run_with_cache_and_grad Tests
# =============================================================================


class TestRunWithCacheAndGrad:
    """Test run_with_cache_and_grad for attribution patching."""

    def test_returns_logits_and_cache(self, runner):
        """run_with_cache_and_grad returns (logits, cache) tuple."""
        prompt = "Hello"
        logits, cache = runner.run_with_cache_and_grad(prompt)

        assert isinstance(logits, torch.Tensor)
        # Cache can be dict or dict-like (e.g., ActivationCache from TransformerLens)
        assert isinstance(cache, Mapping) or hasattr(cache, "__getitem__")

    def test_activations_have_grad(self, runner):
        """Cached activations have requires_grad=True."""
        prompt = "Hello"
        hook_name = "blocks.5.hook_resid_post"

        logits, cache = runner.run_with_cache_and_grad(
            prompt, names_filter=lambda n: n == hook_name
        )

        # Activations should have gradients enabled
        assert cache[hook_name].requires_grad

    def test_can_backprop_through_cache(self, runner):
        """Can backpropagate through cached activations."""
        prompt = "Hello"
        hook_name = "blocks.5.hook_resid_post"

        logits, cache = runner.run_with_cache_and_grad(
            prompt, names_filter=lambda n: n == hook_name
        )

        # Create a simple loss and backprop
        loss = cache[hook_name].sum()
        loss.backward()

        # Should not raise an error
        assert True
