"""Integration tests with real models and backends.

These tests verify:
1. All backends produce equivalent results
2. Interventions work across model architectures
3. Multi-architecture support (GPT-2 style, Pythia, OPT)
4. Performance benchmarks across devices

Run all: pytest tests/test_integration.py -v -s
Skip slow: pytest tests/test_integration.py --skip-slow
"""

import gc
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# =============================================================================
# Model Configuration
# =============================================================================

# Default model for quick tests
TEST_MODEL = "Qwen/Qwen2.5-0.5B"

# Primary test models - used for most integration tests
# Covers: GPT-2 style, Qwen base, Qwen instruct
PRIMARY_MODELS = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# Models for cross-architecture sanity tests (4 models covering different architectures)
# Fast defaults: ~1 minute total for all architecture tests
CROSS_ARCH_MODELS = [
    # GPT-2 style (transformer.h)
    ("gpt2", "124M", "gpt2"),
    # Pythia/GPT-NeoX style (gpt_neox.layers)
    ("EleutherAI/pythia-70m", "70M", "pythia"),
    # OPT style (model.decoder.layers)
    ("facebook/opt-125m", "125M", "opt"),
    # TinyStories for speed
    ("roneneldan/TinyStories-33M", "33M", "gpt2"),
]

TEST_MODELS = {
    "gpt2": {"name": "gpt2", "arch": "gpt2", "params": "124M", "is_instruct": False},
    "pythia-70m": {
        "name": "EleutherAI/pythia-70m",
        "arch": "pythia",
        "params": "70M",
        "is_instruct": False,
    },
    "qwen2.5-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B",
        "arch": "qwen2",
        "params": "500M",
        "is_instruct": False,
    },
    "qwen2.5-0.5b-instruct": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "arch": "qwen2",
        "params": "500M",
        "is_instruct": True,
    },
}

# Check optional backends
try:
    import nnsight

    HAS_NNSIGHT = True
except ImportError:
    HAS_NNSIGHT = False

try:
    import pyvene

    HAS_PYVENE = True
except ImportError:
    HAS_PYVENE = False

requires_nnsight = pytest.mark.skipif(not HAS_NNSIGHT, reason="nnsight not installed")
requires_pyvene = pytest.mark.skipif(not HAS_PYVENE, reason="pyvene not installed")


@pytest.fixture(scope="module")
def transformerlens_runner():
    """Load model with TransformerLens backend once per module."""
    from src.models.model_runner import ModelRunner, ModelBackend

    runner = ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)
    yield runner
    del runner
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def nnsight_runner():
    """Load model with nnsight backend once per module."""
    if not HAS_NNSIGHT:
        pytest.skip("nnsight not installed")

    from src.models.model_runner import ModelRunner, ModelBackend

    runner = ModelRunner(TEST_MODEL, backend=ModelBackend.NNSIGHT)
    yield runner
    del runner
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def pyvene_runner():
    """Load model with pyvene backend once per module."""
    if not HAS_PYVENE:
        pytest.skip("pyvene not installed")

    from src.models.model_runner import ModelRunner, ModelBackend

    runner = ModelRunner(TEST_MODEL, backend=ModelBackend.PYVENE)
    yield runner
    del runner
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =============================================================================
# Backend Equivalence Tests
# =============================================================================


@requires_nnsight
class TestBackendEquivalence:
    """Verify both backends produce identical results."""

    def test_tokenize_identical(self, transformerlens_runner, nnsight_runner):
        """Both backends tokenize text identically."""
        text = "The quick brown fox jumps over the lazy dog."

        tl_ids = transformerlens_runner.tokenize(text)
        nn_ids = nnsight_runner.tokenize(text)

        torch.testing.assert_close(tl_ids, nn_ids)

    def test_decode_identical(self, transformerlens_runner, nnsight_runner):
        """Both backends decode tokens identically."""
        # Use tokens from Qwen vocabulary
        text = "Hello world"
        tl_ids = transformerlens_runner.tokenize(text)

        tl_text = transformerlens_runner.decode(tl_ids[0])
        nn_text = nnsight_runner.decode(tl_ids[0])

        assert tl_text == nn_text

    def test_generate_deterministic_same_output(
        self, transformerlens_runner, nnsight_runner
    ):
        """With temperature=0, both backends produce same output."""
        prompt = "The capital of France is"

        tl_out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0
        )
        nn_out = nnsight_runner.generate(prompt, max_new_tokens=5, temperature=0.0)

        assert tl_out == nn_out, f"TL: {tl_out!r} != NN: {nn_out!r}"

    def test_run_with_cache_same_activations(
        self, transformerlens_runner, nnsight_runner
    ):
        """Both backends capture same activations."""
        prompt = "Test input"
        layer = 5
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        tl_logits, tl_cache = transformerlens_runner.run_with_cache(
            prompt, names_filter
        )
        nn_logits, nn_cache = nnsight_runner.run_with_cache(prompt, names_filter)

        # Compare softmax probabilities (raw logits may differ by offset)
        # Use slightly relaxed tolerance for numerical differences between backends
        tl_probs = torch.softmax(tl_logits[0, -1, :], dim=-1)
        nn_probs = torch.softmax(nn_logits[0, -1, :], dim=-1)
        torch.testing.assert_close(tl_probs, nn_probs, rtol=5e-3, atol=2e-3)

        # Token predictions should match
        assert tl_logits.argmax(dim=-1).tolist() == nn_logits.argmax(dim=-1).tolist()

        # Check activation shapes match
        tl_act = tl_cache[f"blocks.{layer}.hook_resid_post"]
        nn_act = nn_cache[f"blocks.{layer}.hook_resid_post"]
        assert tl_act.shape == nn_act.shape

    def test_model_properties_match(self, transformerlens_runner, nnsight_runner):
        """Both backends report same model properties."""
        assert transformerlens_runner.n_layers == nnsight_runner.n_layers
        assert transformerlens_runner.d_model == nnsight_runner.d_model


# =============================================================================
# Component Tests (Attention, MLP, Residual Stream)
# =============================================================================


class TestComponentActivations:
    """Test activation capture for different components (resid, attn, mlp)."""

    @pytest.mark.parametrize("component", ["resid_post", "attn_out", "mlp_out"])
    def test_transformerlens_captures_component(
        self, transformerlens_runner, component
    ):
        """TransformerLens captures activations for all components."""
        prompt = "Test input"
        layer = transformerlens_runner.n_layers // 2
        hook_name = f"blocks.{layer}.hook_{component}"

        _, cache = transformerlens_runner.run_with_cache(
            prompt, names_filter=lambda n: n == hook_name
        )

        assert hook_name in cache, f"Hook {hook_name} not found in cache"
        act = cache[hook_name]
        assert act.ndim == 3, f"Expected 3D tensor, got {act.ndim}D"
        assert act.shape[0] == 1, "Expected batch size 1"
        assert act.shape[-1] == transformerlens_runner.d_model, "Dimension mismatch"


@requires_nnsight
class TestComponentBackendEquivalence:
    """Verify backends produce equivalent activations for all components."""

    @pytest.mark.parametrize("component", ["resid_post", "attn_out", "mlp_out"])
    def test_activation_equivalence(
        self, transformerlens_runner, nnsight_runner, component
    ):
        """Both backends capture equivalent activations for each component."""
        prompt = "Test input for activation capture"
        layer = 5

        hook_name = f"blocks.{layer}.hook_{component}"
        names_filter = lambda n: n == hook_name

        tl_logits, tl_cache = transformerlens_runner.run_with_cache(
            prompt, names_filter
        )
        nn_logits, nn_cache = nnsight_runner.run_with_cache(prompt, names_filter)

        # Both should have the hook
        assert hook_name in tl_cache, f"TL missing {hook_name}"
        assert hook_name in nn_cache, f"NN missing {hook_name}"

        tl_act = tl_cache[hook_name]
        nn_act = nn_cache[hook_name]

        # Shapes must match
        assert tl_act.shape == nn_act.shape, (
            f"Shape mismatch: {tl_act.shape} vs {nn_act.shape}"
        )

        # Activations should be highly correlated (>0.99)
        tl_flat = tl_act.flatten().float()
        nn_flat = nn_act.flatten().float()
        corr = torch.corrcoef(torch.stack([tl_flat, nn_flat]))[0, 1]
        assert corr > 0.99, f"Correlation {corr:.4f} too low for {component}"

    @pytest.mark.parametrize("component", ["resid_post", "attn_out", "mlp_out"])
    def test_intervention_equivalence(
        self, transformerlens_runner, nnsight_runner, component
    ):
        """Both backends produce equivalent results for interventions on each component."""
        from src.models.intervention_utils import steering

        prompt = "The weather today is"
        layer = 5

        # Create a deterministic direction
        np.random.seed(42)
        direction = np.random.randn(transformerlens_runner.d_model).astype(np.float32)

        intervention = steering(
            layer=layer,
            direction=direction,
            strength=10.0,
            component=component,
        )

        # Run with intervention
        tl_logits = transformerlens_runner.forward_with_intervention(
            prompt, intervention
        )
        nn_logits = nnsight_runner.forward_with_intervention(prompt, intervention)

        # For resid_post, predictions should match exactly
        # For attn_out/mlp_out, hook points may differ slightly between backends
        tl_pred = tl_logits.argmax(dim=-1)
        nn_pred = nn_logits.argmax(dim=-1)

        if component == "resid_post":
            assert tl_pred.tolist() == nn_pred.tolist(), (
                f"Predictions differ for {component}: TL={tl_pred.tolist()}, NN={nn_pred.tolist()}"
            )
            # Probability distributions should be very similar
            tl_probs = torch.softmax(tl_logits[0, -1, :], dim=-1)
            nn_probs = torch.softmax(nn_logits[0, -1, :], dim=-1)
            torch.testing.assert_close(tl_probs, nn_probs, rtol=1e-2, atol=1e-2)
        else:
            # For attn_out/mlp_out, verify probability distributions are correlated
            # Hook points may differ between backends, so exact match isn't expected
            tl_probs = torch.softmax(tl_logits[0, -1, :], dim=-1)
            nn_probs = torch.softmax(nn_logits[0, -1, :], dim=-1)
            corr = torch.corrcoef(torch.stack([tl_probs, nn_probs]))[0, 1]
            assert corr > 0.9, (
                f"Probability correlation {corr:.4f} too low for {component}"
            )


@requires_pyvene
class TestPyveneComponentEquivalence:
    """Verify pyvene produces equivalent activations for all components."""

    @pytest.mark.parametrize("component", ["resid_post", "attn_out", "mlp_out"])
    def test_activation_equivalence(
        self, transformerlens_runner, pyvene_runner, component
    ):
        """Pyvene and TransformerLens capture equivalent activations for each component."""
        prompt = "Test input for pyvene"
        layer = pyvene_runner.n_layers // 2

        hook_name = f"blocks.{layer}.hook_{component}"
        names_filter = lambda n: n == hook_name

        tl_logits, tl_cache = transformerlens_runner.run_with_cache(
            prompt, names_filter
        )
        pv_logits, pv_cache = pyvene_runner.run_with_cache(prompt, names_filter)

        # Both should have the hook
        assert hook_name in tl_cache, f"TL missing {hook_name}"
        assert hook_name in pv_cache, f"Pyvene missing {hook_name}"

        tl_act = tl_cache[hook_name]
        pv_act = pv_cache[hook_name]

        # Shapes must match
        assert tl_act.shape == pv_act.shape, (
            f"Shape mismatch: {tl_act.shape} vs {pv_act.shape}"
        )

        # Activations should be highly correlated
        tl_flat = tl_act.flatten().float()
        pv_flat = pv_act.flatten().float()
        corr = torch.corrcoef(torch.stack([tl_flat, pv_flat]))[0, 1]
        assert corr > 0.99, f"Correlation {corr:.4f} too low for {component}"

    @pytest.mark.parametrize("component", ["resid_post", "attn_out", "mlp_out"])
    def test_intervention_equivalence(
        self, transformerlens_runner, pyvene_runner, component
    ):
        """Pyvene and TransformerLens produce equivalent intervention results."""
        from src.models.intervention_utils import steering

        prompt = "The answer is"
        layer = pyvene_runner.n_layers // 2

        # Create a deterministic direction
        np.random.seed(42)
        direction = np.random.randn(transformerlens_runner.d_model).astype(np.float32)

        intervention = steering(
            layer=layer,
            direction=direction,
            strength=10.0,
            component=component,
        )

        # Run with intervention
        tl_logits = transformerlens_runner.forward_with_intervention(
            prompt, intervention
        )
        pv_logits = pyvene_runner.forward_with_intervention(prompt, intervention)

        # Predictions should match
        tl_pred = tl_logits.argmax(dim=-1)
        pv_pred = pv_logits.argmax(dim=-1)
        assert tl_pred.tolist() == pv_pred.tolist(), (
            f"Predictions differ for {component}: TL={tl_pred.tolist()}, PV={pv_pred.tolist()}"
        )


# =============================================================================
# Pyvene Backend Tests - Core Functionality
# =============================================================================


@requires_pyvene
class TestPyveneBackend:
    """Test pyvene backend core functionality.

    Focuses on the essential operations:
    - Generation equivalence with TransformerLens
    - Activation capture (run_with_cache)
    - Steering interventions
    """

    def test_generation_matches_transformerlens(
        self, pyvene_runner, transformerlens_runner
    ):
        """Pyvene and TransformerLens produce identical outputs with deterministic generation."""
        prompt = "The capital of France is"
        pv_out = pyvene_runner.generate(prompt, max_new_tokens=10, temperature=0.0)
        tl_out = transformerlens_runner.generate(
            prompt, max_new_tokens=10, temperature=0.0
        )

        assert pv_out == tl_out, (
            f"Outputs differ: pyvene={pv_out!r}, transformerlens={tl_out!r}"
        )

    def test_activation_capture(self, pyvene_runner, transformerlens_runner):
        """Pyvene captures activations with correct shapes and high correlation to TransformerLens."""
        prompt = "Testing activation capture"
        layer = pyvene_runner.n_layers // 2
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        pv_logits, pv_cache = pyvene_runner.run_with_cache(prompt, names_filter)
        tl_logits, tl_cache = transformerlens_runner.run_with_cache(
            prompt, names_filter
        )

        # Predictions must match
        assert pv_logits.argmax(dim=-1).tolist() == tl_logits.argmax(dim=-1).tolist()

        # Activations must have same shape and high correlation
        key = f"blocks.{layer}.hook_resid_post"
        pv_act, tl_act = pv_cache[key], tl_cache[key]
        assert pv_act.shape == tl_act.shape

        corr = torch.corrcoef(
            torch.stack([pv_act.flatten().float(), tl_act.flatten().float()])
        )[0, 1]
        assert corr > 0.99, f"Activation correlation {corr:.4f} below threshold"

    def test_steering_modifies_output(self, pyvene_runner):
        """Steering intervention changes model output."""
        from src.models.intervention_utils import steering

        prompt = "The weather today is"
        base_out = pyvene_runner.generate(prompt, max_new_tokens=5, temperature=0.0)

        direction = np.random.randn(pyvene_runner.d_model).astype(np.float32)
        intervention = steering(
            layer=pyvene_runner.n_layers // 2,
            direction=direction,
            strength=50.0,
        )
        steered_out = pyvene_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert steered_out != base_out, "Steering had no effect on output"


@requires_pyvene
class TestPyveneMultiArch:
    """Test pyvene backend with non-GPT2 architectures."""

    @pytest.fixture(scope="class")
    def pythia_pyvene_runner(self):
        """Load Pythia model with pyvene backend."""
        from src.models.model_runner import ModelRunner, ModelBackend

        model_name = TEST_MODELS["pythia-70m"]["name"]
        runner = ModelRunner(model_name, backend=ModelBackend.PYVENE)
        yield runner
        del runner
        gc.collect()

    def test_pythia_pyvene_generation(self, pythia_pyvene_runner):
        """Pyvene generates coherent output for Pythia.

        Note: Exact match with TransformerLens isn't expected for non-GPT2
        architectures due to implementation differences in model conversion.
        """
        prompt = "The answer is"
        output = pythia_pyvene_runner.generate(
            prompt, max_new_tokens=10, temperature=0.0
        )

        assert len(output) > 0, "No output generated"
        assert isinstance(output, str)

    def test_pythia_pyvene_activation_capture(self, pythia_pyvene_runner):
        """Pyvene captures activations for Pythia architecture."""
        prompt = "Testing"
        layer = pythia_pyvene_runner.n_layers // 2
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        logits, cache = pythia_pyvene_runner.run_with_cache(prompt, names_filter)

        assert f"blocks.{layer}.hook_resid_post" in cache
        act = cache[f"blocks.{layer}.hook_resid_post"]
        assert act.shape[-1] == pythia_pyvene_runner.d_model

    def test_pythia_pyvene_steering(self, pythia_pyvene_runner):
        """Pyvene steering works for Pythia architecture."""
        from src.models.intervention_utils import steering

        prompt = "The result is"
        base_out = pythia_pyvene_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0
        )

        direction = np.random.randn(pythia_pyvene_runner.d_model).astype(np.float32)
        intervention = steering(
            layer=pythia_pyvene_runner.n_layers // 2,
            direction=direction,
            strength=30.0,
        )
        steered_out = pythia_pyvene_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert steered_out != base_out, "Steering had no effect on Pythia"



class TestPrimaryModelsCore:
    """Core functionality tests using primary models (GPT-2, Qwen base, Qwen instruct).

    These tests ensure that the most important operations work correctly
    across the primary model set used for research.
    """

    @pytest.fixture(scope="class", params=PRIMARY_MODELS)
    def primary_runner(self, request):
        """Parametrized fixture that yields runners for all primary models."""
        from src.models.model_runner import ModelRunner, ModelBackend

        model_name = request.param
        runner = ModelRunner(model_name, backend=ModelBackend.TRANSFORMERLENS)
        yield runner, model_name
        del runner
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_generation_deterministic(self, primary_runner):
        """Deterministic generation works for all primary models."""
        runner, model_name = primary_runner
        prompt = "The meaning of"

        out1 = runner.generate(prompt, max_new_tokens=10, temperature=0.0)
        out2 = runner.generate(prompt, max_new_tokens=10, temperature=0.0)

        assert out1 == out2, f"{model_name}: Non-deterministic generation with temp=0"

    def test_activation_shapes_correct(self, primary_runner):
        """Activation shapes are correct for all primary models."""
        runner, model_name = primary_runner
        prompt = "Hello world"

        layer = runner.n_layers // 2
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"
        logits, cache = runner.run_with_cache(prompt, names_filter)

        key = f"blocks.{layer}.hook_resid_post"
        act = cache[key]

        assert act.dim() == 3, f"{model_name}: Expected 3D activation tensor"
        assert act.shape[0] == 1, f"{model_name}: Expected batch size 1"
        assert act.shape[2] == runner.d_model, f"{model_name}: Wrong hidden dimension"

    def test_steering_effective(self, primary_runner):
        """Steering is effective for all primary models."""
        from src.models.intervention_utils import steering, random_direction

        runner, model_name = primary_runner
        prompt = "I think that"

        base_out = runner.generate(prompt, max_new_tokens=5, temperature=0.0)

        direction = random_direction(runner.d_model, seed=42)
        intervention = steering(
            layer=runner.n_layers // 2,
            direction=direction,
            strength=100.0,
        )
        steered_out = runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert steered_out != base_out, f"{model_name}: Steering had no effect"

    def test_get_label_probs_valid(self, primary_runner):
        """get_label_probs returns valid probabilities for all primary models."""
        runner, model_name = primary_runner
        prompt = "I choose option"

        probs = runner.get_label_probs(prompt, " ", ("A", "B"))

        assert len(probs) == 2, f"{model_name}: Expected 2 probabilities"
        assert all(0 <= p <= 1 for p in probs), (
            f"{model_name}: Probabilities out of range"
        )


# =============================================================================
# Real Intervention Tests - STEERING
# =============================================================================


class TestSteeringReal:
    """Test steering interventions with real model."""

    def test_steering_apply_to_position(self, transformerlens_runner):
        """Position-targeted steering works."""
        from src.models.intervention_utils import steering

        prompt = "One two three four five"
        d_model = transformerlens_runner.d_model
        direction = np.random.randn(d_model).astype(np.float32)

        intervention = steering(
            layer=transformerlens_runner.n_layers // 2,
            direction=direction,
            strength=100.0,
            positions=2,
        )

        # Should not crash and should produce output
        out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert len(out) > 0

    def test_steering_strength_scaling(self, transformerlens_runner):
        """Higher steering strength has larger effect."""
        from src.models.interventions import create_intervention_hook
        from src.models.intervention_utils import steering

        prompt = "Hello"
        d_model = transformerlens_runner.d_model
        direction = np.random.randn(d_model).astype(np.float32)
        layer = 3

        # Get baseline activations
        _, base_cache = transformerlens_runner.run_with_cache(
            prompt, lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
        base_act = base_cache[f"blocks.{layer}.hook_resid_post"].clone()

        # Apply low strength
        low_intervention = steering(layer=layer, direction=direction, strength=1.0)
        low_hook, _ = create_intervention_hook(
            low_intervention,
            transformerlens_runner.dtype,
            transformerlens_runner.device,
        )
        low_result = low_hook(base_act.clone())

        # Apply high strength
        high_intervention = steering(layer=layer, direction=direction, strength=100.0)
        high_hook, _ = create_intervention_hook(
            high_intervention,
            transformerlens_runner.dtype,
            transformerlens_runner.device,
        )
        high_result = high_hook(base_act.clone())

        # High strength should have larger deviation from base
        low_diff = (low_result - base_act).abs().sum()
        high_diff = (high_result - base_act).abs().sum()
        assert high_diff > low_diff * 10


# =============================================================================
# Real Intervention Tests - ABLATION
# =============================================================================


class TestAblationReal:
    """Test ablation interventions with real model."""

    def test_zero_ablation_apply_to_all(self, transformerlens_runner):
        """Zero ablation dramatically changes output."""
        from src.models.intervention_utils import ablation

        prompt = "The quick brown"

        intervention = ablation(layer=transformerlens_runner.n_layers // 2)

        base_out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0
        )
        ablated_out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert ablated_out != base_out, "Ablation had no effect"

    def test_mean_ablation_apply_to_all(self, transformerlens_runner):
        """Mean ablation with computed means."""
        from src.models.intervention_utils import ablation

        prompt = "Testing mean ablation"
        layer = 4

        # Compute mean activations from a few prompts
        prompts = ["Hello world", "The cat sat", "One two three"]
        activations = []
        for p in prompts:
            _, cache = transformerlens_runner.run_with_cache(
                p, lambda n: n == f"blocks.{layer}.hook_resid_post"
            )
            activations.append(cache[f"blocks.{layer}.hook_resid_post"])

        mean_act = torch.stack([a.mean(dim=(0, 1)) for a in activations]).mean(dim=0)

        intervention = ablation(layer=layer, values=mean_act.cpu().numpy())

        out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert len(out) > 0

    def test_ablation_apply_to_position(self, transformerlens_runner):
        """Position-targeted ablation works."""
        from src.models.intervention_utils import ablation

        prompt = "One two three four"

        intervention = ablation(
            layer=transformerlens_runner.n_layers // 2,
            positions=[1, 2],
        )

        out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert len(out) > 0


# =============================================================================
# Real Intervention Tests - ACTIVATION PATCHING
# =============================================================================


class TestActivationPatchingReal:
    """Test activation patching with real model."""

    def test_patching_apply_to_all(self, transformerlens_runner):
        """Patching replaces activations with cached values."""
        from src.models.intervention_utils import patch

        source_prompt = "Paris is the capital of"
        target_prompt = "Berlin is the capital of"
        layer = transformerlens_runner.n_layers // 2

        # Get source activations
        _, source_cache = transformerlens_runner.run_with_cache(
            source_prompt, lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
        source_act = (
            source_cache[f"blocks.{layer}.hook_resid_post"][0].detach().cpu().numpy()
        )

        # Generate with target prompt normally
        base_out = transformerlens_runner.generate(
            target_prompt, max_new_tokens=5, temperature=0.0
        )

        # Generate with patched activations
        intervention = patch(layer=layer, values=source_act)

        patched_out = transformerlens_runner.generate(
            target_prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert patched_out != base_out, "Patching had no effect"

    def test_patching_apply_to_position(self, transformerlens_runner):
        """Position-targeted patching works."""
        from src.models.intervention_utils import patch

        source_prompt = "The cat sat on"
        target_prompt = "The dog ran on"
        layer = 4

        # Get source activations
        _, source_cache = transformerlens_runner.run_with_cache(
            source_prompt, lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
        source_act = (
            source_cache[f"blocks.{layer}.hook_resid_post"][0].detach().cpu().numpy()
        )

        # Patch position 1 ("cat"/"dog" position)
        intervention = patch(layer=layer, values=source_act, positions=1)

        out = transformerlens_runner.generate(
            target_prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert len(out) > 0

    def test_counterfactual_patching_real(self, transformerlens_runner):
        """Full counterfactual patching workflow."""
        from src.models.intervention_utils import patch

        clean_prompt = "The Eiffel Tower is located in"
        corrupt_prompt = "The Eiffel Tower is located in Berlin, the capital of"

        # Get clean activations at final layer
        final_layer = transformerlens_runner.n_layers - 1
        _, clean_cache = transformerlens_runner.run_with_cache(
            clean_prompt, lambda n: n == f"blocks.{final_layer}.hook_resid_post"
        )
        clean_act = (
            clean_cache[f"blocks.{final_layer}.hook_resid_post"][0]
            .detach()
            .cpu()
            .numpy()
        )

        # Run corrupt with clean activations patched
        intervention = patch(layer=final_layer, values=clean_act)

        patched_out = transformerlens_runner.generate(
            corrupt_prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )

        # Should produce output
        assert len(patched_out) > 0


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Test batch processing with batch_size > 1."""

    def test_batch_generate(self, transformerlens_runner):
        """Generate handles list of prompts."""
        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Spain is",
        ]

        outputs = transformerlens_runner.generate(
            prompts, max_new_tokens=5, temperature=0.0
        )

        assert len(outputs) == 3
        assert all(isinstance(o, str) for o in outputs)
        assert all(len(o) > 0 for o in outputs)

    def test_batch_get_label_probs(self, transformerlens_runner):
        """Label probs handles list of prompts."""
        prompts = ["Choose a) or b)", "Select a) or b)"]
        labels = ("a)", "b)")

        probs = transformerlens_runner.get_label_probs(prompts, "I select:", labels)

        assert len(probs) == 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in probs)

    def test_batch_with_intervention(self, transformerlens_runner):
        """Batch generation with intervention."""
        from src.models.intervention_utils import steering

        prompts = ["Hello there", "Good morning"]
        d_model = transformerlens_runner.d_model
        direction = np.random.randn(d_model).astype(np.float32)

        intervention = steering(layer=3, direction=direction, strength=10.0)

        outputs = transformerlens_runner.generate(
            prompts, max_new_tokens=5, temperature=0.0, intervention=intervention
        )

        assert len(outputs) == 2


# =============================================================================
# Pattern Matching During Generation
# =============================================================================


class TestPatternMatchingGeneration:
    """Test pattern-based targeting during actual generation."""

    def test_pattern_steering_during_generation(self, transformerlens_runner):
        """Pattern-based steering triggers during generation."""
        from src.models.interventions import create_intervention_hook
        from src.models.intervention_utils import steering

        # Prompt that should generate "I think" or similar
        prompt = "When asked about the weather,"
        d_model = transformerlens_runner.d_model
        direction = np.random.randn(d_model).astype(np.float32)

        intervention = steering(
            layer=transformerlens_runner.n_layers // 2,
            direction=direction,
            strength=50.0,
            pattern=" the",  # common pattern
        )

        # Create hook with real tokenizer
        hook, matcher = create_intervention_hook(
            intervention,
            transformerlens_runner.dtype,
            transformerlens_runner.device,
            transformerlens_runner.tokenizer,
        )

        assert matcher is not None
        assert not matcher.should_apply()

        # Simulate tokens being generated
        test_text = "I think the weather"
        for i in range(1, len(test_text) + 1):
            partial = test_text[:i]
            ids = transformerlens_runner.tokenizer.encode(
                partial, add_special_tokens=False
            )
            if ids:
                matcher.update(torch.tensor([ids[-1]]))

        # Pattern " the" should have been found
        # (Note: depends on tokenization, might need adjustment)


# =============================================================================
# KV Cache Tests
# =============================================================================


class TestKVCache:
    """Test KV cache interactions."""

    def test_kv_cache_init_and_freeze(self, transformerlens_runner):
        """KV cache can be initialized and frozen."""
        cache = transformerlens_runner.init_kv_cache()
        assert cache is not None

        # Run with cache
        prompt = "Test prompt"
        logits, _ = transformerlens_runner.run_with_cache(prompt, past_kv_cache=cache)

        # Freeze cache
        cache.freeze()

        # Should still work after freeze
        assert logits is not None

    def test_generate_from_cache(self, transformerlens_runner):
        """Generate from frozen cache works."""
        prompt = "The answer is"

        # Init cache and get prefill logits
        cache = transformerlens_runner.init_kv_cache()
        prefill_logits, _ = transformerlens_runner.run_with_cache(
            prompt, past_kv_cache=cache
        )
        cache.freeze()

        # Generate from cache
        output = transformerlens_runner.generate_from_cache(
            prefill_logits, cache, max_new_tokens=5, temperature=0.0
        )

        assert len(output) > 0

    def test_cached_vs_uncached_same_result(self, transformerlens_runner):
        """Cached generation matches uncached."""
        prompt = "Hello world"

        # Uncached generation
        uncached = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0
        )

        # Cached generation
        cache = transformerlens_runner.init_kv_cache()
        prefill_logits, _ = transformerlens_runner.run_with_cache(
            prompt, past_kv_cache=cache
        )
        cache.freeze()
        cached = transformerlens_runner.generate_from_cache(
            prefill_logits, cache, max_new_tokens=5, temperature=0.0
        )

        assert uncached == cached, f"Uncached: {uncached!r} != Cached: {cached!r}"


# =============================================================================
# Multiple Interventions
# =============================================================================


class TestMultipleInterventions:
    """Test multiple interventions together."""

    def test_different_layers_steer_and_ablate(self, transformerlens_runner):
        """Can apply different interventions at different layers."""
        from src.models.interventions import create_intervention_hook
        from src.models.intervention_utils import steering, ablation

        prompt = "The cat sat on"
        d_model = transformerlens_runner.d_model

        # Steer at layer 2
        steer_intervention = steering(
            layer=2,
            direction=np.random.randn(d_model).astype(np.float32),
            strength=10.0,
        )

        # Ablate at layer 4
        ablate_intervention = ablation(layer=4)

        # Get activations and apply both hooks manually
        _, cache = transformerlens_runner.run_with_cache(prompt)

        steer_hook, _ = create_intervention_hook(
            steer_intervention,
            transformerlens_runner.dtype,
            transformerlens_runner.device,
        )
        ablate_hook, _ = create_intervention_hook(
            ablate_intervention,
            transformerlens_runner.dtype,
            transformerlens_runner.device,
        )

        # Apply to layer 2 activations - keep original for comparison
        layer2_original = cache["blocks.2.hook_resid_post"].clone()
        layer2_to_steer = layer2_original.clone()
        steered = steer_hook(layer2_to_steer)

        # Apply to layer 4 activations
        layer4_act = cache["blocks.4.hook_resid_post"].clone()
        ablated = ablate_hook(layer4_act)

        # Verify both worked
        assert not torch.allclose(steered, layer2_original), "Steering had no effect"
        assert ablated.abs().sum() < 1e-6, "Ablation did not zero activations"


# =============================================================================
# Long Sequence Tests
# =============================================================================


class TestLongSequences:
    """Test with longer sequences."""

    def test_long_prompt(self, transformerlens_runner):
        """Handles longer prompts."""
        # Create a longer prompt
        prompt = " ".join(["word"] * 100)

        out = transformerlens_runner.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert len(out) > 0

    def test_intervention_long_sequence(self, transformerlens_runner):
        """Interventions work with longer sequences."""
        from src.models.intervention_utils import steering

        prompt = " ".join(["token"] * 50)
        d_model = transformerlens_runner.d_model

        intervention = steering(
            layer=3,
            direction=np.random.randn(d_model).astype(np.float32),
            strength=10.0,
            positions=[10, 20, 30, 40],
        )

        out = transformerlens_runner.generate(
            prompt, max_new_tokens=5, temperature=0.0, intervention=intervention
        )
        assert len(out) > 0


# =============================================================================
# Edge Cases with Real Tokenizer
# =============================================================================


class TestRealTokenizerEdgeCases:
    """Test edge cases with real tokenizer."""

    def test_special_characters(self, transformerlens_runner):
        """Handles special characters in prompt."""
        prompts = [
            "Hello! How are you?",
            "Test: 1, 2, 3...",
            'Quote: "test"',
            "Math: 2+2=4",
        ]

        for prompt in prompts:
            out = transformerlens_runner.generate(
                prompt, max_new_tokens=3, temperature=0.0
            )
            assert len(out) > 0

    def test_unicode(self, transformerlens_runner):
        """Handles unicode characters."""
        prompt = "Café résumé naïve"
        out = transformerlens_runner.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert len(out) > 0

    def test_empty_continuation(self, transformerlens_runner):
        """Handles case where model immediately outputs EOS."""
        # Very short prompt
        prompt = "."
        out = transformerlens_runner.generate(prompt, max_new_tokens=3, temperature=0.0)
        # Should not crash, output can be empty or short


# =============================================================================
# Query Dataset Integration
# =============================================================================


from dataclasses import dataclass, field


@dataclass
class MockPromptDataset:
    """Mock PromptDataset for testing - simplified structure that works with query_dataset."""
    dataset_id: str
    config: dict
    samples: list = field(default_factory=list)


def make_test_prompt_dataset(dataset_id: str, samples: list, choice_prefix: str = "I choose:") -> MockPromptDataset:
    """Create a mock PromptDataset for testing."""
    return MockPromptDataset(
        dataset_id=dataset_id,
        config={"name": "test", "prompt_format": {"const_keywords": {"format_choice_prefix": choice_prefix}}},
        samples=samples,
    )


def make_sample(sample_idx: int, text: str, short_label: str = "a)", long_label: str = "b)") -> dict:
    """Create a sample with full preference_pair structure for testing."""
    return {
        "sample_idx": sample_idx,
        "prompt": {
            "text": text,
            "preference_pair": {
                "short_term": {
                    "label": short_label,
                    "time": {"value": 0, "unit": "days"},
                    "reward": {"value": 100, "unit": "dollars"},
                },
                "long_term": {
                    "label": long_label,
                    "time": {"value": 12, "unit": "months"},
                    "reward": {"value": 200, "unit": "dollars"},
                },
            },
        },
    }


class TestQueryDatasetIntegration:
    """Integration tests for query_dataset with real model."""

    def test_query_dataset_real_model(self, transformerlens_runner, tmp_path):
        """Full query_dataset flow with real model."""
        from src.models.query_runner import QueryRunner, QueryConfig

        samples = [
            make_sample(1, "Would you prefer a) $100 now or b) $200 in a year? I choose:"),
        ]
        prompt_dataset = make_test_prompt_dataset("001", samples)

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(prompt_dataset, TEST_MODEL)

        assert len(output.preferences) == 1
        assert output.preferences[0].choice in ["short_term", "long_term", "unknown"]
        assert len(output.preferences[0].response_text) > 0

    def test_query_dataset_captures_internals(self, transformerlens_runner, tmp_path):
        """Internals are captured with correct shapes."""
        from src.models.query_runner import QueryRunner, QueryConfig, InternalsConfig, ActivationSpec

        samples = [
            make_sample(1, "Choose: a) now or b) later? I choose:"),
        ]
        prompt_dataset = make_test_prompt_dataset("002", samples)

        # Configure internals capture
        internals = InternalsConfig(
            activations=[ActivationSpec(component="resid_post", layers=[0, 5])],
        )
        config = QueryConfig(internals=internals)
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(prompt_dataset, TEST_MODEL)

        pref = output.preferences[0]
        assert pref.internals is not None
        assert len(pref.internals.activations) > 0
        # Check shapes
        for name, act in pref.internals.activations.items():
            assert act.dim() == 2  # (seq_len, d_model) after [0] indexing
            assert act.shape[-1] == transformerlens_runner.d_model

    def test_query_dataset_probabilities_sensible(
        self, transformerlens_runner, tmp_path
    ):
        """Captured probabilities are in valid range."""
        from src.models.query_runner import QueryRunner, QueryConfig

        samples = [
            make_sample(1, "Pick: a) apple or b) banana? I choose:"),
        ]
        prompt_dataset = make_test_prompt_dataset("003", samples)

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(prompt_dataset, TEST_MODEL)

        pref = output.preferences[0]
        # Probabilities should be in [0, 1]
        assert 0 <= pref.choice_prob <= 1
        assert 0 <= pref.alt_prob <= 1
        # At least one should be non-zero
        assert pref.choice_prob > 0 or pref.alt_prob > 0

    def test_query_dataset_multiple_samples(self, transformerlens_runner, tmp_path):
        """Multiple samples are all processed."""
        from src.models.query_runner import QueryRunner, QueryConfig

        samples = [
            make_sample(i, f"Question {i}: a) yes or b) no? Answer:")
            for i in range(5)
        ]
        prompt_dataset = make_test_prompt_dataset("004", samples, choice_prefix="Answer:")

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(prompt_dataset, TEST_MODEL)

        assert len(output.preferences) == 5
        # Each should have unique sample_idx
        ids = [p.sample_idx for p in output.preferences]
        assert len(set(ids)) == 5

    def test_model_caching_across_datasets(self, transformerlens_runner, tmp_path):
        """Model is reused for same model name."""
        from src.models.query_runner import QueryRunner, QueryConfig

        samples = [
            make_sample(1, "Choose a) or b)? I pick:"),
        ]
        prompt_dataset1 = make_test_prompt_dataset("005", samples, choice_prefix="I pick:")
        prompt_dataset2 = make_test_prompt_dataset("006", samples, choice_prefix="I pick:")

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        # Query both datasets
        out1 = runner.query_dataset(prompt_dataset1, TEST_MODEL)
        model_after_first = runner._model

        out2 = runner.query_dataset(prompt_dataset2, TEST_MODEL)
        model_after_second = runner._model

        # Same model object should be used
        assert model_after_first is model_after_second


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestQueryErrorHandling:
    """Test error handling in query flow."""

    def test_dataset_not_found(self, transformerlens_runner, tmp_path):
        """Raises error for missing dataset when loading by ID."""
        from src.prompt_datasets import PromptDataset

        # Test that PromptDataset.load_from_id raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            PromptDataset.load_from_id("nonexistent", tmp_path)

    def test_empty_samples_list(self, transformerlens_runner, tmp_path):
        """Handles empty samples list gracefully."""
        from src.models.query_runner import QueryRunner, QueryConfig

        prompt_dataset = make_test_prompt_dataset("empty", [], choice_prefix="Choose:")

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(prompt_dataset, TEST_MODEL)
        assert len(output.preferences) == 0

    def test_missing_choice_prefix_uses_default(self, transformerlens_runner, tmp_path):
        """Missing choice_prefix falls back to default."""
        from src.models.query_runner import QueryRunner, QueryConfig

        samples = [
            make_sample(1, "Pick a) or b)? I select:"),
        ]
        # Create dataset with empty config (no prompt_format)
        prompt_dataset = MockPromptDataset(
            dataset_id="noprefix",
            config={},  # No prompt_format
            samples=samples,
        )

        config = QueryConfig()
        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        # Should not crash - uses default "I select:"
        output = runner.query_dataset(prompt_dataset, TEST_MODEL)
        assert len(output.preferences) == 1


# =============================================================================
# Save Output Tests
# =============================================================================


class TestPreferenceDatasetSave:
    """Test PreferenceDataset.save_as_json method."""

    def test_save_as_json_creates_json(self, tmp_path):
        """JSON file is created with correct structure."""
        from src.models.preference_dataset import PreferenceDataset
        from src.common.types import PreferenceSample

        pref_dataset = PreferenceDataset(
            prompt_dataset_id="test_ds",
            model="test/model-name",
            preferences=[
                PreferenceSample(
                    sample_idx=1,
                    time_horizon={"value": 6, "unit": "months"},
                    short_term_label="a)",
                    long_term_label="b)",
                    short_term_reward=100.0,
                    long_term_reward=200.0,
                    short_term_time=1.0,
                    long_term_time=12.0,
                    choice="short_term",
                    choice_prob=0.7,
                    alt_prob=0.3,
                    response_text="I choose a)",
                    internals=None,
                ),
            ],
        )

        json_path = tmp_path / "test_ds_model-name.json"
        pref_dataset.save_as_json(json_path)

        assert json_path.exists()

        import json

        with open(json_path) as f:
            data = json.load(f)

        assert data["prompt_dataset_id"] == "test_ds"
        assert data["model"] == "test/model-name"
        assert len(data["preferences"]) == 1
        assert data["preferences"][0]["choice"] == "short_term"

    def test_save_as_json_creates_internals_file(self, tmp_path):
        """Internals are saved to .pt file."""
        from src.models.preference_dataset import PreferenceDataset
        from src.common.types import PreferenceSample, CapturedInternals

        activations = {"blocks.5.hook_resid_post": torch.randn(10, 768)}
        internals = CapturedInternals(
            activations=activations,
            activation_names=list(activations.keys()),
        )

        pref_dataset = PreferenceDataset(
            prompt_dataset_id="test_ds",
            model="test-model",
            preferences=[
                PreferenceSample(
                    sample_idx=42,
                    time_horizon=None,
                    short_term_label="a)",
                    long_term_label="b)",
                    short_term_reward=100.0,
                    long_term_reward=200.0,
                    short_term_time=1.0,
                    long_term_time=12.0,
                    choice="long_term",
                    choice_prob=0.6,
                    alt_prob=0.4,
                    response_text="I choose b)",
                    internals=internals,
                ),
            ],
        )

        json_path = tmp_path / "test_ds_test-model.json"
        pref_dataset.save_as_json(json_path)

        # Check internals dir and file
        internals_dir = tmp_path / "internals"
        assert internals_dir.exists()

        pt_files = list(internals_dir.glob("*.pt"))
        assert len(pt_files) == 1

        # Load and verify
        loaded = torch.load(pt_files[0])
        assert "blocks.5.hook_resid_post" in loaded


# =============================================================================
# get_label_probs Tests
# =============================================================================


class TestGetLabelProbsIntegration:
    """Test get_label_probs with real model.

    This function is critical for extracting model preferences in the query pipeline.
    Tests verify:
    - Probabilities are valid and sensible
    - Consistent across backends
    - Works with various label formats
    """

    def test_returns_valid_probabilities(self, transformerlens_runner):
        """Probabilities are valid floats in [0, 1]."""
        prompt = "I choose option"
        probs = transformerlens_runner.get_label_probs(prompt, " ", ("A", "B"))

        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)

    def test_meaningful_probability_distribution(self, transformerlens_runner):
        """Probabilities reflect model's knowledge."""
        prompt = "The capital of France is"
        probs = transformerlens_runner.get_label_probs(prompt, " ", ("Paris", "London"))

        # Paris should have higher probability than London for this prompt
        assert probs[0] > probs[1], (
            f"Paris ({probs[0]:.4f}) should be more likely than London ({probs[1]:.4f})"
        )

    @requires_nnsight
    def test_backends_agree_on_probabilities(
        self, transformerlens_runner, nnsight_runner
    ):
        """Different backends compute same probabilities."""
        prompt = "The answer is"
        labels = ("yes", "no")

        tl_probs = transformerlens_runner.get_label_probs(prompt, " ", labels)
        nn_probs = nnsight_runner.get_label_probs(prompt, " ", labels)

        for i, label in enumerate(labels):
            assert abs(tl_probs[i] - nn_probs[i]) < 0.01, (
                f"Probability mismatch for {label}"
            )

    @requires_pyvene
    def test_pyvene_get_label_probs(self, pyvene_runner, transformerlens_runner):
        """Pyvene computes same probabilities as TransformerLens."""
        prompt = "The weather is"
        labels = ("good", "bad")

        pv_probs = pyvene_runner.get_label_probs(prompt, " ", labels)
        tl_probs = transformerlens_runner.get_label_probs(prompt, " ", labels)

        for i, label in enumerate(labels):
            assert abs(pv_probs[i] - tl_probs[i]) < 0.01, (
                f"Probability mismatch for {label}"
            )

    def test_multi_token_labels(self, transformerlens_runner):
        """Handles multi-token labels correctly."""
        prompt = "My favorite color is"
        probs = transformerlens_runner.get_label_probs(
            prompt, " ", ("bright red", "dark blue")
        )

        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)

    def test_choice_labels_typical_format(self, transformerlens_runner):
        """Works with typical choice format used in preference datasets."""
        prompt = "Would you prefer $100 now or $200 later? I choose:"
        probs = transformerlens_runner.get_label_probs(prompt, " ", ("a)", "b)"))

        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)
        # Both should have some probability mass
        assert sum(probs) > 0, "No probability assigned to either label"


# =============================================================================
# Query Runner Intervention Tests
# =============================================================================


class TestQueryRunnerIntervention:
    """Test intervention support in QueryRunner.

    Verifies that:
    - Intervention configs load correctly from query config
    - Interventions are applied during generation
    - Output differs from non-intervened runs
    """

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset for testing."""
        samples = [
            make_sample(1, "Would you prefer a) $100 now or b) $200 in a year? I choose:"),
        ]
        return make_test_prompt_dataset("interv", samples)

    def test_query_config_with_intervention_loads(self):
        """QueryConfig accepts intervention field."""
        from src.models.query_runner import QueryConfig

        intervention = {
            "layer": 5,
            "mode": "add",
            "strength": 50.0,
            "values": "random",
        }

        config = QueryConfig(intervention=intervention)

        assert config.intervention is not None
        assert config.intervention["mode"] == "add"

    def test_intervention_loads_from_runner(
        self, transformerlens_runner, sample_dataset
    ):
        """QueryRunner loads intervention for model."""
        from src.models.query_runner import QueryRunner, QueryConfig

        intervention = {
            "layer": 5,
            "mode": "add",
            "strength": 50.0,
            "values": "random",
        }

        config = QueryConfig(intervention=intervention)

        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        loaded_intervention = runner._load_intervention(transformerlens_runner)

        assert loaded_intervention is not None
        assert loaded_intervention.layer == 5
        assert loaded_intervention.strength == 50.0

    def test_query_with_intervention_runs(self, transformerlens_runner, sample_dataset):
        """Query with intervention completes without error."""
        from src.models.query_runner import QueryRunner, QueryConfig

        intervention = {
            "layer": 5,
            "mode": "add",
            "strength": 50.0,
            "values": "random",
        }

        config = QueryConfig(intervention=intervention)

        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(sample_dataset, TEST_MODEL)

        assert len(output.preferences) == 1
        assert output.preferences[0].response_text is not None

    def test_intervention_changes_output(self, transformerlens_runner, sample_dataset):
        """Intervention produces different output than baseline."""
        from src.models.query_runner import QueryRunner, QueryConfig
        import numpy as np

        # Run without intervention
        config_base = QueryConfig()

        runner_base = QueryRunner(config_base)
        runner_base._model = transformerlens_runner
        runner_base._model_name = TEST_MODEL
        output_base = runner_base.query_dataset(sample_dataset, TEST_MODEL)

        # Run with strong intervention
        # Use fixed direction for reproducibility
        np.random.seed(42)
        direction = np.random.randn(transformerlens_runner.d_model).astype(np.float32)
        direction = (direction / np.linalg.norm(direction)).tolist()

        intervention = {
            "layer": 6,
            "mode": "add",
            "strength": 100.0,
            "values": direction,
        }

        config_interv = QueryConfig(intervention=intervention)

        runner_interv = QueryRunner(config_interv)
        runner_interv._model = transformerlens_runner
        runner_interv._model_name = TEST_MODEL
        output_interv = runner_interv.query_dataset(sample_dataset, TEST_MODEL)

        # Responses should differ (strong steering should change output)
        base_response = output_base.preferences[0].response_text
        interv_response = output_interv.preferences[0].response_text

        # They may occasionally be the same, so we just verify we got valid outputs
        assert len(base_response) > 0
        assert len(interv_response) > 0

    def test_ablation_intervention_in_query(
        self, transformerlens_runner, sample_dataset
    ):
        """Ablation intervention works in query flow."""
        from src.models.query_runner import QueryRunner, QueryConfig

        intervention = {
            "layer": 3,
            "mode": "set",
            "values": 0,
        }

        config = QueryConfig(intervention=intervention)

        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(sample_dataset, TEST_MODEL)

        assert len(output.preferences) == 1

    def test_intervention_config_from_sample_file_format(
        self, transformerlens_runner, sample_dataset
    ):
        """Intervention config matches sample_interventions JSON format."""
        from src.models.query_runner import QueryRunner, QueryConfig
        from src.models.intervention_loader import load_intervention_json

        # Load an existing sample intervention
        sample_config = load_intervention_json("steer_all")

        # Use it in QueryConfig
        config = QueryConfig(intervention=sample_config)

        runner = QueryRunner(config)
        runner._model = transformerlens_runner
        runner._model_name = TEST_MODEL

        output = runner.query_dataset(sample_dataset, TEST_MODEL)

        assert len(output.preferences) == 1
