"""Cross-architecture sanity tests.

Tests basic operations across diverse model families:
- GPT-2 style (gpt2)
- Pythia/GPT-NeoX style (pythia-70m)
- OPT style (opt-125m)
- TinyStories

These tests are slow because they load multiple different model architectures.
Skip with: pytest --skip-slow
"""

import pytest
import torch

from src.models.model_runner import ModelRunner, ModelBackend
from src.models.interventions import steering, random_direction


# Models for cross-architecture sanity tests (4 models covering different architectures)
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


@pytest.mark.slow
class TestCrossArchitectureSanity:
    """Sanity tests across different model architectures.

    Verifies basic operations work across diverse model families.
    """

    @pytest.fixture(scope="class")
    def model_cache(self):
        """Cache for loaded models to avoid reloading."""
        return {}

    def _get_runner(self, model_name, model_cache):
        """Get or create a runner for the given model."""
        if model_name not in model_cache:
            try:
                runner = ModelRunner(model_name, backend=ModelBackend.TRANSFORMERLENS)
                model_cache[model_name] = runner
            except Exception as e:
                pytest.skip(f"Could not load {model_name}: {e}")
        return model_cache[model_name]

    @pytest.mark.parametrize("model_name,params,arch", CROSS_ARCH_MODELS)
    def test_model_loads_and_generates(self, model_name, params, arch, model_cache):
        """Each model can be loaded and generates text."""
        runner = self._get_runner(model_name, model_cache)

        prompt = "The answer is"
        output = runner.generate(prompt, max_new_tokens=5, temperature=0.0)

        assert isinstance(output, str), f"{model_name} did not return string"
        assert len(output) > 0, f"{model_name} generated empty output"

    @pytest.mark.parametrize("model_name,params,arch", CROSS_ARCH_MODELS)
    def test_model_captures_activations(self, model_name, params, arch, model_cache):
        """Each model can capture activations."""
        runner = self._get_runner(model_name, model_cache)

        prompt = "Test"
        layer = runner.n_layers // 2
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"

        logits, cache = runner.run_with_cache(prompt, names_filter)

        key = f"blocks.{layer}.hook_resid_post"
        assert key in cache, f"{model_name} missing activation {key}"
        assert cache[key].shape[-1] == runner.d_model, f"{model_name} wrong d_model"

    @pytest.mark.parametrize("model_name,params,arch", CROSS_ARCH_MODELS)
    def test_model_steering_changes_output(self, model_name, params, arch, model_cache):
        """Each model responds to steering intervention."""
        runner = self._get_runner(model_name, model_cache)

        prompt = "The result is"
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

        assert steered_out != base_out, f"{model_name} steering had no effect"
