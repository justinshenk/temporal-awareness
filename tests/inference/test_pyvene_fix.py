"""Test that pyvene backend now applies interventions correctly using set_source_representation."""

import numpy as np
import pytest
import torch

from src.inference import ModelRunner
from src.inference.model_runner import ModelBackend
from src.inference.interventions import Intervention, InterventionTarget


TEST_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def runner_pyvene():
    """Pyvene backend."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.PYVENE)


@pytest.fixture(scope="module")
def runner_huggingface():
    """HuggingFace backend."""
    return ModelRunner(TEST_MODEL, backend=ModelBackend.HUGGINGFACE)


class TestPyveneInterventionFix:
    """Test that pyvene interventions are actually applied (non-zero diff)."""

    def test_pyvene_set_intervention_applied(self, runner_pyvene):
        """Test that SET intervention produces non-zero diff from baseline."""
        prompt = "The capital of France is"
        input_ids = runner_pyvene._backend.encode(prompt)

        # Get baseline
        with torch.no_grad():
            baseline = runner_pyvene._backend.forward(input_ids)

        # Create intervention that sets activations to zeros at layer 5, position 2
        hidden_size = runner_pyvene.d_model
        zero_values = np.zeros((1, hidden_size), dtype=np.float32)

        intervention = Intervention(
            layer=5,
            component="resid_post",
            target=InterventionTarget(positions=[2]),
            mode="set",
            values=zero_values,
            strength=1.0,
        )

        # Run with intervention
        intervened = runner_pyvene._backend.run_with_intervention(input_ids, [intervention])

        # Check diff - should be non-zero
        diff = (baseline - intervened).abs()
        max_diff = diff.max().item()

        print(f"Pyvene SET intervention diff: max={max_diff:.4f}, mean={diff.mean().item():.4f}")

        # The key assertion: intervention MUST produce a difference
        assert max_diff > 0.01, f"Intervention was NOT applied! max_diff={max_diff}"

    def test_pyvene_add_intervention_applied(self, runner_pyvene):
        """Test that ADD intervention produces non-zero diff from baseline."""
        prompt = "The capital of France is"
        input_ids = runner_pyvene._backend.encode(prompt)

        # Get baseline
        with torch.no_grad():
            baseline = runner_pyvene._backend.forward(input_ids)

        # Create steering intervention
        hidden_size = runner_pyvene.d_model
        direction = np.random.randn(hidden_size).astype(np.float32)
        direction = direction / np.linalg.norm(direction)  # normalize

        intervention = Intervention(
            layer=5,
            component="resid_post",
            target=InterventionTarget(positions=[2]),
            mode="add",
            values=direction,
            strength=10.0,
        )

        # Run with intervention
        intervened = runner_pyvene._backend.run_with_intervention(input_ids, [intervention])

        # Check diff - should be non-zero
        diff = (baseline - intervened).abs()
        max_diff = diff.max().item()

        print(f"Pyvene ADD intervention diff: max={max_diff:.4f}, mean={diff.mean().item():.4f}")

        # The key assertion: intervention MUST produce a difference
        assert max_diff > 0.01, f"Intervention was NOT applied! max_diff={max_diff}"


class TestPyveneVsHuggingFace:
    """Compare pyvene and huggingface backend intervention results."""

    def test_baselines_match(self, runner_pyvene, runner_huggingface):
        """Test that baselines match between backends."""
        prompt = "The capital of France is"
        input_ids = runner_pyvene._backend.encode(prompt)

        with torch.no_grad():
            pv_baseline = runner_pyvene._backend.forward(input_ids)
            hf_baseline = runner_huggingface._backend.forward(input_ids)

        diff = (pv_baseline - hf_baseline).abs()
        max_diff = diff.max().item()

        print(f"Baseline diff: max={max_diff:.6f}")
        assert max_diff < 1e-4, f"Baselines should match, got max_diff={max_diff}"

    def test_both_backends_apply_set_intervention(self, runner_pyvene, runner_huggingface):
        """Test that both backends apply SET interventions."""
        prompt = "The capital of France is"
        input_ids = runner_pyvene._backend.encode(prompt)

        # Get baselines
        with torch.no_grad():
            pv_baseline = runner_pyvene._backend.forward(input_ids)
            hf_baseline = runner_huggingface._backend.forward(input_ids)

        # Create intervention
        hidden_size = runner_pyvene.d_model
        set_values = np.random.randn(1, hidden_size).astype(np.float32) * 5.0

        intervention = Intervention(
            layer=5,
            component="resid_post",
            target=InterventionTarget(positions=[2]),
            mode="set",
            values=set_values,
            strength=1.0,
        )

        # Run with intervention
        pv_intervened = runner_pyvene._backend.run_with_intervention(input_ids, [intervention])
        hf_intervened = runner_huggingface._backend.run_with_intervention(input_ids, [intervention])

        # Both should have non-zero effects
        pv_effect = (pv_baseline - pv_intervened).abs()
        hf_effect = (hf_baseline - hf_intervened).abs()

        pv_max = pv_effect.max().item()
        hf_max = hf_effect.max().item()

        print(f"Pyvene intervention effect: max={pv_max:.4f}")
        print(f"HuggingFace intervention effect: max={hf_max:.4f}")

        assert pv_max > 0.01, f"Pyvene intervention NOT applied! max={pv_max}"
        assert hf_max > 0.01, f"HuggingFace intervention NOT applied! max={hf_max}"

    def test_interventions_produce_similar_results(self, runner_pyvene, runner_huggingface):
        """Test that both backends produce similar intervention effects."""
        prompt = "The capital of France is"
        input_ids = runner_pyvene._backend.encode(prompt)

        # Create intervention
        np.random.seed(42)
        hidden_size = runner_pyvene.d_model
        set_values = np.random.randn(1, hidden_size).astype(np.float32) * 5.0

        intervention = Intervention(
            layer=5,
            component="resid_post",
            target=InterventionTarget(positions=[2]),
            mode="set",
            values=set_values,
            strength=1.0,
        )

        # Run with intervention
        pv_intervened = runner_pyvene._backend.run_with_intervention(input_ids, [intervention])
        hf_intervened = runner_huggingface._backend.run_with_intervention(input_ids, [intervention])

        # Compare intervened outputs
        diff = (pv_intervened - hf_intervened).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Intervened outputs diff: max={max_diff:.4f}, mean={mean_diff:.4f}")

        # The outputs should be reasonably similar
        # Note: There may be some small differences due to implementation details
        assert max_diff < 1.0, f"Outputs too different! max_diff={max_diff}"
