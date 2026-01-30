"""Tests for experiment pipeline functions with production model.

Tests activation patching, attribution patching, and steering vector computation
using Qwen2.5-1.5B. Marked slow because they require model loading.
"""

import pytest

from src.data import PreferenceData, PreferenceItem, build_prompt_pairs
from src.models import ModelRunner
from src.experiments import run_activation_patching, run_attribution_patching, compute_steering_vector
from src.viz import plot_layer_position_heatmap
from src.common.io import save_json, ensure_dir


TEST_MODEL = "Qwen/Qwen2.5-1.5B"


@pytest.fixture(scope="module")
def synthetic_pref_data():
    """Create synthetic preference data with guaranteed contrastive pairs."""
    preferences = []
    for i in range(4):
        # Short-term choice samples
        preferences.append(
            PreferenceItem(
                sample_id=i,
                time_horizon=None,
                short_term_label="A",
                long_term_label="B",
                choice="short_term",
                choice_prob=0.8,
                alt_prob=0.2,
                response="I select: A.",
                prompt_text="SITUATION: Choose.\nOPTION A: $100 now\nOPTION B: $500 later\nI select:",
            )
        )
        # Long-term choice samples
        preferences.append(
            PreferenceItem(
                sample_id=i + 10,
                time_horizon=None,
                short_term_label="A",
                long_term_label="B",
                choice="long_term",
                choice_prob=0.8,
                alt_prob=0.2,
                response="I select: B.",
                prompt_text="SITUATION: Choose.\nOPTION A: $100 now\nOPTION B: $500 later\nI select:",
            )
        )

    return PreferenceData(
        model=TEST_MODEL,
        preferences=preferences,
        dataset_id="test_synthetic",
    )


@pytest.fixture(scope="module")
def runner():
    """Shared model runner for experiment tests."""
    return ModelRunner(TEST_MODEL)


# =============================================================================
# Prompt Pair Building
# =============================================================================


@pytest.mark.slow
class TestPromptPairs:
    """Tests for building contrastive prompt pairs."""

    def test_build_prompt_pairs(self, synthetic_pref_data):
        """build_prompt_pairs creates contrastive pairs."""
        pairs = build_prompt_pairs(synthetic_pref_data, max_pairs=3)

        assert len(pairs) > 0
        assert len(pairs) <= 3

        for clean_text, corrupted_text, clean_sample, corrupted_sample in pairs:
            assert clean_sample.choice == "short_term"
            assert corrupted_sample.choice == "long_term"
            assert len(clean_text) > 0
            assert len(corrupted_text) > 0


# =============================================================================
# Activation Patching
# =============================================================================


@pytest.mark.slow
class TestActivationPatching:
    """Tests for run_activation_patching function."""

    def test_returns_expected_shapes(self, runner, synthetic_pref_data):
        """Activation patching returns arrays with expected shapes."""
        pos_sweep, full_sweeps, filtered_pos, labels, markers = run_activation_patching(
            runner,
            synthetic_pref_data,
            max_pairs=1,
            threshold=0.01,
        )

        assert pos_sweep.ndim == 1
        assert len(pos_sweep) > 0
        assert isinstance(full_sweeps, dict)
        assert len(full_sweeps) > 0
        for comp, sweep in full_sweeps.items():
            assert sweep.ndim == 2
        assert len(filtered_pos) > 0
        assert isinstance(markers, dict)

    def test_produces_visualization(self, runner, synthetic_pref_data, tmp_path):
        """Activation patching can produce visualization outputs."""
        output_dir = tmp_path / "patching"
        ensure_dir(output_dir)

        pos_sweep, full_sweeps, filtered_pos, labels, markers = run_activation_patching(
            runner,
            synthetic_pref_data,
            max_pairs=1,
            threshold=0.01,
        )

        # Save metadata
        metadata = {
            "model": runner.model_name,
            "n_pairs": 1,
            "seq_len": len(pos_sweep),
        }
        save_json(metadata, output_dir / "metadata.json")

        # Plot heatmap for first component
        first_comp = list(full_sweeps.keys())[0]
        full_sweep = full_sweeps[first_comp]
        plot_layer_position_heatmap(
            full_sweep,
            list(range(full_sweep.shape[0])),
            labels[:full_sweep.shape[1]] if len(labels) >= full_sweep.shape[1] else labels,
            save_path=output_dir / "heatmap.png",
            title="Test Patching",
        )

        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "heatmap.png").exists()


# =============================================================================
# Attribution Patching (EAP)
# =============================================================================


@pytest.mark.slow
class TestAttributionPatching:
    """Tests for run_attribution_patching function (EAP methods)."""

    def test_returns_expected_keys(self, runner, synthetic_pref_data):
        """Returns dict with expected attribution methods."""
        results, labels, markers = run_attribution_patching(
            runner,
            synthetic_pref_data,
            max_pairs=1,
            ig_steps=2,  # Few steps for speed
        )

        # Should have standard attribution methods
        expected_keys = {
            "resid",
            "attn",
            "mlp",
            "eap_attn",
            "eap_mlp",
            "eap_ig_attn",
            "eap_ig_mlp",
        }
        assert expected_keys.issubset(set(results.keys()))

        # Each result is 2D (layers x positions)
        for key, arr in results.items():
            assert arr.ndim == 2
            assert arr.shape[0] == runner.n_layers


# =============================================================================
# Steering Vector Computation
# =============================================================================


@pytest.mark.slow
class TestSteeringVectorComputation:
    """Tests for compute_steering_vector function."""

    def test_returns_direction_and_stats(self, runner, synthetic_pref_data):
        """Returns direction vector and stats dict."""
        direction, stats = compute_steering_vector(
            runner,
            synthetic_pref_data,
            layer=5,
            position=10,
            max_samples=4,
        )

        assert direction.shape == (runner.d_model,)
        assert "layer" in stats
        assert "position" in stats
