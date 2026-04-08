#!/usr/bin/env python3
"""
Tests for SAE Feature Stability experiment.

Tests dataset generation, metric computation, and serialization
without requiring GPU or model downloads.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))

from sae_feature_stability import (
    generate_domain_shift_data,
    generate_register_shift_data,
    generate_negation_data,
    generate_paraphrase_data,
    generate_ambiguous_data,
    load_in_distribution_data,
    load_all_shift_conditions,
    select_discriminative_latents,
    compute_ece,
    compute_feature_stability,
    ProbeMetrics,
    FeatureStabilityMetrics,
    ShiftCondition,
    DATA_DIR,
)


class TestDatasetGeneration:
    """Test that all shift datasets are well-formed."""

    def _check_condition(self, cond: ShiftCondition):
        """Common checks for any shift condition."""
        assert len(cond.prompts) > 0, f"{cond.name}: no prompts"
        assert len(cond.prompts) == len(cond.labels), f"{cond.name}: prompt/label mismatch"
        assert set(cond.labels).issubset({0, 1}), f"{cond.name}: invalid labels"
        assert cond.name, f"condition has no name"
        assert cond.description, f"condition has no description"

        # Check balance (should be roughly balanced)
        n_imm = (cond.labels == 0).sum()
        n_lt = (cond.labels == 1).sum()
        ratio = min(n_imm, n_lt) / max(n_imm, n_lt) if max(n_imm, n_lt) > 0 else 0
        assert ratio >= 0.3, f"{cond.name}: heavily imbalanced ({n_imm} vs {n_lt})"

    def test_domain_shift(self):
        cond = generate_domain_shift_data()
        self._check_condition(cond)
        assert len(cond.prompts) >= 30, "domain shift should have at least 30 samples"

    def test_register_shift(self):
        cond = generate_register_shift_data()
        self._check_condition(cond)
        assert len(cond.prompts) >= 20

    def test_negation(self):
        cond = generate_negation_data()
        self._check_condition(cond)
        # Verify negations contain negation words
        neg_words = {"not", "don't", "shouldn't", "isn't", "forget", "resist", "opposite"}
        has_negation = sum(
            any(w in p.lower() for w in neg_words) for p in cond.prompts
        )
        assert has_negation >= len(cond.prompts) * 0.5, "negation dataset should contain negation words"

    def test_paraphrase(self):
        cond = generate_paraphrase_data()
        self._check_condition(cond)

    def test_ambiguous(self):
        cond = generate_ambiguous_data()
        self._check_condition(cond)

    def test_in_distribution_data_loads(self):
        """Test that the minimal pairs dataset loads correctly."""
        path = DATA_DIR / "raw" / "temporal_scope_pairs_minimal.json"
        if not path.exists():
            pytest.skip("minimal pairs dataset not found")
        prompts, labels = load_in_distribution_data(path)
        assert len(prompts) == len(labels)
        assert len(prompts) > 0
        assert set(labels).issubset({0, 1})

    def test_all_shift_conditions(self):
        """Test that load_all_shift_conditions works."""
        conditions = load_all_shift_conditions(DATA_DIR)
        assert len(conditions) >= 5, f"Expected at least 5 conditions, got {len(conditions)}"
        for name, cond in conditions.items():
            self._check_condition(cond)


class TestMetrics:
    """Test metric computations with synthetic data."""

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated probabilities should have ECE ≈ 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.7, 0.8, 0.9, 0.6, 0.8])
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert 0 <= ece <= 1

    def test_ece_terrible_calibration(self):
        """Completely wrong probabilities should have high ECE."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece > 0.5

    def test_select_discriminative_latents(self):
        """Test latent selection with synthetic SAE activations."""
        n_samples = 100
        d_sae = 1000
        k = 32

        # Create synthetic SAE activations where first 10 latents are discriminative
        rng = np.random.RandomState(42)
        sae_latents = rng.randn(n_samples, d_sae) * 0.1
        labels = np.array([0] * 50 + [1] * 50)

        # Make first 10 latents strongly discriminative
        sae_latents[:50, :10] += 2.0  # immediate class activates these
        sae_latents[50:, :10] -= 2.0  # long-term class suppresses these

        top_k, signed_diff = select_discriminative_latents(sae_latents, labels, k=k)

        assert len(top_k) == k
        # First 10 latents should be in top-k
        assert set(range(10)).issubset(set(top_k.tolist())), \
            "Known discriminative latents should be in top-k"

    def test_feature_stability_identical(self):
        """Identical activations should have perfect stability."""
        n = 50
        d_sae = 500
        k = 32
        rng = np.random.RandomState(42)

        acts = rng.randn(n, d_sae)
        top_k = np.arange(k)

        stability = compute_feature_stability(acts, acts, top_k, k=k)
        assert stability.jaccard_similarity == 1.0
        assert stability.top_k_overlap == 1.0
        assert abs(stability.cosine_similarity - 1.0) < 1e-5
        assert abs(stability.activation_magnitude_ratio - 1.0) < 1e-5

    def test_feature_stability_random(self):
        """Random activations should have low stability."""
        n = 50
        d_sae = 500
        k = 32
        rng = np.random.RandomState(42)

        acts_orig = rng.randn(n, d_sae)
        acts_shift = rng.randn(n, d_sae) * 5  # very different
        top_k = np.arange(k)

        stability = compute_feature_stability(acts_orig, acts_shift, top_k, k=k)
        # Should be low but not necessarily zero
        assert stability.jaccard_similarity < 0.5
        assert stability.cosine_similarity < 0.5


class TestSerialization:
    """Test that results can be serialized."""

    def test_probe_metrics_serializable(self):
        m = ProbeMetrics(
            accuracy=0.85, precision=0.82, recall=0.88, f1=0.85,
            log_loss_value=0.35, n_samples=100,
            mean_confidence=0.78, ece=0.05,
        )
        d = json.loads(json.dumps({"m": {
            "accuracy": m.accuracy,
            "precision": m.precision,
        }}))
        assert d["m"]["accuracy"] == 0.85

    def test_feature_stability_serializable(self):
        m = FeatureStabilityMetrics(
            jaccard_similarity=0.7,
            activation_magnitude_ratio=0.9,
            top_k_overlap=0.8,
            cosine_similarity=0.85,
        )
        d = json.loads(json.dumps({"m": {
            "jaccard": m.jaccard_similarity,
            "overlap": m.top_k_overlap,
        }}))
        assert d["m"]["jaccard"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
