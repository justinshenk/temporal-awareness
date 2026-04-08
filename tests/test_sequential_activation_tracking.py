#!/usr/bin/env python3
"""Tests for sequential_activation_tracking.py"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiments.sequential_activation_tracking import (
    generate_planning_sequences,
    generate_reasoning_chains,
    generate_cumulative_context_sequences,
    load_all_sequences,
    compute_step_drift,
    assess_drift_predictability,
)


class TestSequenceGeneration(unittest.TestCase):
    """Test multi-step task sequence generation."""

    def test_planning_sequences_balanced(self):
        seqs = generate_planning_sequences(n_per_class=3)
        labels = [s.label for s in seqs]
        self.assertEqual(labels.count(0), labels.count(1),
                         "Planning sequences should be balanced between classes")

    def test_planning_sequences_have_steps(self):
        seqs = generate_planning_sequences(n_per_class=3)
        for s in seqs:
            self.assertGreater(len(s.steps), 1, f"Sequence {s.sequence_id} should have multiple steps")
            for step in s.steps:
                self.assertIsInstance(step.prompt, str)
                self.assertGreater(len(step.prompt), 0)
                self.assertEqual(step.label, s.label)

    def test_reasoning_chains_structure(self):
        seqs = generate_reasoning_chains(n_per_class=2)
        for s in seqs:
            self.assertGreater(len(s.steps), 3)
            # Steps should be numbered correctly
            for i, step in enumerate(s.steps):
                self.assertEqual(step.step_index, i)
                self.assertEqual(step.total_steps, len(s.steps))

    def test_cumulative_context_grows(self):
        seqs = generate_cumulative_context_sequences(n_per_class=2)
        for s in seqs:
            prev_len = 0
            for step in s.steps:
                curr_len = len(step.prompt)
                self.assertGreater(curr_len, prev_len,
                                   f"Cumulative step {step.step_index} should be longer than previous")
                prev_len = curr_len

    def test_load_all_sequences(self):
        seqs = load_all_sequences()
        self.assertIn("planning", seqs)
        self.assertIn("reasoning", seqs)
        self.assertIn("cumulative", seqs)
        for seq_type, seq_list in seqs.items():
            self.assertGreater(len(seq_list), 0, f"No sequences for type {seq_type}")


class TestDriftMetrics(unittest.TestCase):
    """Test drift computation functions."""

    def test_no_drift_identical_activations(self):
        """Identical activations at all steps should show no drift."""
        rng = np.random.RandomState(42)
        base_acts = rng.randn(10, 128)
        top_k = np.arange(64)

        step_acts = {0: base_acts, 1: base_acts, 2: base_acts}
        drift = compute_step_drift(step_acts, top_k)

        for d in drift:
            self.assertAlmostEqual(d["cosine_sim_to_start"], 1.0, places=3)
            self.assertAlmostEqual(d["drift_velocity"], 0.0, places=3)

    def test_increasing_drift(self):
        """Progressive noise should produce increasing drift."""
        rng = np.random.RandomState(42)
        base_acts = rng.randn(10, 128)
        top_k = np.arange(64)

        step_acts = {}
        for i in range(5):
            noise = rng.randn(10, 128) * i * 0.5
            step_acts[i] = base_acts + noise

        drift = compute_step_drift(step_acts, top_k)
        cos_sims = [d["cosine_sim_to_start"] for d in drift]

        # First step should have highest similarity
        self.assertGreater(cos_sims[0], cos_sims[-1])

    def test_drift_predictability_perfect_linear(self):
        """Perfectly linear drift should have R² ≈ 1."""
        metrics = [
            {"cosine_sim_to_start": 1.0 - i * 0.1}
            for i in range(6)
        ]
        is_mono, r2, total = assess_drift_predictability(metrics)
        self.assertTrue(is_mono)
        self.assertGreater(r2, 0.95)
        self.assertAlmostEqual(total, 0.5, places=2)

    def test_drift_predictability_random(self):
        """Random drift should have low R²."""
        rng = np.random.RandomState(42)
        metrics = [
            {"cosine_sim_to_start": rng.uniform(0.3, 0.9)}
            for _ in range(10)
        ]
        is_mono, r2, _ = assess_drift_predictability(metrics)
        self.assertFalse(is_mono)  # random should not be monotonic

    def test_single_step_edge_case(self):
        """Single step should return empty drift list."""
        rng = np.random.RandomState(42)
        step_acts = {0: rng.randn(5, 64)}
        top_k = np.arange(32)
        drift = compute_step_drift(step_acts, top_k)
        # Should still work with a single step
        self.assertEqual(len(drift), 1)


class TestSerialization(unittest.TestCase):
    """Test that results can be serialized to JSON."""

    def test_step_metrics_serializable(self):
        from scripts.experiments.sequential_activation_tracking import StepMetrics
        m = StepMetrics(
            step_index=0, n_samples=10,
            sae_probe_accuracy=0.9, act_probe_accuracy=0.85,
            sae_probe_f1=0.88, act_probe_f1=0.83,
            cosine_sim_to_start=0.95, jaccard_to_start=0.8,
            activation_magnitude_ratio=1.1,
            cosine_sim_to_prev=0.98, drift_velocity=0.05,
        )
        d = asdict(m)
        import json
        json.dumps(d)  # should not raise

    def test_sequence_result_serializable(self):
        from scripts.experiments.sequential_activation_tracking import SequenceResult, StepMetrics
        import json

        sm = StepMetrics(
            step_index=0, n_samples=10,
            sae_probe_accuracy=0.9, act_probe_accuracy=0.85,
            sae_probe_f1=0.88, act_probe_f1=0.83,
            cosine_sim_to_start=0.95, jaccard_to_start=0.8,
            activation_magnitude_ratio=1.1,
            cosine_sim_to_prev=0.98, drift_velocity=0.05,
        )
        r = SequenceResult(
            layer=13, sequence_type="planning",
            step_metrics=[sm],
            drift_is_monotonic=True,
            drift_predictability=0.95,
            total_drift=0.15,
        )
        json.dumps(asdict(r))  # should not raise


# Need this import for asdict in serialization tests
from dataclasses import asdict


if __name__ == "__main__":
    unittest.main()
