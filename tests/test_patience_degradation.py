#!/usr/bin/env python3
"""Tests for patience_degradation.py"""

import sys
import json
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiments.patience_degradation import (
    generate_repetitive_sequences,
    _repeat_task_prefix,
    compute_neuron_concentration,
    compute_feature_entropy,
    classify_representation,
    find_degradation_onset,
    find_activation_drift_onset,
    compute_precursor_gap,
    RepetitionMetrics,
    DegradationResult,
    REPETITION_COUNTS,
    QUICK_REPETITION_COUNTS,
)


class TestRepetitiveSequenceGeneration(unittest.TestCase):
    """Test repetitive task sequence generation."""

    def test_all_domains_present(self):
        seqs = generate_repetitive_sequences([1, 5, 10])
        domains = set(seqs.keys())
        self.assertIn("scheduling", domains)
        self.assertIn("writing", domains)
        self.assertIn("analysis", domains)

    def test_all_stakes_present(self):
        seqs = generate_repetitive_sequences([1, 5])
        for domain, stakes_dict in seqs.items():
            self.assertIn("low", stakes_dict, f"Missing 'low' stakes for {domain}")
            self.assertIn("high", stakes_dict, f"Missing 'high' stakes for {domain}")

    def test_repetition_levels_correct(self):
        rep_counts = [1, 5, 10]
        seqs = generate_repetitive_sequences(rep_counts)
        for domain in seqs:
            for stakes in seqs[domain]:
                levels = seqs[domain][stakes]
                actual_reps = [l.repetition_count for l in levels]
                self.assertEqual(actual_reps, rep_counts)

    def test_balanced_labels(self):
        seqs = generate_repetitive_sequences([1])
        for domain in seqs:
            for stakes in seqs[domain]:
                for level in seqs[domain][stakes]:
                    n_imm = (level.labels == 0).sum()
                    n_lt = (level.labels == 1).sum()
                    self.assertEqual(n_imm, n_lt,
                                     f"Labels not balanced for {domain}/{stakes} rep={level.repetition_count}")

    def test_repeat_prefix_grows(self):
        base = "Do the task."
        p1 = _repeat_task_prefix(base, 1)
        p5 = _repeat_task_prefix(base, 5)
        p10 = _repeat_task_prefix(base, 10)
        self.assertEqual(p1, base)
        self.assertGreater(len(p5), len(p1))
        self.assertGreater(len(p10), len(p5))
        self.assertTrue(p5.endswith(base))
        self.assertTrue(p10.endswith(base))

    def test_prompts_match_samples(self):
        seqs = generate_repetitive_sequences([1, 3])
        for domain in seqs:
            for stakes in seqs[domain]:
                for level in seqs[domain][stakes]:
                    self.assertEqual(len(level.prompts), len(level.samples))
                    self.assertEqual(len(level.labels), len(level.samples))


class TestNeuronAnalysis(unittest.TestCase):
    """Test neuron-level vs. distributed analysis."""

    def test_concentrated_signal(self):
        """Signal concentrated in few neurons should give high concentration."""
        rng = np.random.RandomState(42)
        acts = rng.randn(20, 200) * 0.01  # low baseline
        acts[:, :5] = rng.randn(20, 5) * 10  # 5 strong neurons
        conc = compute_neuron_concentration(acts, top_k=10)
        self.assertGreater(conc, 0.5, "Concentrated signal should have high concentration")

    def test_distributed_signal(self):
        """Evenly distributed signal should give low concentration."""
        rng = np.random.RandomState(42)
        acts = rng.randn(20, 200)  # all neurons roughly equal
        conc = compute_neuron_concentration(acts, top_k=10)
        self.assertLess(conc, 0.2, "Distributed signal should have low concentration")

    def test_entropy_increases_with_distribution(self):
        """More spread-out activations should have higher entropy."""
        rng = np.random.RandomState(42)
        # Concentrated
        conc_acts = np.zeros((10, 100))
        conc_acts[:, 0] = 1.0
        e_conc = compute_feature_entropy(conc_acts)

        # Distributed
        dist_acts = np.ones((10, 100)) * 0.01
        e_dist = compute_feature_entropy(dist_acts)

        self.assertGreater(e_dist, e_conc)

    def test_classify_neuron_level(self):
        result = classify_representation([0.5, 0.4, 0.6], [3.0, 2.8, 3.2])
        self.assertEqual(result, "neuron-level")

    def test_classify_distributed(self):
        result = classify_representation([0.05, 0.08, 0.06], [8.0, 7.5, 8.2])
        self.assertEqual(result, "distributed")

    def test_classify_mixed(self):
        result = classify_representation([0.2, 0.18, 0.22], [5.0, 5.5, 5.2])
        self.assertEqual(result, "mixed")


class TestDegradationDetection(unittest.TestCase):
    """Test leading indicator and degradation onset detection."""

    def test_degradation_onset_clear(self):
        reps = [1, 5, 10, 15, 20]
        accs = [0.95, 0.93, 0.88, 0.82, 0.75]
        onset = find_degradation_onset(reps, accs, threshold=0.05)
        self.assertEqual(onset, 10, "Should detect onset at rep=10 (0.95-0.88 > 0.05)")

    def test_no_degradation(self):
        reps = [1, 5, 10, 15, 20]
        accs = [0.95, 0.94, 0.93, 0.93, 0.92]
        onset = find_degradation_onset(reps, accs, threshold=0.05)
        self.assertEqual(onset, 20, "No degradation should return last rep")

    def test_activation_drift_onset(self):
        reps = [1, 5, 10, 15, 20]
        cos_sims = [1.0, 0.98, 0.93, 0.88, 0.80]
        onset = find_activation_drift_onset(reps, cos_sims, threshold=0.05)
        self.assertEqual(onset, 10, "Should detect drift onset at rep=10 (1-0.93 > 0.05)")

    def test_positive_precursor_gap(self):
        gap = compute_precursor_gap(behavioral_onset=15, activation_onset=10)
        self.assertEqual(gap, 5, "Gap should be 5 (activations changed 5 reps earlier)")
        self.assertGreater(gap, 0, "Positive gap means activations are leading indicator")

    def test_negative_precursor_gap(self):
        gap = compute_precursor_gap(behavioral_onset=10, activation_onset=15)
        self.assertEqual(gap, -5, "Negative gap means behavior degrades before activations shift")


class TestSerialization(unittest.TestCase):
    """Test JSON serialization of result types."""

    def test_repetition_metrics_serializable(self):
        m = RepetitionMetrics(
            repetition_count=5, n_samples=12,
            sae_probe_accuracy=0.9, act_probe_accuracy=0.85,
            sae_probe_f1=0.88, act_probe_f1=0.83,
            cosine_sim_to_baseline=0.95, jaccard_to_baseline=0.8,
            magnitude_ratio=1.05,
            top_neuron_concentration=0.25, feature_entropy=6.5,
            mean_confidence=0.87, confidence_std=0.12,
        )
        json.dumps(asdict(m))

    def test_degradation_result_serializable(self):
        m = RepetitionMetrics(
            repetition_count=1, n_samples=12,
            sae_probe_accuracy=0.9, act_probe_accuracy=0.85,
            sae_probe_f1=0.88, act_probe_f1=0.83,
            cosine_sim_to_baseline=1.0, jaccard_to_baseline=1.0,
            magnitude_ratio=1.0,
            top_neuron_concentration=0.25, feature_entropy=6.5,
            mean_confidence=0.87, confidence_std=0.12,
        )
        r = DegradationResult(
            layer=13, domain="scheduling", stakes="high",
            repetition_metrics=[m],
            degradation_onset_rep=15,
            behavioral_precursor_gap=5,
            is_domain_general=True,
            neuron_vs_distributed="distributed",
        )
        json.dumps(asdict(r))


if __name__ == "__main__":
    unittest.main()
