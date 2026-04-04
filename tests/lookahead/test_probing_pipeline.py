"""End-to-end pipeline tests with synthetic activations.

These tests verify that the probing and commitment curve machinery
works correctly using synthetic data with KNOWN ground truth.

We create fake activations where:
- At early positions: activations are random (no planning signal)
- At position T-k: a linear signal encoding the target appears
- At position T: the target is generated

If our probes can recover this synthetic commitment curve correctly,
the machinery is sound. If not, there's a bug.

This catches:
- Off-by-one errors in position indexing
- Scaler/probe interaction bugs
- Commitment point detection edge cases
- Baseline calibration issues
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.lookahead.utils.types import (
    PlanningExample, TaskType, ActivationCache,
    CommitmentCurve, CommitmentPoint,
)
from src.lookahead.probing.commitment_probes import (
    ProbeConfig,
    train_commitment_probes,
    compute_commitment_curves,
    find_commitment_points,
    run_shuffled_label_baseline,
    run_permutation_test,
)


def _make_synthetic_data(
    n_examples: int = 60,
    seq_len: int = 20,
    d_model: int = 64,
    n_classes: int = 3,
    commitment_position: int = 12,
    signal_strength: float = 3.0,
    seed: int = 42,
) -> tuple[list[ActivationCache], list[PlanningExample]]:
    """Create synthetic data with a known commitment point.
    
    Before commitment_position: activations are pure noise.
    At and after commitment_position: a linear signal encoding the
    class label is added to the activations.
    
    This simulates a model that "commits" to a target at a specific
    position — exactly what we want our probes to detect.
    
    Args:
        n_examples: Number of examples
        seq_len: Sequence length
        d_model: Hidden dimension
        n_classes: Number of target classes
        commitment_position: Position where signal appears
        signal_strength: SNR of the injected signal
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    
    # Create class-specific direction vectors (what the probe should find)
    class_directions = rng.randn(n_classes, d_model)
    class_directions /= np.linalg.norm(class_directions, axis=1, keepdims=True)
    
    target_names = [f"target_{i}" for i in range(n_classes)]
    
    caches = []
    examples = []
    
    for i in range(n_examples):
        label = i % n_classes
        
        # Base activations: random noise
        activations = rng.randn(seq_len, d_model).astype(np.float32)
        
        # Inject signal at and after commitment_position
        # Signal ramps up: partial at commitment, full afterward
        for pos in range(seq_len):
            if pos >= commitment_position:
                ramp = min(1.0, (pos - commitment_position + 1) / 3.0)
                activations[pos] += signal_strength * ramp * class_directions[label]
        
        cache = ActivationCache(
            example_id=f"synth_{i}",
            token_ids=list(range(seq_len)),
            token_strings=[f"tok_{j}" for j in range(seq_len)],
            activations={0: activations},  # single layer
        )
        caches.append(cache)
        
        example = PlanningExample(
            task_type=TaskType.RHYME,
            prompt=f"synthetic prompt {i}",
            target_value=target_names[label],
            target_token_positions=[seq_len - 1],
            metadata={"is_control": False},
            example_id=f"synth_{i}",
        )
        examples.append(example)
    
    return caches, examples


class TestProbeOnSyntheticData:
    """Test that probes correctly recover known commitment signals."""
    
    def test_probes_detect_signal_at_commitment_position(self):
        """Probe accuracy should jump at the commitment position."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=12, signal_strength=3.0
        )
        
        config = ProbeConfig(
            n_folds=3,
            use_cv_regularization=False,
            random_state=42,
        )
        
        probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
        
        # Before commitment: accuracy should be near chance
        chance = 1.0 / 3  # 3 classes
        pre_accs = [
            probe_results[p]["cv_accuracy_mean"]
            for p in range(0, 10)
            if p in probe_results
        ]
        assert np.mean(pre_accs) < chance + 0.15, (
            f"Pre-commitment accuracy too high: {np.mean(pre_accs):.3f} "
            f"(chance={chance:.3f}). Signal leaking?"
        )
        
        # After commitment: accuracy should be well above chance
        post_accs = [
            probe_results[p]["cv_accuracy_mean"]
            for p in range(14, 20)
            if p in probe_results
        ]
        assert np.mean(post_accs) > 0.7, (
            f"Post-commitment accuracy too low: {np.mean(post_accs):.3f}. "
            f"Probe not detecting injected signal?"
        )
    
    def test_commitment_curve_shape_is_sigmoidal(self):
        """Commitment curves should rise from low to high at the commitment point."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=10, signal_strength=4.0
        )
        
        config = ProbeConfig(n_folds=3, use_cv_regularization=False, random_state=42)
        probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
        curves = compute_commitment_curves(caches, examples, layer=0, probe_results=probe_results, config=config)
        
        assert len(curves) > 0, "Should produce commitment curves"
        
        # Check that curves rise
        for curve in curves[:5]:
            early = curve.confidences[:8].mean()
            late = curve.confidences[12:].mean()
            assert late > early + 0.1, (
                f"Curve should rise: early={early:.3f}, late={late:.3f}"
            )
    
    def test_commitment_point_detected_near_true_position(self):
        """Detected commitment point should be near the true injection point."""
        true_commitment = 10
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=true_commitment, signal_strength=4.0
        )
        
        config = ProbeConfig(
            n_folds=3, use_cv_regularization=False,
            commitment_threshold=0.6,
            stability_window=2,
            random_state=42,
        )
        probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
        curves = compute_commitment_curves(caches, examples, layer=0, probe_results=probe_results, config=config)
        points = find_commitment_points(curves, threshold=0.6, stability_window=2)
        
        valid_points = [p for p in points if p.is_valid]
        assert len(valid_points) > 0, "Should find valid commitment points"
        
        detected_positions = [p.position for p in valid_points]
        mean_detected = np.mean(detected_positions)
        
        # Should be within 3 positions of truth
        assert abs(mean_detected - true_commitment) <= 3, (
            f"Detected commitment at {mean_detected:.1f}, true is {true_commitment}"
        )
    
    def test_no_signal_produces_chance_accuracy(self):
        """With no injected signal, probes should be at chance."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=100,  # beyond seq_len → no signal
            signal_strength=0.0,
        )
        
        config = ProbeConfig(n_folds=3, use_cv_regularization=False, random_state=42)
        probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
        
        chance = 1.0 / 3
        all_accs = [r["cv_accuracy_mean"] for r in probe_results.values()]
        assert np.mean(all_accs) < chance + 0.10, (
            f"No-signal accuracy too high: {np.mean(all_accs):.3f}"
        )
    
    def test_shuffled_baseline_is_at_chance(self):
        """Shuffled-label baseline should be near chance regardless of signal."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=10, signal_strength=4.0
        )
        
        config = ProbeConfig(
            n_folds=3, use_cv_regularization=False,
            n_shuffle_iterations=30,
            random_state=42,
        )
        
        baselines = run_shuffled_label_baseline(caches, examples, layer=0, config=config)
        assert len(baselines) > 0, "Should produce baseline results"
        
        chance = 1.0 / 3
        baseline_accs = [b.metric_value for b in baselines]
        
        # Shuffled baseline should be near chance (within 10%)
        assert np.mean(baseline_accs) < chance + 0.10, (
            f"Shuffled baseline too high: {np.mean(baseline_accs):.3f}. "
            f"Probe is overfitting to activation structure, not labels."
        )
    
    def test_real_probe_significantly_beats_shuffled(self):
        """Real probe accuracy should be significantly above shuffled baseline at commitment positions."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=10, signal_strength=4.0
        )
        
        config = ProbeConfig(
            n_folds=3, use_cv_regularization=False,
            n_shuffle_iterations=30,
            random_state=42,
        )
        
        probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
        baselines = run_shuffled_label_baseline(caches, examples, layer=0, config=config)
        
        # At post-commitment positions, real accuracy should beat shuffled
        post_commit_accs = [
            probe_results[p]["cv_accuracy_mean"]
            for p in range(12, 20)
            if p in probe_results
        ]
        post_commit_baselines = [
            b.metric_value for b in baselines
            if int(b.metric_name.split("_")[-1]) >= 12
        ]
        
        real_mean = np.mean(post_commit_accs) if post_commit_accs else 0
        baseline_mean = np.mean(post_commit_baselines) if post_commit_baselines else 0.33
        
        gap = real_mean - baseline_mean
        assert gap > 0.2, (
            f"Real probe ({real_mean:.3f}) should beat shuffled ({baseline_mean:.3f}) "
            f"by >0.2. Gap is only {gap:.3f}"
        )
    
    def test_commitment_detection_with_different_strengths(self):
        """Stronger signals should produce earlier, more confident commitment."""
        results_by_strength = {}
        
        for strength in [1.0, 2.0, 4.0]:
            caches, examples = _make_synthetic_data(
                n_examples=90, commitment_position=10, signal_strength=strength
            )
            config = ProbeConfig(n_folds=3, use_cv_regularization=False, random_state=42)
            probe_results = train_commitment_probes(caches, examples, layer=0, config=config)
            
            post_accs = [
                probe_results[p]["cv_accuracy_mean"]
                for p in range(13, 20)
                if p in probe_results
            ]
            results_by_strength[strength] = np.mean(post_accs) if post_accs else 0
        
        # Stronger signal → higher accuracy
        assert results_by_strength[4.0] > results_by_strength[1.0], (
            f"Stronger signal ({results_by_strength[4.0]:.3f}) should give higher "
            f"accuracy than weaker ({results_by_strength[1.0]:.3f})"
        )


class TestCommitmentPointEdgeCases:
    """Test edge cases in commitment point detection."""
    
    def test_no_commitment_when_below_threshold(self):
        """If confidence never exceeds threshold, point should be invalid."""
        curve = CommitmentCurve(
            example_id="test",
            layer=0,
            positions=np.arange(20),
            confidences=np.full(20, 0.3),  # always below threshold
            target_value="target",
            target_position=19,
        )
        points = find_commitment_points([curve], threshold=0.8)
        assert len(points) == 1
        assert not points[0].is_valid
    
    def test_commitment_requires_stability(self):
        """A single spike above threshold shouldn't count as commitment."""
        confs = np.full(20, 0.3)
        confs[10] = 0.95  # single spike
        confs[11] = 0.3   # drops back
        
        curve = CommitmentCurve(
            example_id="test",
            layer=0,
            positions=np.arange(20),
            confidences=confs,
            target_value="target",
            target_position=19,
        )
        points = find_commitment_points([curve], threshold=0.8, stability_window=3)
        assert not points[0].is_valid, "Single spike shouldn't count as commitment"
    
    def test_commitment_with_sustained_high(self):
        """Sustained high confidence should be detected."""
        confs = np.full(20, 0.3)
        confs[10:] = 0.95  # sustained high
        
        curve = CommitmentCurve(
            example_id="test",
            layer=0,
            positions=np.arange(20),
            confidences=confs,
            target_value="target",
            target_position=19,
        )
        points = find_commitment_points([curve], threshold=0.8, stability_window=3)
        assert points[0].is_valid
        assert points[0].position == 10
    
    def test_tokens_before_target_calculation(self):
        """tokens_before_target should correctly measure distance."""
        confs = np.full(20, 0.3)
        confs[12:] = 0.95
        
        curve = CommitmentCurve(
            example_id="test",
            layer=0,
            positions=np.arange(20),
            confidences=confs,
            target_value="target",
            target_position=17,  # target at position 17
        )
        points = find_commitment_points([curve], threshold=0.8, stability_window=2)
        assert points[0].is_valid
        assert points[0].tokens_before_target == 5  # 17 - 12 = 5


class TestPermutationTest:
    """Test the permutation testing framework."""
    
    def test_significant_signal_detected(self):
        """Strong synthetic signal should be statistically significant."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=8, signal_strength=4.0
        )
        
        config = ProbeConfig(n_folds=3, use_cv_regularization=False, random_state=42)
        
        result = run_permutation_test(
            caches, examples, layer=0, position=15,
            n_permutations=100, config=config,
        )
        
        assert result["p_value"] < 0.05, (
            f"Strong signal should be significant: p={result['p_value']:.4f}"
        )
        assert result["observed_accuracy"] > result["null_distribution_mean"] + 0.1
    
    def test_no_signal_not_significant(self):
        """No-signal positions should not be significant."""
        caches, examples = _make_synthetic_data(
            n_examples=90, commitment_position=15, signal_strength=4.0
        )
        
        config = ProbeConfig(n_folds=3, use_cv_regularization=False, random_state=42)
        
        # Test at position 3 (well before commitment)
        result = run_permutation_test(
            caches, examples, layer=0, position=3,
            n_permutations=100, config=config,
        )
        
        assert result["p_value"] > 0.01, (
            f"Pre-commitment position should not be significant: p={result['p_value']:.4f}"
        )
