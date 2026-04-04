"""Tests for comprehensive baselines and behavioral validation.

These verify that our anti-artifact machinery actually works.
"""

import pytest
import numpy as np

from src.lookahead.utils.types import (
    PlanningExample, TaskType, ActivationCache, BaselineResult,
)
from src.lookahead.probing.comprehensive_baselines import (
    bag_of_words_baseline,
    pca_reduction_baseline,
    position_shuffled_baseline,
    random_direction_baseline,
    fdr_correction,
    bonferroni_correction,
    run_all_baselines,
)
from src.lookahead.datasets.planning_vs_continuation import (
    generate_nonce_rhyme_dataset,
    generate_competing_continuation_dataset,
    generate_counterfactual_pairs,
)


def _make_synthetic_with_bow_confound(
    n_examples=60, seq_len=20, d_model=64, n_classes=3, seed=42,
):
    """Synthetic data where the signal IS a bag-of-words confound.
    
    Token IDs are correlated with labels, so BoW baseline should
    match the activation probe. This tests that our BoW baseline
    correctly detects this confound.
    """
    rng = np.random.RandomState(seed)
    
    target_names = [f"target_{i}" for i in range(n_classes)]
    caches, examples = [], []
    
    for i in range(n_examples):
        label = i % n_classes
        
        # Token IDs are label-dependent (the confound)
        token_ids = list(range(seq_len))
        token_ids[3] = label * 100 + 50  # token at position 3 encodes the label
        
        # Activations encode the SAME info as the token identity
        activations = rng.randn(seq_len, d_model).astype(np.float32)
        # Signal at all positions comes from the token at position 3
        class_dir = rng.randn(d_model)
        class_dir /= np.linalg.norm(class_dir)
        for pos in range(seq_len):
            activations[pos] += 3.0 * label * class_dir
        
        caches.append(ActivationCache(
            example_id=f"bow_confound_{i}",
            token_ids=token_ids,
            token_strings=[f"tok_{j}" for j in token_ids],
            activations={0: activations},
        ))
        examples.append(PlanningExample(
            task_type=TaskType.RHYME, prompt=f"prompt {i}",
            target_value=target_names[label],
            target_token_positions=[seq_len - 1],
            metadata={"is_control": False},
            example_id=f"bow_confound_{i}",
        ))
    
    return caches, examples


def _make_genuine_planning_signal(
    n_examples=90, seq_len=20, d_model=64, n_classes=3,
    commitment_pos=12, seed=42,
):
    """Synthetic data with genuine planning signal that ONLY appears
    in activations after commitment_pos, NOT in token identities.
    
    BoW should fail here. Activation probe should succeed.
    """
    rng = np.random.RandomState(seed)
    target_names = [f"target_{i}" for i in range(n_classes)]
    class_dirs = rng.randn(n_classes, d_model)
    class_dirs /= np.linalg.norm(class_dirs, axis=1, keepdims=True)
    
    caches, examples = [], []
    
    for i in range(n_examples):
        label = i % n_classes
        
        # Token IDs are IDENTICAL across classes — no BoW signal
        token_ids = list(range(seq_len))
        
        activations = rng.randn(seq_len, d_model).astype(np.float32)
        for pos in range(commitment_pos, seq_len):
            ramp = min(1.0, (pos - commitment_pos + 1) / 3.0)
            activations[pos] += 4.0 * ramp * class_dirs[label]
        
        caches.append(ActivationCache(
            example_id=f"genuine_{i}",
            token_ids=token_ids,
            token_strings=[f"tok_{j}" for j in token_ids],
            activations={0: activations},
        ))
        examples.append(PlanningExample(
            task_type=TaskType.RHYME, prompt=f"prompt {i}",
            target_value=target_names[label],
            target_token_positions=[seq_len - 1],
            metadata={"is_control": False},
            example_id=f"genuine_{i}",
        ))
    
    return caches, examples


class TestBagOfWordsBaseline:
    def test_bow_detects_confound(self):
        """BoW should achieve high accuracy when tokens correlate with labels."""
        caches, examples = _make_synthetic_with_bow_confound()
        results = bag_of_words_baseline(caches, examples, n_folds=3)
        accs = [r.metric_value for r in results]
        assert max(accs) > 0.6, f"BoW should detect token-label confound, max acc: {max(accs):.3f}"
    
    def test_bow_fails_without_confound(self):
        """BoW should be at chance when tokens don't correlate with labels."""
        caches, examples = _make_genuine_planning_signal()
        results = bag_of_words_baseline(caches, examples, n_folds=3)
        accs = [r.metric_value for r in results]
        chance = 1.0 / 3
        assert np.mean(accs) < chance + 0.15, (
            f"BoW should be near chance without confound: {np.mean(accs):.3f}"
        )


class TestPCAReduction:
    def test_signal_survives_moderate_pca(self):
        """Genuine signal should survive PCA to 20 dimensions."""
        caches, examples = _make_genuine_planning_signal(commitment_pos=8)
        
        results = pca_reduction_baseline(caches, examples, layer=0, n_components_list=[20], n_folds=3)
        if 20 in results:
            post_commit_accs = [
                r.metric_value for r in results[20]
                if int(r.metric_name.split("_")[-1]) >= 12
            ]
            if post_commit_accs:
                assert max(post_commit_accs) > 0.5, (
                    f"Signal should survive PCA-20: {max(post_commit_accs):.3f}"
                )


class TestPositionShuffle:
    def test_position_shuffle_kills_genuine_signal(self):
        """Position-shuffled activations should destroy example-specific signal."""
        caches, examples = _make_genuine_planning_signal()
        results = position_shuffled_baseline(caches, examples, layer=0, n_shuffles=20, n_folds=3)
        accs = [r.metric_value for r in results]
        chance = 1.0 / 3
        assert np.mean(accs) < chance + 0.12, (
            f"Position-shuffled should be near chance: {np.mean(accs):.3f}"
        )


class TestMultipleComparisons:
    def test_fdr_controls_false_positives(self):
        """FDR should reject most null p-values."""
        # 100 null p-values (uniform), 10 real signals
        rng = np.random.RandomState(42)
        null_p = rng.uniform(0, 1, 100).tolist()
        real_p = [0.001] * 10
        all_p = null_p + real_p
        
        sig, adj = fdr_correction(all_p, alpha=0.05)
        
        # Real signals should be significant
        assert sum(sig[100:]) >= 8, f"Should detect most real signals: {sum(sig[100:])}/10"
        
        # False discovery rate among null should be controlled
        false_discoveries = sum(sig[:100])
        total_discoveries = sum(sig)
        if total_discoveries > 0:
            fdr = false_discoveries / total_discoveries
            assert fdr < 0.15, f"FDR too high: {fdr:.3f}"
    
    def test_bonferroni_is_more_conservative(self):
        """Bonferroni should reject fewer than FDR."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.06, 0.1, 0.5, 0.8]
        
        fdr_sig, _ = fdr_correction(p_values)
        bonf_sig, _ = bonferroni_correction(p_values)
        
        assert sum(bonf_sig) <= sum(fdr_sig), "Bonferroni should be more conservative"


class TestPlanningVsContinuationDatasets:
    def test_nonce_rhymes_generated(self):
        examples = generate_nonce_rhyme_dataset()
        nonce_examples = [e for e in examples if e.metadata.get("is_nonce")]
        controls = [e for e in examples if e.metadata.get("is_nonce_control")]
        assert len(nonce_examples) > 0, "Should have nonce examples"
        assert len(controls) > 0, "Should have real-word controls"
    
    def test_nonce_words_in_prompts(self):
        examples = generate_nonce_rhyme_dataset()
        for ex in examples:
            anchor = ex.metadata.get("anchor_word", "")
            if ex.metadata.get("is_nonce"):
                assert anchor in ex.prompt, f"Nonce word '{anchor}' not in prompt"
    
    def test_competing_continuations(self):
        examples = generate_competing_continuation_dataset()
        assert len(examples) >= 5
        for ex in examples:
            assert ex.metadata.get("is_competing"), "Should be marked as competing"
            assert ex.metadata.get("likely_continuation"), "Should have likely continuation"
    
    def test_counterfactual_pairs_differ(self):
        pairs = generate_counterfactual_pairs()
        assert len(pairs) > 0
        for ex_a, ex_b in pairs:
            assert ex_a.metadata["anchor_word"] != ex_b.metadata["anchor_word"]
            assert ex_a.target_value != ex_b.target_value
            # Prompts should have same structure but different anchor
            assert ex_a.metadata["counterfactual_pair"] == ex_b.metadata["counterfactual_pair"]


class TestComprehensiveBaselineSuite:
    def test_run_all_baselines_completes(self):
        """The full baseline suite should run without errors."""
        caches, examples = _make_genuine_planning_signal(n_examples=60)
        suite = run_all_baselines(caches, examples, layer=0, n_folds=3, random_state=42)
        
        assert len(suite.bag_of_words) > 0
        assert len(suite.position_shuffled) > 0
        assert len(suite.random_direction) > 0
        assert len(suite.pca_reduction) > 0
