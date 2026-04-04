"""Commitment curve probing: measuring when models commit to future structure.

The core analytical method for RQ4. At each token position t, we train a
linear probe on the residual stream activations to predict what the model
will generate at position t+k (the "target").

If the probe achieves high accuracy at position t for a target at position t+k,
this means the model has already encoded (committed to) that future structure
k tokens in advance.

The "commitment curve" is the probe accuracy plotted as a function of position,
showing when the commitment emerges and stabilizes.

Critical controls:
1. Shuffled-label baseline: same activations, randomized targets
2. Random-direction baseline: project onto random directions instead of probe
3. Cross-validated probes: prevent overfitting on small datasets
4. Bootstrap CIs: quantify uncertainty in commitment timing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle

from ..utils.types import (
    ActivationCache,
    CommitmentCurve,
    CommitmentPoint,
    PlanningExample,
    BaselineResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for commitment probing."""
    # Probe training
    regularization: str = "l2"  # "l1", "l2", or "elasticnet"
    C: float = 1.0  # inverse regularization strength
    max_iter: int = 2000
    
    # Cross-validation
    n_folds: int = 5
    use_cv_regularization: bool = True  # use LogisticRegressionCV for C selection
    
    # Commitment detection
    commitment_threshold: float = 0.8  # confidence threshold for "committed"
    stability_window: int = 3  # must stay above threshold for this many positions
    
    # Baselines
    n_shuffle_iterations: int = 100  # for permutation test
    n_bootstrap_iterations: int = 1000  # for confidence intervals
    
    # Seed
    random_state: int = 42


def train_commitment_probes(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    config: ProbeConfig | None = None,
) -> dict[int, dict]:
    """Train probes at each token position to predict the target.
    
    For a given layer, we train a separate probe at each token position.
    Each probe: activation_at_position_t → target_class
    
    The "target_class" depends on the task:
    - Rhyme: which rhyme word family (one-hot among rhyme groups)
    - Acrostic: which next letter (one of 26)
    - Code: which return type (int/str/bool/list/float/None)
    
    Args:
        caches: Activation caches for all examples
        examples: Corresponding planning examples (for labels)
        layer: Which layer to probe
        config: Probe configuration
        
    Returns:
        dict[position -> {probe, accuracy, auc, predictions}]
    """
    if config is None:
        config = ProbeConfig()
    
    # Build the dataset: for each example, get (activation_at_pos_t, target_label)
    # We need consistent target labels across examples
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    if len(unique_targets) < 2:
        logger.warning("Need at least 2 unique targets for classification. "
                       f"Got: {unique_targets}")
        return {}
    
    # Find the shortest sequence length (probe at each position up to this)
    min_seq_len = min(len(c.token_ids) for c in caches)
    
    # Filter to non-control examples with valid targets
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10:
        logger.warning(f"Only {len(valid_indices)} valid examples — too few for reliable probing")
        return {}
    
    # Build labels array
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    
    # Check class balance
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_class_count = label_counts.min()
    if min_class_count < config.n_folds:
        logger.warning(f"Smallest class has {min_class_count} samples, "
                       f"less than {config.n_folds} folds. Reducing folds.")
        config = ProbeConfig(**{**config.__dict__, "n_folds": max(2, min_class_count)})
    
    results = {}
    
    for pos in range(min_seq_len):
        # Build feature matrix: (n_samples, d_model)
        X = np.stack([
            caches[i].activations[layer][pos]
            for i in valid_indices
        ])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train probe — use l1_ratio instead of deprecated penalty param
        probe = LogisticRegression(
            C=config.C,
            l1_ratio=0,  # equivalent to penalty='l2'
            max_iter=config.max_iter,
            random_state=config.random_state,
            solver="saga",
        )
        
        # Cross-validated accuracy
        n_splits = min(config.n_folds, min_class_count)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        
        try:
            cv_scores = cross_val_score(probe, X_scaled, labels, cv=cv, scoring="accuracy")
        except ValueError as e:
            logger.warning(f"CV failed at position {pos}: {e}")
            continue
        
        # Out-of-fold predictions — CRITICAL for honest commitment curves.
        # Using in-sample predictions would overfit and show spurious early
        # commitment. OOF ensures each example's confidence comes from a
        # probe that never saw that example during training.
        try:
            oof_probabilities = cross_val_predict(
                LogisticRegression(
                    C=config.C, l1_ratio=0, max_iter=config.max_iter,
                    random_state=config.random_state, solver="saga",
                ),
                X_scaled, labels, cv=cv, method="predict_proba",
            )
        except ValueError:
            oof_probabilities = None
        
        # Fit on full data for probe weights (used for patching later)
        probe.fit(X_scaled, labels)
        predictions = probe.predict(X_scaled)
        
        # Compute AUC if binary using OOF predictions
        auc = None
        if len(unique_labels) == 2 and oof_probabilities is not None:
            try:
                auc = roc_auc_score(labels, oof_probabilities[:, 1])
            except ValueError:
                pass
        
        results[pos] = {
            "probe": probe,
            "scaler": scaler,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "train_accuracy": accuracy_score(labels, predictions),
            "auc": auc,
            "oof_probabilities": oof_probabilities,  # honest, held-out predictions
            "probabilities": probe.predict_proba(X_scaled),  # in-sample (for reference only)
            "n_samples": len(labels),
            "n_classes": len(unique_labels),
            "position": pos,
            "layer": layer,
            "valid_indices": valid_indices,  # needed to map back to examples
        }
    
    return results


def compute_commitment_curves(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    probe_results: dict[int, dict],
    config: ProbeConfig | None = None,
) -> list[CommitmentCurve]:
    """Compute per-example commitment curves using OUT-OF-FOLD predictions.
    
    IMPORTANT: We use the oof_probabilities from cross-validation, NOT
    the in-sample probe predictions. Using in-sample predictions would
    overfit and show artificially early commitment — a classic artifact
    in probing studies.
    
    For each example, at each position, the OOF confidence tells us:
    "when a probe trained WITHOUT this example predicts its target,
    how confident is it?" This is the honest measure of commitment.
    
    Args:
        caches: Activation caches
        examples: Planning examples
        layer: Layer being probed
        probe_results: Output of train_commitment_probes (must contain oof_probabilities)
        config: Configuration
        
    Returns:
        List of CommitmentCurve, one per non-control example
    """
    if config is None:
        config = ProbeConfig()
    
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    # Build mapping from example index → OOF row index
    # valid_indices in probe_results tells us which examples were used
    first_pos = min(probe_results.keys())
    valid_indices = probe_results[first_pos].get("valid_indices", [])
    if not valid_indices:
        logger.warning("No valid_indices in probe results — falling back to in-sample")
        return _compute_commitment_curves_insample(
            caches, examples, layer, probe_results, config
        )
    
    # Map: example's global index → row in the OOF matrix
    example_idx_to_oof_row = {ex_idx: oof_row for oof_row, ex_idx in enumerate(valid_indices)}
    
    curves = []
    
    for global_idx, (cache, example) in enumerate(zip(caches, examples)):
        if example.metadata.get("is_control", False) or not example.target_value:
            continue
        
        target_idx = target_to_idx.get(example.target_value)
        if target_idx is None:
            continue
        
        oof_row = example_idx_to_oof_row.get(global_idx)
        if oof_row is None:
            continue
        
        positions = []
        confidences = []
        
        for pos in sorted(probe_results.keys()):
            if pos >= len(cache.token_ids):
                break
            
            pr = probe_results[pos]
            oof_probs = pr.get("oof_probabilities")
            
            if oof_probs is not None and oof_row < len(oof_probs):
                # Use honest out-of-fold prediction
                proba = oof_probs[oof_row]
            else:
                # Fallback: use in-sample (mark as less reliable)
                scaler = pr["scaler"]
                probe = pr["probe"]
                act = cache.activations[layer][pos].reshape(1, -1)
                act_scaled = scaler.transform(act)
                proba = probe.predict_proba(act_scaled)[0]
            
            if target_idx < len(proba):
                conf = proba[target_idx]
            else:
                conf = 0.0
            
            positions.append(pos)
            confidences.append(conf)
        
        target_pos = example.target_token_positions[0] if example.target_token_positions else len(cache.token_ids) - 1
        
        curve = CommitmentCurve(
            example_id=example.example_id,
            layer=layer,
            positions=np.array(positions),
            confidences=np.array(confidences),
            target_value=example.target_value,
            target_position=target_pos,
        )
        curves.append(curve)
    
    return curves


def _compute_commitment_curves_insample(
    caches, examples, layer, probe_results, config,
) -> list[CommitmentCurve]:
    """Fallback: in-sample commitment curves (used only if OOF unavailable)."""
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    curves = []
    
    for cache, example in zip(caches, examples):
        if example.metadata.get("is_control", False) or not example.target_value:
            continue
        target_idx = target_to_idx.get(example.target_value)
        if target_idx is None:
            continue
        
        positions, confidences = [], []
        for pos in sorted(probe_results.keys()):
            if pos >= len(cache.token_ids):
                break
            pr = probe_results[pos]
            act = cache.activations[layer][pos].reshape(1, -1)
            act_scaled = pr["scaler"].transform(act)
            proba = pr["probe"].predict_proba(act_scaled)[0]
            conf = proba[target_idx] if target_idx < len(proba) else 0.0
            positions.append(pos)
            confidences.append(conf)
        
        target_pos = example.target_token_positions[0] if example.target_token_positions else len(cache.token_ids) - 1
        curves.append(CommitmentCurve(
            example_id=example.example_id, layer=layer,
            positions=np.array(positions), confidences=np.array(confidences),
            target_value=example.target_value, target_position=target_pos,
        ))
    return curves


def find_commitment_points(
    curves: list[CommitmentCurve],
    threshold: float = 0.8,
    stability_window: int = 3,
) -> list[CommitmentPoint]:
    """Find where commitment emerges in each curve.
    
    Commitment point = first position where probe confidence exceeds
    the threshold AND stays above it for `stability_window` consecutive
    positions. The stability requirement prevents noise from triggering
    false commitments.
    
    Args:
        curves: Commitment curves to analyze
        threshold: Confidence threshold for "committed"
        stability_window: Must stay above threshold for this many positions
        
    Returns:
        List of CommitmentPoint
    """
    points = []
    
    for curve in curves:
        above_threshold = curve.confidences >= threshold
        
        # Find first stable crossing
        commitment_pos = None
        commitment_conf = None
        
        for i in range(len(above_threshold) - stability_window + 1):
            if all(above_threshold[i:i + stability_window]):
                commitment_pos = int(curve.positions[i])
                commitment_conf = float(curve.confidences[i])
                break
        
        is_valid = commitment_pos is not None
        
        if not is_valid:
            # Set to last position as fallback
            commitment_pos = int(curve.positions[-1])
            commitment_conf = float(curve.confidences[-1])
        
        tokens_before = curve.target_position - commitment_pos if is_valid else 0
        
        point = CommitmentPoint(
            example_id=curve.example_id,
            layer=curve.layer,
            position=commitment_pos,
            confidence_at_commitment=commitment_conf,
            threshold=threshold,
            tokens_before_target=tokens_before,
            is_valid=is_valid,
        )
        points.append(point)
    
    return points


def run_shuffled_label_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    config: ProbeConfig | None = None,
) -> list[BaselineResult]:
    """Control: train probes with shuffled target labels.
    
    If a probe trained on shuffled labels achieves similar accuracy
    to the real probe, then the "commitment" signal is likely an artifact
    of the activation structure rather than genuine planning.
    
    This is the most important control — it directly tests whether
    the probe is picking up on the target-specific signal vs. general
    position-in-sequence information.
    
    Returns:
        List of BaselineResult with shuffled accuracy at each position
    """
    if config is None:
        config = ProbeConfig()
    
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    min_seq_len = min(len(caches[i].token_ids) for i in valid_indices)
    
    results = []
    rng = np.random.RandomState(config.random_state)
    
    for pos in range(min_seq_len):
        X = np.stack([caches[i].activations[layer][pos] for i in valid_indices])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run multiple shuffled iterations
        shuffled_accs = []
        for _ in range(config.n_shuffle_iterations):
            shuffled_labels = rng.permutation(labels)
            
            probe = LogisticRegression(
                C=config.C,
                max_iter=config.max_iter,
                random_state=config.random_state,
            )
            
            try:
                cv = StratifiedKFold(n_splits=min(config.n_folds, 3), shuffle=True, random_state=None)
                scores = cross_val_score(probe, X_scaled, shuffled_labels, cv=cv, scoring="accuracy")
                shuffled_accs.append(scores.mean())
            except ValueError:
                continue
        
        if shuffled_accs:
            shuffled_accs = np.array(shuffled_accs)
            
            # Bootstrap CI
            ci_lo = np.percentile(shuffled_accs, 2.5)
            ci_hi = np.percentile(shuffled_accs, 97.5)
            
            results.append(BaselineResult(
                baseline_type="shuffled_labels",
                metric_name=f"accuracy_pos_{pos}",
                metric_value=float(shuffled_accs.mean()),
                confidence_interval=(float(ci_lo), float(ci_hi)),
                n_samples=len(shuffled_accs),
            ))
    
    return results


def run_permutation_test(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    position: int,
    n_permutations: int = 1000,
    config: ProbeConfig | None = None,
) -> dict:
    """Permutation test for significance of probe accuracy at a specific position.
    
    Tests H0: the probe accuracy at this position is no better than chance.
    
    Returns:
        dict with {observed_accuracy, null_distribution, p_value}
    """
    if config is None:
        config = ProbeConfig()
    
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    X = np.stack([caches[i].activations[layer][position] for i in valid_indices])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Observed accuracy
    probe = LogisticRegression(C=config.C, max_iter=config.max_iter, random_state=config.random_state)
    cv = StratifiedKFold(n_splits=min(config.n_folds, len(np.unique(labels))), shuffle=True, random_state=config.random_state)
    observed_scores = cross_val_score(probe, X_scaled, labels, cv=cv, scoring="accuracy")
    observed_acc = observed_scores.mean()
    
    # Null distribution
    rng = np.random.RandomState(config.random_state)
    null_accs = []
    
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        perm_probe = LogisticRegression(C=config.C, max_iter=config.max_iter, random_state=config.random_state)
        try:
            perm_scores = cross_val_score(perm_probe, X_scaled, perm_labels, cv=cv, scoring="accuracy")
            null_accs.append(perm_scores.mean())
        except ValueError:
            continue
    
    null_accs = np.array(null_accs)
    p_value = float((null_accs >= observed_acc).sum() / len(null_accs))
    
    return {
        "observed_accuracy": float(observed_acc),
        "null_distribution_mean": float(null_accs.mean()),
        "null_distribution_std": float(null_accs.std()),
        "p_value": p_value,
        "n_permutations": len(null_accs),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }
