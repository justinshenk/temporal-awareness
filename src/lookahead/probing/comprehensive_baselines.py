"""Comprehensive baselines to rule out artifacts.

Every claimed signal MUST survive all of these controls. If any
baseline matches the real probe accuracy, the signal is an artifact.

Baselines implemented:
1. BAG_OF_WORDS: Probe using token-identity features instead of activations.
   If this matches real probe → signal is surface-level, not representational.

2. PCA_REDUCTION: Reduce activations to k dimensions, then probe.
   If signal vanishes at k=10 → probe was fitting noise in high-dim space.

3. POSITION_CONTROL: Shuffle which position's activations map to which label.
   If this matches → signal is position-specific, not content-specific.

4. RANDOM_DIRECTION: Project activations onto random directions and probe.
   If this matches → signal is structural (e.g., activation norm), not directional.

5. ANCHOR_WORD_ONLY: For rhyme task, probe using ONLY the anchor word token's
   activation. If this matches → probe just reads the anchor, no planning.

6. MULTIPLE_COMPARISONS: FDR correction across all position × layer tests.
   Without this, 5% of positions will be "significant" by chance.

7. UNTRAINED_MODEL: Same architecture, random weights. If signal exists →
   it's architectural, not learned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..utils.types import ActivationCache, PlanningExample, BaselineResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. BAG-OF-WORDS BASELINE
# ═══════════════════════════════════════════════════════════════════════

def bag_of_words_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    n_folds: int = 5,
    random_state: int = 42,
) -> list[BaselineResult]:
    """Probe using token-identity features instead of activations.
    
    This is the MOST IMPORTANT baseline for the rhyme task.
    
    If a BoW probe (which just knows which tokens are in the prompt)
    achieves similar accuracy to the activation probe, then the
    activation probe is just reading surface features — not detecting
    planning in the model's representations.
    
    Features: binary vector of size vocab_size indicating which tokens
    appear in the prompt up to position t.
    """
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10 or len(unique_targets) < 2:
        return []
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    min_seq_len = min(len(caches[i].token_ids) for i in valid_indices)
    
    # Get vocab size from first cache
    max_token_id = max(max(caches[i].token_ids) for i in valid_indices) + 1
    # Cap at reasonable size for BoW
    bow_dim = min(max_token_id, 50257)  # GPT-2 vocab size
    
    results = []
    
    for pos in range(min_seq_len):
        # Build BoW feature: which tokens have appeared up to position pos
        X_bow = np.zeros((len(valid_indices), bow_dim), dtype=np.float32)
        for row, idx in enumerate(valid_indices):
            for t in range(pos + 1):
                tid = caches[idx].token_ids[t]
                if tid < bow_dim:
                    X_bow[row, tid] = 1.0
        
        # Remove zero-variance features
        nonzero_cols = X_bow.sum(axis=0) > 0
        X_bow_filtered = X_bow[:, nonzero_cols]
        
        if X_bow_filtered.shape[1] == 0:
            continue
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bow_filtered)
        
        probe = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="saga")
        
        try:
            n_splits = min(n_folds, min(np.bincount(labels)))
            if n_splits < 2:
                continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(probe, X_scaled, labels, cv=cv, scoring="accuracy")
        except ValueError:
            continue
        
        results.append(BaselineResult(
            baseline_type="bag_of_words",
            metric_name=f"accuracy_pos_{pos}",
            metric_value=float(scores.mean()),
            confidence_interval=(
                float(scores.mean() - 1.96 * scores.std()),
                float(scores.mean() + 1.96 * scores.std()),
            ),
            n_samples=len(valid_indices),
        ))
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. PCA DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════

def pca_reduction_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    n_components_list: list[int] | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[int, list[BaselineResult]]:
    """Probe after PCA reduction to k dimensions.
    
    If the signal survives at k=10, it's concentrated in a low-dimensional
    subspace (consistent with a genuine direction in activation space).
    If it vanishes below k=50, it was distributed noise.
    
    Returns: dict[n_components -> list of BaselineResult per position]
    """
    if n_components_list is None:
        n_components_list = [5, 10, 20, 50, 100]
    
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10 or len(unique_targets) < 2:
        return {}
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    min_seq_len = min(len(caches[i].token_ids) for i in valid_indices)
    
    all_results = {}
    
    for n_comp in n_components_list:
        results = []
        
        for pos in range(min_seq_len):
            X = np.stack([caches[i].activations[layer][pos] for i in valid_indices])
            
            # Cap n_components at min(n_samples, n_features)
            actual_n = min(n_comp, X.shape[0] - 1, X.shape[1])
            if actual_n < 2:
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=actual_n, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            probe = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="saga")
            
            try:
                n_splits = min(n_folds, min(np.bincount(labels)))
                if n_splits < 2:
                    continue
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                scores = cross_val_score(probe, X_pca, labels, cv=cv, scoring="accuracy")
            except ValueError:
                continue
            
            results.append(BaselineResult(
                baseline_type=f"pca_{actual_n}",
                metric_name=f"accuracy_pos_{pos}",
                metric_value=float(scores.mean()),
                confidence_interval=(
                    float(scores.mean() - 1.96 * scores.std()),
                    float(scores.mean() + 1.96 * scores.std()),
                ),
                n_samples=len(valid_indices),
            ))
        
        all_results[n_comp] = results
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# 3. POSITION-SHUFFLED CONTROL
# ═══════════════════════════════════════════════════════════════════════

def position_shuffled_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    n_shuffles: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[BaselineResult]:
    """Shuffle which position's activations map to which example.
    
    For each example, replace position t's activation with the activation
    from position t of a RANDOM OTHER EXAMPLE. This preserves position-level
    statistics but destroys example-specific content.
    
    If probe still works → it's detecting position-level features,
    not example-specific planning.
    """
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10 or len(unique_targets) < 2:
        return []
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    min_seq_len = min(len(caches[i].token_ids) for i in valid_indices)
    rng = np.random.RandomState(random_state)
    
    results = []
    
    for pos in range(min_seq_len):
        shuffled_accs = []
        
        for _ in range(n_shuffles):
            # For each example, grab position t from a random other example
            shuffle_order = rng.permutation(len(valid_indices))
            X_shuffled = np.stack([
                caches[valid_indices[shuffle_order[row]]].activations[layer][pos]
                for row in range(len(valid_indices))
            ])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_shuffled)
            
            probe = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="saga")
            
            try:
                n_splits = min(n_folds, min(np.bincount(labels)))
                if n_splits < 2:
                    continue
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
                scores = cross_val_score(probe, X_scaled, labels, cv=cv, scoring="accuracy")
                shuffled_accs.append(scores.mean())
            except ValueError:
                continue
        
        if shuffled_accs:
            arr = np.array(shuffled_accs)
            results.append(BaselineResult(
                baseline_type="position_shuffled",
                metric_name=f"accuracy_pos_{pos}",
                metric_value=float(arr.mean()),
                confidence_interval=(float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
                n_samples=len(shuffled_accs),
            ))
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. RANDOM DIRECTION BASELINE
# ═══════════════════════════════════════════════════════════════════════

def random_direction_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    n_directions: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[BaselineResult]:
    """Project activations onto random directions and probe.
    
    If projecting onto a random 1D direction gives similar accuracy
    to the full probe, the signal is just about activation norm
    or global statistics, not a specific direction.
    """
    unique_targets = sorted(set(ex.target_value for ex in examples if ex.target_value))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.target_value in target_to_idx
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10 or len(unique_targets) < 2:
        return []
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    min_seq_len = min(len(caches[i].token_ids) for i in valid_indices)
    d_model = caches[valid_indices[0]].activations[layer].shape[1]
    rng = np.random.RandomState(random_state)
    
    results = []
    
    for pos in range(min_seq_len):
        X = np.stack([caches[i].activations[layer][pos] for i in valid_indices])
        
        rand_accs = []
        for _ in range(n_directions):
            # Random unit direction
            direction = rng.randn(d_model)
            direction /= np.linalg.norm(direction)
            
            # Project: (n_samples, 1)
            X_proj = (X @ direction).reshape(-1, 1)
            
            probe = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="saga")
            try:
                n_splits = min(n_folds, min(np.bincount(labels)))
                if n_splits < 2:
                    continue
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
                scores = cross_val_score(probe, X_proj, labels, cv=cv, scoring="accuracy")
                rand_accs.append(scores.mean())
            except ValueError:
                continue
        
        if rand_accs:
            arr = np.array(rand_accs)
            results.append(BaselineResult(
                baseline_type="random_direction",
                metric_name=f"accuracy_pos_{pos}",
                metric_value=float(arr.mean()),
                confidence_interval=(float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
                n_samples=len(rand_accs),
            ))
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. ANCHOR-WORD-ONLY BASELINE (rhyme-specific)
# ═══════════════════════════════════════════════════════════════════════

def anchor_word_only_baseline(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[BaselineResult]:
    """Probe using ONLY the activation at the anchor word position.
    
    For rhyme task: if a single probe at the anchor position predicts
    the target just as well as probes at later positions, then
    "commitment" is just "reading the anchor word" — not planning.
    
    We use the SAME anchor-position activation at every "test position"
    to show whether post-anchor probes add information beyond the anchor.
    """
    from ..utils.types import TaskType as _TaskType
    
    valid_indices = [
        i for i, ex in enumerate(examples)
        if ex.target_value and ex.task_type == _TaskType.RHYME
        and not ex.metadata.get("is_control", False)
    ]
    
    if len(valid_indices) < 10:
        return []
    
    unique_targets = sorted(set(examples[i].target_value for i in valid_indices))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    if len(unique_targets) < 2:
        return []
    
    labels = np.array([target_to_idx[examples[i].target_value] for i in valid_indices])
    
    # Find anchor word position for each example
    anchor_positions = []
    for idx in valid_indices:
        ex = examples[idx]
        anchor = ex.metadata.get("anchor_word", "")
        # Find the last occurrence of anchor in token strings
        tokens = caches[idx].token_strings
        anchor_pos = 0
        for t, tok in enumerate(tokens):
            if anchor.lower() in tok.lower():
                anchor_pos = t
        anchor_positions.append(anchor_pos)
    
    # Build feature matrix: anchor activation only
    X_anchor = np.stack([
        caches[valid_indices[row]].activations[layer][anchor_positions[row]]
        for row in range(len(valid_indices))
    ])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anchor)
    
    probe = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="saga")
    
    try:
        n_splits = min(n_folds, min(np.bincount(labels)))
        if n_splits < 2:
            return []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(probe, X_scaled, labels, cv=cv, scoring="accuracy")
    except ValueError:
        return []
    
    return [BaselineResult(
        baseline_type="anchor_word_only",
        metric_name="accuracy_anchor_position",
        metric_value=float(scores.mean()),
        confidence_interval=(
            float(scores.mean() - 1.96 * scores.std()),
            float(scores.mean() + 1.96 * scores.std()),
        ),
        n_samples=len(valid_indices),
    )]


# ═══════════════════════════════════════════════════════════════════════
# 6. MULTIPLE COMPARISONS CORRECTION
# ═══════════════════════════════════════════════════════════════════════

def fdr_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: Raw p-values from permutation tests
        alpha: FDR threshold
        
    Returns:
        (significant_mask, adjusted_p_values)
    """
    n = len(p_values)
    if n == 0:
        return [], []
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    # BH procedure
    adjusted = np.zeros(n)
    significant = np.zeros(n, dtype=bool)
    
    # Step-up procedure
    prev_adj = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted_p = min(prev_adj, sorted_p[i] * n / rank)
        adjusted[sorted_idx[i]] = adjusted_p
        prev_adj = adjusted_p
        significant[sorted_idx[i]] = adjusted_p < alpha
    
    return significant.tolist(), adjusted.tolist()


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Bonferroni correction (more conservative than FDR).
    
    Use this when you want to be extra safe about false positives.
    """
    n = len(p_values)
    adjusted = [min(1.0, p * n) for p in p_values]
    significant = [p < alpha for p in adjusted]
    return significant, adjusted


# ═══════════════════════════════════════════════════════════════════════
# 7. COMPREHENSIVE BASELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BaselineSuite:
    """Results from running all baselines."""
    bag_of_words: list[BaselineResult]
    pca_reduction: dict[int, list[BaselineResult]]
    position_shuffled: list[BaselineResult]
    random_direction: list[BaselineResult]
    anchor_word_only: list[BaselineResult]  # rhyme only
    
    # Corrected p-values (from permutation tests)
    fdr_significant: list[bool] | None = None
    fdr_adjusted_p: list[float] | None = None
    bonferroni_significant: list[bool] | None = None
    bonferroni_adjusted_p: list[float] | None = None


def run_all_baselines(
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    layer: int,
    p_values: list[float] | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> BaselineSuite:
    """Run the complete baseline suite.
    
    This is the function you call. It runs every baseline and returns
    a suite that you compare against your real probe results.
    
    The paper should include a table like:
    
    | Position | Real Probe | Shuffled Labels | BoW | PCA-10 | Pos-Shuffled | Random Dir | Anchor Only |
    |----------|-----------|-----------------|-----|--------|-------------|-----------|-------------|
    | 5        | 0.35      | 0.33            | 0.89| 0.34   | 0.33        | 0.33      | 0.85        |
    | 10       | 0.72      | 0.34            | 0.90| 0.71   | 0.35        | 0.34      | 0.85        |
    | 15       | 0.91      | 0.33            | 0.90| 0.88   | 0.34        | 0.34      | 0.85        |
    
    The signal is genuine planning only if:
    - Real Probe >> Shuffled Labels (not random)
    - Real Probe >> BoW (not surface features)
    - PCA-10 ≈ Real Probe (low-dimensional, not noise)
    - Real Probe >> Position-Shuffled (example-specific, not position-general)
    - Real Probe >> Random Direction (specific direction, not norm)
    - Real Probe > Anchor Only at later positions (adds info beyond anchor)
    """
    logger.info(f"Running comprehensive baselines on layer {layer}...")
    
    logger.info("  Bag-of-words baseline...")
    bow = bag_of_words_baseline(caches, examples, n_folds, random_state)
    
    logger.info("  PCA reduction baseline...")
    pca = pca_reduction_baseline(caches, examples, layer, [5, 10, 20, 50], n_folds, random_state)
    
    logger.info("  Position-shuffled baseline...")
    pos_shuf = position_shuffled_baseline(caches, examples, layer, 30, n_folds, random_state)
    
    logger.info("  Random direction baseline...")
    rand_dir = random_direction_baseline(caches, examples, layer, 30, n_folds, random_state)
    
    logger.info("  Anchor-word-only baseline...")
    anchor = anchor_word_only_baseline(caches, examples, layer, n_folds, random_state)
    
    suite = BaselineSuite(
        bag_of_words=bow,
        pca_reduction=pca,
        position_shuffled=pos_shuf,
        random_direction=rand_dir,
        anchor_word_only=anchor,
    )
    
    # Apply multiple comparisons correction if p-values provided
    if p_values:
        logger.info("  Applying multiple comparisons correction...")
        fdr_sig, fdr_adj = fdr_correction(p_values)
        bonf_sig, bonf_adj = bonferroni_correction(p_values)
        suite.fdr_significant = fdr_sig
        suite.fdr_adjusted_p = fdr_adj
        suite.bonferroni_significant = bonf_sig
        suite.bonferroni_adjusted_p = bonf_adj
    
    logger.info("  All baselines complete.")
    return suite
