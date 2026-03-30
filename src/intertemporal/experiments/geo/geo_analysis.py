"""Main analysis function for geometric (PCA) analysis.

Performs PCA on residual stream activations to analyze:
1. At which layer clean/corrupted become separable
2. How many dimensions the separation requires
3. Whether separation direction aligns with logit difference
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

from ....common.logging import log
from .geo_results import GeoPCALayerResult, GeoPCAPositionResult, GeoPairResult

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


def run_geo_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    pair_idx: int = 0,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    n_components: int = 3,
) -> GeoPairResult:
    """Run geometric (PCA) analysis for a single pair.

    Computes PCA at each layer for specified positions, tracking:
    1. Explained variance ratios (how concentrated is information)
    2. Clean/corrupted separation in PC space
    3. Alignment with logit difference direction

    Args:
        runner: Model runner
        pair: Contrastive pair
        pair_idx: Pair index for tracking
        positions: Positions to analyze (None = last token only)
        layers: Layers to analyze (None = all layers)
        n_components: Number of PCA components to track

    Returns:
        GeoPairResult with per-position, per-layer analysis
    """
    import torch

    n_layers = runner.n_layers

    # Get logit difference direction for alignment check
    logit_diff_direction = _get_logit_diff_direction(runner, pair)

    # Determine positions to analyze
    if positions is None:
        # Default: last token position (decision point)
        last_pos = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1
        positions = [last_pos]

    # Determine layers to analyze
    if layers is None:
        layers = list(range(n_layers))

    # Get activations for both clean and corrupted
    clean_acts = _get_resid_post_activations(runner, pair.clean_traj.token_ids)
    corrupted_acts = _get_resid_post_activations(runner, pair.corrupted_traj.token_ids)

    position_results = []
    for pos in positions:
        layer_results = []

        for layer in layers:
            if layer not in clean_acts or layer not in corrupted_acts:
                continue

            # Get activations at this position
            clean_vec = clean_acts[layer][pos]
            corrupted_vec = corrupted_acts[layer][pos]

            # Stack for PCA (2 samples in this single-pair case)
            # For aggregated analysis, we'd stack across pairs
            X = np.stack([clean_vec, corrupted_vec], axis=0)

            # Run PCA
            n_comp = min(n_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X)

            # Replace NaN explained variance with 0 (happens when variance is near zero)
            explained_variance = pca.explained_variance_ratio_.tolist()
            explained_variance = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in explained_variance]

            # Clean is sample 0, corrupted is sample 1
            clean_pc = X_pca[0].tolist()
            corrupted_pc = X_pca[1].tolist()

            # Compute separation metrics
            separation_distance = float(np.linalg.norm(X_pca[0] - X_pca[1]))
            separation_pc1 = float(X_pca[0, 0] - X_pca[1, 0]) if X_pca.shape[1] > 0 else 0.0

            # Compute alignment with logit diff direction
            logit_diff_alignment = None
            if logit_diff_direction is not None:
                # Project logit diff direction onto PC1
                pc1_direction = pca.components_[0]
                alignment = _cosine_similarity(pc1_direction, logit_diff_direction)
                logit_diff_alignment = float(alignment)

            layer_results.append(GeoPCALayerResult(
                layer=layer,
                explained_variance_ratio=explained_variance,
                clean_mean_pc=clean_pc,
                corrupted_mean_pc=corrupted_pc,
                separation_distance=separation_distance,
                separation_pc1=separation_pc1,
                logit_diff_alignment=logit_diff_alignment,
            ))

        # Get position label if available
        pos_label = None
        if pos < len(pair.clean_traj.token_ids):
            pos_label = f"pos_{pos}"

        position_results.append(GeoPCAPositionResult(
            position=pos,
            position_label=pos_label,
            layer_results=layer_results,
        ))

    return GeoPairResult(
        pair_idx=pair_idx,
        position_results=position_results,
    )


def _get_resid_post_activations(
    runner: "BinaryChoiceRunner",
    tokens: list[int],
) -> dict[int, np.ndarray]:
    """Get resid_post activations at all layers.

    Returns:
        Dict mapping layer -> activations [seq_len, d_model]
    """
    import torch

    result = {}

    # Run model with cache
    input_ids = torch.tensor([tokens], device=runner.device)

    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=None)

    # Parse cache results
    for key, value in cache.items():
        if not key.startswith("blocks."):
            continue

        parts = key.split(".")
        if len(parts) < 3:
            continue

        try:
            layer = int(parts[1])
        except ValueError:
            continue

        hook_part = parts[2]
        if hook_part == "hook_resid_post":
            # value is [batch, seq_len, d_model], take first batch element
            result[layer] = value[0].cpu().numpy()

    return result


def _get_logit_diff_direction(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> np.ndarray | None:
    """Get the logit difference direction in embedding space.

    This is the direction from long_term token embedding to short_term token embedding.
    Used to check if PCA separation aligns with the task-relevant direction.

    Returns:
        Normalized direction vector or None if unavailable
    """
    try:
        # Get the unembedding matrix
        W_U = runner._backend.get_unembedding_matrix()  # [vocab, d_model]

        # Get token IDs for short and long labels
        short_token = pair.clean_traj.token_ids[-1]  # Last token is the choice
        long_token = pair.corrupted_traj.token_ids[-1]

        # Get embeddings
        short_emb = W_U[short_token].cpu().numpy()
        long_emb = W_U[long_token].cpu().numpy()

        # Direction from long to short (positive = more short-like)
        direction = short_emb - long_emb

        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            return direction / norm
        return None
    except (AttributeError, IndexError):
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
