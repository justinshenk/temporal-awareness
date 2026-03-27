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
from .geo_utils import (
    cosine_similarity,
    explained_variance_to_list,
    get_logit_diff_direction,
    get_resid_post_activations,
)

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
    short_token = pair.clean_traj.token_ids[-1]
    long_token = pair.corrupted_traj.token_ids[-1]
    logit_diff_direction = get_logit_diff_direction(runner, short_token, long_token)

    # Determine positions to analyze
    if positions is None:
        # Default: last token position (decision point)
        last_pos = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1
        positions = [last_pos]

    # Determine layers to analyze
    if layers is None:
        layers = list(range(n_layers))

    # Get activations for both clean and corrupted
    clean_acts = get_resid_post_activations(runner, pair.clean_traj.token_ids)
    corrupted_acts = get_resid_post_activations(runner, pair.corrupted_traj.token_ids)

    position_results = []
    clean_len = len(pair.clean_traj.token_ids)
    corrupted_len = len(pair.corrupted_traj.token_ids)

    for pos in positions:
        layer_results = []

        # Map position from clean to corrupted using position mapping
        # position_mapping convention: src = clean, dst = corrupted
        corrupted_pos = pair.position_mapping.get(pos, pos)
        # Clamp to valid range
        corrupted_pos = max(0, min(int(corrupted_pos), corrupted_len - 1))

        # Skip if clean position is out of bounds
        if pos < 0 or pos >= clean_len:
            continue

        for layer in layers:
            if layer not in clean_acts or layer not in corrupted_acts:
                continue

            # Get activations at this position
            clean_vec = clean_acts[layer][pos]
            corrupted_vec = corrupted_acts[layer][corrupted_pos]

            # Stack for PCA (2 samples in this single-pair case)
            # For aggregated analysis, we'd stack across pairs
            X = np.stack([clean_vec, corrupted_vec], axis=0)

            # Run PCA
            n_comp = min(n_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X)

            # Replace NaN explained variance with 0 (happens when variance is near zero)
            explained_variance = explained_variance_to_list(pca.explained_variance_ratio_)

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
                logit_diff_alignment = cosine_similarity(pc1_direction, logit_diff_direction)

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
