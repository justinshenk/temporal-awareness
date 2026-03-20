"""SVD analysis for difference-in-means directions."""

from __future__ import annotations

import numpy as np

from .diffmeans_results import SVDResult


def compute_svd_analysis(
    diff_matrix: np.ndarray,
    layer: int,
    n_components: int = 10,
) -> SVDResult:
    """Compute SVD analysis for difference vectors at a layer.

    Args:
        diff_matrix: Matrix of shape [n_pairs, d_model] where each row is a
                    difference vector (clean - corrupted) for one pair
        layer: Layer index
        n_components: Number of top components to track

    Returns:
        SVDResult with singular values and dimensionality metrics
    """
    if diff_matrix.shape[0] < 2:
        return SVDResult(layer=layer)

    # Center the data
    centered = diff_matrix - np.mean(diff_matrix, axis=0, keepdims=True)

    # Compute SVD
    try:
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return SVDResult(layer=layer)

    # Normalize singular values
    total_sv = np.sum(S)
    if total_sv < 1e-10:
        return SVDResult(layer=layer)

    normalized_sv = S / total_sv

    # Compute metrics
    # Top-k singular values
    top_k = min(n_components, len(S))
    top_singular_values = normalized_sv[:top_k].tolist()

    # Explained variance ratio (squared singular values)
    variance = S ** 2
    total_variance = np.sum(variance)
    if total_variance > 0:
        explained_variance = (variance / total_variance)[:top_k].tolist()
    else:
        explained_variance = [0.0] * top_k

    # Effective rank: 1 / sum(p_i^2) where p_i = s_i / sum(s)
    # Also known as participation ratio
    effective_rank = 1.0 / np.sum(normalized_sv ** 2) if np.sum(normalized_sv ** 2) > 0 else 0.0

    # Top-1 ratio: how much the top singular value dominates
    top1_ratio = float(normalized_sv[0]) if len(normalized_sv) > 0 else 0.0

    return SVDResult(
        layer=layer,
        singular_values=top_singular_values,
        explained_variance_ratio=explained_variance,
        effective_rank=float(effective_rank),
        top1_ratio=top1_ratio,
    )


def compute_svd_trajectory(
    layer_diff_matrices: dict[int, np.ndarray],
    n_components: int = 10,
) -> list[SVDResult]:
    """Compute SVD analysis for each layer.

    Args:
        layer_diff_matrices: Dict mapping layer -> [n_pairs, d_model] difference matrix
        n_components: Number of top components to track

    Returns:
        List of SVDResult for each layer
    """
    results = []
    for layer in sorted(layer_diff_matrices.keys()):
        result = compute_svd_analysis(layer_diff_matrices[layer], layer, n_components)
        results.append(result)
    return results
