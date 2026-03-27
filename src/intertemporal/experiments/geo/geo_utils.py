"""Shared utilities for geometric analysis.

Common functions used by both:
- experiments/geo/ (per-pair contrastive PCA analysis)
- geometry/ (dataset-wide linear probe analysis)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if either vector has near-zero norm
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_resid_post_activations(
    runner: "BinaryChoiceRunner",
    tokens: list[int],
) -> dict[int, np.ndarray]:
    """Get resid_post activations at all layers.

    Args:
        runner: Model runner with TransformerLens backend
        tokens: Input token IDs

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


def get_logit_diff_direction(
    runner: "BinaryChoiceRunner",
    short_token: int,
    long_token: int,
) -> np.ndarray | None:
    """Get the logit difference direction in embedding space.

    This is the direction from long_term token embedding to short_term token embedding.
    Used to check if PCA separation aligns with the task-relevant direction.

    Args:
        runner: Model runner
        short_token: Token ID for short-term choice
        long_token: Token ID for long-term choice

    Returns:
        Normalized direction vector or None if unavailable
    """
    try:
        # Get the unembedding matrix
        W_U = runner._backend.get_unembedding_matrix()  # [vocab, d_model]

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


def explained_variance_to_list(variance_ratio: np.ndarray) -> list[float]:
    """Convert explained variance ratio to list, handling NaN values.

    Args:
        variance_ratio: Numpy array of variance ratios

    Returns:
        List of floats with NaN replaced by 0.0
    """
    result = variance_ratio.tolist()
    return [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in result]
