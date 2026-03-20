"""Rotation analysis for difference-in-means directions."""

from __future__ import annotations

import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    denom = norm1 * norm2
    if denom < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle in degrees between two vectors."""
    cos_sim = cosine_similarity(v1, v2)
    # Clamp to avoid numerical issues with arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


def compute_rotation_decomposition(
    diff_resid_pre: np.ndarray,
    diff_attn_out: np.ndarray,
    diff_mlp_out: np.ndarray,
    diff_resid_post: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose rotation into attention and MLP contributions.

    The residual stream evolves as:
        resid_post = resid_pre + attn_out + mlp_out

    We compute:
        1. Direction after attention: diff(resid_pre + attn_out)
        2. Angle from diff(resid_pre) to diff(resid_pre + attn_out) = attention's rotation
        3. Angle from diff(resid_pre + attn_out) to diff(resid_post) = MLP's rotation

    Args:
        diff_resid_pre: Difference vector at resid_pre[L]
        diff_attn_out: Difference vector for attn_out[L]
        diff_mlp_out: Difference vector for mlp_out[L]
        diff_resid_post: Difference vector at resid_post[L]

    Returns:
        Tuple of (attn_rotation_angle, mlp_rotation_angle, total_rotation_angle)
    """
    # Direction at resid_pre
    dir_pre = diff_resid_pre

    # Direction after attention: resid_pre + attn_out
    dir_post_attn = diff_resid_pre + diff_attn_out

    # Direction at resid_post (after MLP)
    dir_post = diff_resid_post

    # Compute angles
    attn_angle = angle_between(dir_pre, dir_post_attn)
    mlp_angle = angle_between(dir_post_attn, dir_post)
    total_angle = angle_between(dir_pre, dir_post)

    return attn_angle, mlp_angle, total_angle


def compute_layer_direction_similarity(
    diff_vectors: dict[int, np.ndarray],
) -> dict[int, tuple[float | None, float | None]]:
    """Compute cosine similarity between consecutive layer directions.

    Args:
        diff_vectors: Dict mapping layer -> difference vector

    Returns:
        Dict mapping layer -> (cosine_to_prev, cosine_to_next)
    """
    layers = sorted(diff_vectors.keys())
    result = {}

    for i, layer in enumerate(layers):
        cos_prev = None
        cos_next = None

        if i > 0:
            prev_layer = layers[i - 1]
            cos_prev = cosine_similarity(diff_vectors[prev_layer], diff_vectors[layer])

        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            cos_next = cosine_similarity(diff_vectors[layer], diff_vectors[next_layer])

        result[layer] = (cos_prev, cos_next)

    return result
