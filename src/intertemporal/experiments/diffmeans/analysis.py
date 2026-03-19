"""Main analysis function for difference-in-means."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from .results import DiffMeansLayerResult, DiffMeansPairResult
from .rotation import (
    compute_layer_direction_similarity,
    compute_rotation_decomposition,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


def run_diffmeans_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    pair_idx: int = 0,
    position: int | None = None,
) -> DiffMeansPairResult:
    """Run difference-in-means analysis for a single pair.

    Computes:
    1. Difference vectors (clean - corrupted) at each layer
    2. Cosine similarity between consecutive layers
    3. Rotation decomposition (attention vs MLP contribution)

    Args:
        runner: Model runner
        pair: Contrastive pair
        pair_idx: Pair index for tracking
        position: Specific position to analyze (None = use last token)

    Returns:
        DiffMeansPairResult with per-layer analysis
    """
    n_layers = runner.n_layers

    # Get activations for clean and corrupted
    clean_acts = _get_all_activations(runner, pair.clean_traj.token_ids)
    corrupted_acts = _get_all_activations(runner, pair.corrupted_traj.token_ids)

    # Determine position to analyze
    if position is None:
        # Use last token position (decision point)
        pos = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1
    else:
        pos = position

    # Compute difference vectors at each layer for resid_post
    diff_resid_post = {}
    for layer in range(n_layers):
        if layer in clean_acts.get("resid_post", {}) and layer in corrupted_acts.get("resid_post", {}):
            clean_vec = clean_acts["resid_post"][layer][pos]
            corrupted_vec = corrupted_acts["resid_post"][layer][pos]
            diff_resid_post[layer] = clean_vec - corrupted_vec

    if not diff_resid_post:
        # No activations captured, return empty result
        return DiffMeansPairResult(pair_idx=pair_idx, layer_results=[], positions_analyzed=0)

    # Compute cosine similarities between consecutive layers
    layer_similarities = compute_layer_direction_similarity(diff_resid_post)

    # Compute rotation decomposition for each layer
    layer_results = []
    for layer in range(n_layers):
        if layer not in diff_resid_post:
            continue

        cos_prev, cos_next = layer_similarities.get(layer, (None, None))

        # Rotation decomposition (need resid_pre, attn_out, mlp_out)
        attn_angle = None
        mlp_angle = None
        total_angle = None

        if (
            layer > 0
            and layer in clean_acts.get("resid_pre", {})
            and layer in clean_acts.get("attn_out", {})
            and layer in clean_acts.get("mlp_out", {})
        ):
            # Get difference vectors for decomposition
            diff_resid_pre = (
                clean_acts["resid_pre"][layer][pos] - corrupted_acts["resid_pre"][layer][pos]
            )
            diff_attn_out = (
                clean_acts["attn_out"][layer][pos] - corrupted_acts["attn_out"][layer][pos]
            )
            diff_mlp_out = (
                clean_acts["mlp_out"][layer][pos] - corrupted_acts["mlp_out"][layer][pos]
            )
            diff_resid_post_layer = diff_resid_post[layer]

            attn_angle, mlp_angle, total_angle = compute_rotation_decomposition(
                diff_resid_pre, diff_attn_out, diff_mlp_out, diff_resid_post_layer
            )

        layer_results.append(DiffMeansLayerResult(
            layer=layer,
            cosine_to_next=cos_next,
            cosine_to_prev=cos_prev,
            diff_norm=float(np.linalg.norm(diff_resid_post[layer])),
            attn_rotation_angle=attn_angle,
            mlp_rotation_angle=mlp_angle,
            total_rotation_angle=total_angle,
        ))

    return DiffMeansPairResult(
        pair_idx=pair_idx,
        layer_results=layer_results,
        positions_analyzed=1,
    )


def _get_all_activations(
    runner: "BinaryChoiceRunner",
    tokens: list[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Get activations at all layers for all components.

    Returns:
        Dict mapping component -> layer -> activations [seq_len, d_model]
    """
    components = ["resid_post", "resid_pre", "attn_out", "mlp_out"]
    result = {comp: {} for comp in components}

    # Run model with cache
    input_ids = torch.tensor([tokens], device=runner.device)

    with torch.no_grad():
        # Use backend's run_with_cache method
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=None)

    # Parse cache results into our format
    # Cache keys are like "blocks.{layer}.hook_{component}"
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

        # Extract component from hook name (e.g., "hook_resid_post" -> "resid_post")
        hook_part = parts[2]
        if hook_part.startswith("hook_"):
            comp = hook_part[5:]  # Remove "hook_" prefix
            if comp in components:
                # value is [batch, seq_len, d_model], take first batch element
                result[comp][layer] = value[0].cpu().numpy()

    return result
