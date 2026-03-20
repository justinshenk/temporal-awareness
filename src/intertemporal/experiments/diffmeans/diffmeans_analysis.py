"""Main analysis function for difference-in-means."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from .diffmeans_results import DiffMeansLayerResult, DiffMeansPairResult
from .diffmeans_rotation import (
    compute_layer_direction_similarity,
    compute_rotation_decomposition,
    cosine_similarity,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


def run_diffmeans_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    pair_idx: int = 0,
    position: int | None = None,
    additional_positions: list[int] | None = None,
) -> DiffMeansPairResult:
    """Run difference-in-means analysis for a single pair.

    Computes:
    1. Difference vectors (clean - corrupted) at each layer
    2. Cosine similarity between consecutive layers
    3. Rotation decomposition (attention vs MLP contribution)
    4. Cosine similarity to logit direction
    5. Cosine similarity to initial direction (cumulative drift)
    6. Component-wise (attn, mlp) difference norms

    Args:
        runner: Model runner (with W_U for logit direction)
        pair: Contrastive pair
        pair_idx: Pair index for tracking
        position: Primary position to analyze (None = use last token)
        additional_positions: Additional positions to analyze

    Returns:
        DiffMeansPairResult with per-layer analysis
    """
    n_layers = runner.n_layers

    # Get activations for clean and corrupted
    clean_acts = _get_all_activations(runner, pair.clean_traj.token_ids)
    corrupted_acts = _get_all_activations(runner, pair.corrupted_traj.token_ids)

    # Determine primary position to analyze
    if position is None:
        # Use last token position (decision point)
        primary_pos = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1
    else:
        primary_pos = position

    # Get logit direction (W_U[clean_choice] - W_U[corrupted_choice])
    logit_direction = _compute_logit_direction(runner, pair)

    # Compute difference vectors at primary position
    diff_resid_post = {}
    diff_attn_out = {}
    diff_mlp_out = {}

    for layer in range(n_layers):
        if layer in clean_acts.get("resid_post", {}) and layer in corrupted_acts.get("resid_post", {}):
            clean_vec = clean_acts["resid_post"][layer][primary_pos]
            corrupted_vec = corrupted_acts["resid_post"][layer][primary_pos]
            diff_resid_post[layer] = clean_vec - corrupted_vec

        if layer in clean_acts.get("attn_out", {}) and layer in corrupted_acts.get("attn_out", {}):
            diff_attn_out[layer] = (
                clean_acts["attn_out"][layer][primary_pos]
                - corrupted_acts["attn_out"][layer][primary_pos]
            )

        if layer in clean_acts.get("mlp_out", {}) and layer in corrupted_acts.get("mlp_out", {}):
            diff_mlp_out[layer] = (
                clean_acts["mlp_out"][layer][primary_pos]
                - corrupted_acts["mlp_out"][layer][primary_pos]
            )

    if not diff_resid_post:
        return DiffMeansPairResult(pair_idx=pair_idx, layer_results=[], positions_analyzed=0)

    # Get initial direction for cumulative drift computation
    initial_layer = min(diff_resid_post.keys())
    initial_direction = diff_resid_post[initial_layer]

    # Compute cosine similarities between consecutive layers
    layer_similarities = compute_layer_direction_similarity(diff_resid_post)

    # Build layer results for primary position
    layer_results = _build_layer_results(
        n_layers=n_layers,
        diff_resid_post=diff_resid_post,
        diff_attn_out=diff_attn_out,
        diff_mlp_out=diff_mlp_out,
        clean_acts=clean_acts,
        corrupted_acts=corrupted_acts,
        primary_pos=primary_pos,
        layer_similarities=layer_similarities,
        logit_direction=logit_direction,
        initial_direction=initial_direction,
    )

    # Compute position-specific results for additional positions
    position_results = {}
    all_positions = set()
    if additional_positions:
        all_positions.update(additional_positions)
    # Always include the primary position
    all_positions.add(primary_pos)

    seq_len = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids))
    for pos in all_positions:
        if pos < 0 or pos >= seq_len:
            continue
        position_results[pos] = _build_position_layer_results(
            n_layers=n_layers,
            clean_acts=clean_acts,
            corrupted_acts=corrupted_acts,
            pos=pos,
        )

    return DiffMeansPairResult(
        pair_idx=pair_idx,
        layer_results=layer_results,
        positions_analyzed=len(position_results),
        position_results=position_results,
        primary_position=primary_pos,
    )


def _compute_logit_direction(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> np.ndarray | None:
    """Compute logit direction from unembedding matrix.

    Returns W_U[clean_choice_token] - W_U[corrupted_choice_token],
    which is the direction in activation space that points toward
    the clean choice in logit space.
    """
    W_U = runner.W_U
    if W_U is None:
        return None

    # Get the divergent tokens (first tokens where clean/corrupted differ)
    clean_div_pos = pair.clean_divergent_position
    corrupted_div_pos = pair.corrupted_divergent_position

    if clean_div_pos is None or corrupted_div_pos is None:
        # Fallback: use last token
        clean_token = pair.clean_traj.token_ids[-1]
        corrupted_token = pair.corrupted_traj.token_ids[-1]
    else:
        clean_token = pair.clean_traj.token_ids[clean_div_pos]
        corrupted_token = pair.corrupted_traj.token_ids[corrupted_div_pos]

    if clean_token == corrupted_token:
        # Same token, no logit difference
        return None

    # W_U is [d_model, vocab_size] or [vocab_size, d_model] depending on backend
    # We want the column/row for each token
    if hasattr(W_U, "detach"):
        W_U_np = W_U.detach().cpu().numpy()
    elif hasattr(W_U, "cpu"):
        W_U_np = W_U.cpu().numpy()
    else:
        W_U_np = np.array(W_U)

    # Handle both shapes
    if W_U_np.shape[0] > W_U_np.shape[1]:
        # Shape is [vocab_size, d_model]
        clean_vec = W_U_np[clean_token]
        corrupted_vec = W_U_np[corrupted_token]
    else:
        # Shape is [d_model, vocab_size]
        clean_vec = W_U_np[:, clean_token]
        corrupted_vec = W_U_np[:, corrupted_token]

    return clean_vec - corrupted_vec


def _build_layer_results(
    n_layers: int,
    diff_resid_post: dict[int, np.ndarray],
    diff_attn_out: dict[int, np.ndarray],
    diff_mlp_out: dict[int, np.ndarray],
    clean_acts: dict[str, dict[int, np.ndarray]],
    corrupted_acts: dict[str, dict[int, np.ndarray]],
    primary_pos: int,
    layer_similarities: dict[int, tuple[float | None, float | None]],
    logit_direction: np.ndarray | None,
    initial_direction: np.ndarray,
) -> list[DiffMeansLayerResult]:
    """Build layer results for the primary position with all metrics."""
    layer_results = []

    for layer in range(n_layers):
        if layer not in diff_resid_post:
            continue

        cos_prev, cos_next = layer_similarities.get(layer, (None, None))
        diff_vec = diff_resid_post[layer]

        # Cosine to logit direction
        cosine_to_logit = None
        if logit_direction is not None:
            cosine_to_logit = cosine_similarity(diff_vec, logit_direction)

        # Cosine to initial direction (cumulative drift)
        cosine_to_initial = cosine_similarity(diff_vec, initial_direction)

        # Component diff norms
        attn_out_diff_norm = None
        mlp_out_diff_norm = None
        if layer in diff_attn_out:
            attn_out_diff_norm = float(np.linalg.norm(diff_attn_out[layer]))
        if layer in diff_mlp_out:
            mlp_out_diff_norm = float(np.linalg.norm(diff_mlp_out[layer]))

        # Rotation decomposition
        attn_angle = None
        mlp_angle = None
        total_angle = None

        if (
            layer > 0
            and layer in clean_acts.get("resid_pre", {})
            and layer in clean_acts.get("attn_out", {})
            and layer in clean_acts.get("mlp_out", {})
        ):
            diff_resid_pre = (
                clean_acts["resid_pre"][layer][primary_pos]
                - corrupted_acts["resid_pre"][layer][primary_pos]
            )
            diff_attn = (
                clean_acts["attn_out"][layer][primary_pos]
                - corrupted_acts["attn_out"][layer][primary_pos]
            )
            diff_mlp = (
                clean_acts["mlp_out"][layer][primary_pos]
                - corrupted_acts["mlp_out"][layer][primary_pos]
            )

            attn_angle, mlp_angle, total_angle = compute_rotation_decomposition(
                diff_resid_pre, diff_attn, diff_mlp, diff_vec
            )

        layer_results.append(
            DiffMeansLayerResult(
                layer=layer,
                cosine_to_next=cos_next,
                cosine_to_prev=cos_prev,
                diff_norm=float(np.linalg.norm(diff_vec)),
                attn_rotation_angle=attn_angle,
                mlp_rotation_angle=mlp_angle,
                total_rotation_angle=total_angle,
                cosine_to_logit=cosine_to_logit,
                cosine_to_initial=cosine_to_initial,
                attn_out_diff_norm=attn_out_diff_norm,
                mlp_out_diff_norm=mlp_out_diff_norm,
            )
        )

    return layer_results


def _build_position_layer_results(
    n_layers: int,
    clean_acts: dict[str, dict[int, np.ndarray]],
    corrupted_acts: dict[str, dict[int, np.ndarray]],
    pos: int,
) -> list[DiffMeansLayerResult]:
    """Build simplified layer results for a specific position (just diff norms)."""
    layer_results = []

    for layer in range(n_layers):
        if layer not in clean_acts.get("resid_post", {}):
            continue
        if layer not in corrupted_acts.get("resid_post", {}):
            continue

        clean_vec = clean_acts["resid_post"][layer][pos]
        corrupted_vec = corrupted_acts["resid_post"][layer][pos]
        diff_vec = clean_vec - corrupted_vec

        layer_results.append(
            DiffMeansLayerResult(
                layer=layer,
                diff_norm=float(np.linalg.norm(diff_vec)),
            )
        )

    return layer_results


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
