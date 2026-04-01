"""Main analysis function for difference-in-means."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from ....common.math import cosine_similarity
from ....inference.inference_utils import get_all_activations
from ...common.semantic_positions import ALL_TRAJECTORY_POSITIONS
from .diffmeans_results import DiffMeansLayerResult, DiffMeansPairResult, PositionPair, ResolvedPositions
from .diffmeans_rotation import (
    compute_layer_direction_similarity,
    compute_rotation_decomposition,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair
    from ...common.sample_position_mapping import SamplePositionMapping


def run_diffmeans_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    clean_mapping: "SamplePositionMapping",
    corrupted_mapping: "SamplePositionMapping",
    pair_idx: int = 0,
    positions: list[str] | None = None,
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
        clean_mapping: SamplePositionMapping for clean trajectory
        corrupted_mapping: SamplePositionMapping for corrupted trajectory
        pair_idx: Pair index for tracking
        positions: Semantic position names to analyze (default: ALL_TRAJECTORY_POSITIONS)

    Returns:
        DiffMeansPairResult with per-layer analysis
    """
    if positions is None:
        positions = list(ALL_TRAJECTORY_POSITIONS)

    n_layers = runner.n_layers
    log(f"[diffmeans] Model has {n_layers} layers")

    # Get activations for clean and corrupted
    clean_acts = get_all_activations(runner, pair.clean_traj.token_ids)
    corrupted_acts = get_all_activations(runner, pair.corrupted_traj.token_ids)

    # Debug: log which layers were captured
    if clean_acts.get("resid_post"):
        captured_layers = sorted(clean_acts["resid_post"].keys())
        log(f"[diffmeans] Captured layers: {min(captured_layers)} to {max(captured_layers)} ({len(captured_layers)} total)")

    # Resolve semantic positions to absolute positions
    resolved_positions = _resolve_positions(
        positions, clean_mapping, corrupted_mapping, pair
    )

    if not resolved_positions:
        raise ValueError(
            f"No positions resolved for pair {pair_idx}. "
            f"Requested: {positions}. "
            f"clean_mapping has: {list(clean_mapping.named_positions.keys())}. "
            f"corrupted_mapping has: {list(corrupted_mapping.named_positions.keys())}."
        )

    # Get logit direction (W_U[clean_choice] - W_U[corrupted_choice])
    logit_direction = _compute_logit_direction(runner, pair)

    clean_len = len(pair.clean_traj.token_ids)
    corrupted_len = len(pair.corrupted_traj.token_ids)

    # Build full layer results for each format_pos
    # Generate both per-rel_pos results (e.g., "time_horizon:0") and combined (e.g., "time_horizon")
    position_results: dict[str, list[DiffMeansLayerResult]] = {}

    for format_pos, pos_pairs in resolved_positions.items():
        # Collect diff vectors for each rel_pos
        per_rel_pos_diffs: list[tuple[int, dict, dict, dict]] = []  # (rel_pos, resid, attn, mlp)

        for rel_pos, pos_pair in enumerate(pos_pairs):
            clean_pos = pos_pair.clean_pos
            corrupted_pos = pos_pair.corrupted_pos

            if clean_pos < 0 or clean_pos >= clean_len:
                continue
            if corrupted_pos < 0 or corrupted_pos >= corrupted_len:
                continue

            # Compute difference vectors at this position
            diff_resid_post = {}
            diff_attn_out = {}
            diff_mlp_out = {}

            for layer in range(n_layers):
                if layer in clean_acts.get("resid_post", {}) and layer in corrupted_acts.get("resid_post", {}):
                    clean_vec = clean_acts["resid_post"][layer][clean_pos]
                    corrupted_vec = corrupted_acts["resid_post"][layer][corrupted_pos]
                    diff_resid_post[layer] = clean_vec - corrupted_vec

                if layer in clean_acts.get("attn_out", {}) and layer in corrupted_acts.get("attn_out", {}):
                    diff_attn_out[layer] = (
                        clean_acts["attn_out"][layer][clean_pos]
                        - corrupted_acts["attn_out"][layer][corrupted_pos]
                    )

                if layer in clean_acts.get("mlp_out", {}) and layer in corrupted_acts.get("mlp_out", {}):
                    diff_mlp_out[layer] = (
                        clean_acts["mlp_out"][layer][clean_pos]
                        - corrupted_acts["mlp_out"][layer][corrupted_pos]
                    )

            if diff_resid_post:
                per_rel_pos_diffs.append((rel_pos, diff_resid_post, diff_attn_out, diff_mlp_out))

        if not per_rel_pos_diffs:
            continue

        # Generate per-rel_pos results (e.g., "time_horizon:0", "time_horizon:1")
        for rel_pos, diff_resid_post, diff_attn_out, diff_mlp_out in per_rel_pos_diffs:
            initial_layer = min(diff_resid_post.keys())
            initial_direction = diff_resid_post[initial_layer]
            layer_similarities = compute_layer_direction_similarity(diff_resid_post)

            # Use first position pair for clean/corrupted pos (for component extraction)
            clean_pos = pos_pairs[rel_pos].clean_pos
            corrupted_pos = pos_pairs[rel_pos].corrupted_pos

            layer_results = _build_layer_results(
                n_layers=n_layers,
                diff_resid_post=diff_resid_post,
                diff_attn_out=diff_attn_out,
                diff_mlp_out=diff_mlp_out,
                clean_acts=clean_acts,
                corrupted_acts=corrupted_acts,
                clean_pos=clean_pos,
                corrupted_pos=corrupted_pos,
                layer_similarities=layer_similarities,
                logit_direction=logit_direction,
                initial_direction=initial_direction,
            )
            position_results[f"{format_pos}:{rel_pos}"] = layer_results

        # Generate combined result by averaging diff vectors across all rel_pos
        combined_resid = {}
        combined_attn = {}
        combined_mlp = {}

        def _to_tensor(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v)
            return v

        for layer in range(n_layers):
            resid_vecs = [_to_tensor(d[1][layer]) for d in per_rel_pos_diffs if layer in d[1]]
            attn_vecs = [_to_tensor(d[2][layer]) for d in per_rel_pos_diffs if layer in d[2]]
            mlp_vecs = [_to_tensor(d[3][layer]) for d in per_rel_pos_diffs if layer in d[3]]

            if resid_vecs:
                combined_resid[layer] = torch.stack(resid_vecs).mean(dim=0)
            if attn_vecs:
                combined_attn[layer] = torch.stack(attn_vecs).mean(dim=0)
            if mlp_vecs:
                combined_mlp[layer] = torch.stack(mlp_vecs).mean(dim=0)

        if combined_resid:
            initial_layer = min(combined_resid.keys())
            initial_direction = combined_resid[initial_layer]
            layer_similarities = compute_layer_direction_similarity(combined_resid)

            # Use first position for clean/corrupted (doesn't matter much for combined)
            clean_pos = pos_pairs[0].clean_pos
            corrupted_pos = pos_pairs[0].corrupted_pos

            layer_results = _build_layer_results(
                n_layers=n_layers,
                diff_resid_post=combined_resid,
                diff_attn_out=combined_attn,
                diff_mlp_out=combined_mlp,
                clean_acts=clean_acts,
                corrupted_acts=corrupted_acts,
                clean_pos=clean_pos,
                corrupted_pos=corrupted_pos,
                layer_similarities=layer_similarities,
                logit_direction=logit_direction,
                initial_direction=initial_direction,
            )
            position_results[format_pos] = layer_results

    return DiffMeansPairResult(
        pair_idx=pair_idx,
        position_results=position_results,
    )


def _resolve_positions(
    positions: list[str],
    clean_mapping: "SamplePositionMapping",
    corrupted_mapping: "SamplePositionMapping",
    pair: "ContrastivePair",
) -> ResolvedPositions:
    """Resolve semantic position names to (clean_pos, corrupted_pos) pairs.

    Args:
        positions: Semantic position names (e.g., "time_horizon", "response_choice")
        clean_mapping: SamplePositionMapping for clean trajectory
        corrupted_mapping: SamplePositionMapping for corrupted trajectory
        pair: ContrastivePair with position_mapping

    Returns:
        ResolvedPositions mapping format_pos name -> list of PositionPair

    Raises:
        ValueError: If clean_mapping or corrupted_mapping is None
    """
    if clean_mapping is None:
        raise ValueError("clean_mapping is required")
    if corrupted_mapping is None:
        raise ValueError("corrupted_mapping is required")

    result: dict[str, list[PositionPair]] = {}

    for format_pos in positions:
        clean_positions = clean_mapping.named_positions.get(format_pos, [])
        corrupted_positions = corrupted_mapping.named_positions.get(format_pos, [])

        if not clean_positions or not corrupted_positions:
            continue

        # Pair up positions: use min length, or map using PairPositionMapping
        pairs = []
        for i, clean_pos in enumerate(clean_positions):
            if i < len(corrupted_positions):
                corrupted_pos = corrupted_positions[i]
            else:
                # Use PairPositionMapping to find corresponding corrupted position
                corrupted_pos = pair.position_mapping.src_to_dst(clean_pos, clean_pos)
            pairs.append(PositionPair(clean_pos=clean_pos, corrupted_pos=corrupted_pos))

        if pairs:
            result[format_pos] = pairs

    return ResolvedPositions(positions=result)


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
    clean_pos: int,
    corrupted_pos: int,
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
                clean_acts["resid_pre"][layer][clean_pos]
                - corrupted_acts["resid_pre"][layer][corrupted_pos]
            )
            diff_attn = (
                clean_acts["attn_out"][layer][clean_pos]
                - corrupted_acts["attn_out"][layer][corrupted_pos]
            )
            diff_mlp = (
                clean_acts["mlp_out"][layer][clean_pos]
                - corrupted_acts["mlp_out"][layer][corrupted_pos]
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
