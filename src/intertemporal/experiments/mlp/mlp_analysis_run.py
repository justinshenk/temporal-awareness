"""Main analysis function for MLP neuron analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from ....inference.inference_utils import get_mlp_neuron_activations, get_mlp_w_out
from ...common.semantic_positions import ALL_TRAJECTORY_POSITIONS

from .mlp_analysis_results import (
    MLPNeuronLayerResult,
    MLPPairResult,
    NeuronInfo,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair
    from ...common.sample_position_mapping import SamplePositionMapping


def run_mlp_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    clean_mapping: "SamplePositionMapping",
    corrupted_mapping: "SamplePositionMapping",
    pair_idx: int = 0,
    layers: list[int] | None = None,
    positions: list[str] | None = None,
    n_top_neurons: int = 50,
) -> MLPPairResult:
    """Run MLP neuron analysis for a single pair at all semantic positions.

    Computes per-position:
    1. Per-neuron activation differences (clean - corrupted)
    2. Per-neuron logit contributions (activation_diff * W_out @ logit_direction)
    3. Sparsity metrics (how many neurons explain the effect)
    4. Max-activating prompt tracking

    Args:
        runner: Model runner with access to W_U and model internals
        pair: Contrastive pair
        clean_mapping: SamplePositionMapping for clean trajectory
        corrupted_mapping: SamplePositionMapping for corrupted trajectory
        pair_idx: Pair index for tracking
        layers: Layers to analyze (default: [35, 31, 28, 19])
        positions: Semantic position names to analyze (default: ALL_TRAJECTORY_POSITIONS)
        n_top_neurons: Number of top neurons to track per layer

    Returns:
        MLPPairResult with per-position, per-layer neuron analysis
    """
    if layers is None:
        layers = [35, 31, 28, 19]
    if positions is None:
        positions = list(ALL_TRAJECTORY_POSITIONS)

    # Get logit direction
    logit_direction = _compute_logit_direction(runner, pair)
    if logit_direction is None:
        log(f"[mlp] Warning: logit_direction is None for pair {pair_idx}")
        return MLPPairResult(pair_idx=pair_idx, position_results={})

    # Get full sequence MLP neuron activations for both trajectories
    clean_neuron_acts = get_mlp_neuron_activations(
        runner, pair.clean_traj.token_ids, layers
    )
    corrupted_neuron_acts = get_mlp_neuron_activations(
        runner, pair.corrupted_traj.token_ids, layers
    )

    # Pre-fetch W_out matrices for each layer
    w_out_by_layer: dict[int, np.ndarray | None] = {}
    for layer in layers:
        w_out_by_layer[layer] = get_mlp_w_out(runner, layer)

    # Resolve semantic positions to (clean_pos, corrupted_pos) pairs
    resolved_positions = _resolve_positions(
        positions, clean_mapping, corrupted_mapping, pair
    )

    if not resolved_positions:
        log(
            f"[mlp] Warning: No positions resolved for pair {pair_idx}. "
            f"Requested: {positions}. "
            f"clean_mapping has: {list(clean_mapping.named_positions.keys())}."
        )
        return MLPPairResult(pair_idx=pair_idx, position_results={})

    clean_len = len(pair.clean_traj.token_ids)
    corrupted_len = len(pair.corrupted_traj.token_ids)

    position_results: dict[str, list[MLPNeuronLayerResult]] = {}
    neuron_activations: dict[str, float] = {}

    for format_pos, pos_pairs in resolved_positions.items():
        # Collect layer results for each rel_pos
        per_rel_pos_results: list[tuple[int, list[MLPNeuronLayerResult]]] = []

        for rel_pos, (clean_pos, corrupted_pos) in enumerate(pos_pairs):
            if clean_pos < 0 or clean_pos >= clean_len:
                continue
            if corrupted_pos < 0 or corrupted_pos >= corrupted_len:
                continue

            layer_results = []

            for layer in layers:
                if layer not in clean_neuron_acts or layer not in corrupted_neuron_acts:
                    continue

                W_out = w_out_by_layer.get(layer)
                if W_out is None:
                    continue

                # Get activations at this position
                # clean_neuron_acts[layer] is [seq_len, d_mlp]
                clean_h = clean_neuron_acts[layer][clean_pos]
                corrupted_h = corrupted_neuron_acts[layer][corrupted_pos]

                # Analyze this layer
                layer_result = _analyze_layer_neurons(
                    layer=layer,
                    clean_h=torch.from_numpy(clean_h),
                    corrupted_h=torch.from_numpy(corrupted_h),
                    W_out=torch.from_numpy(W_out),
                    logit_direction=logit_direction,
                    n_top=n_top_neurons,
                )
                layer_results.append(layer_result)

            if layer_results:
                per_rel_pos_results.append((rel_pos, layer_results))

        if not per_rel_pos_results:
            continue

        # Store per-rel_pos results (e.g., "time_horizon:0", "time_horizon:1")
        for rel_pos, layer_results in per_rel_pos_results:
            position_results[f"{format_pos}:{rel_pos}"] = layer_results

            # Track top neuron activations for interpretability
            for lr in layer_results:
                for ni in lr.top_neurons[:5]:
                    key = f"{format_pos}:{rel_pos}_{lr.layer}_{ni.neuron_idx}"
                    neuron_activations[key] = ni.logit_contribution

        # Compute combined result by averaging across all rel_pos
        if len(per_rel_pos_results) > 0:
            combined_layer_results = _combine_layer_results(
                [lr_list for _, lr_list in per_rel_pos_results], layers, n_top_neurons
            )
            if combined_layer_results:
                position_results[format_pos] = combined_layer_results

    return MLPPairResult(
        pair_idx=pair_idx,
        position_results=position_results,
        neuron_activations=neuron_activations,
    )


def _resolve_positions(
    positions: list[str],
    clean_mapping: "SamplePositionMapping",
    corrupted_mapping: "SamplePositionMapping",
    pair: "ContrastivePair",
) -> dict[str, list[tuple[int, int]]]:
    """Resolve semantic position names to (clean_pos, corrupted_pos) pairs.

    Args:
        positions: Semantic position names (e.g., "time_horizon", "response_choice")
        clean_mapping: SamplePositionMapping for clean trajectory
        corrupted_mapping: SamplePositionMapping for corrupted trajectory
        pair: ContrastivePair with position_mapping

    Returns:
        Dict mapping format_pos name -> list of (clean_pos, corrupted_pos) tuples
    """
    if clean_mapping is None or corrupted_mapping is None:
        return {}

    result: dict[str, list[tuple[int, int]]] = {}

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
            pairs.append((clean_pos, corrupted_pos))

        if pairs:
            result[format_pos] = pairs

    return result


def _compute_logit_direction(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> torch.Tensor | None:
    """Compute normalized logit direction (clean_token - corrupted_token in W_U space)."""
    W_U = runner.W_U
    if W_U is None:
        log("[mlp] Warning: W_U is None - cannot compute logit direction")
        return None

    # Get the divergent tokens
    clean_div_pos = pair.clean_divergent_position
    corrupted_div_pos = pair.corrupted_divergent_position

    if clean_div_pos is None or corrupted_div_pos is None:
        clean_token = pair.clean_traj.token_ids[-1]
        corrupted_token = pair.corrupted_traj.token_ids[-1]
    else:
        clean_token = pair.clean_traj.token_ids[clean_div_pos]
        corrupted_token = pair.corrupted_traj.token_ids[corrupted_div_pos]

    if clean_token == corrupted_token:
        log(f"[mlp] Warning: clean_token == corrupted_token ({clean_token})")
        return None

    # Extract from W_U (handle both shapes)
    if W_U.shape[0] > W_U.shape[1]:
        # [vocab_size, d_model]
        clean_vec = W_U[clean_token]
        corrupted_vec = W_U[corrupted_token]
    else:
        # [d_model, vocab_size]
        clean_vec = W_U[:, clean_token]
        corrupted_vec = W_U[:, corrupted_token]

    direction = clean_vec - corrupted_vec
    return direction / torch.norm(direction)


def _analyze_layer_neurons(
    layer: int,
    clean_h: torch.Tensor,
    corrupted_h: torch.Tensor,
    W_out: torch.Tensor,
    logit_direction: torch.Tensor,
    n_top: int,
) -> MLPNeuronLayerResult:
    """Analyze neurons for a single layer.

    Args:
        layer: Layer index
        clean_h: Clean hidden activations [d_mlp]
        corrupted_h: Corrupted hidden activations [d_mlp]
        W_out: Output projection matrix [d_mlp, d_model]
        logit_direction: Normalized logit direction [d_model]
        n_top: Number of top neurons to return

    Returns:
        MLPNeuronLayerResult with neuron analysis
    """
    # Move all tensors to the same device as logit_direction
    device = logit_direction.device
    clean_h = clean_h.to(device)
    corrupted_h = corrupted_h.to(device)
    W_out = W_out.to(device)

    d_mlp = clean_h.shape[0]
    act_diff = clean_h - corrupted_h  # [d_mlp]

    # Compute per-neuron logit contributions
    # contribution[n] = act_diff[n] * (W_out[n] @ logit_direction)
    w_out_projections = W_out @ logit_direction  # [d_mlp]
    logit_contributions = act_diff * w_out_projections  # [d_mlp]

    # Convert to numpy for analysis (detach in case grad is required)
    act_diff_np = act_diff.detach().cpu().numpy()
    clean_h_np = clean_h.detach().cpu().numpy()
    corrupted_h_np = corrupted_h.detach().cpu().numpy()
    logit_contrib_np = logit_contributions.detach().cpu().numpy()
    w_out_proj_np = w_out_projections.detach().cpu().numpy()

    # Find top neurons by absolute contribution
    top_indices = np.argsort(np.abs(logit_contrib_np))[::-1][:n_top]

    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append(NeuronInfo(
            neuron_idx=idx,
            activation_diff=float(act_diff_np[idx]),
            clean_activation=float(clean_h_np[idx]),
            corrupted_activation=float(corrupted_h_np[idx]),
            logit_contribution=float(logit_contrib_np[idx]),
            w_out_logit_alignment=float(w_out_proj_np[idx]),
        ))

    # Compute aggregate statistics
    total_contrib = float(np.sum(np.abs(logit_contrib_np)))
    top_k_contrib = float(np.sum(np.abs(logit_contrib_np[top_indices])))
    sparsity_ratio = top_k_contrib / total_contrib if total_contrib > 0 else 0.0

    n_positive = int(np.sum(logit_contrib_np > 0))
    n_negative = int(np.sum(logit_contrib_np < 0))

    return MLPNeuronLayerResult(
        layer=layer,
        n_neurons=d_mlp,
        top_neurons=top_neurons,
        total_logit_contribution=total_contrib,
        top_k_contribution_frac=sparsity_ratio,
        sparsity_ratio=sparsity_ratio,
        n_positive_contributors=n_positive,
        n_negative_contributors=n_negative,
    )


def _combine_layer_results(
    per_rel_pos_results: list[list[MLPNeuronLayerResult]],
    layers: list[int],
    n_top: int,
) -> list[MLPNeuronLayerResult]:
    """Combine layer results from multiple rel_pos by averaging.

    Args:
        per_rel_pos_results: List of layer result lists, one per rel_pos
        layers: Layers that were analyzed
        n_top: Number of top neurons to keep

    Returns:
        Combined layer results with averaged statistics
    """
    if not per_rel_pos_results:
        return []

    combined_results = []

    for layer in layers:
        # Collect layer results for this layer across all rel_pos
        layer_results_for_layer = []
        for lr_list in per_rel_pos_results:
            for lr in lr_list:
                if lr.layer == layer:
                    layer_results_for_layer.append(lr)
                    break

        if not layer_results_for_layer:
            continue

        # Average the aggregate statistics
        mean_total_contrib = np.mean([lr.total_logit_contribution for lr in layer_results_for_layer])
        mean_sparsity = np.mean([lr.sparsity_ratio for lr in layer_results_for_layer])
        mean_n_positive = int(np.mean([lr.n_positive_contributors for lr in layer_results_for_layer]))
        mean_n_negative = int(np.mean([lr.n_negative_contributors for lr in layer_results_for_layer]))

        # Aggregate top neurons: collect all and re-rank by mean contribution
        neuron_data: dict[int, list[NeuronInfo]] = {}
        for lr in layer_results_for_layer:
            for ni in lr.top_neurons:
                if ni.neuron_idx not in neuron_data:
                    neuron_data[ni.neuron_idx] = []
                neuron_data[ni.neuron_idx].append(ni)

        # Compute mean NeuronInfo for each neuron
        combined_neurons = []
        for neuron_idx, infos in neuron_data.items():
            combined_neurons.append(NeuronInfo(
                neuron_idx=neuron_idx,
                activation_diff=float(np.mean([ni.activation_diff for ni in infos])),
                clean_activation=float(np.mean([ni.clean_activation for ni in infos])),
                corrupted_activation=float(np.mean([ni.corrupted_activation for ni in infos])),
                logit_contribution=float(np.mean([ni.logit_contribution for ni in infos])),
                w_out_logit_alignment=float(np.mean([ni.w_out_logit_alignment for ni in infos])),
            ))

        # Sort by absolute contribution and take top n
        combined_neurons.sort(key=lambda x: abs(x.logit_contribution), reverse=True)
        top_neurons = combined_neurons[:n_top]

        n_neurons = layer_results_for_layer[0].n_neurons if layer_results_for_layer else 0

        combined_results.append(MLPNeuronLayerResult(
            layer=layer,
            n_neurons=n_neurons,
            top_neurons=top_neurons,
            total_logit_contribution=float(mean_total_contrib),
            top_k_contribution_frac=float(mean_sparsity),
            sparsity_ratio=float(mean_sparsity),
            n_positive_contributors=mean_n_positive,
            n_negative_contributors=mean_n_negative,
        ))

    return combined_results
