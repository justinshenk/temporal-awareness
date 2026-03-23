"""Main analysis function for MLP neuron analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log

from .mlp_analysis_results import (
    MLPNeuronLayerResult,
    MLPPairResult,
    NeuronInfo,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


def run_mlp_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    pair_idx: int = 0,
    layers: list[int] | None = None,
    position: int | None = None,
    n_top_neurons: int = 50,
) -> MLPPairResult:
    """Run MLP neuron analysis for a single pair.

    Computes:
    1. Per-neuron activation differences (clean - corrupted)
    2. Per-neuron logit contributions (activation_diff * W_out @ logit_direction)
    3. Sparsity metrics (how many neurons explain the effect)
    4. Max-activating prompt tracking

    Args:
        runner: Model runner with access to W_U and model internals
        pair: Contrastive pair
        pair_idx: Pair index for tracking
        layers: Layers to analyze (default: [35, 31, 28, 19])
        position: Position to analyze (None = last token / P_dest)
        n_top_neurons: Number of top neurons to track per layer

    Returns:
        MLPPairResult with per-layer neuron analysis
    """
    if layers is None:
        layers = [35, 31, 28, 19]

    # Determine position to analyze
    if position is None:
        position = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1

    # Get logit direction
    logit_direction = _compute_logit_direction(runner, pair)
    if logit_direction is None:
        log(f"[mlp] Warning: logit_direction is None for pair {pair_idx}")
        return MLPPairResult(pair_idx=pair_idx, position=position, layer_results=[])

    # Try to get per-neuron activations first (requires mlp.hook_post)
    neuron_clean_acts, neuron_corrupted_acts = _get_mlp_neuron_activations(
        runner, pair, layers, position
    )

    # Also get MLP output activations for fallback
    clean_acts, corrupted_acts = _get_mlp_hidden_activations(runner, pair, layers, position)

    layer_results = []
    neuron_activations = {}

    for layer in layers:
        # Check if we have per-neuron activations
        has_neuron_acts = (
            layer in neuron_clean_acts
            and layer in neuron_corrupted_acts
        )

        # Try to get W_out for neuron-level analysis
        W_out = _get_mlp_w_out(runner, layer) if has_neuron_acts else None

        if has_neuron_acts and W_out is not None:
            # Full per-neuron analysis
            layer_result = _analyze_layer_neurons(
                layer=layer,
                clean_h=neuron_clean_acts[layer],
                corrupted_h=neuron_corrupted_acts[layer],
                W_out=W_out,
                logit_direction=logit_direction,
                n_top=n_top_neurons,
            )
            layer_results.append(layer_result)

            # Track top neuron activations
            for ni in layer_result.top_neurons[:10]:
                neuron_activations[f"{layer}_{ni.neuron_idx}"] = ni.logit_contribution

        elif layer in clean_acts and layer in corrupted_acts:
            # Fallback: aggregate MLP output analysis only
            clean_h = clean_acts[layer]
            corrupted_h = corrupted_acts[layer]
            mlp_diff = (clean_h - corrupted_h).detach()
            logit_contribution = float(torch.dot(mlp_diff, logit_direction).item())

            layer_result = MLPNeuronLayerResult(
                layer=layer,
                n_neurons=0,
                top_neurons=[],
                total_logit_contribution=logit_contribution,
                top_k_contribution_frac=0.0,
                sparsity_ratio=0.0,
                n_positive_contributors=1 if logit_contribution > 0 else 0,
                n_negative_contributors=1 if logit_contribution < 0 else 0,
            )
            layer_results.append(layer_result)
            neuron_activations[f"{layer}_mlp_out"] = logit_contribution

    return MLPPairResult(
        pair_idx=pair_idx,
        position=position,
        layer_results=layer_results,
        neuron_activations=neuron_activations,
    )


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


def _get_mlp_hidden_activations(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    layers: list[int],
    position: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Get MLP output activations for both trajectories.

    Uses the mlp_out hook which captures the MLP layer output [batch, seq, d_model].
    Works with all backends (HuggingFace, TransformerLens, etc.).

    Returns:
        (clean_acts, corrupted_acts) where each is {layer: activations[d_model]}
    """
    # Build hook filter for mlp_out (standard hook name across backends)
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.hook_mlp_out")

    names_filter = lambda name: name in hooks

    # Run clean
    clean_input = torch.tensor([pair.clean_traj.token_ids], device=runner.device)
    with torch.no_grad():
        _, clean_cache = runner._backend.run_with_cache(clean_input, names_filter=names_filter)

    # Run corrupted
    corrupted_input = torch.tensor([pair.corrupted_traj.token_ids], device=runner.device)
    with torch.no_grad():
        _, corrupted_cache = runner._backend.run_with_cache(corrupted_input, names_filter=names_filter)

    # Extract activations at position
    clean_acts = {}
    corrupted_acts = {}

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_mlp_out"

        if hook_name in clean_cache:
            # [batch, seq, d_model] -> [d_model] at position
            clean_h = clean_cache[hook_name][0, position, :]
            clean_acts[layer] = clean_h

        if hook_name in corrupted_cache:
            # Map position if needed (corrupted may have different length)
            corr_pos = pair.position_mapping.get(position, position)
            corr_len = corrupted_cache[hook_name].shape[1]
            corr_pos = max(0, min(int(corr_pos), corr_len - 1))
            corrupted_h = corrupted_cache[hook_name][0, corr_pos, :]
            corrupted_acts[layer] = corrupted_h

    return clean_acts, corrupted_acts


def _get_mlp_neuron_activations(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    layers: list[int],
    position: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Get per-neuron MLP activations (after activation function) for both trajectories.

    Uses mlp.hook_post which captures neuron activations [batch, seq, d_mlp].

    Returns:
        (clean_acts, corrupted_acts) where each is {layer: activations[d_mlp]}
    """
    # Build hook filter for mlp.hook_post (neuron activations)
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.mlp.hook_post")

    names_filter = lambda name: name in hooks

    # Run clean
    clean_input = torch.tensor([pair.clean_traj.token_ids], device=runner.device)
    with torch.no_grad():
        _, clean_cache = runner._backend.run_with_cache(clean_input, names_filter=names_filter)

    # Run corrupted
    corrupted_input = torch.tensor([pair.corrupted_traj.token_ids], device=runner.device)
    with torch.no_grad():
        _, corrupted_cache = runner._backend.run_with_cache(corrupted_input, names_filter=names_filter)

    # Extract activations at position
    clean_acts = {}
    corrupted_acts = {}

    for layer in layers:
        hook_name = f"blocks.{layer}.mlp.hook_post"

        if hook_name in clean_cache:
            # [batch, seq, d_mlp] -> [d_mlp] at position
            clean_h = clean_cache[hook_name][0, position, :]
            clean_acts[layer] = clean_h

        if hook_name in corrupted_cache:
            # Map position if needed (corrupted may have different length)
            corr_pos = pair.position_mapping.get(position, position)
            corr_len = corrupted_cache[hook_name].shape[1]
            corr_pos = max(0, min(int(corr_pos), corr_len - 1))
            corrupted_h = corrupted_cache[hook_name][0, corr_pos, :]
            corrupted_acts[layer] = corrupted_h

    return clean_acts, corrupted_acts


def _get_mlp_w_out(runner: "BinaryChoiceRunner", layer: int) -> torch.Tensor | None:
    """Get W_out matrix [d_mlp, d_model] for a layer.

    For TransformerLens models: blocks[layer].mlp.W_out
    For HuggingFace models: model.layers[layer].mlp.down_proj.weight
    """
    # Try TransformerLens-style first
    try:
        if hasattr(runner._model, "blocks"):
            return runner._model.blocks[layer].mlp.W_out.detach()
    except (AttributeError, IndexError):
        pass

    # Try HuggingFace-style (Qwen, Llama, etc.)
    try:
        if hasattr(runner._model, "model") and hasattr(runner._model.model, "layers"):
            mlp = runner._model.model.layers[layer].mlp
            if hasattr(mlp, "down_proj"):
                # down_proj.weight is [d_model, d_mlp], we need [d_mlp, d_model]
                return mlp.down_proj.weight.T.detach()
    except (AttributeError, IndexError):
        pass

    return None


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
