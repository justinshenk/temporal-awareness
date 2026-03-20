"""MLP neuron-level analysis: identify important neurons in MLP layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...common.contrastive_pair import ContrastivePair
from ...common.device_utils import clear_gpu_memory
from ...common.hook_utils import hook_name
from ...common.profiler import profile
from ...common.patching_types import PatchingMode

from .fine_config import FineConfig
from .fine_results import NeuronResult, MLPNeuronResults

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner


def _get_mlp_hook_filter(layers: list[int]) -> callable:
    """Create filter for MLP-related hooks at specific layers."""
    hooks = set()
    for layer in layers:
        # hook_pre: MLP input (after layer norm)
        hooks.add(f"blocks.{layer}.mlp.hook_pre")
        # hook_post: After activation function (neuron activations)
        hooks.add(f"blocks.{layer}.mlp.hook_post")
        # mlp_out: Final MLP output
        hooks.add(hook_name(layer, "mlp_out"))
    return lambda name: name in hooks


@profile
def run_mlp_neuron_analysis(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    mode: PatchingMode,
    config: FineConfig | None = None,
) -> dict[int, MLPNeuronResults]:
    """Analyze MLP neurons to find most important contributors.

    For each MLP layer:
    1. Get mlp_out_clean - mlp_out_corrupted at the metric position
    2. Decompose into individual neuron contributions
    3. Rank neurons by contribution magnitude

    Neuron contribution = activation_diff * W_out_row
    where activation_diff = neuron_activation_clean - neuron_activation_corrupted

    Args:
        runner: Model runner
        pair: Contrastive pair
        mode: "denoising" or "noising"
        config: Fine patching configuration

    Returns:
        Dict mapping layer -> MLPNeuronResults
    """
    if config is None:
        from .fine_config import DEFAULT_FINE_CONFIG
        config = DEFAULT_FINE_CONFIG

    results: dict[int, MLPNeuronResults] = {}

    layers = config.mlp_layers
    names_filter = _get_mlp_hook_filter(layers)

    # Get divergent position
    clean_div_pos, corrupted_div_pos = pair.choice_divergent_positions
    metric_pos = corrupted_div_pos - 1 if mode == "denoising" else clean_div_pos - 1

    # Run clean trajectory with cache
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    clean_cache = clean_choice.cache

    # Run corrupted trajectory with cache
    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        pair.corrupted_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    corrupted_cache = corrupted_choice.cache

    # Get logit direction for projection
    W_U = runner.W_U
    labels = pair.clean_labels
    label_a_id = runner.encode_ids(labels[0], add_special_tokens=False)[0]
    label_b_id = runner.encode_ids(labels[1], add_special_tokens=False)[0]
    logit_direction = W_U[:, label_a_id] - W_U[:, label_b_id]
    logit_direction = logit_direction / torch.norm(logit_direction)

    for layer in layers:
        # Try to get neuron activations directly
        hook_post_name = f"blocks.{layer}.mlp.hook_post"
        clean_post = clean_cache.get(hook_post_name)
        corrupted_post = corrupted_cache.get(hook_post_name)

        if clean_post is None or corrupted_post is None:
            # Fallback: use mlp_out difference directly
            results[layer] = _analyze_from_mlp_out(
                clean_cache, corrupted_cache, layer, metric_pos, logit_direction
            )
            continue

        # Get neuron activations at metric position
        # hook_post: [batch, pos, d_mlp]
        clean_acts = clean_post[0, metric_pos, :]  # [d_mlp]
        corrupted_acts = corrupted_post[0, metric_pos, :]  # [d_mlp]
        act_diff = clean_acts - corrupted_acts  # [d_mlp]

        d_mlp = act_diff.shape[0]

        # Get W_out matrix from model
        # W_out: [d_mlp, d_model] - transforms neuron activations to output
        W_out = _get_mlp_w_out(runner, layer)

        if W_out is None:
            results[layer] = _analyze_from_mlp_out(
                clean_cache, corrupted_cache, layer, metric_pos, logit_direction
            )
            continue

        # Compute per-neuron contributions
        # contribution[i] = act_diff[i] * W_out[i, :]
        # Project each contribution onto logit direction
        neuron_results = []
        for neuron_idx in range(d_mlp):
            neuron_contribution = act_diff[neuron_idx] * W_out[neuron_idx, :]  # [d_model]
            logit_proj = torch.dot(neuron_contribution, logit_direction).item()
            contribution_norm = torch.norm(neuron_contribution).item()

            neuron_results.append(NeuronResult(
                layer=layer,
                neuron_idx=neuron_idx,
                contribution=contribution_norm,
                activation_diff=act_diff[neuron_idx].item(),
                logit_projection=logit_proj,
            ))

        results[layer] = MLPNeuronResults(
            layer=layer,
            n_neurons=d_mlp,
            neuron_results=neuron_results,
        )

    # Clean up
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    clear_gpu_memory()

    return results


def _get_mlp_w_out(runner: "BinaryChoiceRunner", layer: int) -> torch.Tensor | None:
    """Get the MLP output projection matrix W_out for a layer."""
    try:
        # TransformerLens stores this as W_out in the MLP block
        if hasattr(runner._model, "blocks"):
            return runner._model.blocks[layer].mlp.W_out.detach()
    except (AttributeError, IndexError):
        pass
    return None


def _analyze_from_mlp_out(
    clean_cache: dict,
    corrupted_cache: dict,
    layer: int,
    metric_pos: int,
    logit_direction: torch.Tensor,
) -> MLPNeuronResults:
    """Fallback analysis using mlp_out difference.

    When per-neuron hooks aren't available, we can still compute
    the total MLP contribution and project onto logit direction.
    """
    mlp_out_name = hook_name(layer, "mlp_out")
    clean_out = clean_cache.get(mlp_out_name)
    corrupted_out = corrupted_cache.get(mlp_out_name)

    if clean_out is None or corrupted_out is None:
        return MLPNeuronResults(layer=layer, n_neurons=0, neuron_results=[])

    # Get mlp_out difference at metric position
    clean_at_pos = clean_out[0, metric_pos, :]  # [d_model]
    corrupted_at_pos = corrupted_out[0, metric_pos, :]  # [d_model]
    diff = clean_at_pos - corrupted_at_pos

    # Total contribution (not per-neuron)
    total_contribution = torch.norm(diff).item()
    total_logit_proj = torch.dot(diff, logit_direction).item()

    # Return single "aggregate" neuron representing total MLP contribution
    return MLPNeuronResults(
        layer=layer,
        n_neurons=1,
        neuron_results=[
            NeuronResult(
                layer=layer,
                neuron_idx=-1,  # -1 indicates aggregate
                contribution=total_contribution,
                activation_diff=0.0,
                logit_projection=total_logit_proj,
            )
        ],
    )
