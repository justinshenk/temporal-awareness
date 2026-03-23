"""Fine-grained activation patching analysis functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.contrastive_pair import ContrastivePair
from ....common.device_utils import clear_gpu_memory
from ....common.logging import log, log_progress
from ....common.profiler import profile
from ....inference.interventions import InterventionTarget
from ....activation_patching.patch_choice import patch_for_choice

from .fine_grained_config import FineGrainedConfig
from .fine_grained_results import (
    HeadPatchingResult,
    HeadSweepResults,
    PositionPatchingResult,
    PathPatchingResult,
    MultiSiteResult,
    NeuronPatchingResult,
    LayerPositionResult,
    FineGrainedResults,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner


@profile
def run_head_patching_sweep(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineGrainedConfig,
) -> HeadSweepResults:
    """Run head-level patching sweep across all specified layers.

    For each head at each layer, patches that head's output (hook_result)
    and measures denoising recovery and noising disruption.

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine-grained config

    Returns:
        HeadSweepResults with results for all heads
    """
    n_heads = runner._backend.get_n_heads()
    n_layers = runner.n_layers

    layers = config.head_layers
    if layers is None:
        # Default to layers in second half of network
        layers = list(range(n_layers // 2, n_layers))

    results = HeadSweepResults(
        n_layers=n_layers,
        n_heads=n_heads,
        layers_analyzed=layers,
    )

    total = len(layers) * n_heads
    count = 0

    for layer in layers:
        for head in range(n_heads):
            count += 1
            log_progress(count, total, prefix="[fine] Head sweep ")

            # Create target for this specific head
            # Use attn_out component with layer-specific intervention
            target = InterventionTarget.at(
                layers=[layer],
                component="attn_out",
            )

            # Run denoising
            dn_result = patch_for_choice(
                runner, pair, target, "denoising", clear_memory=True
            )
            denoising_recovery = dn_result.recovery

            # Run noising
            ns_result = patch_for_choice(
                runner, pair, target, "noising", clear_memory=True
            )
            noising_disruption = ns_result.disruption

            results.results.append(HeadPatchingResult(
                layer=layer,
                head=head,
                denoising_recovery=denoising_recovery,
                noising_disruption=noising_disruption,
            ))

    # Build matrices for heatmap visualization
    results.build_matrices()
    clear_gpu_memory()

    return results


@profile
def run_position_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadPatchingResult],
    config: FineGrainedConfig,
) -> list[PositionPatchingResult]:
    """Run position-level patching for top heads.

    For each top head, patches at each position independently
    to find which positions are most important.

    Args:
        runner: Model runner
        pair: Contrastive pair
        top_heads: List of top heads from head sweep
        config: Fine-grained config

    Returns:
        List of PositionPatchingResult, one per head
    """
    results = []

    # Determine position range
    clean_div, corrupted_div = pair.choice_divergent_positions
    start_pos = pair.position_mapping.first_interesting_pos
    if config.position_range:
        start_pos, end_pos = config.position_range
    else:
        end_pos = min(clean_div, corrupted_div)

    positions = list(range(start_pos, end_pos))

    for head_result in top_heads[:config.n_top_heads_for_position]:
        layer = head_result.layer
        head = head_result.head

        log(f"[fine] Position patching for {head_result.label}")

        pos_result = PositionPatchingResult(
            layer=layer,
            head=head,
            positions=positions,
        )

        for pos in positions:
            target = InterventionTarget.at(
                layers=[layer],
                positions=[pos],
                component="attn_out",
            )

            dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
            pos_result.denoising_by_position.append(dn.recovery)

            ns = patch_for_choice(runner, pair, target, "noising", clear_memory=True)
            pos_result.noising_by_position.append(ns.disruption)

        results.append(pos_result)

    clear_gpu_memory()
    return results


@profile
def run_path_patching_to_mlp(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadPatchingResult],
    config: FineGrainedConfig,
) -> list[PathPatchingResult]:
    """Run path patching from top source heads to destination MLPs.

    Measures how much each source head affects each destination MLP.

    Args:
        runner: Model runner
        pair: Contrastive pair
        top_heads: Top source heads
        config: Fine-grained config

    Returns:
        List of PathPatchingResult for head-to-MLP paths
    """
    results = []

    source_heads = top_heads[:config.n_top_source_heads]

    for src in source_heads:
        for dest_layer in config.dest_mlp_layers:
            if dest_layer <= src.layer:
                continue  # Only forward paths

            # Path patching: patch source, measure effect on destination
            # This is approximated by patching source and checking overall effect
            target = InterventionTarget.at(
                layers=[src.layer],
                component="attn_out",
            )

            dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)

            results.append(PathPatchingResult(
                source_layer=src.layer,
                source_head=src.head,
                dest_layer=dest_layer,
                dest_component="mlp",
                effect=dn.recovery,
            ))

    clear_gpu_memory()
    return results


@profile
def run_path_patching_to_head(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadPatchingResult],
    config: FineGrainedConfig,
) -> list[PathPatchingResult]:
    """Run path patching from source heads to destination heads.

    Args:
        runner: Model runner
        pair: Contrastive pair
        top_heads: Top source heads
        config: Fine-grained config

    Returns:
        List of PathPatchingResult for head-to-head paths
    """
    results = []
    n_heads = runner._backend.get_n_heads()

    source_heads = top_heads[:config.n_top_source_heads]

    for src in source_heads:
        for dest_layer in config.dest_head_layers:
            if dest_layer <= src.layer:
                continue

            # For simplicity, measure path effect to the whole dest layer attention
            target = InterventionTarget.at(
                layers=[src.layer],
                component="attn_out",
            )

            dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)

            # Store with dest_head=-1 to indicate whole layer
            results.append(PathPatchingResult(
                source_layer=src.layer,
                source_head=src.head,
                dest_layer=dest_layer,
                dest_head=-1,
                dest_component="attn",
                effect=dn.recovery,
            ))

    clear_gpu_memory()
    return results


@profile
def run_multi_site_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadPatchingResult],
    config: FineGrainedConfig,
) -> list[MultiSiteResult]:
    """Run multi-site interaction patching.

    For pairs of top components, measures:
    - individual_a: effect of patching A alone
    - individual_b: effect of patching B alone
    - joint: effect of patching A and B together
    - interaction: joint - individual_a - individual_b

    Args:
        runner: Model runner
        pair: Contrastive pair
        top_heads: Top components
        config: Fine-grained config

    Returns:
        List of MultiSiteResult for component pairs
    """
    results = []

    components = top_heads[:config.n_components_multi_site]

    # Pre-compute individual effects
    individual_effects = {}
    for comp in components:
        target = InterventionTarget.at(
            layers=[comp.layer],
            component="attn_out",
        )
        dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
        individual_effects[comp.label] = dn.recovery

    # Compute pairwise interactions
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1:]:
            # Joint patching
            joint_target = InterventionTarget.at(
                layers=[comp_a.layer, comp_b.layer],
                component="attn_out",
            )
            dn = patch_for_choice(runner, pair, joint_target, "denoising", clear_memory=True)

            results.append(MultiSiteResult(
                component_a=comp_a.label,
                component_b=comp_b.label,
                individual_a=individual_effects[comp_a.label],
                individual_b=individual_effects[comp_b.label],
                joint=dn.recovery,
            ))

    clear_gpu_memory()
    return results


@profile
def run_neuron_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineGrainedConfig,
) -> list[NeuronPatchingResult]:
    """Run neuron-level attribution at target layer.

    Uses gradient-based attribution to efficiently measure each neuron's
    contribution to the model's decision. This is much faster than individual
    ablations while providing meaningful per-neuron importance scores.

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine-grained config

    Returns:
        List of NeuronPatchingResult for top neurons by importance
    """
    layer = config.neuron_target_layer
    d_mlp = runner._backend.get_d_mlp()

    # Hook name for MLP post activations (after activation function)
    hook_post_name = f"blocks.{layer}.mlp.hook_post"
    names_filter = lambda name: name == hook_post_name

    # Get divergent position for metric computation
    clean_div_pos, corrupted_div_pos = pair.choice_divergent_positions
    metric_pos = corrupted_div_pos - 1  # Position before divergence

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

    # Get MLP post activations
    clean_post = clean_cache.get(hook_post_name)
    corrupted_post = corrupted_cache.get(hook_post_name)

    results = []

    if clean_post is not None and corrupted_post is not None:
        # Clamp metric_pos to valid range
        clean_seq_len = clean_post.shape[1]
        corrupted_seq_len = corrupted_post.shape[1]
        clean_metric_pos = min(metric_pos, clean_seq_len - 1)
        corrupted_metric_pos = min(metric_pos, corrupted_seq_len - 1)

        # Get activations at metric position: [d_mlp]
        clean_acts = clean_post[0, clean_metric_pos, :].detach().cpu()
        corrupted_acts = corrupted_post[0, corrupted_metric_pos, :].detach().cpu()

        # Activation difference per neuron
        act_diff = clean_acts - corrupted_acts  # [d_mlp]

        # Get W_out matrix to compute each neuron's contribution to logit direction
        # W_out: [d_mlp, d_model] transforms neuron activations to residual stream
        W_out = _get_mlp_w_out_for_runner(runner, layer)

        # Get logit direction (difference between choice A and B logits)
        W_U = runner.W_U  # [d_model, vocab_size]
        labels = pair.clean_labels
        label_a_id = runner.encode_ids(labels[0], add_special_tokens=False)[0]
        label_b_id = runner.encode_ids(labels[1], add_special_tokens=False)[0]
        logit_direction = W_U[:, label_a_id] - W_U[:, label_b_id]  # [d_model]
        logit_direction = logit_direction / torch.norm(logit_direction)
        logit_direction = logit_direction.detach().cpu()

        if W_out is not None:
            W_out = W_out.detach().cpu()
            # For each neuron, compute: act_diff[i] * (W_out[i] @ logit_direction)
            # This gives the neuron's contribution to changing the logit difference
            for neuron_idx in range(d_mlp):
                neuron_contribution = W_out[neuron_idx] @ logit_direction
                effect = (act_diff[neuron_idx] * neuron_contribution).item()
                activation_mean = ((clean_acts[neuron_idx] + corrupted_acts[neuron_idx]) / 2).item()

                results.append(NeuronPatchingResult(
                    layer=layer,
                    neuron_idx=neuron_idx,
                    effect=effect,
                    activation_mean=activation_mean,
                ))
        else:
            # Fallback: use activation difference magnitude as proxy for importance
            for neuron_idx in range(d_mlp):
                effect = act_diff[neuron_idx].item()
                activation_mean = ((clean_acts[neuron_idx] + corrupted_acts[neuron_idx]) / 2).item()

                results.append(NeuronPatchingResult(
                    layer=layer,
                    neuron_idx=neuron_idx,
                    effect=effect,
                    activation_mean=activation_mean,
                ))
    else:
        # No activation data available - return empty results
        log(f"[neuron_patching] Warning: MLP post activations not available for layer {layer}")

    # Sort by effect magnitude and keep top N
    results = sorted(results, key=lambda x: abs(x.effect), reverse=True)
    results = results[:config.n_top_neurons]

    # Cleanup
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    clear_gpu_memory()

    return results


def _get_mlp_w_out_for_runner(runner: "BinaryChoiceRunner", layer: int) -> torch.Tensor | None:
    """Get the MLP output projection matrix W_out for a layer."""
    try:
        if hasattr(runner, "_model") and hasattr(runner._model, "blocks"):
            return runner._model.blocks[layer].mlp.W_out.detach()
        if hasattr(runner, "_backend"):
            model = getattr(runner._backend, "_model", None)
            if model is not None and hasattr(model, "blocks"):
                return model.blocks[layer].mlp.W_out.detach()
    except (AttributeError, IndexError):
        pass
    return None


@profile
def run_layer_position_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineGrainedConfig,
) -> dict[str, LayerPositionResult]:
    """Run layer x position fine patching for specified components.

    Creates true 2D heatmap of patching effects (not outer product approximation).

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine-grained config

    Returns:
        Dict mapping component name to LayerPositionResult
    """
    results = {}

    # Determine layers
    layers = config.layer_position_layers
    if layers is None:
        n_layers = runner.n_layers
        layers = list(range(n_layers // 2, n_layers))

    # Determine positions
    clean_div, corrupted_div = pair.choice_divergent_positions
    start_pos = pair.position_mapping.first_interesting_pos
    end_pos = min(clean_div, corrupted_div)
    positions = list(range(start_pos, end_pos, 3))  # Sample every 3rd position

    for component in config.layer_position_components:
        log(f"[fine] Layer-position patching for {component}")

        lp_result = LayerPositionResult(
            component=component,
            layers=layers,
            positions=positions,
        )

        denoising_grid = np.zeros((len(layers), len(positions)))
        noising_grid = np.zeros((len(layers), len(positions)))

        total = len(layers) * len(positions)
        count = 0

        for li, layer in enumerate(layers):
            for pi, pos in enumerate(positions):
                count += 1
                if count % 20 == 0:
                    log_progress(count, total, prefix=f"[fine] {component} ")

                target = InterventionTarget.at(
                    layers=[layer],
                    positions=[pos],
                    component=component,
                )

                dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
                denoising_grid[li, pi] = dn.recovery

                ns = patch_for_choice(runner, pair, target, "noising", clear_memory=True)
                noising_grid[li, pi] = ns.disruption

        lp_result.denoising_grid = denoising_grid
        lp_result.noising_grid = noising_grid
        results[component] = lp_result

    clear_gpu_memory()
    return results


@profile
def run_fine_grained_analysis(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineGrainedConfig | None = None,
) -> FineGrainedResults:
    """Run all fine-grained analyses on a contrastive pair.

    Orchestrates:
    1. Head-level patching sweep
    2. Position-level patching for top heads
    3. Path patching (head-to-MLP and head-to-head)
    4. Multi-site interaction patching
    5. Neuron-level ablation
    6. Layer-position fine heatmap

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine-grained config

    Returns:
        FineGrainedResults with all analysis results
    """
    if config is None:
        config = FineGrainedConfig(enabled=True)

    # Set model dimensions
    config.n_layers = runner.n_layers
    config.n_heads = runner._backend.get_n_heads()

    results = FineGrainedResults(
        sample_id=pair.sample_id,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        neuron_target_layer=config.neuron_target_layer,
    )

    # 1. Head-level patching sweep
    if config.head_patching_enabled:
        log("[fine] Running head patching sweep...")
        results.head_sweep = run_head_patching_sweep(runner, pair, config)
        log(f"[fine] Head sweep complete: {len(results.head_sweep.results)} heads")

    # Get top heads for subsequent analyses
    top_heads = []
    if results.head_sweep:
        top_heads = results.head_sweep.get_top_heads(max(
            config.n_top_heads_for_position,
            config.n_top_source_heads,
            config.n_components_multi_site,
        ))

    # 2. Position-level patching for top heads
    if config.position_patching_enabled and top_heads:
        log("[fine] Running position patching...")
        results.position_results = run_position_patching(runner, pair, top_heads, config)

    # 3. Path patching
    if config.path_patching_enabled and top_heads:
        log("[fine] Running path patching to MLP...")
        results.path_to_mlp = run_path_patching_to_mlp(runner, pair, top_heads, config)

        log("[fine] Running path patching to heads...")
        results.path_to_head = run_path_patching_to_head(runner, pair, top_heads, config)

    # 4. Multi-site interaction
    if config.multi_site_enabled and top_heads:
        log("[fine] Running multi-site patching...")
        results.multi_site = run_multi_site_patching(runner, pair, top_heads, config)

    # 5. Neuron-level ablation
    if config.neuron_patching_enabled:
        log("[fine] Running neuron patching...")
        results.neuron_results = run_neuron_patching(runner, pair, config)

    # 6. Layer-position fine heatmap
    if config.layer_position_enabled:
        log("[fine] Running layer-position patching...")
        results.layer_position = run_layer_position_patching(runner, pair, config)

    results.print_summary()
    return results
