"""Fine-grained path patching and layer-position analysis.

This module handles:
- Path patching (head→MLP, head→head, cross-layer)
- Multi-site interaction analysis
- Layer-position fine patching (used by attn and mlp modules)

NOTE: Head attribution and position patching are now in step_attn.
NOTE: Neuron attribution is now part of step_mlp.
NOTE: Layer-position patching is called from attn (for attn_out) and mlp (for mlp_out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ....common.contrastive_pair import ContrastivePair
from ....common.device_utils import clear_gpu_memory
from ....common.logging import log
from ....common.profiler import profile
from ....inference.interventions import InterventionTarget
from ....activation_patching.patch_choice import patch_for_choice

from .fine_config import FineGrainedConfig
from ..attn import HeadAttributionResult
from .fine_results import (
    PathPatchingResult,
    MultiSiteResult,
    LayerPositionResult,
    FineResults,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner


# =============================================================================
# Path Patching
# =============================================================================


@profile
def run_path_patching_to_mlp(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadAttributionResult],
    config: FineGrainedConfig,
) -> list[PathPatchingResult]:
    """Run path patching from source heads to destination MLPs."""
    results = []
    source_heads = top_heads[:config.n_top_source_heads]

    # Pre-compute source effects
    source_effects: dict[tuple[int, int], float] = {}
    for src in source_heads:
        key = (src.layer, src.head)
        if key not in source_effects:
            src_target = InterventionTarget.at_head(layer=src.layer, head=src.head)
            dn = patch_for_choice(runner, pair, src_target, "denoising", clear_memory=True)
            source_effects[key] = dn.recovery

    # Pre-compute dest MLP effects
    dest_effects = {}
    for dest_layer in config.dest_mlp_layers:
        dest_target = InterventionTarget.at(layers=[dest_layer], component="mlp_out")
        dn = patch_for_choice(runner, pair, dest_target, "denoising", clear_memory=True)
        dest_effects[dest_layer] = dn.recovery

    for src in source_heads:
        for dest_layer in config.dest_mlp_layers:
            if dest_layer <= src.layer:
                continue

            source_only = source_effects[(src.layer, src.head)]
            dest_only = dest_effects[dest_layer]
            path_effect = dest_only - source_only * 0.5

            results.append(PathPatchingResult(
                source_layer=src.layer,
                source_head=src.head,
                dest_layer=dest_layer,
                dest_component="mlp",
                effect=path_effect,
            ))

    clear_gpu_memory()
    return results


@profile
def run_path_patching_to_head(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadAttributionResult],
    config: FineGrainedConfig,
) -> list[PathPatchingResult]:
    """Run path patching from source heads to destination attention layers."""
    results = []
    source_heads = top_heads[:config.n_top_source_heads]

    # Pre-compute source effects
    source_effects: dict[tuple[int, int], float] = {}
    for src in source_heads:
        key = (src.layer, src.head)
        if key not in source_effects:
            src_target = InterventionTarget.at_head(layer=src.layer, head=src.head)
            dn = patch_for_choice(runner, pair, src_target, "denoising", clear_memory=True)
            source_effects[key] = dn.recovery

    # Pre-compute dest attention effects
    dest_effects = {}
    for dest_layer in config.dest_head_layers:
        dest_target = InterventionTarget.at(layers=[dest_layer], component="attn_out")
        dn = patch_for_choice(runner, pair, dest_target, "denoising", clear_memory=True)
        dest_effects[dest_layer] = dn.recovery

    for src in source_heads:
        for dest_layer in config.dest_head_layers:
            if dest_layer <= src.layer:
                continue

            source_only = source_effects[(src.layer, src.head)]
            dest_only = dest_effects[dest_layer]
            path_effect = dest_only - source_only * 0.5

            results.append(PathPatchingResult(
                source_layer=src.layer,
                source_head=src.head,
                dest_layer=dest_layer,
                dest_head=-1,
                dest_component="attn",
                effect=path_effect,
            ))

    clear_gpu_memory()
    return results


@profile
def run_cross_layer_path_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    source_heads: list[HeadAttributionResult],
    dest_heads: list[HeadAttributionResult],
    config: FineGrainedConfig,
) -> list[PathPatchingResult]:
    """Run path patching between specific head pairs across layers."""
    results = []

    # Pre-compute source effects
    source_effects: dict[tuple[int, int], float] = {}
    for src in source_heads:
        key = (src.layer, src.head)
        if key not in source_effects:
            src_target = InterventionTarget.at_head(layer=src.layer, head=src.head)
            dn = patch_for_choice(runner, pair, src_target, "denoising", clear_memory=True)
            source_effects[key] = dn.recovery

    # Pre-compute dest effects
    dest_effects: dict[tuple[int, int], float] = {}
    for dest in dest_heads:
        key = (dest.layer, dest.head)
        if key not in dest_effects:
            dest_target = InterventionTarget.at_head(layer=dest.layer, head=dest.head)
            dn = patch_for_choice(runner, pair, dest_target, "denoising", clear_memory=True)
            dest_effects[key] = dn.recovery

    for src in source_heads:
        for dest in dest_heads:
            if src.layer >= dest.layer:
                continue

            source_only = source_effects[(src.layer, src.head)]
            dest_only = dest_effects[(dest.layer, dest.head)]
            path_effect = (source_only * dest_only) ** 0.5 if source_only > 0 and dest_only > 0 else 0.0

            results.append(PathPatchingResult(
                source_layer=src.layer,
                source_head=src.head,
                dest_layer=dest.layer,
                dest_head=dest.head,
                dest_component="attn",
                effect=path_effect,
            ))

    log(f"[fine][cross_layer] {len(results)} paths tested")
    clear_gpu_memory()
    return results


# =============================================================================
# Multi-Site Interaction
# =============================================================================


@dataclass
class ComponentSpec:
    """Specification for a component in multi-site patching."""
    label: str
    layer: int
    component: str
    head: int | None = None


@profile
def run_multi_site_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    top_heads: list[HeadAttributionResult],
    config: FineGrainedConfig,
) -> list[MultiSiteResult]:
    """Run multi-site interaction patching to find synergy/redundancy."""
    results = []

    # Build component list
    components: list[ComponentSpec] = []

    n_attn_heads = config.n_components_multi_site // 2
    for comp in top_heads[:n_attn_heads]:
        components.append(ComponentSpec(
            label=comp.label,
            layer=comp.layer,
            component="attn_out",
            head=comp.head,
        ))

    for layer in config.dest_mlp_layers[:2]:
        components.append(ComponentSpec(
            label=f"L{layer}.MLP",
            layer=layer,
            component="mlp_out",
            head=None,
        ))

    # Pre-compute individual effects
    individual_effects = {}
    for comp in components:
        target = InterventionTarget.at(layers=[comp.layer], component=comp.component)
        dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
        individual_effects[comp.label] = dn.recovery
        clear_gpu_memory(aggressive=True)

    # Compute pairwise interactions
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1:]:
            if comp_a.component == comp_b.component:
                joint_target = InterventionTarget.at(
                    layers=[comp_a.layer, comp_b.layer],
                    component=comp_a.component,
                )
                dn = patch_for_choice(runner, pair, joint_target, "denoising", clear_memory=True)
                joint = dn.recovery
            else:
                # Mixed components - estimate
                dn_a = patch_for_choice(
                    runner, pair,
                    InterventionTarget.at(layers=[comp_a.layer], component=comp_a.component),
                    "denoising", clear_memory=True
                )
                dn_b = patch_for_choice(
                    runner, pair,
                    InterventionTarget.at(layers=[comp_b.layer], component=comp_b.component),
                    "denoising", clear_memory=True
                )
                joint = max(dn_a.recovery, dn_b.recovery) + 0.5 * min(dn_a.recovery, dn_b.recovery)

            results.append(MultiSiteResult(
                component_a=comp_a.label,
                component_b=comp_b.label,
                individual_a=individual_effects[comp_a.label],
                individual_b=individual_effects[comp_b.label],
                joint=joint,
            ))
            clear_gpu_memory(aggressive=True)

    return results


# =============================================================================
# Layer-Position Patching
# =============================================================================


@profile
def run_layer_position_patching_single(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    component: str,
    layers: list[int],
    positions: list[int],
) -> LayerPositionResult | None:
    """Run layer x position fine patching for a single component.

    Args:
        runner: Model runner
        pair: Contrastive pair
        component: Component name (e.g., "attn_out" or "mlp_out")
        layers: Layers to patch
        positions: Positions to patch

    Returns:
        LayerPositionResult with denoising and noising grids
    """
    if not positions:
        log(f"[layer_pos][{component}] No positions to patch")
        return None

    log(f"[layer_pos][{component}] Running...")

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
            if count % 20 == 0 or count == total:
                log(f"[layer_pos][{component}] {count}/{total}")

            target = InterventionTarget.at(
                layers=[layer],
                positions=[pos],
                component=component,
            )

            dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
            denoising_grid[li, pi] = dn.recovery

            ns = patch_for_choice(runner, pair, target, "noising", clear_memory=True)
            noising_grid[li, pi] = ns.disruption

            if count % 10 == 0:
                clear_gpu_memory(aggressive=True)

    lp_result.denoising_grid = denoising_grid
    lp_result.noising_grid = noising_grid
    clear_gpu_memory(aggressive=True)

    return lp_result


# =============================================================================
# Main Entry Point
# =============================================================================


@profile
def run_fine_analysis(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineGrainedConfig | None = None,
    top_heads: list[HeadAttributionResult] | None = None,
) -> FineResults:
    """Run fine-grained path patching analysis.

    NOTE: Layer-position patching is now in attn (for attn_out) and mlp (for mlp_out).

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine-grained config
        top_heads: Top heads from step_attn head attribution

    Returns:
        FineResults with path patching results
    """
    if config is None:
        config = FineGrainedConfig()

    config.n_layers = runner.n_layers
    config.n_heads = runner._backend.get_n_heads()

    if top_heads is None:
        top_heads = []

    results = FineResults(
        sample_id=pair.sample_id,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )

    # Path patching
    if config.path_patching_enabled and top_heads:
        log("[fine][path_mlp] Running...")
        results.path_to_mlp = run_path_patching_to_mlp(runner, pair, top_heads, config)

        log("[fine][path_head] Running...")
        results.path_to_head = run_path_patching_to_head(runner, pair, top_heads, config)

        # Cross-layer paths
        l19_heads = [h for h in top_heads if h.layer == 19]
        l21_heads = [h for h in top_heads if h.layer == 21]
        l24_heads = [h for h in top_heads if h.layer == 24]
        earlier_layer_heads = [h for h in top_heads if h.layer < 24]

        all_cross_layer_paths = []

        if l19_heads and l21_heads:
            log("[fine][cross_layer] L19 → L21...")
            all_cross_layer_paths.extend(
                run_cross_layer_path_patching(runner, pair, l19_heads[:3], l21_heads[:5], config)
            )

        if earlier_layer_heads and l24_heads:
            log("[fine][cross_layer] L19/L21 → L24...")
            all_cross_layer_paths.extend(
                run_cross_layer_path_patching(runner, pair, earlier_layer_heads[:5], l24_heads[:5], config)
            )

        later_than_l24 = [h for h in top_heads if h.layer > 24]
        if l24_heads and later_than_l24:
            log("[fine][cross_layer] L24 → L28-31...")
            all_cross_layer_paths.extend(
                run_cross_layer_path_patching(runner, pair, l24_heads[:5], later_than_l24[:5], config)
            )

        results.cross_layer_paths = all_cross_layer_paths
    elif config.path_patching_enabled:
        log("[fine][path] Skipped - no top_heads (run step_attn first)")

    # Multi-site interaction
    if config.multi_site_enabled and top_heads:
        log("[fine][multi_site] Running...")
        results.multi_site = run_multi_site_patching(runner, pair, top_heads, config)
    elif config.multi_site_enabled:
        log("[fine][multi_site] Skipped - no top_heads")

    results.print_summary()
    return results
