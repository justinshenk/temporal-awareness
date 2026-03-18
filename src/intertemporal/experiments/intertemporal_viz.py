"""Visualization generation for intertemporal experiments.

This module provides functions to generate visualizations, either from
in-memory data or cached results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...activation_patching.coarse import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...common.logging import log
from ..viz import (
    visualize_all_aggregated,
    visualize_att_patching,
    visualize_fine_patching,
    visualize_pair_results,
)

if TYPE_CHECKING:
    from ...activation_patching import ActPatchAggregatedResult, ActPatchPairResult
    from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults
    from ...binary_choice import BinaryChoiceRunner
    from ...common.contrastive_pair import ContrastivePair


def detect_cached_components(exp_dir: Path) -> list[str]:
    """Detect which components have cached coarse patching results.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        List of component names with cached results
    """
    components = []
    pair_0 = exp_dir / "pair_0"
    if pair_0.exists():
        for d in pair_0.iterdir():
            if d.is_dir() and d.name.startswith("sweep_"):
                comp = d.name.replace("sweep_", "")
                if (d / "coarse_results.json").exists():
                    components.append(comp)
    return components


def load_coarse_results_for_pair(
    pair_dir: Path,
    components: list[str],
) -> dict[str, CoarseActPatchResults]:
    """Load coarse patching results for a single pair from cache.

    Args:
        pair_dir: Path to pair directory (e.g., exp_dir/pair_0)
        components: List of component names to load

    Returns:
        Dict mapping component name to results
    """
    coarse_results = {}
    for component in components:
        results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
        if results_path.exists():
            coarse_results[component] = CoarseActPatchResults.from_json(results_path)
    return coarse_results


def load_coarse_aggregated(
    exp_dir: Path,
    components: list[str],
) -> dict[str, CoarseActPatchAggregatedResults]:
    """Load aggregated coarse patching results from cache.

    Args:
        exp_dir: Path to experiment directory
        components: List of component names to load

    Returns:
        Dict mapping component name to aggregated results
    """
    coarse_agg = {}
    for component in components:
        agg_path = exp_dir / f"coarse_agg_{component}.json"
        if agg_path.exists():
            coarse_agg[component] = CoarseActPatchAggregatedResults.from_json(agg_path)
    return coarse_agg


def rebuild_coarse_aggregated(
    exp_dir: Path,
    components: list[str],
) -> dict[str, CoarseActPatchAggregatedResults]:
    """Rebuild aggregated results from all per-pair cached results.

    This is used when regenerating visualizations to ensure all pairs
    are included in the aggregation, not just what was in the stale agg file.

    Args:
        exp_dir: Path to experiment directory
        components: List of component names to aggregate

    Returns:
        Dict mapping component name to freshly aggregated results
    """
    coarse_agg = {comp: CoarseActPatchAggregatedResults() for comp in components}

    # Find all pair directories
    pair_idx = 0
    while True:
        pair_dir = exp_dir / f"pair_{pair_idx}"
        if not pair_dir.exists():
            break

        # Load and add results for each component
        for component in components:
            results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
            if results_path.exists():
                result = CoarseActPatchResults.from_json(results_path)
                result.sample_id = pair_idx
                coarse_agg[component].add(result)

        pair_idx += 1

    # Filter out empty aggregations
    return {comp: agg for comp, agg in coarse_agg.items() if agg.n_samples > 0}


def generate_viz(
    exp_dir: Path,
    *,
    # Optional in-memory data (if None, loads from cache)
    coarse_agg_by_component: dict[str, CoarseActPatchAggregatedResults] | None = None,
    coarse_patching: dict[tuple[int, str], CoarseActPatchResults] | None = None,
    att_agg: "AttrPatchAggregatedResults | None" = None,
    att_patching: dict[int, "AttrPatchPairResult"] | None = None,
    fine_agg: "ActPatchAggregatedResult | None" = None,
    fine_patching: dict[int, "ActPatchPairResult"] | None = None,
    # Optional context for richer visualizations
    pairs: list["ContrastivePair"] | None = None,
    runner: "BinaryChoiceRunner | None" = None,
    save_token_trees_fn: callable | None = None,
    # Components to process (auto-detected from cache if None)
    components: list[str] | None = None,
) -> None:
    """Generate all visualizations for an experiment.

    Can work with either in-memory data or load from cache. If in-memory
    data is provided, it takes precedence over cached data.

    Args:
        exp_dir: Path to experiment directory
        coarse_agg_by_component: In-memory aggregated coarse results
        coarse_patching: In-memory per-pair coarse results keyed by (pair_idx, component)
        att_agg: In-memory aggregated attribution results
        att_patching: In-memory per-pair attribution results
        fine_agg: In-memory aggregated fine patching results
        fine_patching: In-memory per-pair fine patching results
        pairs: List of contrastive pairs (for tokenization viz)
        runner: Model runner (for tokenization viz)
        save_token_trees_fn: Function to save token trees
        components: List of components to process (auto-detected if None)
    """
    exp_dir = Path(exp_dir)

    # Detect components from cache if not provided
    if components is None:
        components = detect_cached_components(exp_dir)
        if not components and coarse_agg_by_component:
            components = list(coarse_agg_by_component.keys())

    if not components:
        log(f"[viz] No components found for {exp_dir}")
        return

    log(f"[viz] Processing components: {components}")

    # Rebuild aggregated data from per-pair results if not provided in-memory
    # This ensures all pairs are included (the agg file might be stale)
    if coarse_agg_by_component is None:
        coarse_agg_by_component = rebuild_coarse_aggregated(exp_dir, components)
        log(f"[viz] Rebuilt aggregation from {coarse_agg_by_component[components[0]].n_samples if coarse_agg_by_component else 0} pairs")

    # Generate aggregated visualizations
    agg_out_dir = exp_dir / "agg"

    if att_agg:
        visualize_att_patching(
            att_agg.denoising_agg,
            agg_out_dir / "all" / "att_patching" / "denoising",
        )
        visualize_att_patching(
            att_agg.noising_agg,
            agg_out_dir / "all" / "att_patching" / "noising",
        )

    if coarse_agg_by_component:
        visualize_all_aggregated(coarse_agg_by_component, agg_out_dir)
        log("[viz] Generated aggregated visualizations")

    if fine_agg:
        visualize_fine_patching(fine_agg, agg_out_dir)

    # Generate per-pair visualizations
    pair_idx = 0
    while True:
        pair_dir = exp_dir / f"pair_{pair_idx}"
        if not pair_dir.exists():
            break

        # Get coarse results from in-memory data or load from cache
        if coarse_patching:
            pair_coarse = {
                comp: coarse_patching[(pair_idx, comp)]
                for comp in components
                if (pair_idx, comp) in coarse_patching
            }
        else:
            pair_coarse = load_coarse_results_for_pair(pair_dir, components)

        pair = pairs[pair_idx] if pairs and pair_idx < len(pairs) else None
        att_result = att_patching.get(pair_idx) if att_patching else None
        fine_result = fine_patching.get(pair_idx) if fine_patching else None

        if pair_coarse or att_result or fine_result:
            visualize_pair_results(
                pair_idx=pair_idx,
                pair_out_dir=pair_dir,
                pair=pair,
                runner=runner,
                att_result=att_result,
                coarse_results=pair_coarse if pair_coarse else None,
                fine_result=fine_result,
                try_loading_cache=True,
                save_token_trees_fn=save_token_trees_fn,
            )
            log(f"[viz] Generated pair {pair_idx} visualizations")

        pair_idx += 1

    log(f"[viz] Done generating visualizations for {exp_dir.name}")


def regenerate_all_visualizations(experiments_dir: Path) -> None:
    """Regenerate visualizations for all experiments in the directory.

    Args:
        experiments_dir: Path to the experiments directory containing experiment folders
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        log(f"[viz] Experiments directory does not exist: {experiments_dir}")
        return

    # Find all experiment directories (those with pair_0 subdirectory)
    exp_dirs = []
    for d in experiments_dir.iterdir():
        if d.is_dir() and (d / "pair_0").exists():
            exp_dirs.append(d)

    if not exp_dirs:
        log("[viz] No experiment directories found")
        return

    log(f"[viz] Found {len(exp_dirs)} experiment directories")

    for exp_dir in sorted(exp_dirs):
        log(f"\n[viz] Regenerating visualizations for: {exp_dir.name}")
        generate_viz(exp_dir)
