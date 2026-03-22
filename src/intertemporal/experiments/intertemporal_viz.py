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
from ...attribution_patching import AttrPatchAggregatedResults, AttrPatchPairResult
from ...common.file_io import load_json
from ...common.logging import log
from ..viz import (
    visualize_all_aggregated,
    visualize_all_att_aggregated_slices,
    visualize_fine_patching,
    visualize_pair_results,
)
from ..viz.diffmeans_viz import visualize_diffmeans
from ..viz.geo_viz import visualize_geo, visualize_geo_pair
from .diffmeans import DiffMeansAggregatedResults, DiffMeansPairResult
from .geo import GeoAggregatedResults, GeoPairResult
from .processing import ProcessedResults

if TYPE_CHECKING:
    from ...activation_patching import ActPatchAggregatedResult, ActPatchPairResult
    from ...attribution_patching import AttrPatchPairResult
    from ...binary_choice import BinaryChoiceRunner
    from ...common.contrastive_pair import ContrastivePair
    from ..common.contrastive_preferences import ContrastivePreferences


def get_pairs_dir(exp_dir: Path) -> Path:
    """Get the pairs directory for an experiment."""
    return exp_dir / "pairs"


def get_pair_dir(exp_dir: Path, pair_idx: int) -> Path:
    """Get the directory for a specific pair."""
    return get_pairs_dir(exp_dir) / f"pair_{pair_idx}"


def detect_cached_components(exp_dir: Path) -> list[str]:
    """Detect which components have cached coarse patching results.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        List of component names with cached results

    Note: This is a standalone version for use without ExperimentContext.
    When using ExperimentContext, prefer ctx.detect_cached_components().
    """
    components = []
    pair_0 = get_pair_dir(exp_dir, 0)
    if pair_0.exists():
        for d in pair_0.iterdir():
            if d.is_dir() and d.name.startswith("sweep_"):
                comp = d.name.replace("sweep_", "")
                if (d / "coarse_results.json").exists():
                    components.append(comp)
    return components


def _extract_label_pairs_from_preference(pref_data: dict) -> tuple[tuple[str, str], ...] | None:
    """Extract label pairs from contrastive preference data.

    For multilabel experiments, short_term_labels and long_term_labels
    are lists where each index corresponds to a fork.

    Returns tuple of (short_label, long_label) pairs for each fork.
    """
    short_labels = pref_data.get("short_term_labels", [])
    long_labels = pref_data.get("long_term_labels", [])

    if not isinstance(short_labels, list) or not isinstance(long_labels, list):
        return None

    if len(short_labels) <= 1 and len(long_labels) <= 1:
        return None  # Single label, not multilabel

    # Create pairs: (short_label[i], long_label[i]) for each fork
    pairs = []
    for i in range(min(len(short_labels), len(long_labels))):
        pairs.append((short_labels[i], long_labels[i]))

    return tuple(pairs) if pairs else None


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
    # Load label_pairs from contrastive_preference.json if available
    label_pairs = None
    pref_path = pair_dir / "contrastive_preference.json"
    if pref_path.exists():
        pref_data = load_json(pref_path)
        label_pairs = _extract_label_pairs_from_preference(pref_data)

    coarse_results = {}
    for component in components:
        results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
        if results_path.exists():
            result = CoarseActPatchResults.from_json(results_path)
            # Add label_pairs if not already present
            if label_pairs and result.label_pairs is None:
                result.label_pairs = label_pairs
            coarse_results[component] = result
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
    coarse_dir = exp_dir / "agg_coarse"
    for component in components:
        agg_path = coarse_dir / f"{component}.json"
        # Fallback to legacy paths
        if not agg_path.exists():
            agg_path = exp_dir / "agg" / "coarse" / f"{component}.json"
        if not agg_path.exists():
            agg_path = exp_dir / "coarse_agg" / f"{component}.json"
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
        pair_dir = get_pair_dir(exp_dir, pair_idx)
        if not pair_dir.exists():
            break

        # Load label_pairs from contrastive_preference.json if available
        label_pairs = None
        pref_path = pair_dir / "contrastive_preference.json"
        if pref_path.exists():
            pref_data = load_json(pref_path)
            label_pairs = _extract_label_pairs_from_preference(pref_data)

        # Load and add results for each component
        for component in components:
            results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
            if results_path.exists():
                result = CoarseActPatchResults.from_json(results_path)
                result.sample_id = pair_idx
                # Add label_pairs if not already present
                if label_pairs and result.label_pairs is None:
                    result.label_pairs = label_pairs
                coarse_agg[component].add(result)

        pair_idx += 1

    # Filter out empty aggregations
    return {comp: agg for comp, agg in coarse_agg.items() if agg.n_samples > 0}


def load_att_agg(exp_dir: Path) -> AttrPatchAggregatedResults | None:
    """Load aggregated attribution results from cache.

    Tries new folder structure first, then legacy path.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        AttrPatchAggregatedResults or None if not found
    """
    # Try new folder structure first
    att_dir = exp_dir / "agg_att"
    if (att_dir / "att_agg.json").exists():
        return AttrPatchAggregatedResults.from_json(att_dir / "att_agg.json")

    # Fallback to legacy paths
    legacy_paths = [
        exp_dir / "att_agg" / "att_agg.json",
        exp_dir / "agg" / "att" / "att_agg.json",
        exp_dir / "att_agg.json",
    ]
    for path in legacy_paths:
        if path.exists():
            return AttrPatchAggregatedResults.from_json(path)

    return None


def load_att_pair_result(pair_dir: Path) -> AttrPatchPairResult | None:
    """Load per-pair attribution results from cache.

    Args:
        pair_dir: Path to pair directory (e.g., exp_dir/pair_0)

    Returns:
        AttrPatchPairResult or None if not found
    """
    # Try new path first
    att_path = pair_dir / "att_patching" / "att_results.json"
    if att_path.exists():
        return AttrPatchPairResult.from_json(att_path)
    # Fallback to legacy path
    legacy_path = pair_dir / "att" / "att_results.json"
    if legacy_path.exists():
        return AttrPatchPairResult.from_json(legacy_path)
    return None


def load_diffmeans_agg(exp_dir: Path) -> DiffMeansAggregatedResults | None:
    """Load aggregated diffmeans results from cache.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        DiffMeansAggregatedResults or None if not found
    """
    path = exp_dir / "agg_diffmeans" / "diffmeans_agg.json"
    # Fallback to legacy path
    if not path.exists():
        path = exp_dir / "agg" / "diffmeans" / "diffmeans_agg.json"
    if path.exists():
        return DiffMeansAggregatedResults.from_json(path)
    return None


def load_diffmeans_pair(pair_dir: Path) -> DiffMeansPairResult | None:
    """Load per-pair diffmeans results from cache.

    Args:
        pair_dir: Path to pair directory (e.g., exp_dir/pair_0)

    Returns:
        DiffMeansPairResult or None if not found
    """
    path = pair_dir / "diffmeans" / "diffmeans.json"
    if path.exists():
        return DiffMeansPairResult.from_json(path)
    return None


def load_geo_agg(exp_dir: Path) -> GeoAggregatedResults | None:
    """Load aggregated geo results from cache.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        GeoAggregatedResults or None if not found
    """
    path = exp_dir / "agg_geo" / "geo_agg.json"
    if path.exists():
        return GeoAggregatedResults.from_json(path)
    return None


def load_geo_pair(pair_dir: Path) -> GeoPairResult | None:
    """Load per-pair geo results from cache.

    Args:
        pair_dir: Path to pair directory (e.g., exp_dir/pair_0)

    Returns:
        GeoPairResult or None if not found
    """
    path = pair_dir / "geo" / "geo_results.json"
    if path.exists():
        return GeoPairResult.from_json(path)
    return None


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
    diffmeans_agg: DiffMeansAggregatedResults | None = None,
    diffmeans_patching: dict[int, DiffMeansPairResult] | None = None,
    geo_agg: GeoAggregatedResults | None = None,
    geo_patching: dict[int, GeoPairResult] | None = None,
    processed_results: ProcessedResults | None = None,
    # Optional context for richer visualizations
    pairs: list["ContrastivePair"] | None = None,
    pref_pairs: list["ContrastivePreferences"] | None = None,
    runner: "BinaryChoiceRunner | None" = None,
    save_token_trees_fn: callable | None = None,
    # Components to process (auto-detected from cache if None)
    components: list[str] | None = None,
    # If True, only generate aggregated visualizations (skip per-pair)
    only_agg: bool = False,
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
        diffmeans_patching: In-memory per-pair diffmeans results
        geo_agg: In-memory aggregated geo results
        geo_patching: In-memory per-pair geo results
        processed_results: Pre-computed analysis results from step_process_results
        pairs: List of contrastive pairs (for tokenization viz)
        pref_pairs: List of ContrastivePreferences for slice filtering
        runner: Model runner (for tokenization viz)
        save_token_trees_fn: Function to save token trees
        components: List of components to process (auto-detected if None)
        only_agg: If True, skip per-pair visualizations
    """
    exp_dir = Path(exp_dir)

    # Detect components from cache if not provided
    if components is None:
        components = detect_cached_components(exp_dir)
        if not components and coarse_agg_by_component:
            components = list(coarse_agg_by_component.keys())

    if components:
        log(f"[viz] Processing components: {components}")
    else:
        log(f"[viz] No coarse components found, will process other viz types")

    # Rebuild aggregated data from per-pair results if not provided in-memory
    # This ensures all pairs are included (the agg file might be stale)
    if coarse_agg_by_component is None and components:
        coarse_agg_by_component = rebuild_coarse_aggregated(exp_dir, components)
        if coarse_agg_by_component and components:
            log(f"[viz] Rebuilt aggregation from {coarse_agg_by_component[components[0]].n_samples} pairs")

    # Load attribution patching aggregated results from cache if not provided
    if att_agg is None:
        att_agg = load_att_agg(exp_dir)
        if att_agg:
            log("[viz] Loaded attribution aggregated results from cache")

    # Load diffmeans aggregated results from cache if not provided
    if diffmeans_agg is None:
        diffmeans_agg = load_diffmeans_agg(exp_dir)
        if diffmeans_agg:
            log("[viz] Loaded diffmeans aggregated results from cache")

    # Generate aggregated visualizations
    if att_agg:
        visualize_all_att_aggregated_slices(att_agg, exp_dir / "agg_att")
        log("[viz] Generated attribution aggregated visualizations")

    if coarse_agg_by_component:
        visualize_all_aggregated(
            coarse_agg_by_component,
            exp_dir / "agg_coarse",
            pref_pairs,
            exp_dir,
            processed_results,
        )
        log("[viz] Generated aggregated visualizations")

    if fine_agg:
        visualize_fine_patching(fine_agg, exp_dir / "agg_fine")

    if diffmeans_agg:
        visualize_diffmeans(diffmeans_agg, exp_dir / "agg_diffmeans")
        log("[viz] Generated diffmeans visualizations")

    # Load geo aggregated results from cache if not provided
    if geo_agg is None:
        geo_agg = load_geo_agg(exp_dir)
        if geo_agg:
            log("[viz] Loaded geo aggregated results from cache")

    if geo_agg:
        visualize_geo(geo_agg, exp_dir / "agg_geo", pref_pairs=pref_pairs)
        log("[viz] Generated geo visualizations")

    # Skip per-pair visualizations if only_agg is True
    if only_agg:
        log(f"[viz] Skipping per-pair visualizations (only_agg=True)")
        log(f"[viz] Done generating visualizations for {exp_dir.name}")
        return

    # Generate per-pair visualizations
    pair_idx = 0
    while True:
        pair_dir = get_pair_dir(exp_dir, pair_idx)
        if not pair_dir.exists():
            break

        # Get coarse results from in-memory data or load from cache
        if coarse_patching and components:
            pair_coarse = {
                comp: coarse_patching[(pair_idx, comp)]
                for comp in components
                if (pair_idx, comp) in coarse_patching
            }
        elif components:
            pair_coarse = load_coarse_results_for_pair(pair_dir, components)
        else:
            pair_coarse = {}

        pair = pairs[pair_idx] if pairs and pair_idx < len(pairs) else None
        # Try in-memory first, then load from cache
        att_result = att_patching.get(pair_idx) if att_patching else load_att_pair_result(pair_dir)
        fine_result = fine_patching.get(pair_idx) if fine_patching else None
        diffmeans_result = (
            diffmeans_patching.get(pair_idx)
            if diffmeans_patching
            else load_diffmeans_pair(pair_dir)
        )
        geo_result = (
            geo_patching.get(pair_idx)
            if geo_patching
            else load_geo_pair(pair_dir)
        )

        if pair_coarse or att_result or fine_result or diffmeans_result or geo_result:
            visualize_pair_results(
                pair_idx=pair_idx,
                pair_out_dir=pair_dir,
                pair=pair,
                runner=runner,
                att_result=att_result,
                coarse_results=pair_coarse if pair_coarse else None,
                fine_result=fine_result,
                diffmeans_result=diffmeans_result,
                geo_result=geo_result,
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

    # Find all experiment directories (those with pairs/pair_0 subdirectory)
    exp_dirs = []
    for d in experiments_dir.iterdir():
        if d.is_dir() and get_pair_dir(d, 0).exists():
            exp_dirs.append(d)

    if not exp_dirs:
        log("[viz] No experiment directories found")
        return

    log(f"[viz] Found {len(exp_dirs)} experiment directories")

    for exp_dir in sorted(exp_dirs):
        log(f"\n[viz] Regenerating visualizations for: {exp_dir.name}")
        generate_viz(exp_dir)
