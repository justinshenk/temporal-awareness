"""Visualization orchestration helpers for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...common.logging import log
from ...viz.token_coloring import PairTokenColoring, get_token_coloring_for_pair

from .att_patching_viz import visualize_att_patching
from .coarse_patching_viz import visualize_coarse_patching, visualize_component_comparison
from .fine_patching_viz import visualize_fine_patching
from .tokenization_viz import (
    TokenizationVizData,
    visualize_tokenization,
    visualize_tokenization_from_cache,
)

if TYPE_CHECKING:
    from ...activation_patching import ActPatchResult
    from ...activation_patching.coarse import CoarseActPatchResult
    from ...attribution_patching import AttrPatchResult
    from ...common.contrastive_pair import ContrastivePair


def visualize_pair_results(
    pair_idx: int,
    pair_out_dir: Path,
    *,
    pair: "ContrastivePair | None" = None,
    runner: Any = None,
    att_result: "AttrPatchResult | None" = None,
    coarse_results: dict[str, "CoarseActPatchResult"] | None = None,
    fine_result: "ActPatchResult | None" = None,
    try_loading_cache: bool = False,
    save_token_trees_fn: Any = None,
) -> None:
    """Visualize all results for a single contrastive pair.

    This helper orchestrates visualization of tokenization, attribution patching,
    coarse activation patching, fine activation patching, and component comparisons
    for a single pair.

    Args:
        pair_idx: Index of the pair (for logging)
        pair_out_dir: Output directory for this pair's visualizations
        pair: ContrastivePair (optional, needed if cache unavailable)
        runner: Model runner (optional, needed if cache unavailable)
        att_result: Attribution patching result for this pair
        coarse_results: Dict mapping component name to coarse patching result
        fine_result: Fine activation patching result for this pair
        try_loading_cache: If True, try loading tokenization from cache
        save_token_trees_fn: Optional callback to save token trees
    """
    pair_out_dir = Path(pair_out_dir)
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure tokenization visualization exists and get coloring
    coloring = _ensure_tokenization_and_get_coloring(
        pair_idx=pair_idx,
        pair_out_dir=pair_out_dir,
        pair=pair,
        runner=runner,
        try_loading_cache=try_loading_cache,
        save_token_trees_fn=save_token_trees_fn,
    )

    if coloring is None:
        log(f"[viz] Pair {pair_idx}: no coloring available (position labels will be numeric)")

    # In intertemporal experiments, "short" corresponds to "clean" trajectory
    position_labels = coloring.get_position_labels("clean") if coloring else None
    section_markers = coloring.get_section_markers("clean") if coloring else None

    # Step 2: Attribution patching visualizations
    if att_result is not None:
        if att_result.result.denoising:
            visualize_att_patching(
                att_result.result.denoising,
                pair_out_dir / "denoising",
                position_labels,
                section_markers,
            )
        if att_result.result.noising:
            visualize_att_patching(
                att_result.result.noising,
                pair_out_dir / "noising",
                position_labels,
                section_markers,
            )

    # Step 3: Coarse patching per component
    if coarse_results:
        for component, result in coarse_results.items():
            sweep_dir = pair_out_dir / f"sweep_{component}"
            visualize_coarse_patching(result, sweep_dir, coloring, pair=pair)

    # Step 4: Fine patching
    if fine_result is not None:
        visualize_fine_patching(
            fine_result,
            pair_out_dir,
            position_labels,
            section_markers,
        )

    # Step 5: Component comparison
    if coarse_results and len(coarse_results) > 0:
        visualize_component_comparison(
            coarse_results,
            pair_out_dir / "component_comparison",
        )


def _ensure_tokenization_and_get_coloring(
    pair_idx: int,
    pair_out_dir: Path,
    pair: "ContrastivePair | None",
    runner: Any,
    try_loading_cache: bool,
    save_token_trees_fn: Any,
) -> PairTokenColoring | None:
    """Ensure tokenization viz exists and return coloring.

    Tries multiple sources in order of preference:
    1. tokenization_viz_cache.json (full data with decoded tokens)
    2. token_tree.json (fallback with token IDs and prompt lengths)
    3. ContrastivePair object (if provided)

    The function degrades gracefully - visualizations will work even with
    partial data, just with less rich annotations.

    Returns:
        PairTokenColoring or None if unavailable
    """
    tokenization_png = pair_out_dir / "tokenization.png"

    # Try to load coloring from cache first
    viz_data = TokenizationVizData.load(pair_out_dir)

    if not tokenization_png.exists():
        # Need to create the tokenization plot
        if try_loading_cache and viz_data is not None:
            # Regenerate plot from cached data
            if visualize_tokenization_from_cache(pair_out_dir):
                log(f"[viz] Regenerated tokenization plot from cache for pair {pair_idx}")
        elif pair is not None and runner is not None:
            # Create fresh visualization with the model
            if save_token_trees_fn is not None:
                save_token_trees_fn(pair_idx, pair, pair_out_dir)
            visualize_tokenization([pair], runner, pair_out_dir, max_pairs=1)
            # Reload viz_data after creating
            viz_data = TokenizationVizData.load(pair_out_dir)

    # Get coloring - try multiple sources in order of preference
    if viz_data is not None:
        return viz_data.get_coloring()

    # Fallback 1: Try to load from token_tree.json
    viz_data_from_tree = TokenizationVizData.from_token_tree(pair_out_dir)
    if viz_data_from_tree is not None:
        log(f"[viz] Pair {pair_idx}: using token_tree.json fallback for coloring")
        return viz_data_from_tree.get_coloring()

    # Fallback 2: Use ContrastivePair if provided
    if pair is not None:
        return get_token_coloring_for_pair(pair)

    # No coloring available
    return None
