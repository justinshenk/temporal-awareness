"""Main entry point for aggregated coarse patching visualization.

Creates structured output organized by analysis slice at top level:
- agg/all/sweep_<component>/layer_sweep/denoising/...
- agg/same_labels/sweep_<component>/position_sweep/noising/...
- etc.

All plots use short=clean perspective (long is just the inverse).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import LabelPerspective
from ...viz_config import CORE_SLICES, GENERATE_ALL_SLICES
from .analysis_slices import ANALYSIS_SLICES
from .data_extraction import extract_all_columns
from .metric_plots import plot_column
from .style import COLUMN_METRICS

if TYPE_CHECKING:
    from ....common.contrastive_preferences import ContrastivePreferences


def _get_n_labels(result: CoarseActPatchAggregatedResults) -> int:
    """Get the number of label pairs from the first sample."""
    for sample_result in result.by_sample.values():
        if sample_result.sanity_result:
            return sample_result.sanity_result.n_labels
        for step_results in sample_result.layer_results.values():
            for target_result in step_results.values():
                return target_result.n_labels
    return 1


def plot_aggregated_structured(
    result: CoarseActPatchAggregatedResults,
    output_dir: Path,
    analysis_slice: str = "all",
) -> None:
    """Create structured aggregated visualization for a single analysis slice.

    Directory structure:
        layer_sweep/
          denoising/
            core.png, probs.png, logits.png, fork.png, vocab.png, trajectory.png
          noising/
        position_sweep/
          denoising/
          noising/

    All plots use short=clean perspective (long=clean is redundant).

    Args:
        result: Aggregated coarse patching results
        output_dir: Output directory for this slice (e.g., agg/all/sweep_resid_post/)
        analysis_slice: Name of the analysis slice for title
    """
    output_dir = Path(output_dir)

    sweep_types: list[Literal["layer", "position"]] = ["layer", "position"]
    modes: list[Literal["denoising", "noising"]] = ["denoising", "noising"]
    columns = list(COLUMN_METRICS.keys())

    # Always use short=clean perspective
    perspective: Literal["short", "long"] = "short"

    # Determine if this is multilabel
    n_labels = _get_n_labels(result)
    is_multilabel = n_labels > 1

    if is_multilabel:
        label_perspectives: list[LabelPerspective] = ["clean", "corrupted", "combined"]
    else:
        label_perspectives = ["clean"]

    # Get component from result
    component = result.component

    for sweep_type in sweep_types:
        sweep_dir_name = f"{sweep_type}_sweep"

        for label_persp in label_perspectives:
            for mode in modes:
                # Build directory path
                if is_multilabel:
                    dir_path = output_dir / sweep_dir_name / label_persp / mode
                else:
                    dir_path = output_dir / sweep_dir_name / mode
                dir_path.mkdir(parents=True, exist_ok=True)

                # Build title prefix (compact format)
                sweep_label = "Layer" if sweep_type == "layer" else "Pos"
                mode_label = "Denoise" if mode == "denoising" else "Noise"
                if is_multilabel:
                    label_label = {
                        "clean": "CleanLbl",
                        "corrupted": "CorruptLbl",
                        "combined": "Combined",
                    }[label_persp]
                    title_prefix = (
                        f"[{component}] {sweep_label} | {label_label} | "
                        f"{mode_label} | {analysis_slice} | n={result.n_samples}"
                    )
                else:
                    title_prefix = (
                        f"[{component}] {sweep_label} | "
                        f"{mode_label} | {analysis_slice} | n={result.n_samples}"
                    )

                # Extract ALL columns at once (much more efficient)
                all_column_data = extract_all_columns(
                    result,
                    sweep_type,
                    perspective,
                    mode,
                    label_persp,
                )

                for column in columns:
                    column_data = all_column_data.get(column)
                    if not column_data or not column_data.metrics:
                        continue

                    # Plot
                    output_path = dir_path / f"{column}.png"
                    plot_column(
                        column_data,
                        output_path,
                        title_prefix,
                    )


def _load_horizon_indices_from_cache(
    exp_dir: Path | None,
    slice_name: str,
) -> list[int] | None:
    """Try to load slice pair indices from precomputed horizon analysis files.

    Args:
        exp_dir: Experiment directory containing horizon analysis files
        slice_name: Name of the slice (horizon, no_horizon, half_horizon)

    Returns:
        List of pair indices if found, None otherwise
    """
    import json

    if exp_dir is None:
        return None

    # Map slice name to horizon analysis file and key
    horizon_mapping = {
        "horizon": ("horizon.json", ["clean_greater", "corrupted_greater", "equal"]),
        "no_horizon": ("no_horizon.json", ["pair_indices"]),
        "half_horizon": ("half_horizon.json", ["clean_has_horizon", "corrupted_has_horizon"]),
    }

    if slice_name not in horizon_mapping:
        return None

    filename, keys = horizon_mapping[slice_name]
    filepath = exp_dir / filename

    if not filepath.exists():
        return None

    try:
        with open(filepath) as f:
            data = json.load(f)
        # Collect all indices from relevant keys
        indices = []
        for key in keys:
            if key in data and isinstance(data[key], list):
                indices.extend(data[key])
        return sorted(set(indices))
    except (json.JSONDecodeError, KeyError):
        return None


def _get_slice_pair_indices(
    slice_name: str,
    pref_pairs: list["ContrastivePreferences"] | None,
    exp_dir: Path | None = None,
) -> tuple[list[int] | None, bool]:
    """Get pair indices that pass the slice requirements.

    Args:
        slice_name: Name of the analysis slice
        pref_pairs: List of ContrastivePreferences, indexed by pair_idx
        exp_dir: Experiment directory (for loading cached horizon analysis)

    Returns:
        Tuple of (pair_indices, can_compute):
        - pair_indices: List of pair indices that pass the slice, or None if no filtering needed
        - can_compute: True if we can compute this slice (False when pref_pairs needed but missing)
    """
    from .analysis_slices import get_analysis_slice

    # "all" slice means no filtering - always computable
    if slice_name == "all":
        return None, True

    # Try to compute from pref_pairs first (most accurate)
    if pref_pairs is not None:
        analysis_slice = get_analysis_slice(slice_name)
        if analysis_slice is not None:
            indices = []
            for pair_idx, pref in enumerate(pref_pairs):
                if analysis_slice.req.passes(pref):
                    indices.append(pair_idx)
            return indices, True

    # Fallback: try to load from cached horizon analysis
    cached_indices = _load_horizon_indices_from_cache(exp_dir, slice_name)
    if cached_indices is not None:
        return cached_indices, True

    # Can't compute this slice
    return None, False


def plot_all_aggregated_slices(
    agg_by_component: dict[str, CoarseActPatchAggregatedResults],
    output_dir: Path,
    pref_pairs: list["ContrastivePreferences"] | None = None,
    exp_dir: Path | None = None,
) -> None:
    """Create aggregated visualizations for all analysis slices and components.

    Directory structure:
        output_dir/
          all/
            sweep_resid_post/
              layer_sweep/denoising/...
            sweep_attn_out/
            component_comparison/
          same_labels/
            sweep_resid_post/
            ...

    Args:
        agg_by_component: Dict mapping component name to aggregated results
        output_dir: Base output directory (e.g., agg/)
        pref_pairs: List of ContrastivePreferences for slice filtering
        exp_dir: Experiment directory for loading cached horizon analysis
    """
    from ..component_comparison import plot_all_component_comparisons

    output_dir = Path(output_dir)

    from .....activation_patching.coarse import CoarseActPatchResults

    # Determine which slices to generate
    first_agg = next(iter(agg_by_component.values()))
    n_samples = first_agg.n_samples
    if not GENERATE_ALL_SLICES or n_samples <= 2:
        # Only generate core slices (all, horizon, no_horizon, half_horizon)
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name in CORE_SLICES]
    else:
        slices_to_generate = ANALYSIS_SLICES

    # Generate plots for each analysis slice
    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name

        # Get pair indices for this slice
        pair_indices, can_compute = _get_slice_pair_indices(slice_name, pref_pairs, exp_dir)

        # Skip slices we can't compute (missing pref_pairs)
        if not can_compute:
            continue

        # Skip slices with no matching pairs
        if pair_indices is not None and len(pair_indices) == 0:
            continue

        # Filter aggregated results for this slice
        if pair_indices is not None:
            filtered_agg = {
                comp: agg.filter_by_pairs(pair_indices)
                for comp, agg in agg_by_component.items()
            }
            # Skip if all components are empty after filtering
            if all(agg.n_samples == 0 for agg in filtered_agg.values()):
                continue
        else:
            filtered_agg = agg_by_component

        # Per-component aggregated sweep plots
        for component, agg_result in filtered_agg.items():
            if agg_result.n_samples == 0:
                continue
            comp_dir = slice_dir / f"sweep_{component}"
            plot_aggregated_structured(agg_result, comp_dir, slice_name)

        # Multi-component comparison plots
        has_multi_component = len(filtered_agg) > 1
        if has_multi_component:
            # Get first sample from each component for component_comparison plots
            results_by_component: dict[str, CoarseActPatchResults] = {}
            for comp, agg_result in filtered_agg.items():
                if agg_result.by_sample:
                    first_sample = next(iter(agg_result.by_sample.values()))
                    results_by_component[comp] = first_sample

            if results_by_component:
                comp_comparison_dir = slice_dir / "sweep_component_comparison"
                comp_comparison_dir.mkdir(parents=True, exist_ok=True)
                plot_all_component_comparisons(results_by_component, comp_comparison_dir)

    print(f"[viz] All aggregated slices saved to {output_dir}")
