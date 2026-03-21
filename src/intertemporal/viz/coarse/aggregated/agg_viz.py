"""Main entry point for aggregated coarse patching visualization.

Creates structured output organized by analysis slice at top level:
- agg/all/sweep_<component>/layer_sweep/denoising/...
- agg/same_labels/sweep_<component>/position_sweep/noising/...
- etc.

All plots use short=clean perspective (long is just the inverse).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import LabelPerspective
from .analysis_slices import ANALYSIS_SLICES
from .data_extraction import extract_all_columns
from .metric_plots import plot_column
from .style import COLUMN_METRICS


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


def plot_all_aggregated_slices(
    agg_by_component: dict[str, CoarseActPatchAggregatedResults],
    output_dir: Path,
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
    """
    from ..component_comparison import plot_all_component_comparisons

    output_dir = Path(output_dir)

    from .....activation_patching.coarse import CoarseActPatchResults

    # Get first sample from each component for component_comparison plots
    results_by_component: dict[str, CoarseActPatchResults] = {}
    for comp, agg_result in agg_by_component.items():
        if agg_result.by_sample:
            first_sample = next(iter(agg_result.by_sample.values()))
            results_by_component[comp] = first_sample

    has_multi_component = len(results_by_component) > 1

    # Determine which slices to generate based on sample count
    # For small datasets, only generate the "all" slice
    first_agg = next(iter(agg_by_component.values()))
    n_samples = first_agg.n_samples
    if n_samples <= 2:
        # Only generate "all" slice for small datasets
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name == "all"]
    else:
        slices_to_generate = ANALYSIS_SLICES

    # Generate plots for each analysis slice
    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name

        # Per-component aggregated sweep plots
        for component, agg_result in agg_by_component.items():
            comp_dir = slice_dir / f"sweep_{component}"
            plot_aggregated_structured(agg_result, comp_dir, slice_name)

        # Multi-component comparison plots
        if has_multi_component:
            comp_comparison_dir = slice_dir / "component_comparison"
            comp_comparison_dir.mkdir(parents=True, exist_ok=True)
            plot_all_component_comparisons(results_by_component, comp_comparison_dir)

    print(f"[viz] All aggregated slices saved to {output_dir}")
