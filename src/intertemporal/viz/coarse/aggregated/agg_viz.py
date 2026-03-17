"""Main entry point for aggregated coarse patching visualization.

Creates structured output with per-metric-column plots organized by:
- Sweep type (layer, position)
- Pair grouping (same_labels, different_labels)
- Label perspective (clean, corrupted, combined) - for multilabel
- Perspective (short/long clean)
- Mode (denoising/noising)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import LabelPerspective
from .data_extraction import extract_column_data
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
) -> None:
    """Create structured aggregated visualization.

    Directory structure for multilabel (n_labels > 1):
        agg_layer_sweep/
          different_labels/
            clean/                    # metrics using clean label tokens
              short/                  # clean=short perspective
                denoising/
                  core.png, probs.png, logits.png, fork.png, vocab.png, trajectory.png
                noising/
              long/
            corrupted/                # metrics using corrupted label tokens
              short/
              long/
            combined/                 # aggregated metrics across both label systems
              short/
              long/
        agg_pos_sweep/
          (same structure)

    For single-label (n_labels == 1):
        agg_layer_sweep/
          same_labels/
            short/
            long/
        ...

    Args:
        result: Aggregated coarse patching results
        output_dir: Base output directory
    """
    output_dir = Path(output_dir)

    sweep_types: list[Literal["layer", "position"]] = ["layer", "position"]
    perspectives: list[Literal["short", "long"]] = ["short", "long"]
    modes: list[Literal["denoising", "noising"]] = ["denoising", "noising"]
    columns = list(COLUMN_METRICS.keys())

    # Determine if this is multilabel
    n_labels = _get_n_labels(result)
    is_multilabel = n_labels > 1

    if is_multilabel:
        pair_groupings = ["different_labels"]
        label_perspectives: list[LabelPerspective] = ["clean", "corrupted", "combined"]
    else:
        pair_groupings = ["same_labels"]
        label_perspectives = ["clean"]

    # Get component from result
    component = result.component

    for sweep_type in sweep_types:
        sweep_dir_name = f"agg_{sweep_type}_sweep"

        for grouping in pair_groupings:
            for label_persp in label_perspectives:
                for perspective in perspectives:
                    for mode in modes:
                        # Build directory path
                        if is_multilabel:
                            dir_path = (
                                output_dir
                                / sweep_dir_name
                                / grouping
                                / label_persp
                                / perspective
                                / mode
                            )
                        else:
                            dir_path = (
                                output_dir
                                / sweep_dir_name
                                / grouping
                                / perspective
                                / mode
                            )
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
                                f"[{component}] {sweep_label} | {label_label} | {perspective.upper()}=clean | "
                                f"{mode_label} | n={result.n_samples}"
                            )
                        else:
                            title_prefix = (
                                f"[{component}] {sweep_label} | {perspective.upper()}=clean | "
                                f"{mode_label} | n={result.n_samples}"
                            )

                        for column in columns:
                            # Extract data
                            column_data = extract_column_data(
                                result,
                                column,
                                sweep_type,
                                perspective,
                                mode,
                                label_persp,
                            )

                            if not column_data.metrics:
                                continue

                            # Plot
                            output_path = dir_path / f"{column}.png"
                            plot_column(
                                column_data,
                                output_path,
                                title_prefix,
                            )

    print(f"[viz] Aggregated structured plots saved to {output_dir}")
