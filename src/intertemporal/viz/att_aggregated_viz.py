"""Aggregated visualization for attribution patching results."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ...attribution_patching import AttrPatchAggregatedResults, AttributionSummary
from ...attribution_patching.attribution_key import AttributionKey
from ...common import profile
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_patching_heatmap,
)
from .analysis_plots.analysis_plots_main import generate_analysis_plots
from .slice_config import CORE_SLICES, GENERATE_ALL_SLICES


# Method priority for selecting best result per component
METHOD_PRIORITY = {"eap_ig": 3, "eap": 2, "standard": 1}

# Normalization modes for matrix visualization
NormMode = Literal["raw", "percentile", "zscore", "topk"]


def _normalize_scores(scores: np.ndarray, mode: NormMode, topk_percentile: float = 95.0) -> np.ndarray:
    """Normalize attribution scores according to the specified mode.

    Args:
        scores: Raw attribution scores [n_layers, n_positions]
        mode: Normalization mode
        topk_percentile: Percentile threshold for topk mode (default 95th)

    Returns:
        Normalized scores array
    """
    if mode == "raw":
        return scores

    if mode == "percentile":
        # Map values to percentiles (0-100)
        flat = scores.flatten()
        ranks = stats.rankdata(flat, method="average")
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100 if len(ranks) > 1 else ranks * 100
        return percentiles.reshape(scores.shape)

    if mode == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(scores)
        std = np.std(scores)
        if std > 0:
            return (scores - mean) / std
        return scores - mean

    if mode == "topk":
        # Threshold at percentile, show only top values
        threshold = np.percentile(np.abs(scores), topk_percentile)
        result = scores.copy()
        # Set values below threshold to 0
        result[np.abs(result) < threshold] = 0
        return result

    return scores


def _get_norm_colormap(mode: NormMode) -> tuple[str, float | None, float | None]:
    """Get colormap and limits for normalization mode.

    Returns:
        (colormap_name, vmin, vmax) - vmin/vmax are None for auto-scaling
    """
    if mode == "percentile":
        return "RdBu_r", 0, 100
    if mode == "zscore":
        return "RdBu_r", -3, 3  # Standard z-score range
    if mode == "topk":
        return "RdBu_r", None, None  # Auto-scale for top values
    return "RdBu_r", None, None  # Raw mode


@profile
def visualize_att_aggregated(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    slice_name: str = "all",
    norm_mode: NormMode = "topk",
    topk_percentile: float = 95.0,
) -> None:
    """Visualize aggregated attribution patching results.

    Creates:
        output_dir/
          {component}_{mode}.png  (best method per component)
          matrix/
            {component}_{mode}.png  (grid of all methods for component)
          analysis/
            ...

    Args:
        agg: Aggregated attribution results
        output_dir: Output directory
        slice_name: Name of the analysis slice
        norm_mode: Normalization mode for matrix visualization:
            - "raw": No normalization, use raw attribution scores
            - "percentile": Map values to percentiles (0-100) per panel
            - "zscore": Z-score normalization (mean=0, std=1) per panel
            - "topk": Show only values above percentile threshold (default)
        topk_percentile: Percentile threshold for topk mode (default 95)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simple per-component plots (best method)
    for mode, summary, n_pairs in [
        ("denoising", agg.denoising_agg, len(agg.denoising)),
        ("noising", agg.noising_agg, len(agg.noising)),
    ]:
        if summary and summary.results:
            _visualize_best_per_component(summary, output_dir, mode, n_pairs, slice_name)

    # Matrix: consolidated grid per component/mode
    matrix_dir = output_dir / "matrix"
    _generate_matrix_plots(agg, matrix_dir, slice_name, norm_mode, topk_percentile)

    # Analysis plots
    analysis_dir = output_dir / "analysis"
    generate_analysis_plots(agg, analysis_dir, title_prefix=f"{slice_name} | ")


def _visualize_best_per_component(
    summary: AttributionSummary,
    output_dir: Path,
    mode: str,
    n_pairs: int,
    slice_name: str,
) -> None:
    """Create one plot per component using best available method."""
    if not summary.results:
        return

    first_result = next(iter(summary.results.values()))
    layers = first_result.layers
    n_positions = first_result.n_positions
    pos_labels = [f"p{i}" for i in range(n_positions)]

    mode_label = "Denoise" if mode == "denoising" else "Noise"

    # Find best method per component
    best_per_component: dict[str, tuple[str, object]] = {}

    for key_str, attr_result in summary.results.items():
        if attr_result.scores.size == 0:
            continue
        key = AttributionKey.from_str(key_str)
        if key.component not in best_per_component:
            best_per_component[key.component] = (key.method, attr_result)
        elif METHOD_PRIORITY.get(key.method, 0) > METHOD_PRIORITY.get(best_per_component[key.component][0], 0):
            best_per_component[key.component] = (key.method, attr_result)

    for component, (method, attr_result) in best_per_component.items():
        filename = f"{component}_{mode}.png"
        title = f"{method} | {component} | {mode_label} | {slice_name} | n={n_pairs}"

        config = PatchingHeatmapConfig(
            title=title,
            subtitle=f"{attr_result.n_layers} layers, {n_positions} positions",
            cbar_label="Attribution Score",
            cmap="RdBu_r",
        )
        plot_patching_heatmap(
            attr_result.scores,
            layers,
            pos_labels,
            config=config,
            save_path=output_dir / filename,
        )

    print(f"[viz] Attribution aggregated ({mode}) plots saved to {output_dir}")


def _generate_matrix_plots(
    agg: AttrPatchAggregatedResults,
    matrix_dir: Path,
    slice_name: str,
    norm_mode: NormMode = "topk",
    topk_percentile: float = 95.0,
) -> None:
    """Generate consolidated matrix plots: {component}_{mode}.png with all methods.

    Args:
        agg: Aggregated attribution results
        matrix_dir: Output directory
        slice_name: Name of the analysis slice
        norm_mode: Normalization mode for visualization:
            - "raw": No normalization, use raw attribution scores
            - "percentile": Map values to percentiles (0-100) per panel
            - "zscore": Z-score normalization (mean=0, std=1) per panel
            - "topk": Show only values above percentile threshold (default)
        topk_percentile: Percentile threshold for topk mode (default 95)
    """
    matrix_dir.mkdir(parents=True, exist_ok=True)

    cmap, vmin, vmax = _get_norm_colormap(norm_mode)
    norm_suffix = f" ({norm_mode})" if norm_mode != "raw" else ""

    for mode, summary, n_pairs in [
        ("denoising", agg.denoising_agg, len(agg.denoising)),
        ("noising", agg.noising_agg, len(agg.noising)),
    ]:
        if not summary or not summary.results:
            continue

        mode_label = "Denoise" if mode == "denoising" else "Noise"

        # Group by component
        by_component: dict[str, list[tuple[AttributionKey, object]]] = defaultdict(list)
        for key_str, attr_result in summary.results.items():
            if attr_result.scores.size == 0:
                continue
            key = AttributionKey.from_str(key_str)
            by_component[key.component].append((key, attr_result))

        # Create one consolidated plot per component
        for component, results_list in by_component.items():
            # Sort: standard first, then eap, then eap_ig
            method_order = {"standard": 0, "eap": 1, "eap_ig": 2}
            results_list.sort(key=lambda x: (method_order.get(x[0].method, 99), x[0].quadrature or ""))

            n_plots = len(results_list)
            if n_plots == 0:
                continue

            # Grid layout
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                squeeze=False,
            )

            for idx, (key, attr_result) in enumerate(results_list):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # Normalize scores according to mode
                normalized_scores = _normalize_scores(
                    attr_result.scores, norm_mode, topk_percentile
                )

                im = ax.imshow(
                    normalized_scores,
                    aspect="auto",
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)

                # Colorbar label
                if norm_mode == "percentile":
                    cbar.set_label("Percentile", fontsize=7)
                elif norm_mode == "zscore":
                    cbar.set_label("Z-score", fontsize=7)
                elif norm_mode == "topk":
                    cbar.set_label(f"Top {100-topk_percentile:.0f}%", fontsize=7)

                subtitle = key.method
                if key.quadrature:
                    subtitle += f" | {key.quadrature}"
                ax.set_title(subtitle, fontsize=10)
                ax.set_xlabel("Position", fontsize=8)
                ax.set_ylabel("Layer", fontsize=8)

            # Hide unused subplots
            for idx in range(n_plots, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].axis("off")

            fig.suptitle(
                f"{component} | {mode_label} | {slice_name} | n={n_pairs}{norm_suffix}",
                fontsize=12,
            )
            plt.tight_layout()

            plt.savefig(matrix_dir / f"{component}_{mode}.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"[viz] Attribution matrix plots saved to {matrix_dir}")


@profile
def visualize_all_att_aggregated_slices(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
) -> None:
    """Visualize aggregated attribution results for all analysis slices."""
    from .coarse.aggregated.analysis_slices import ANALYSIS_SLICES

    output_dir = Path(output_dir)

    n_samples = len(agg.denoising) or len(agg.noising)
    if not GENERATE_ALL_SLICES or n_samples <= 2:
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name in CORE_SLICES]
    else:
        slices_to_generate = ANALYSIS_SLICES

    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name
        visualize_att_aggregated(agg, slice_dir, slice_name)

    print(f"[viz] All attribution aggregated slices saved to {output_dir}")
