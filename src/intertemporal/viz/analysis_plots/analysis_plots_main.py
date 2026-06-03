"""Main entry point for generating analysis plots.

Orchestrates generation of all analysis plot types for attribution patching results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ....common import profile
from ....common.logging import log
from ..slice_config import CORE_SLICES, GENERATE_ALL_SLICES

from .layer_scores import plot_aggregated_layer_attribution
from .mode_comparison import plot_aggregated_mode_comparison
from .score_histograms import plot_aggregated_histograms
from .top_scores import plot_aggregated_top_scores

if TYPE_CHECKING:
    from ....attribution_patching import AttrPatchAggregatedResults


@profile
def generate_analysis_plots(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Generate all analysis plots for attribution patching results.

    Creates:
        output_dir/
          layer_attribution/
            layer_attribution_denoising.png
            layer_attribution_noising.png
            layer_attribution_*_resid_post.png
            ...
          histograms/
            score_histogram_denoising.png
            score_histogram_noising.png
          comparison/
            mode_bar_chart.png
            mode_scatter.png
            mode_scatter_by_component.png
          top_scores/
            top_scores_text_denoising.png
            top_scores_bar_denoising.png
            ...

    Args:
        agg: Aggregated attribution results
        output_dir: Base output directory
        title_prefix: Optional prefix for plot titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[analysis] Generating analysis plots in {output_dir}")

    # 1. Layer attribution line plots
    plot_aggregated_layer_attribution(
        agg,
        output_dir / "layer_attribution",
        title_prefix=title_prefix,
    )
    log("[analysis] Generated layer attribution plots")

    # 2. Score histograms
    plot_aggregated_histograms(
        agg,
        output_dir / "histograms",
        title_prefix=title_prefix,
    )
    log("[analysis] Generated score histograms")

    # 3. Mode comparison (denoising vs noising)
    plot_aggregated_mode_comparison(
        agg,
        output_dir / "comparison",
        title_prefix=title_prefix,
    )
    log("[analysis] Generated mode comparison plots")

    # 4. Top scores
    plot_aggregated_top_scores(
        agg,
        output_dir / "top_scores",
        title_prefix=title_prefix,
    )
    log("[analysis] Generated top scores plots")

    log(f"[analysis] All analysis plots saved to {output_dir}")


@profile
def generate_all_analysis_plots(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
) -> None:
    """Generate analysis plots for all analysis slices.

    Creates:
        output_dir/
          all/analysis/...
          same_labels/analysis/...
          ...

    Args:
        agg: Aggregated attribution results
        output_dir: Base output directory (typically agg/)
    """
    from ..coarse.aggregated.analysis_slices import ANALYSIS_SLICES

    output_dir = Path(output_dir)

    # Determine which slices to generate
    n_samples = len(agg.denoising) + len(agg.noising)
    if not GENERATE_ALL_SLICES or n_samples <= 2:
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name in CORE_SLICES]
    else:
        slices_to_generate = ANALYSIS_SLICES

    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name / "analysis"

        generate_analysis_plots(agg, slice_dir, title_prefix=f"{slice_name} | ")

    log(f"[analysis] All analysis slices saved to {output_dir}")
