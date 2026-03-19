"""Score distribution histograms for attribution patching.

Creates histograms showing the distribution of EAP-IG scores,
matching the style from the research PDF (page 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ....common import profile

if TYPE_CHECKING:
    from ....attribution_patching import AttributionSummary, AttrPatchAggregatedResults


def _collect_all_scores(summary: AttributionSummary) -> np.ndarray:
    """Collect all attribution scores from a summary.

    Args:
        summary: Attribution summary

    Returns:
        1D array of all finite scores (NaN/inf filtered out)
    """
    all_scores = []
    for result in summary.results.values():
        all_scores.append(result.scores.flatten())

    if not all_scores:
        return np.array([])

    combined = np.concatenate(all_scores)
    # Filter out NaN and infinite values
    return combined[np.isfinite(combined)]


@profile
def plot_score_histogram(
    summary: AttributionSummary,
    output_path: Path,
    title: str = "EAP-IG Score Distribution",
    n_bins: int = 50,
) -> None:
    """Plot histogram of attribution scores.

    Args:
        summary: Attribution summary to plot
        output_path: Path to save the plot
        title: Plot title
        n_bins: Number of histogram bins
    """
    scores = _collect_all_scores(summary)
    if len(scores) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores, bins=n_bins, color="#1976D2", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Attribution Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6, axis="y")
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.3, axis="y")
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_scaling_histograms(
    summaries: list[tuple[str, AttributionSummary]],
    output_path: Path,
    title: str = "Scaling",
) -> None:
    """Plot histograms showing score distribution at different scales.

    Recreates the scaling analysis from the PDF (page 6), showing how
    score distributions narrow as N_sample and N_steps increase.

    Args:
        summaries: List of (label, summary) tuples, e.g. [("N=5, steps=10", summary1), ...]
        output_path: Path to save the plot
        title: Plot title
    """
    n_plots = len(summaries)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    for idx, (label, summary) in enumerate(summaries):
        ax = axes[idx]
        scores = _collect_all_scores(summary)

        if len(scores) > 0:
            ax.hist(scores, bins=30, color="#1976D2", edgecolor="white", linewidth=0.5)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.5, linewidth=0.6, axis="y")
        ax.grid(True, which="minor", alpha=0.25, linewidth=0.3, axis="y")
        ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_aggregated_histograms(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot score histograms for aggregated results.

    Args:
        agg: Aggregated attribution results
        output_dir: Directory to save plots
        title_prefix: Optional prefix for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = len(agg.denoising) + len(agg.noising)

    for mode, summary in [("denoising", agg.denoising_agg), ("noising", agg.noising_agg)]:
        if summary is None:
            continue

        title = f"{title_prefix}Score distribution ({mode}, n={summary.n_pairs})"
        plot_score_histogram(
            summary,
            output_dir / f"score_histogram_{mode}.png",
            title=title,
        )
