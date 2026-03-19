"""Top scores visualization for attribution patching.

Shows the top positive and negative scoring components,
matching the style from the research PDF (page 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ....common import profile

if TYPE_CHECKING:
    from ....attribution_patching import AttributionSummary, AttrPatchAggregatedResults


def _get_layer_component_scores(
    summary: AttributionSummary,
) -> list[tuple[int, str, float]]:
    """Get scores aggregated by (layer, component).

    Returns:
        List of (layer, component, score) tuples
    """
    scores: dict[tuple[int, str], float] = {}

    for key, result in summary.results.items():
        comp_type = "attn" if "attn" in result.component else "mlp" if "mlp" in result.component else "resid"

        for layer_idx, layer in enumerate(result.layers):
            key_tuple = (layer, comp_type)
            score = float(np.sum(result.scores[layer_idx]))
            scores[key_tuple] = scores.get(key_tuple, 0.0) + score

    return [(layer, comp, score) for (layer, comp), score in scores.items()]


@profile
def plot_top_scores_text(
    summary: AttributionSummary,
    output_path: Path,
    n_top: int = 5,
    title: str = "Top Attribution Scores",
) -> None:
    """Plot top positive and negative scores as text boxes.

    Args:
        summary: Attribution summary
        output_path: Path to save the plot
        n_top: Number of top scores to show
        title: Plot title
    """
    all_scores = _get_layer_component_scores(summary)
    if not all_scores:
        return

    # Sort by score
    sorted_scores = sorted(all_scores, key=lambda x: x[2])
    top_negative = sorted_scores[:n_top]
    top_positive = sorted_scores[-n_top:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    # Format as text
    def format_scores(scores_list: list) -> str:
        lines = []
        for layer, comp, score in scores_list:
            lines.append(f"({layer}, '{comp}'): {score:.6f}")
        return "\n".join(lines)

    # Top negative
    ax = axes[0]
    ax.text(
        0.1, 0.5,
        format_scores(top_negative),
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="#FFCDD2", alpha=0.8),
    )
    ax.set_title("Top Negative Scores")
    ax.axis("off")

    # Top positive
    ax = axes[1]
    ax.text(
        0.1, 0.5,
        format_scores(top_positive),
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="#C8E6C9", alpha=0.8),
    )
    ax.set_title("Top Positive Scores")
    ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_top_scores_bar(
    summary: AttributionSummary,
    output_path: Path,
    n_top: int = 10,
    title: str = "Top Attribution Scores by Layer",
) -> None:
    """Plot top positive and negative scores as horizontal bar chart.

    Args:
        summary: Attribution summary
        output_path: Path to save the plot
        n_top: Number of top scores to show per side
        title: Plot title
    """
    all_scores = _get_layer_component_scores(summary)
    if not all_scores:
        return

    sorted_scores = sorted(all_scores, key=lambda x: x[2])
    top_negative = sorted_scores[:n_top]
    top_positive = sorted_scores[-n_top:][::-1]

    # Combine and plot
    combined = top_negative + top_positive
    labels = [f"L{layer} ({comp})" for layer, comp, _ in combined]
    values = [score for _, _, score in combined]
    colors = ["#EF5350" if v < 0 else "#4CAF50" for v in values]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attribution Score")
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6, axis="x")
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.3, axis="x")
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_aggregated_top_scores(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot top scores for aggregated results.

    Args:
        agg: Aggregated attribution results
        output_dir: Directory to save plots
        title_prefix: Optional prefix for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode, summary in [("denoising", agg.denoising_agg), ("noising", agg.noising_agg)]:
        if summary is None:
            continue

        plot_top_scores_text(
            summary,
            output_dir / f"top_scores_text_{mode}.png",
            title=f"{title_prefix}Top Scores ({mode})",
        )

        plot_top_scores_bar(
            summary,
            output_dir / f"top_scores_bar_{mode}.png",
            title=f"{title_prefix}Top Scores by Layer ({mode})",
        )
