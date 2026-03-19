"""Layer-wise attribution score plots.

Creates line plots showing attribution scores aggregated by layer,
matching the style from the research PDF (pages 8, 10).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ....common import profile

if TYPE_CHECKING:
    from ....attribution_patching import AttributionSummary, AttrPatchAggregatedResults


def _sum_by_layer(summary: AttributionSummary, component: str | None = None) -> tuple[list[int], list[float]]:
    """Sum attribution scores by layer.

    Args:
        summary: Attribution summary
        component: Optional component filter (e.g., 'resid_post', 'attn_out', 'mlp_out')

    Returns:
        Tuple of (layers, scores) where scores[i] is the sum for layer layers[i]
    """
    layer_scores: dict[int, float] = {}

    for key, result in summary.results.items():
        # Filter by component if specified
        if component and component not in result.component:
            continue

        for layer_idx, layer in enumerate(result.layers):
            # Filter out NaN/inf values before summing
            valid_scores = result.scores[layer_idx][np.isfinite(result.scores[layer_idx])]
            if len(valid_scores) > 0:
                score = float(np.sum(valid_scores))
                layer_scores[layer] = layer_scores.get(layer, 0.0) + score

    if not layer_scores:
        return [], []

    layers = sorted(layer_scores.keys())
    scores = [layer_scores[layer] for layer in layers]
    return layers, scores


def _mean_by_layer(summary: AttributionSummary, component: str | None = None) -> tuple[list[int], list[float]]:
    """Mean attribution scores by layer.

    Args:
        summary: Attribution summary
        component: Optional component filter

    Returns:
        Tuple of (layers, scores) where scores[i] is the mean for layer layers[i]
    """
    layer_scores: dict[int, list[float]] = {}

    for key, result in summary.results.items():
        if component and component not in result.component:
            continue

        for layer_idx, layer in enumerate(result.layers):
            score = float(np.mean(result.scores[layer_idx]))
            if layer not in layer_scores:
                layer_scores[layer] = []
            layer_scores[layer].append(score)

    if not layer_scores:
        return [], []

    layers = sorted(layer_scores.keys())
    scores = [np.mean(layer_scores[layer]) for layer in layers]
    return layers, scores


@profile
def plot_layer_attribution_line(
    summary: AttributionSummary,
    output_path: Path,
    title: str = "Residual stream attribution scores",
    component: str | None = None,
    aggregation: str = "sum",
) -> None:
    """Plot attribution scores aggregated by layer as a line plot.

    Args:
        summary: Attribution summary to plot
        output_path: Path to save the plot
        title: Plot title
        component: Optional component filter
        aggregation: 'sum' or 'mean'
    """
    if aggregation == "sum":
        layers, scores = _sum_by_layer(summary, component)
    else:
        layers, scores = _mean_by_layer(summary, component)

    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, scores, linewidth=1.5, color="#1976D2")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Attribution Score")
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_multi_model_layer_attribution(
    results_by_model: dict[str, AttributionSummary],
    output_path: Path,
    title: str = "Attribution scores of residual stream at each layer",
    component: str | None = None,
) -> None:
    """Plot layer attribution for multiple models in a grid.

    Args:
        results_by_model: Dict mapping model name to attribution summary
        output_path: Path to save the plot
        title: Plot title
        component: Optional component filter
    """
    n_models = len(results_by_model)
    if n_models == 0:
        return

    # Calculate grid dimensions
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=14)

    for idx, (model_name, summary) in enumerate(results_by_model.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        layers, scores = _sum_by_layer(summary, component)
        if layers:
            ax.plot(layers, scores, linewidth=1.5, color="#1976D2")
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_title(model_name)
        ax.set_xlabel("Layer")
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.5, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.25, linewidth=0.3)
        ax.set_axisbelow(True)

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_aggregated_layer_attribution(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot layer attribution for aggregated results.

    Creates plots for denoising and noising modes.

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

        title = f"{title_prefix}Layer attribution scores ({mode})"
        plot_layer_attribution_line(
            summary,
            output_dir / f"layer_attribution_{mode}.png",
            title=title,
        )

        # Also plot by component
        for comp in ["resid_post", "attn_out", "mlp_out"]:
            layers, _ = _sum_by_layer(summary, comp)
            if layers:
                plot_layer_attribution_line(
                    summary,
                    output_dir / f"layer_attribution_{mode}_{comp}.png",
                    title=f"{title_prefix}{comp} attribution ({mode})",
                    component=comp,
                )
