"""Mode comparison plots for attribution patching.

Creates bar charts and scatter plots comparing denoising vs noising
attribution scores by component and layer.

In activation/attribution patching:
- Denoising: inject clean activations into corrupted run
- Noising: inject corrupted activations into clean run
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from ....common import profile

if TYPE_CHECKING:
    from ....attribution_patching import AttributionSummary, AttrPatchAggregatedResults


ComponentType = Literal["attn", "mlp", "resid"]


def _get_component_scores_by_layer(
    summary: AttributionSummary,
    component_type: ComponentType,
) -> dict[int, float]:
    """Get summed scores by layer for a component type.

    Args:
        summary: Attribution summary
        component_type: 'attn', 'mlp', or 'resid'

    Returns:
        Dict mapping layer to summed score
    """
    layer_scores: dict[int, float] = {}

    component_patterns = {
        "attn": ["attn_out", "attn"],
        "mlp": ["mlp_out", "mlp"],
        "resid": ["resid_post", "resid"],
    }
    patterns = component_patterns.get(component_type, [component_type])

    for key, result in summary.results.items():
        if not any(p in result.component for p in patterns):
            continue

        for layer_idx, layer in enumerate(result.layers):
            score = float(np.sum(result.scores[layer_idx]))
            layer_scores[layer] = layer_scores.get(layer, 0.0) + score

    return layer_scores


@profile
def plot_mode_bar_chart(
    denoising_summary: AttributionSummary | None,
    noising_summary: AttributionSummary | None,
    output_path: Path,
    title: str = "Attribution Scores by Layer: Denoising vs Noising",
) -> None:
    """Plot bar chart comparing denoising vs noising attribution by component.

    Creates a 3-panel plot showing Attention, MLP, and Residual components.

    Args:
        denoising_summary: Attribution summary for denoising mode
        noising_summary: Attribution summary for noising mode
        output_path: Path to save the plot
        title: Plot title
    """
    if denoising_summary is None and noising_summary is None:
        return

    components: list[ComponentType] = ["attn", "mlp", "resid"]
    component_labels = ["Attention", "MLP", "Residual"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Get all layers
    all_layers = set()
    for summary in [denoising_summary, noising_summary]:
        if summary:
            for result in summary.results.values():
                all_layers.update(result.layers)
    layers = sorted(all_layers)

    if not layers:
        plt.close(fig)
        return

    bar_width = 0.35
    x = np.arange(len(layers))

    for ax_idx, (comp, comp_label) in enumerate(zip(components, component_labels)):
        ax = axes[ax_idx]

        denoise_scores = [0.0] * len(layers)
        noise_scores = [0.0] * len(layers)

        if denoising_summary:
            denoise_layer_scores = _get_component_scores_by_layer(denoising_summary, comp)
            denoise_scores = [denoise_layer_scores.get(layer, 0.0) for layer in layers]

        if noising_summary:
            noise_layer_scores = _get_component_scores_by_layer(noising_summary, comp)
            noise_scores = [noise_layer_scores.get(layer, 0.0) for layer in layers]

        ax.bar(x - bar_width / 2, denoise_scores, bar_width, label="Denoising", color="#64B5F6")
        ax.bar(x + bar_width / 2, noise_scores, bar_width, label="Noising", color="#EF5350")

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Attribution Score")
        ax.set_title(comp_label)
        ax.legend(loc="upper right")
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.5, linewidth=0.6, axis="y")
        ax.grid(True, which="minor", alpha=0.25, linewidth=0.3, axis="y")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])

    axes[-1].set_xlabel("Layer")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(l) for l in layers])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_mode_scatter(
    denoising_summary: AttributionSummary | None,
    noising_summary: AttributionSummary | None,
    output_path: Path,
    title: str = "Denoising vs Noising Attribution Scores",
) -> None:
    """Plot scatter of denoising vs noising attribution scores.

    Each point is a layer, colored by component type (attn/mlp),
    annotated with layer index.

    Args:
        denoising_summary: Attribution summary for denoising mode
        noising_summary: Attribution summary for noising mode
        output_path: Path to save the plot
        title: Plot title
    """
    if denoising_summary is None or noising_summary is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"attn": "#4CAF50", "mlp": "#FF9800"}
    components: list[ComponentType] = ["attn", "mlp"]

    for comp in components:
        denoise_layer_scores = _get_component_scores_by_layer(denoising_summary, comp)
        noise_layer_scores = _get_component_scores_by_layer(noising_summary, comp)

        common_layers = sorted(set(denoise_layer_scores.keys()) & set(noise_layer_scores.keys()))

        if not common_layers:
            continue

        denoise_vals = [denoise_layer_scores[l] for l in common_layers]
        noise_vals = [noise_layer_scores[l] for l in common_layers]

        ax.scatter(denoise_vals, noise_vals, c=colors[comp], label=comp, s=80, alpha=0.8)

        for layer, d, n in zip(common_layers, denoise_vals, noise_vals):
            ax.annotate(
                str(layer),
                (d, n),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.7,
            )

    # Add diagonal line
    all_vals = []
    for comp in components:
        all_vals.extend(_get_component_scores_by_layer(denoising_summary, comp).values())
        all_vals.extend(_get_component_scores_by_layer(noising_summary, comp).values())

    if all_vals:
        min_val, max_val = min(all_vals), max(all_vals)
        margin = (max_val - min_val) * 0.1
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "k--",
            alpha=0.3,
            label="x = y",
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Denoising Attribution Score")
    ax.set_ylabel("Noising Attribution Score")
    ax.set_title(f"{title}\n(each point = layer, annotated with layer index)")
    ax.legend(title="Component")
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_component_scatter_grid(
    agg: AttrPatchAggregatedResults,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot denoising vs noising scatter for each component in a grid.

    Args:
        agg: Aggregated attribution results
        output_path: Path to save the plot
        title_prefix: Optional prefix for title
    """
    if agg.denoising_agg is None or agg.noising_agg is None:
        return

    components: list[ComponentType] = ["attn", "mlp", "resid"]
    component_labels = ["Attention", "MLP", "Residual"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title_prefix}Denoising vs Noising by Component", fontsize=14)

    for ax, comp, label in zip(axes, components, component_labels):
        denoise_scores = _get_component_scores_by_layer(agg.denoising_agg, comp)
        noise_scores = _get_component_scores_by_layer(agg.noising_agg, comp)

        common_layers = sorted(set(denoise_scores.keys()) & set(noise_scores.keys()))
        if not common_layers:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        denoise_vals = [denoise_scores[l] for l in common_layers]
        noise_vals = [noise_scores[l] for l in common_layers]

        ax.scatter(denoise_vals, noise_vals, c="#1976D2", s=60, alpha=0.7)

        for layer, d, n in zip(common_layers, denoise_vals, noise_vals):
            ax.annotate(str(layer), (d, n), fontsize=7, alpha=0.6)

        # Diagonal
        all_vals = denoise_vals + noise_vals
        if all_vals:
            lims = [min(all_vals), max(all_vals)]
            ax.plot(lims, lims, "k--", alpha=0.3)

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Denoising Score")
        ax.set_ylabel("Noising Score")
        ax.set_title(label)
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.5, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.25, linewidth=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@profile
def plot_aggregated_mode_comparison(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot mode comparison for aggregated results.

    Args:
        agg: Aggregated attribution results
        output_dir: Directory to save plots
        title_prefix: Optional prefix for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart comparing denoising vs noising
    plot_mode_bar_chart(
        agg.denoising_agg,
        agg.noising_agg,
        output_dir / "mode_bar_chart.png",
        title=f"{title_prefix}Attribution by Component",
    )

    # Scatter plot
    if agg.denoising_agg and agg.noising_agg:
        plot_mode_scatter(
            agg.denoising_agg,
            agg.noising_agg,
            output_dir / "mode_scatter.png",
            title=f"{title_prefix}Denoising vs Noising",
        )

        plot_component_scatter_grid(
            agg,
            output_dir / "mode_scatter_by_component.png",
            title_prefix=title_prefix,
        )
