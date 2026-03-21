"""Sanity check plots for component comparison.

These should be checked first - if they fail, other results are suspect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .....activation_patching.coarse import SweepStepResults
from .utils import create_figure, save_plot, setup_grid


def plot_sanity_checks(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Generate all sanity check plots."""
    _plot_resid_delta_comparison(layer_data, output_dir)
    _plot_resid_delta_difference(layer_data, output_dir)


def _plot_resid_delta_comparison(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Compare resid_pre[L+1] vs resid_post[L] - should match."""
    resid_pre = layer_data.get("resid_pre")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    pre_layers = sorted(resid_pre.keys())
    post_layers = sorted(resid_post.keys())

    if not pre_layers or not post_layers:
        return

    fig, axes = create_figure(1, 2, figsize=(14, 6))

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]

        # resid_pre[L+1] values
        pre_next_layers = []
        pre_next_values = []
        for layer in pre_layers[1:]:
            val = resid_pre[layer].recovery if mode == "denoising" else resid_pre[layer].disruption
            if val is not None:
                pre_next_layers.append(layer - 1)
                pre_next_values.append(val)

        # resid_post[L] values
        post_values = []
        post_plot_layers = []
        for layer in post_layers[:-1]:
            val = resid_post[layer].recovery if mode == "denoising" else resid_post[layer].disruption
            if val is not None:
                post_plot_layers.append(layer)
                post_values.append(val)

        if pre_next_values:
            ax.plot(pre_next_layers, pre_next_values, "o-", color="#1f77b4", linewidth=2,
                    markersize=6, label="resid_pre[L+1]", alpha=0.8)
        if post_values:
            ax.plot(post_plot_layers, post_values, "s--", color="#d62728", linewidth=2,
                    markersize=6, label="resid_post[L]", alpha=0.8)

        ax.set_xlabel("Layer Index (L)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Sanity Check: resid_pre[L+1] vs resid_post[L] - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "resid_delta_sanity.png")


def _plot_resid_delta_difference(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot difference (resid_pre[L+1] - resid_post[L]) with tolerance band."""
    resid_pre = layer_data.get("resid_pre")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    pre_layers = sorted(resid_pre.keys())
    post_layers = sorted(resid_post.keys())

    if not pre_layers or not post_layers:
        return

    fig, axes = create_figure(1, 2, figsize=(14, 6))
    tolerance = 0.02

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]

        layers = []
        differences = []

        for layer in post_layers[:-1]:
            next_layer = layer + 1
            if next_layer in pre_layers:
                post_val = resid_post[layer].recovery if mode == "denoising" else resid_post[layer].disruption
                pre_val = resid_pre[next_layer].recovery if mode == "denoising" else resid_pre[next_layer].disruption

                if post_val is not None and pre_val is not None:
                    layers.append(layer)
                    differences.append(pre_val - post_val)

        if not layers:
            continue

        # Tolerance band
        ax.fill_between(layers, -tolerance, tolerance, alpha=0.2, color="green",
                        label=f"±{tolerance} tolerance")

        # Plot differences with color coding
        colors = ["red" if abs(d) > tolerance else "green" for d in differences]
        ax.scatter(layers, differences, c=colors, s=60, edgecolors="black", linewidth=0.5, zorder=3)
        ax.plot(layers, differences, "k-", alpha=0.3, linewidth=1)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        # Annotate worst offenders
        sorted_by_diff = sorted(zip(layers, differences), key=lambda x: abs(x[1]), reverse=True)
        for layer, diff in sorted_by_diff[:3]:
            if abs(diff) > tolerance:
                ax.annotate(f"L{layer}: {diff:.3f}", xy=(layer, diff), xytext=(5, 10),
                            textcoords="offset points", fontsize=8, color="red", fontweight="bold")

        ax.set_xlabel("Layer Index (L)", fontsize=12, fontweight="bold")
        ax.set_ylabel("resid_pre[L+1] - resid_post[L]", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Sanity Check Difference - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "resid_delta_difference.png")
