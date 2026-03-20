"""Visualization for fine-grained activation patching results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...activation_patching.fine import FinePatchingResults, HeadPatchingResults, MLPNeuronResults
from ...common import profile
from ...viz.plot_helpers import finalize_plot as _finalize_plot


@profile
def visualize_fine_patching(
    results: list[FinePatchingResults] | None,
    output_dir: Path,
) -> None:
    """Visualize fine-grained activation patching results.

    Creates:
    - Head importance rankings
    - MLP neuron importance rankings
    - Attention pattern visualizations

    Args:
        results: List of FinePatchingResults from run_fine_patching
        output_dir: Directory to save plots
    """
    if not results:
        print("[viz] No fine patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results across samples
    _plot_head_importance(results, output_dir)
    _plot_mlp_importance(results, output_dir)
    _plot_attention_summary(results, output_dir)

    print(f"[viz] Fine patching plots saved to {output_dir}")


def _plot_head_importance(
    results: list[FinePatchingResults],
    output_dir: Path,
) -> None:
    """Plot head importance rankings across layers."""
    # Collect all head results
    all_heads = []
    for result in results:
        all_heads.extend(result.get_top_heads_all_layers(10))

    if not all_heads:
        return

    # Group by (layer, head) and average scores
    head_scores: dict[tuple[int, int], list[float]] = {}
    for h in all_heads:
        key = (h.layer, h.head)
        if key not in head_scores:
            head_scores[key] = []
        head_scores[key].append(h.score)

    # Average scores
    avg_scores = {k: np.mean(v) for k, v in head_scores.items()}

    # Sort by absolute score
    sorted_heads = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    if not sorted_heads:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"L{layer}.H{head}" for (layer, head), _ in sorted_heads]
    scores = [score for _, score in sorted_heads]
    colors = ["steelblue" if s > 0 else "indianred" for s in scores]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Attribution Score", fontsize=11)
    ax.set_ylabel("Head", fontsize=11)
    ax.set_title(f"Top Attention Heads | n={len(results)} samples", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _finalize_plot(output_dir / "head_importance.png")


def _plot_mlp_importance(
    results: list[FinePatchingResults],
    output_dir: Path,
) -> None:
    """Plot MLP neuron importance rankings."""
    # Collect top neurons from all samples
    all_neurons = []
    for result in results:
        all_neurons.extend(result.get_top_neurons_all_layers(10))

    if not all_neurons:
        return

    # Group by (layer, neuron) and average contributions
    neuron_scores: dict[tuple[int, int], list[float]] = {}
    for n in all_neurons:
        key = (n.layer, n.neuron_idx)
        if key not in neuron_scores:
            neuron_scores[key] = []
        neuron_scores[key].append(n.contribution)

    # Average scores
    avg_scores = {k: np.mean(v) for k, v in neuron_scores.items()}

    # Sort by contribution
    sorted_neurons = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    if not sorted_neurons:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"L{layer}.N{neuron}" if neuron >= 0 else f"L{layer}.MLP" for (layer, neuron), _ in sorted_neurons]
    scores = [score for _, score in sorted_neurons]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, scores, color="steelblue", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Contribution Norm", fontsize=11)
    ax.set_ylabel("Neuron", fontsize=11)
    ax.set_title(f"Top MLP Neurons | n={len(results)} samples", fontsize=12, fontweight="bold")

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _finalize_plot(output_dir / "mlp_importance.png")


def _plot_attention_summary(
    results: list[FinePatchingResults],
    output_dir: Path,
) -> None:
    """Plot attention pattern summary for important heads."""
    # Collect attention patterns
    all_patterns = []
    for result in results:
        all_patterns.extend(result.attention_patterns)

    if not all_patterns:
        return

    # Group by (layer, head) and average attention to source
    attention_by_head: dict[tuple[int, int], list[float]] = {}
    for p in all_patterns:
        key = (p.layer, p.head)
        if key not in attention_by_head:
            attention_by_head[key] = []
        attention_by_head[key].append(p.mean_attention_to_source)

    # Average attention
    avg_attention = {k: np.mean(v) for k, v in attention_by_head.items()}

    # Sort by attention
    sorted_heads = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:15]

    if not sorted_heads:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"L{layer}.H{head}" for (layer, head), _ in sorted_heads]
    attentions = [attn for _, attn in sorted_heads]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, attentions, color="teal", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Attention to Source", fontsize=11)
    ax.set_title(f"Attention: Destination → Source | n={len(results)} samples", fontsize=12, fontweight="bold")

    # Add grid
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _finalize_plot(output_dir / "attention_patterns.png")


def visualize_head_layer_heatmap(
    results: list[FinePatchingResults],
    output_dir: Path,
) -> None:
    """Create heatmap of head importance across layers.

    Creates a [n_layers, n_heads] heatmap showing average importance.
    """
    if not results or not results[0].head_results:
        return

    n_layers = results[0].n_layers
    n_heads = results[0].n_heads

    # Aggregate scores
    scores_sum = np.zeros((n_layers, n_heads))
    counts = np.zeros((n_layers, n_heads))

    for result in results:
        for layer, layer_result in result.head_results.items():
            for hr in layer_result.head_results:
                scores_sum[layer, hr.head] += hr.score
                counts[layer, hr.head] += 1

    # Average (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_scores = np.where(counts > 0, scores_sum / counts, 0)

    # Only plot layers that were analyzed
    analyzed_layers = sorted(results[0].head_results.keys())
    if not analyzed_layers:
        return

    # Extract rows for analyzed layers only
    plot_scores = avg_scores[analyzed_layers, :]

    fig, ax = plt.subplots(figsize=(12, max(4, len(analyzed_layers) * 0.3)))

    im = ax.imshow(plot_scores, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attribution Score")

    ax.set_yticks(range(len(analyzed_layers)))
    ax.set_yticklabels([f"L{l}" for l in analyzed_layers], fontsize=9)
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"Head Importance | n={len(results)} samples", fontsize=12, fontweight="bold")

    plt.tight_layout()
    _finalize_plot(output_dir / "head_layer_heatmap.png")
