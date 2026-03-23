"""Visualization for MLP layer contribution analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ...common.logging import log
from ..experiments.mlp_analysis import MLPAggregatedResults, MLPPairResult, NeuronInfo


# Plot styling
DPI = 150
POSITIVE_COLOR = "#4ECDC4"  # Teal for positive contributions
NEGATIVE_COLOR = "#FF6B6B"  # Red for negative contributions
CLEAN_COLOR = "#2ecc71"  # Green for clean activations
CORRUPTED_COLOR = "#9b59b6"  # Purple for corrupted activations
BAR_ALPHA = 0.8
GRID_ALPHA = 0.3

# Target layer for detailed neuron analysis
TARGET_LAYER = 31


def visualize_mlp_analysis(
    agg: MLPAggregatedResults,
    output_dir: Path,
) -> None:
    """Generate all MLP analysis visualizations.

    Args:
        agg: Aggregated MLP analysis results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[mlp_viz] No pair results to visualize")
        return

    n_plots = 0

    # Layer contribution bar chart
    _plot_layer_contributions(agg, output_dir / "mlp_layer_contributions.png")
    n_plots += 1

    # Layer contribution heatmap across pairs
    if agg.n_pairs > 1:
        _plot_contribution_heatmap(agg, output_dir / "mlp_contribution_heatmap.png")
        n_plots += 1

    # Per-pair layer contributions
    _plot_per_pair_contributions(agg, output_dir / "mlp_per_pair_contributions.png")
    n_plots += 1

    # Check if we have neuron-level data for the target layer
    has_neuron_data = _has_neuron_data(agg, TARGET_LAYER)
    if has_neuron_data:
        # 10. Neuron activation difference heatmap across layers
        _plot_neuron_activation_heatmap_across_layers(
            agg, output_dir / "neuron_activation_heatmap_layers.png"
        )
        n_plots += 1

    log(f"[mlp_viz] Generated {n_plots} plots in {output_dir}")


def _plot_layer_contributions(agg: MLPAggregatedResults, output_path: Path) -> None:
    """Plot mean layer contributions as a bar chart."""
    layers = agg.layers_analyzed
    if not layers:
        return

    # Compute mean contribution per layer
    mean_contribs = []
    std_contribs = []
    for layer in layers:
        contribs = []
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if lr:
                contribs.append(lr.total_logit_contribution)
        if contribs:
            mean_contribs.append(np.mean(contribs))
            std_contribs.append(np.std(contribs) if len(contribs) > 1 else 0)
        else:
            mean_contribs.append(0)
            std_contribs.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    colors = [POSITIVE_COLOR if c >= 0 else NEGATIVE_COLOR for c in mean_contribs]

    bars = ax.bar(x, mean_contribs, color=colors, alpha=BAR_ALPHA,
                  yerr=std_contribs if agg.n_pairs > 1 else None,
                  capsize=3, ecolor='gray')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_contribs)):
        height = bar.get_height()
        sign = "+" if val >= 0 else ""
        ax.annotate(f'{sign}{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title(f'MLP Layer Contributions to Decision\n({agg.n_pairs} pairs)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Supports clean prediction'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Opposes clean prediction')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_contribution_heatmap(agg: MLPAggregatedResults, output_path: Path) -> None:
    """Plot heatmap of layer contributions across pairs."""
    layers = agg.layers_analyzed
    if not layers or agg.n_pairs < 2:
        return

    # Build contribution matrix
    matrix = []
    for pr in agg.pair_results:
        row = []
        for layer in layers:
            lr = pr.get_layer_result(layer)
            row.append(lr.total_logit_contribution if lr else 0)
        matrix.append(row)

    matrix = np.array(matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), max(4, agg.n_pairs * 0.4)))

    # Use diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Logit Contribution', fontsize=10)

    # Labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_yticks(np.arange(agg.n_pairs))
    ax.set_yticklabels([f'Pair {i}' for i in range(agg.n_pairs)])
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Pair', fontsize=12)
    ax.set_title('MLP Layer Contributions Across Pairs', fontsize=14)

    # Add text annotations
    for i in range(agg.n_pairs):
        for j in range(len(layers)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_per_pair_contributions(agg: MLPAggregatedResults, output_path: Path) -> None:
    """Plot layer contributions for each pair as grouped bars."""
    layers = agg.layers_analyzed
    if not layers:
        return

    n_pairs = min(agg.n_pairs, 10)  # Limit to 10 pairs for readability

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.8 / n_pairs

    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    for i, pr in enumerate(agg.pair_results[:n_pairs]):
        contribs = []
        for layer in layers:
            lr = pr.get_layer_result(layer)
            contribs.append(lr.total_logit_contribution if lr else 0)

        offset = (i - n_pairs / 2 + 0.5) * width
        ax.bar(x + offset, contribs, width, label=f'Pair {i}',
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title('MLP Layer Contributions by Pair', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _has_neuron_data(agg: MLPAggregatedResults, layer: int) -> bool:
    """Check if aggregated results have per-neuron data for a layer."""
    for pr in agg.pair_results:
        lr = pr.get_layer_result(layer)
        if lr and lr.top_neurons:
            return True
    return False


def _get_top_neurons_for_layer(
    agg: MLPAggregatedResults, layer: int, n: int = 20
) -> list[tuple[int, float, float, float]]:
    """Get top neurons by mean activation difference across pairs.

    Returns:
        List of (neuron_idx, mean_act_diff, mean_clean_act, mean_corrupted_act)
    """
    neuron_data: dict[int, list[NeuronInfo]] = {}

    for pr in agg.pair_results:
        lr = pr.get_layer_result(layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_data:
                neuron_data[ni.neuron_idx] = []
            neuron_data[ni.neuron_idx].append(ni)

    # Compute mean statistics
    results = []
    for neuron_idx, infos in neuron_data.items():
        mean_act_diff = np.mean([ni.activation_diff for ni in infos])
        mean_clean_act = np.mean([ni.clean_activation for ni in infos])
        mean_corrupted_act = np.mean([ni.corrupted_activation for ni in infos])
        results.append((neuron_idx, mean_act_diff, mean_clean_act, mean_corrupted_act))

    # Sort by absolute activation difference
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:n]


def _plot_neuron_activation_heatmap_across_layers(
    agg: MLPAggregatedResults, output_path: Path
) -> None:
    """Plot 10: Heatmap of top neuron activation differences across layers.

    Rows: top 20 neurons by activation difference at TARGET_LAYER
    Columns: layers analyzed
    Color: activation difference at each layer
    """
    layers = agg.layers_analyzed
    if not layers:
        return

    # Get top neurons at target layer
    top_neurons = _get_top_neurons_for_layer(agg, TARGET_LAYER, n=20)
    if not top_neurons:
        return

    neuron_indices = [n[0] for n in top_neurons]

    # Build matrix: for each neuron, get its activation diff at each layer
    matrix = np.zeros((len(neuron_indices), len(layers)))

    for j, layer in enumerate(layers):
        layer_top = _get_top_neurons_for_layer(agg, layer, n=100)
        layer_map = {n[0]: n[1] for n in layer_top}  # neuron_idx -> act_diff

        for i, neuron_idx in enumerate(neuron_indices):
            matrix[i, j] = layer_map.get(neuron_idx, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.2), max(6, len(neuron_indices) * 0.3)))

    vmax = max(abs(matrix.min()), abs(matrix.max())) if matrix.size > 0 else 1
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Difference (clean - corrupted)', fontsize=10)

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_yticks(np.arange(len(neuron_indices)))
    ax.set_yticklabels([f'N{n}' for n in neuron_indices])
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'Neuron (top 20 from L{TARGET_LAYER})', fontsize=12)
    ax.set_title(f'Neuron Activation Differences Across Layers\n(Top 20 neurons from L{TARGET_LAYER})', fontsize=14)

    # Add text annotations for non-zero values
    for i in range(len(neuron_indices)):
        for j in range(len(layers)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def visualize_mlp_pair(
    result: MLPPairResult,
    output_dir: Path,
) -> None:
    """Generate MLP analysis visualizations for a single pair.

    Args:
        result: MLP analysis results for one pair
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result.layer_results:
        return

    n_plots = 0

    # Layer contribution bar chart for this pair
    _plot_pair_layer_contributions(result, output_dir / "mlp_layer_contributions.png")
    n_plots += 1

    # Check for neuron-level data at target layer
    target_lr = result.get_layer_result(TARGET_LAYER)
    has_neuron_data = target_lr is not None and len(target_lr.top_neurons) > 0

    if has_neuron_data:
        # 9. Neuron activation difference ranked bar chart
        _plot_neuron_activation_diff_ranked(
            result, TARGET_LAYER, output_dir / f"neuron_activation_diff_L{TARGET_LAYER}.png"
        )
        n_plots += 1

        # 12. Neuron activation distribution (clean vs corrupted)
        _plot_neuron_activation_distribution(
            result, TARGET_LAYER, output_dir / f"neuron_activation_dist_L{TARGET_LAYER}.png"
        )
        n_plots += 1

        # 13. Neuron output direction alignment
        _plot_neuron_output_alignment(
            result, TARGET_LAYER, output_dir / f"neuron_output_alignment_L{TARGET_LAYER}.png"
        )
        n_plots += 1

        # 14. Neuron contribution decomposition
        _plot_neuron_contribution_decomposition(
            result, TARGET_LAYER, output_dir / f"neuron_contrib_decomp_L{TARGET_LAYER}.png"
        )
        n_plots += 1

        # 16. Two-dimensional neuron activation scatter
        _plot_neuron_activation_scatter_2d(
            result, TARGET_LAYER, output_dir / f"neuron_activation_scatter_L{TARGET_LAYER}.png"
        )
        n_plots += 1

    if n_plots > 0:
        log(f"[mlp_viz] Generated {n_plots} pair plots in {output_dir}")


def _plot_pair_layer_contributions(result: MLPPairResult, output_path: Path) -> None:
    """Plot layer contributions for a single pair."""
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    # Get contributions
    contribs = [lr.total_logit_contribution for lr in result.layer_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    colors = [POSITIVE_COLOR if c >= 0 else NEGATIVE_COLOR for c in contribs]

    bars = ax.bar(x, contribs, color=colors, alpha=BAR_ALPHA)

    # Add value labels on bars
    for bar, val in zip(bars, contribs):
        height = bar.get_height()
        sign = "+" if val >= 0 else ""
        ax.annotate(f'{sign}{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: MLP Layer Contributions\n(Position {result.position})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    # Add legend
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Supports clean prediction'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Opposes clean prediction')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_neuron_activation_diff_ranked(
    result: MLPPairResult, layer: int, output_path: Path, n_top: int = 50
) -> None:
    """Plot 9: Neuron activation difference ranked bar chart.

    X-axis: neuron index, sorted by |clean_activation - corrupted_activation|
    Y-axis: activation difference (signed)
    """
    lr = result.get_layer_result(layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons sorted by absolute activation difference
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.activation_diff), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(neurons))
    act_diffs = [n.activation_diff for n in neurons]
    neuron_ids = [n.neuron_idx for n in neurons]

    colors = [POSITIVE_COLOR if d >= 0 else NEGATIVE_COLOR for d in act_diffs]
    ax.bar(x, act_diffs, color=colors, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index (sorted by |activation diff|)', fontsize=12)
    ax.set_ylabel('Activation Difference (clean - corrupted)', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Activation Differences\n(Top {len(neurons)} neurons)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Higher in clean'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Higher in corrupted')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_neuron_activation_distribution(
    result: MLPPairResult, layer: int, output_path: Path, n_top: int = 10
) -> None:
    """Plot 12: Neuron activation distribution (clean vs corrupted).

    Each panel shows one neuron with clean vs corrupted activation values.
    """
    lr = result.get_layer_result(layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons by logit contribution
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:n_top]

    if not neurons:
        return

    n_neurons = len(neurons)
    n_cols = min(5, n_neurons)
    n_rows = (n_neurons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_neurons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, neuron in enumerate(neurons):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Plot clean and corrupted activations as vertical lines
        clean_act = neuron.clean_activation
        corrupted_act = neuron.corrupted_activation

        ax.axvline(x=clean_act, color=CLEAN_COLOR, linewidth=3, label='Clean')
        ax.axvline(x=corrupted_act, color=CORRUPTED_COLOR, linewidth=3, label='Corrupted')

        # Add difference annotation
        diff = clean_act - corrupted_act
        mid = (clean_act + corrupted_act) / 2

        # Draw arrow between the two
        ax.annotate('', xy=(clean_act, 0.5), xytext=(corrupted_act, 0.5),
                   arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax.text(mid, 0.6, f'{diff:+.2f}', ha='center', fontsize=9, color='gray')

        ax.set_title(f'N{neuron.neuron_idx}', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Hide empty subplots
    for idx in range(n_neurons, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Pair {result.pair_idx}: L{layer} Top Neuron Activations\n(by patching effect)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_neuron_output_alignment(
    result: MLPPairResult, layer: int, output_path: Path, n_top: int = 30
) -> None:
    """Plot 13: Neuron output direction alignment.

    Bar chart: X = neuron, Y = cosine similarity (W_out[neuron] @ logit_direction)
    """
    lr = result.get_layer_result(layer)
    if not lr or not lr.top_neurons:
        return

    # Get neurons with alignment data
    neurons = [n for n in lr.top_neurons if n.w_out_logit_alignment != 0]
    neurons = sorted(neurons, key=lambda n: abs(n.w_out_logit_alignment), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(neurons))
    alignments = [n.w_out_logit_alignment for n in neurons]
    neuron_ids = [n.neuron_idx for n in neurons]

    colors = [POSITIVE_COLOR if a >= 0 else NEGATIVE_COLOR for a in alignments]
    ax.bar(x, alignments, color=colors, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('W_out alignment with logit direction', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Output Direction Alignment\n(W_out[neuron] @ logit_direction)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Aligned with clean'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Aligned with corrupted')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_neuron_contribution_decomposition(
    result: MLPPairResult, layer: int, output_path: Path, n_top: int = 30
) -> None:
    """Plot 14: Neuron contribution decomposition (clean vs corrupted).

    X-axis: neuron index (top 30)
    Stacked bar: contribution in clean run vs corrupted run
    Contribution = activation * W_out_logit_alignment
    """
    lr = result.get_layer_result(layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons by contribution magnitude
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(neurons))
    neuron_ids = [n.neuron_idx for n in neurons]

    # Compute contributions: activation * w_out_alignment
    clean_contribs = [n.clean_activation * n.w_out_logit_alignment for n in neurons]
    corrupted_contribs = [n.corrupted_activation * n.w_out_logit_alignment for n in neurons]

    width = 0.35
    ax.bar(x - width/2, clean_contribs, width, label='Clean contribution',
           color=CLEAN_COLOR, alpha=BAR_ALPHA)
    ax.bar(x + width/2, corrupted_contribs, width, label='Corrupted contribution',
           color=CORRUPTED_COLOR, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('Contribution (activation * W_out alignment)', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Contribution Decomposition\n(Position {result.position})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_neuron_activation_scatter_2d(
    result: MLPPairResult, layer: int, output_path: Path
) -> None:
    """Plot 16: Two-dimensional neuron activation scatter.

    X-axis: activation of top neuron
    Y-axis: activation of second neuron
    Two points: clean and corrupted
    """
    lr = result.get_layer_result(layer)
    if not lr or len(lr.top_neurons) < 2:
        return

    # Get top 2 neurons by contribution
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:2]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract activations
    n1, n2 = neurons[0], neurons[1]

    clean_x, clean_y = n1.clean_activation, n2.clean_activation
    corr_x, corr_y = n1.corrupted_activation, n2.corrupted_activation

    # Plot points
    ax.scatter([clean_x], [clean_y], s=200, c=CLEAN_COLOR, marker='o',
              label='Clean', edgecolors='black', linewidths=2, zorder=3)
    ax.scatter([corr_x], [corr_y], s=200, c=CORRUPTED_COLOR, marker='s',
              label='Corrupted', edgecolors='black', linewidths=2, zorder=3)

    # Draw arrow from corrupted to clean
    ax.annotate('', xy=(clean_x, clean_y), xytext=(corr_x, corr_y),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    # Add coordinate labels
    ax.annotate(f'({clean_x:.2f}, {clean_y:.2f})', xy=(clean_x, clean_y),
               xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax.annotate(f'({corr_x:.2f}, {corr_y:.2f})', xy=(corr_x, corr_y),
               xytext=(10, -15), textcoords='offset points', fontsize=9)

    ax.set_xlabel(f'Neuron {n1.neuron_idx} activation', fontsize=12)
    ax.set_ylabel(f'Neuron {n2.neuron_idx} activation', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Top 2 Neuron Activation Space', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=GRID_ALPHA)

    # Make plot square
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
