"""Sweep plot visualization for coarse activation patching.

Provides unified plotting for layer and position sweeps with
2x6 subplot grids (denoising on top, noising on bottom).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt

from ....activation_patching import IntervenedChoiceMetrics
from ....activation_patching.coarse import SweepStepResults
from ....viz.token_coloring import PairTokenColoring
from .columns import core, fork, logits, probs, trajectory, vocab
from .helpers import (
    add_boundary_legend,
    add_token_type_legend,
    add_xaxis_boundary_markers,
    color_xaxis_ticks,
    get_tick_spacing,
)


# ============================================================================
#  Data Extraction
# ============================================================================


def _extract_metrics_for_mode(
    sweep_data: SweepStepResults,
    x_values: Sequence[int],
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
) -> list[IntervenedChoiceMetrics]:
    """Extract metrics for each x value in a sweep.

    Uses cached metrics if available (from pop_heavy), otherwise computes from choice.

    Args:
        sweep_data: Results mapping x values to ActPatchTargetResult
        x_values: Sorted list of x values (layers or positions)
        mode: "denoising" or "noising"
        clean_traj: Which trajectory is considered "clean" ("short" or "long")

    Returns:
        List of IntervenedChoiceMetrics, one per x value
    """
    metrics = []
    for x in x_values:
        target_result = sweep_data[x]
        # Switch perspective if viewing from long-term's point of view
        if clean_traj == "long":
            target_result = target_result.switch()
        # Use getter which returns cached metrics if available
        if mode == "denoising":
            metrics.append(target_result.get_denoising_metrics())
        else:
            metrics.append(target_result.get_noising_metrics())
    return metrics


# ============================================================================
#  Row Plotting
# ============================================================================


def _plot_sweep_row(
    axes_row: Sequence[plt.Axes],
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    mode: Literal["denoising", "noising"],
    tick_positions: Sequence[int],
    xlabel: str,
    coloring: PairTokenColoring | None = None,
) -> list[plt.Axes | None]:
    """Plot a single row of sweep metrics (6 columns).

    Args:
        axes_row: 6 axes for this row
        x_values: X-axis values (layers or positions)
        metrics: Metrics for each x value
        mode: "denoising" or "noising"
        tick_positions: X-axis tick positions
        xlabel: X-axis label (e.g., "Layer" or "Position")
        coloring: Optional token coloring for position tick colors

    Returns:
        List of 6 secondary axes (one per column) for y-axis synchronization
    """
    # All columns return their secondary axis for synchronization
    secondary_axes: list[plt.Axes | None] = []

    # Column 0: Core metrics
    secondary_axes.append(core.plot(axes_row[0], x_values, metrics, mode, tick_positions, xlabel))

    # Column 1: Probs/Logprobs
    secondary_axes.append(probs.plot(axes_row[1], x_values, metrics, tick_positions, xlabel))

    # Column 2: Logits
    secondary_axes.append(logits.plot(axes_row[2], x_values, metrics, tick_positions, xlabel))

    # Column 3: Fork metrics
    secondary_axes.append(fork.plot(axes_row[3], x_values, metrics, tick_positions, xlabel))

    # Column 4: Vocab metrics
    secondary_axes.append(vocab.plot(axes_row[4], x_values, metrics, tick_positions, xlabel))

    # Column 5: Trajectory metrics
    secondary_axes.append(trajectory.plot(axes_row[5], x_values, metrics, tick_positions, xlabel))

    # Color x-axis ticks if coloring is provided (for position sweeps)
    # Use tick_positions (already subsampled) not x_values to avoid overwriting
    if coloring:
        for ax in axes_row:
            color_xaxis_ticks(ax, list(tick_positions), coloring)

    return secondary_axes


# ============================================================================
#  Y-Axis Synchronization
# ============================================================================


def _synchronize_y_axes(
    axes: Sequence[Sequence[plt.Axes]],
    secondary_axes_by_row: list[list[plt.Axes | None]],
) -> None:
    """Synchronize y-axes across rows for all columns.

    Both primary (left) and secondary (right) y-axes are synchronized so that
    the same column in different rows shares the same scale.

    Args:
        axes: 2D array of primary axes (rows x cols)
        secondary_axes_by_row: List of [6 secondary axes] for each row
    """
    n_cols = len(axes[0])

    # Synchronize each column's PRIMARY axes between rows
    for col in range(n_cols):
        y_mins = [axes[row][col].get_ylim()[0] for row in range(2)]
        y_maxs = [axes[row][col].get_ylim()[1] for row in range(2)]
        shared_ylim = (min(y_mins), max(y_maxs))
        for row in range(2):
            axes[row][col].set_ylim(shared_ylim)

    # Synchronize each column's SECONDARY axes between rows
    for col in range(n_cols):
        col_sec_axes = [secondary_axes_by_row[row][col] for row in range(2)]
        # Skip if any are None (e.g., logits with no data)
        if None in col_sec_axes:
            continue
        y_mins = [ax.get_ylim()[0] for ax in col_sec_axes]
        y_maxs = [ax.get_ylim()[1] for ax in col_sec_axes]
        shared_ylim = (min(y_mins), max(y_maxs))
        for ax in col_sec_axes:
            ax.set_ylim(shared_ylim)

    # Additionally sync Fork (col 3) and Vocab (col 4) PRIMARY axes together
    fork_vocab_ymins = [axes[r][c].get_ylim()[0] for r in range(2) for c in [3, 4]]
    fork_vocab_ymaxs = [axes[r][c].get_ylim()[1] for r in range(2) for c in [3, 4]]
    fork_vocab_ylim = (min(fork_vocab_ymins), max(fork_vocab_ymaxs))
    for row in range(2):
        axes[row][3].set_ylim(fork_vocab_ylim)
        axes[row][4].set_ylim(fork_vocab_ylim)

    # Additionally sync Fork (col 3) and Vocab (col 4) SECONDARY axes together
    fork_vocab_sec = []
    for row in range(2):
        if secondary_axes_by_row[row][3] is not None:
            fork_vocab_sec.append(secondary_axes_by_row[row][3])
        if secondary_axes_by_row[row][4] is not None:
            fork_vocab_sec.append(secondary_axes_by_row[row][4])
    if fork_vocab_sec:
        sec_ymins = [ax.get_ylim()[0] for ax in fork_vocab_sec]
        sec_ymaxs = [ax.get_ylim()[1] for ax in fork_vocab_sec]
        sec_ylim = (min(sec_ymins), max(sec_ymaxs))
        for ax in fork_vocab_sec:
            ax.set_ylim(sec_ylim)


# ============================================================================
#  Main Plot Functions
# ============================================================================


def plot_layer_sweep(
    layer_data: SweepStepResults,
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    component: str = "resid_post",
) -> None:
    """Plot layer sweep with 2x6 subplots (denoising/noising x 6 metric columns).

    Args:
        layer_data: Results mapping layer -> ActPatchTargetResult
        output_dir: Directory to save output
        step_size: Step size used in the sweep
        clean_traj: Which trajectory is "clean" ("short" or "long")
        component: Component being patched (for plot title)
    """
    layers = sorted(layer_data.keys())
    if not layers:
        return

    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=(52, 18), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Title with baseline info
    baseline_info = _get_baseline_info(layer_data, layers[0])
    fig.suptitle(
        f"Coarse Layer Sweep [{component}], Clean = {clean_traj}, Steps = {step_size}\n{baseline_info}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    tick_positions = layers[:: get_tick_spacing(len(layers))]

    # Adjust spacing: more horizontal padding, room for legends between rows
    fig.subplots_adjust(left=0.045, right=0.96, top=0.91, bottom=0.08, wspace=0.40, hspace=0.48)

    # Plot each row and collect secondary axes for synchronization
    secondary_axes_by_row: list[list[plt.Axes | None]] = []
    for row_idx, mode in enumerate(["denoising", "noising"]):
        metrics = _extract_metrics_for_mode(layer_data, layers, mode, clean_traj)
        row_secondary = _plot_sweep_row(
            axes[row_idx], layers, metrics, mode, tick_positions, "Layer"
        )
        secondary_axes_by_row.append(row_secondary)

        # Row label - positioned based on subplot locations
        row_label = "Denoising" if mode == "denoising" else "Noising"
        fig.text(
            0.012,
            0.72 - row_idx * 0.44,
            row_label,
            fontsize=15,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    _synchronize_y_axes(axes, secondary_axes_by_row)

    # Save
    save_path = output_dir / f"coarse_layer_sweep_{clean_traj}_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_position_sweep(
    pos_data: SweepStepResults,
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    coloring: PairTokenColoring | None = None,
    component: str = "resid_post",
) -> None:
    """Plot position sweep with 2x6 subplots (denoising/noising x 6 metric columns).

    Args:
        pos_data: Results mapping position -> ActPatchTargetResult
        output_dir: Directory to save output
        step_size: Step size used in the sweep
        clean_traj: Which trajectory is "clean" ("short" or "long")
        coloring: Optional token coloring for position tick colors
        component: Component being patched (for plot title)
    """
    positions = sorted(pos_data.keys())
    if not positions:
        return

    # Create figure (extra height for bottom legends)
    fig, axes = plt.subplots(2, 6, figsize=(52, 20), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Title
    fig.suptitle(
        f"Coarse Position Sweep [{component}], Clean = {clean_traj}, Steps = {step_size}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    tick_positions = positions[:: get_tick_spacing(len(positions))]

    # Get boundary markers from BOTH trajectories to ensure consistent x-axis
    # In intertemporal experiments: short=clean, long=corrupted
    prompt_boundary = None
    choice_div_pos = None
    all_boundaries: list[int] = []
    if coloring:
        # Get markers for current trajectory
        traj_type = "clean" if clean_traj == "short" else "corrupted"
        section_markers = coloring.get_section_markers(traj_type)
        prompt_boundary = section_markers.get("prompt_boundary")
        choice_div_pos = section_markers.get("choice_div_pos")

        # Collect ALL boundary positions from both trajectories for consistent x-axis
        for traj in ["clean", "corrupted"]:
            markers = coloring.get_section_markers(traj)
            if markers.get("prompt_boundary") is not None:
                all_boundaries.append(markers["prompt_boundary"])
            if markers.get("choice_div_pos") is not None:
                all_boundaries.append(markers["choice_div_pos"])

    # Calculate x-axis range that includes data AND all boundary markers
    x_min = min(positions)
    x_max = max(positions)
    if all_boundaries:
        x_min = min(x_min, min(all_boundaries))
        x_max = max(x_max, max(all_boundaries))

    # Adjust spacing: extra bottom margin for Token Type and Boundary legends
    fig.subplots_adjust(left=0.045, right=0.96, top=0.91, bottom=0.14, wspace=0.40, hspace=0.48)

    # Plot each row and collect secondary axes for synchronization
    secondary_axes_by_row: list[list[plt.Axes | None]] = []
    for row_idx, mode in enumerate(["denoising", "noising"]):
        metrics = _extract_metrics_for_mode(pos_data, positions, mode, clean_traj)
        row_secondary = _plot_sweep_row(
            axes[row_idx], positions, metrics, mode, tick_positions, "Position", coloring
        )
        secondary_axes_by_row.append(row_secondary)

        # Set consistent x-axis limits and add boundary markers
        for ax in axes[row_idx]:
            ax.set_xlim(x_min - 1, x_max + 1)  # Small padding
            add_xaxis_boundary_markers(ax, prompt_boundary, choice_div_pos)

        # Row label - positioned based on subplot locations
        row_label = "Denoising" if mode == "denoising" else "Noising"
        fig.text(
            0.012,
            0.72 - row_idx * 0.44,
            row_label,
            fontsize=15,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    _synchronize_y_axes(axes, secondary_axes_by_row)

    # Add legends
    add_token_type_legend(fig)
    add_boundary_legend(fig)

    # Save
    save_path = output_dir / f"coarse_position_sweep_{clean_traj}_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _get_baseline_info(sweep_data: SweepStepResults, first_key: int) -> str:
    """Extract baseline info string from first result."""
    first_result = sweep_data[first_key]
    if first_result.denoising and first_result.denoising.baseline_corrupted:
        orig = first_result.denoising.baseline_corrupted
        lp_diff = orig.divergent_logprobs[0] - orig.divergent_logprobs[1]
        return f"Baseline logprob diff: {lp_diff:.2f}"
    return ""
