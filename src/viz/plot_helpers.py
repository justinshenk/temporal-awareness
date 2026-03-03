"""General plotting helpers for visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .palettes import BAR_COLORS


def finalize_plot(
    save_path: Path | None = None,
    dpi: int = 150,
    facecolor: str = "white",
) -> None:
    """Finalize current figure: save or show.

    Uses plt.gcf() to get current figure. Applies tight_layout before saving.

    Args:
        save_path: Path to save the figure. If None, shows the plot.
        dpi: DPI for the saved image
        facecolor: Background color
    """
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def finalize_and_save(
    fig: plt.Figure,
    save_path: Path,
    dpi: int = 150,
    facecolor: str = "white",
) -> None:
    """Finalize and save a specific figure with white background.

    Args:
        fig: Matplotlib figure
        save_path: Path to save the figure
        dpi: DPI for the saved image
        facecolor: Background color
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_comparison_bars(
    ax: plt.Axes,
    labels: list[str],
    denoising_vals: list[float],
    noising_vals: list[float],
    ylabel: str = "Value",
    title: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Create a grouped bar chart comparing denoising vs noising.

    Args:
        ax: Matplotlib axes
        labels: X-axis labels
        denoising_vals: Values for denoising bars
        noising_vals: Values for noising bars
        ylabel: Y-axis label
        title: Plot title
        ylim: Y-axis limits (min, max)
    """
    import numpy as np

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, denoising_vals, width, label="Denoising", color=BAR_COLORS["denoising"])
    ax.bar(x + width / 2, noising_vals, width, label="Noising", color=BAR_COLORS["noising"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(fontsize=9)

    if ylim:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3, axis="y")


def add_value_labels_to_bars(
    ax: plt.Axes,
    bars,
    values: list[float],
    fmt: str = "{:+.2f}",
    fontsize: int = 11,
) -> None:
    """Add value labels on top of bars.

    Args:
        ax: Matplotlib axes
        bars: Bar container from ax.bar()
        values: Values to display
        fmt: Format string for values
        fontsize: Font size for labels
    """
    for bar, val in zip(bars, values):
        ypos = val + (2 if val >= 0 else -2)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            fmt.format(val),
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=fontsize,
            fontweight="bold",
        )


def setup_line_plot_panel(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_outside: bool = True,
    ncol: int = 3,
) -> None:
    """Setup common line plot panel styling.

    Args:
        ax: Matplotlib axes
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        legend_outside: Whether to place legend outside the plot
        ncol: Number of columns in legend
    """
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.12),
            fontsize=9,
            ncol=ncol,
            frameon=False,
        )
    else:
        ax.legend(loc="best", fontsize=9)
