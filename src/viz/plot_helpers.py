"""General plotting helpers for visualization."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from .viz_palettes import BAR_COLORS

# Module-level SVG save setting (can be controlled via context manager)
_SAVE_SVG: bool = False


@contextmanager
def svg_mode(enabled: bool = True):
    """Context manager to enable/disable SVG saving globally.

    Usage:
        with svg_mode(True):
            # All save_figure/finalize_plot calls will also save SVG
            visualize_something(...)

    Args:
        enabled: Whether to save SVG files alongside PNGs
    """
    global _SAVE_SVG
    old_value = _SAVE_SVG
    _SAVE_SVG = enabled
    try:
        yield
    finally:
        _SAVE_SVG = old_value


def set_svg_mode(enabled: bool) -> None:
    """Set global SVG save mode.

    Args:
        enabled: Whether to save SVG files alongside PNGs
    """
    global _SAVE_SVG
    _SAVE_SVG = enabled


def get_svg_mode() -> bool:
    """Get current SVG save mode setting."""
    return _SAVE_SVG


def save_figure(
    fig: plt.Figure | None,
    save_path: Path,
    dpi: int = 150,
    facecolor: str = "white",
    save_svg: bool | None = None,
    close: bool = True,
) -> None:
    """Save a figure to PNG and optionally SVG.

    This is a low-level utility for saving figures. Use finalize_plot or
    finalize_and_save for most cases.

    Args:
        fig: Figure to save, or None to use current figure (plt.gcf())
        save_path: Path to save the PNG file
        dpi: DPI for PNG output
        facecolor: Background color
        save_svg: If True, also save SVG. If None, uses global svg_mode setting.
        close: If True, close the figure after saving
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    # Use global setting if not explicitly specified
    should_save_svg = save_svg if save_svg is not None else _SAVE_SVG

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    if should_save_svg:
        svg_path = save_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=facecolor)

    if close:
        plt.close(fig)


def finalize_plot(
    save_path: Path | None = None,
    dpi: int = 150,
    facecolor: str = "white",
    save_svg: bool | None = None,
) -> None:
    """Finalize current figure: save or show.

    Uses plt.gcf() to get current figure. Applies tight_layout before saving.

    Args:
        save_path: Path to save the figure. If None, shows the plot.
        dpi: DPI for the saved image
        facecolor: Background color
        save_svg: If True, also save SVG. If None, uses global svg_mode setting.
    """
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use global setting if not explicitly specified
        should_save_svg = save_svg if save_svg is not None else _SAVE_SVG

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
        if should_save_svg:
            svg_path = save_path.with_suffix(".svg")
            plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=facecolor)
            print(f"Saved: {save_path} + .svg")
        else:
            print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def finalize_and_save(
    fig: plt.Figure,
    save_path: Path,
    dpi: int = 150,
    facecolor: str = "white",
    save_svg: bool | None = None,
) -> None:
    """Finalize and save a specific figure with white background.

    Args:
        fig: Matplotlib figure
        save_path: Path to save the figure
        dpi: DPI for the saved image
        facecolor: Background color
        save_svg: If True, also save SVG. If None, uses global svg_mode setting.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Use global setting if not explicitly specified
    should_save_svg = save_svg if save_svg is not None else _SAVE_SVG

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    if should_save_svg:
        svg_path = save_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=facecolor)
        print(f"Saved: {save_path} + .svg")
    else:
        print(f"Saved: {save_path}")
    plt.close(fig)


def add_pair_label(fig: plt.Figure, pair_idx: int | None) -> None:
    """Add a subtle pair index label to the figure corner.

    Args:
        fig: Matplotlib figure
        pair_idx: Pair index to display, or None to skip
    """
    if pair_idx is not None:
        fig.text(
            0.99, 0.01, f"Pair {pair_idx}",
            fontsize=10, color="gray", ha="right", va="bottom",
        )


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
