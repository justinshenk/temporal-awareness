"""Visualization: cluster distribution and embedding plots."""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_cluster_distribution(cluster_dist: list[int], title: str, output_path: Path):
    """Bar chart of cluster sizes."""

    colors = ["seagreen" if c > 0 else "lightgray" for c in cluster_dist]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(cluster_dist)), cluster_dist, color=colors)
    ax.set(xlabel="Cluster ID", ylabel="Sample Count", title=title)

    active = sum(1 for c in cluster_dist if c > 0)
    total = sum(cluster_dist)
    ax.text(0.98, 0.98, f"Active: {active}/{len(cluster_dist)}\nSamples: {total}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradient_embedding(
    embeddings_2d: np.ndarray,
    values: list[float | None],
    title: str,
    output_path: Path,
):
    """Scatter plot of 2D embeddings colored by log-scaled continuous gradient.

    Points with value == -1 or None are excluded entirely.

    Args:
        embeddings_2d: (n, 2) array of 2D coordinates
        values: list of numeric values, one per point (-1 or None = excluded)
        title: plot title
        output_path: where to save the PNG
    """
    values_arr = np.array([v if v is not None else -1.0 for v in values], dtype=float)
    valid = values_arr > 0  # exclude -1, None, and 0 (invalid for log)

    if not valid.any():
        return

    coords = embeddings_2d[valid]
    v = values_arr[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    norm = LogNorm(vmin=v.min(), vmax=v.max())
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=v,
        cmap="Spectral_r",
        norm=norm,
        s=15,
        alpha=0.7,
    )
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_embedding(
    embeddings_2d: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
):
    """Scatter plot of 2D embeddings colored by categorical labels.

    Args:
        embeddings_2d: (n, 2) array of 2D coordinates
        labels: list of category strings, one per point
        title: plot title
        output_path: where to save the PNG
    """
    categories = sorted(set(labels))
    # Use tab10 for <=10 categories, tab20 for more
    cmap = plt.cm.tab10 if len(categories) <= 10 else plt.cm.tab20
    color_map = {cat: cmap(i / max(len(categories) - 1, 1)) for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for cat in categories:
        mask = np.array([l == cat for l in labels])
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=color_map[cat],
            label=cat,
            s=15,
            alpha=0.7,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
