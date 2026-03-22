"""Plotting functions for geometric visualization."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .geo_viz_analysis import EmbeddingResult, LinearProbeResult, PCAResult
from .geo_viz_config import GeoVizConfig
from .geo_viz_data import ActivationData, get_time_horizon_months


def _months_to_years(months: float) -> float:
    """Convert months to years."""
    return months / 12.0

logger = logging.getLogger(__name__)

# Color scheme
CMAP_GRADIENT = "plasma"
CMAP_BUCKETS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
BUCKET_LABELS = ["<1yr", "1-5yr", "5-10yr", ">10yr"]


# =============================================================================
# Coloring Schemes
# =============================================================================


@dataclass
class ColoringScheme:
    """A coloring scheme for plots."""

    name: str  # Short name for filenames
    label: str  # Display label for colorbar/legend
    values: np.ndarray  # Values for coloring
    use_log: bool = False
    is_categorical: bool = False
    categories: list[str] | None = None


def get_coloring_schemes(data: ActivationData) -> list[ColoringScheme]:
    """Get all available coloring schemes from the data."""
    schemes = []

    # Time horizon (in years)
    horizons = np.array([_months_to_years(get_time_horizon_months(s)) for s in data.samples])
    schemes.append(
        ColoringScheme(
            name="horizon",
            label="Time Horizon (years)",
            values=horizons,
            use_log=True,
        )
    )

    # Choice-based schemes
    if data.choices:
        chosen_times = np.array([_months_to_years(c.chosen_time_months) for c in data.choices])
        schemes.append(
            ColoringScheme(
                name="chosen_time",
                label="Chosen Delivery Time (years)",
                values=chosen_times,
                use_log=True,
            )
        )

        chosen_rewards = np.array([c.chosen_reward for c in data.choices])
        schemes.append(
            ColoringScheme(
                name="chosen_reward",
                label="Chosen Reward",
                values=chosen_rewards,
                use_log=False,
            )
        )

        chose_long = np.array([1.0 if c.chose_long_term else 0.0 for c in data.choices])
        schemes.append(
            ColoringScheme(
                name="choice_type",
                label="Choice Type",
                values=chose_long,
                is_categorical=True,
                categories=["Short-term", "Long-term"],
            )
        )

        choice_probs = np.array([c.choice_prob for c in data.choices])
        schemes.append(
            ColoringScheme(
                name="choice_prob",
                label="Choice Probability",
                values=choice_probs,
                use_log=False,
            )
        )

    return schemes


def _get_horizons(data: ActivationData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract horizon arrays from data (in years)."""
    horizons = np.array([_months_to_years(get_time_horizon_months(s)) for s in data.samples])
    log_horizons = np.log10(horizons + 0.1)  # Small offset to handle values < 1 year
    buckets = np.digitize(horizons, [1, 5, 10])  # 1yr, 5yr, 10yr thresholds
    return horizons, log_horizons, buckets


# =============================================================================
# Summary Plots (go in plots/ root)
# =============================================================================


def plot_linear_probe_summary(
    data: ActivationData,
    results: dict[str, LinearProbeResult],
    output_dir: Path,
):
    """Plot linear probe summary bar chart."""
    horizons, log_horizons, _ = _get_horizons(data)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    targets = sorted(results.keys(), key=lambda k: results[k].r2_mean, reverse=True)
    r2_values = [results[t].r2_mean for t in targets]
    r2_stds = [results[t].r2_std for t in targets]

    colors = ["#4CAF50" if "Pdest" in t else "#F44336" for t in targets]

    ax.barh(range(len(targets)), r2_values, xerr=r2_stds, color=colors, alpha=0.8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel("R² Score (5-fold CV)")
    ax.set_title("Linear Probe for Time Horizon Decoding\n(Green=Dest, Red=Source)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-0.1, 1.1)

    for i, (t, r2) in enumerate(zip(targets, r2_values)):
        ax.text(max(r2, 0) + 0.02, i, f"{r2:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe_summary.png", dpi=150)
    plt.close()

    # Scatter plots
    n_targets = len(results)
    n_cols = min(4, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, target_key in enumerate(sorted(results.keys())):
        result = results[target_key]
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        ax.scatter(
            log_horizons,
            result.predictions,
            c=log_horizons,
            cmap=CMAP_GRADIENT,
            s=10,
            alpha=0.6,
        )
        ax.plot(
            [log_horizons.min(), log_horizons.max()],
            [log_horizons.min(), log_horizons.max()],
            "r--",
            alpha=0.5,
        )
        ax.set_xlabel("Actual log₁₀(years)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target_key}\nR²={result.r2_mean:.3f}")

    for idx in range(len(results), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe_scatter.png", dpi=150)
    plt.close()


# =============================================================================
# Per-Target Plots
# =============================================================================


def _scatter_with_scheme(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    scheme: ColoringScheme,
    add_colorbar: bool = True,
) -> None:
    """Scatter plot with a coloring scheme."""
    if scheme.is_categorical:
        colors = ["#2196F3", "#F44336"]
        for val, color, label in zip([0, 1], colors, scheme.categories or ["0", "1"]):
            mask = scheme.values == val
            if mask.sum() > 0:
                ax.scatter(x[mask], y[mask], c=color, s=15, alpha=0.7, label=label)
        ax.legend(markerscale=2, fontsize=8)
    else:
        vmin = max(1, scheme.values.min()) if scheme.use_log else scheme.values.min()
        vmax = scheme.values.max()
        norm = LogNorm(vmin=vmin, vmax=vmax) if scheme.use_log else None
        sc = ax.scatter(x, y, c=scheme.values, cmap=CMAP_GRADIENT, s=15, alpha=0.7, norm=norm)
        if add_colorbar:
            plt.colorbar(sc, ax=ax, label=scheme.label, shrink=0.8)


def plot_target_pca(
    target_key: str,
    pca_result: PCAResult,
    schemes: list[ColoringScheme],
    target_dir: Path,
):
    """Generate PCA plots for a single target."""
    pca_dir = target_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    best_pc_idx = pca_result.pc_correlations[0][0]
    best_corr = pca_result.pc_correlations[0][1]
    X_pca = pca_result.transformed

    # Main summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.bar(range(min(20, len(pca_result.explained_variance))),
           pca_result.explained_variance[:20], color="#2196F3", alpha=0.7)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")

    ax = axes[0, 1]
    top_n = min(15, len(pca_result.pc_correlations))
    pc_idxs = [c[0] for c in pca_result.pc_correlations[:top_n]]
    corrs = [c[1] for c in pca_result.pc_correlations[:top_n]]
    colors = ["#4CAF50" if c > 0 else "#F44336" for c in corrs]
    ax.barh(range(top_n), corrs, color=colors, alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"PC{i}" for i in pc_idxs])
    ax.set_xlabel("Spearman Correlation with log(horizon)")
    ax.set_title("Top PC Correlations")
    ax.axvline(0, color="black", linewidth=0.5)

    ax = axes[1, 0]
    _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], schemes[0])
    ax.set_xlabel(f"PC0 ({pca_result.explained_variance[0]*100:.1f}%)")
    ax.set_ylabel(f"PC{best_pc_idx} (r={best_corr:.2f})")
    ax.set_title("PC0 vs Best Time-Correlated PC")

    ax = axes[1, 1]
    _scatter_with_scheme(ax, X_pca[:, best_pc_idx], schemes[0].values, schemes[0])
    ax.set_xlabel(f"PC{best_pc_idx}")
    ax.set_ylabel("Time Horizon (years)")
    if schemes[0].use_log:
        ax.set_yscale("log")
    ax.set_title(f"PC{best_pc_idx} vs Horizon")

    plt.suptitle(f"{target_key} - PCA Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(pca_dir / "summary.png", dpi=150)
    plt.close()

    # Per-coloring plots
    for scheme in schemes:
        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme)
        ax.set_xlabel(f"PC0 ({pca_result.explained_variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC{best_pc_idx} (r={best_corr:.2f})")
        ax.set_title(f"{scheme.label}")
        plt.tight_layout()
        plt.savefig(pca_dir / f"{scheme.name}.png", dpi=150)
        plt.close()


def plot_target_embeddings(
    target_key: str,
    embedding_result: EmbeddingResult,
    schemes: list[ColoringScheme],
    target_dir: Path,
):
    """Generate embedding plots for a single target."""
    emb_dir = target_dir / "embeddings"

    # Collect available embeddings
    embeddings = {}
    if embedding_result.pca_embedding is not None:
        embeddings["pca"] = embedding_result.pca_embedding
    if embedding_result.umap_embedding is not None:
        embeddings["umap"] = embedding_result.umap_embedding
    if embedding_result.tsne_embedding is not None:
        embeddings["tsne"] = embedding_result.tsne_embedding

    if not embeddings:
        return

    # Per-coloring directories
    for scheme in schemes:
        scheme_dir = emb_dir / scheme.name
        scheme_dir.mkdir(parents=True, exist_ok=True)

        for method, coords in embeddings.items():
            fig, ax = plt.subplots(figsize=(8, 7))
            _scatter_with_scheme(ax, coords[:, 0], coords[:, 1], scheme)
            ax.set_xlabel(f"{method.upper()} 1")
            ax.set_ylabel(f"{method.upper()} 2")
            ax.set_title(f"{method.upper()} - {scheme.label}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(scheme_dir / f"{method}.png", dpi=150)
            plt.close()

    # Combined overview
    n_methods = len(embeddings)
    n_schemes = len(schemes)
    fig, axes = plt.subplots(n_schemes, n_methods, figsize=(4 * n_methods, 4 * n_schemes))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if n_schemes == 1:
        axes = axes.reshape(1, -1)

    for row, scheme in enumerate(schemes):
        for col, (method, coords) in enumerate(embeddings.items()):
            ax = axes[row, col]
            _scatter_with_scheme(ax, coords[:, 0], coords[:, 1], scheme, add_colorbar=False)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method.upper())
            if col == 0:
                ax.set_ylabel(scheme.name, fontsize=10)

    plt.suptitle(f"{target_key} - Embeddings Overview", fontsize=14)
    plt.tight_layout()
    plt.savefig(emb_dir / "overview.png", dpi=150)
    plt.close()


def plot_target_3d(
    target_key: str,
    pca_result: PCAResult,
    schemes: list[ColoringScheme],
    target_dir: Path,
):
    """Generate interactive 3D plots for a single target."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly not available, skipping 3D plots")
        return

    html_dir = target_dir / "3d"
    html_dir.mkdir(parents=True, exist_ok=True)

    X_pca = pca_result.transformed
    top_pcs = [c[0] for c in pca_result.pc_correlations[:3]]
    corrs = [pca_result.pc_correlations[i][1] for i in range(min(3, len(pca_result.pc_correlations)))]

    for scheme in schemes:
        if scheme.is_categorical:
            colors = ["blue" if v == 0 else "red" for v in scheme.values]
            hover_text = [scheme.categories[int(v)] if scheme.categories else str(int(v)) for v in scheme.values]
            marker = dict(size=4, color=colors, opacity=0.8)
        else:
            color_vals = np.log10(scheme.values + 1) if scheme.use_log else scheme.values
            hover_text = [f"{scheme.label}: {v:.1f}" for v in scheme.values]
            marker = dict(size=4, color=color_vals, colorscale="Plasma", opacity=0.8,
                          colorbar=dict(title=scheme.label))

        fig = go.Figure(data=[go.Scatter3d(
            x=X_pca[:, top_pcs[0]],
            y=X_pca[:, top_pcs[1]],
            z=X_pca[:, top_pcs[2]] if len(top_pcs) > 2 else np.zeros(len(X_pca)),
            mode="markers",
            marker=marker,
            text=hover_text,
            hovertemplate="PC%d: %%{x:.2f}<br>PC%d: %%{y:.2f}<br>PC%d: %%{z:.2f}<br>%%{text}<extra></extra>"
            % (top_pcs[0], top_pcs[1], top_pcs[2] if len(top_pcs) > 2 else 0),
        )])

        corr_str = ", ".join([f"PC{top_pcs[i]} (r={corrs[i]:.2f})" for i in range(len(corrs))])
        fig.update_layout(
            title=f"{target_key} - {scheme.label}<br>{corr_str}",
            scene=dict(
                xaxis_title=f"PC{top_pcs[0]}",
                yaxis_title=f"PC{top_pcs[1]}",
                zaxis_title=f"PC{top_pcs[2]}" if len(top_pcs) > 2 else "PC0",
            ),
            width=900,
            height=700,
        )

        fig.write_html(html_dir / f"{scheme.name}.html")


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_all_plots(
    data: ActivationData,
    linear_probe_results: dict[str, LinearProbeResult],
    pca_results: dict[str, PCAResult],
    embedding_results: dict[str, EmbeddingResult],
    config: GeoVizConfig,
):
    """Generate all plots with organized directory structure."""
    output_dir = config.output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = get_coloring_schemes(data)
    logger.info(f"Using {len(schemes)} coloring schemes: {[s.name for s in schemes]}")

    # Summary plots in root
    logger.info("Generating summary plots...")
    plot_linear_probe_summary(data, linear_probe_results, output_dir)

    # Per-target plots
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    for target_key in pca_results.keys():
        logger.info(f"  Plotting {target_key}...")
        target_dir = targets_dir / target_key
        target_dir.mkdir(parents=True, exist_ok=True)

        plot_target_pca(target_key, pca_results[target_key], schemes, target_dir)

        if target_key in embedding_results:
            plot_target_embeddings(target_key, embedding_results[target_key], schemes, target_dir)

        plot_target_3d(target_key, pca_results[target_key], schemes, target_dir)

    logger.info(f"All plots saved to {output_dir}")
