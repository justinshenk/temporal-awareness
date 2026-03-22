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

# Time scale colors and labels (weeks, months, years, decades)
TIME_SCALE_COLORS = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]  # purple, blue, green, orange
TIME_SCALE_LABELS = ["Weeks", "Months", "Years", "Decades"]


def _get_time_scale(months: float) -> int:
    """Classify time horizon into scale: 0=weeks, 1=months, 2=years, 3=decades."""
    if months < 1:  # Less than 1 month = weeks
        return 0
    elif months < 12:  # Less than 1 year = months
        return 1
    elif months < 120:  # Less than 10 years = years
        return 2
    else:  # 10+ years = decades
        return 3


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

    # Get raw time values
    horizons_months = np.array([get_time_horizon_months(s) for s in data.samples])
    horizons = np.array([_months_to_years(m) for m in horizons_months])

    # Time scale categories (weeks, months, years, decades)
    time_scales = np.array([_get_time_scale(m) for m in horizons_months])
    schemes.append(
        ColoringScheme(
            name="time_scale",
            label="Time Scale",
            values=time_scales,
            is_categorical=True,
            categories=TIME_SCALE_LABELS,
        )
    )

    # Time in different units (for log-scale coloring)
    # Weeks
    horizons_weeks = horizons_months * (30.44 / 7)  # ~4.35 weeks per month
    schemes.append(
        ColoringScheme(
            name="horizon_weeks",
            label="Time Horizon (weeks)",
            values=horizons_weeks,
            use_log=True,
        )
    )

    # Months
    schemes.append(
        ColoringScheme(
            name="horizon_months",
            label="Time Horizon (months)",
            values=horizons_months,
            use_log=True,
        )
    )

    # Years (already have horizons in years)
    schemes.append(
        ColoringScheme(
            name="horizon_years",
            label="Time Horizon (years)",
            values=horizons,
            use_log=True,
        )
    )

    # Decades
    horizons_decades = horizons / 10.0
    schemes.append(
        ColoringScheme(
            name="horizon_decades",
            label="Time Horizon (decades)",
            values=horizons_decades,
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
        # Use time scale colors for time_scale scheme, otherwise default colors
        if scheme.name == "time_scale":
            colors = TIME_SCALE_COLORS
        else:
            colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]  # Extend if needed

        categories = scheme.categories or [str(i) for i in range(len(colors))]
        unique_vals = sorted(set(scheme.values.astype(int)))

        for val in unique_vals:
            color = colors[val % len(colors)]
            label = categories[val] if val < len(categories) else str(val)
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
            # Map categorical values to colors
            if scheme.name == "time_scale":
                color_map = {0: "purple", 1: "blue", 2: "green", 3: "orange"}
            else:
                color_map = {0: "blue", 1: "red", 2: "green", 3: "orange"}
            colors = [color_map.get(int(v), "gray") for v in scheme.values]
            hover_text = [scheme.categories[int(v)] if scheme.categories and int(v) < len(scheme.categories) else str(int(v)) for v in scheme.values]
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
# Helper Functions
# =============================================================================


def _parse_target_key(target_key: str) -> tuple[str, str]:
    """Parse target key into (base_key, position).

    E.g., "L19_mlp_out_Pdest" -> ("L19_mlp_out", "dest")
    """
    if target_key.endswith("_Pdest"):
        return target_key[:-6], "dest"
    elif target_key.endswith("_Psource"):
        return target_key[:-8], "source"
    else:
        return target_key, "unknown"


def _group_targets_by_base(target_keys: list[str]) -> dict[str, dict[str, str]]:
    """Group target keys by their base (layer+component).

    Returns: {base_key: {position: full_target_key}}
    """
    groups = {}
    for key in target_keys:
        base, pos = _parse_target_key(key)
        if base not in groups:
            groups[base] = {}
        groups[base][pos] = key
    return groups


def plot_cross_layer_summary(
    linear_probe_results: dict[str, LinearProbeResult],
    pca_results: dict[str, PCAResult],
    schemes: list[ColoringScheme],
    output_dir: Path,
):
    """Generate cross-layer comparison plots showing all layers/positions together."""
    # Parse all targets
    targets_by_pos = {"dest": [], "source": []}
    for key in linear_probe_results.keys():
        base, pos = _parse_target_key(key)
        if pos in targets_by_pos:
            targets_by_pos[pos].append((key, base))

    # Sort by layer number
    def extract_layer(item):
        match = re.search(r"L(\d+)", item[1])
        return int(match.group(1)) if match else 0

    for pos in targets_by_pos:
        targets_by_pos[pos].sort(key=extract_layer)

    # Cross-layer R² comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (pos, targets) in enumerate(targets_by_pos.items()):
        if not targets:
            continue
        ax = axes[idx]
        labels = [t[1] for t in targets]  # base keys
        r2_scores = [linear_probe_results[t[0]].r2_mean for t in targets]
        colors = ["#4CAF50" if pos == "dest" else "#F44336"] * len(targets)

        bars = ax.barh(range(len(labels)), r2_scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("R² Score")
        ax.set_title(f"{pos.upper()} Position\nAcross Layers")
        ax.set_xlim(-0.1, 1.1)
        ax.axvline(0, color="black", linewidth=0.5)

        for i, r2 in enumerate(r2_scores):
            ax.text(max(r2, 0) + 0.02, i, f"{r2:.3f}", va="center", fontsize=8)

    plt.suptitle("Linear Probe R² by Layer and Position", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "cross_layer_r2.png", dpi=150)
    plt.close()

    # Cross-layer PCA grid (dest only, showing time_scale coloring)
    dest_targets = targets_by_pos.get("dest", [])
    if dest_targets and len(schemes) > 0:
        n_targets = len(dest_targets)
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        # Use time_scale scheme if available, else first scheme
        scheme = next((s for s in schemes if s.name == "time_scale"), schemes[0])

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, (target_key, base_key) in enumerate(dest_targets):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            if target_key in pca_results:
                pca_result = pca_results[target_key]
                X_pca = pca_result.transformed
                best_pc_idx = pca_result.pc_correlations[0][0]
                best_corr = pca_result.pc_correlations[0][1]

                _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False)
                ax.set_xlabel("PC0")
                ax.set_ylabel(f"PC{best_pc_idx}")
                ax.set_title(f"{base_key}\nr={best_corr:.2f}")

        # Hide empty subplots
        for idx in range(len(dest_targets), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle(f"PCA Across Layers (DEST) - {scheme.label}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "cross_layer_pca_dest.png", dpi=150)
        plt.close()

        # Same for source if available
        source_targets = targets_by_pos.get("source", [])
        if source_targets:
            n_targets = len(source_targets)
            n_cols = min(3, n_targets)
            n_rows = (n_targets + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
            axes = np.atleast_2d(axes)

            for idx, (target_key, base_key) in enumerate(source_targets):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                if target_key in pca_results:
                    pca_result = pca_results[target_key]
                    X_pca = pca_result.transformed
                    best_pc_idx = pca_result.pc_correlations[0][0]
                    best_corr = pca_result.pc_correlations[0][1]

                    _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False)
                    ax.set_xlabel("PC0")
                    ax.set_ylabel(f"PC{best_pc_idx}")
                    ax.set_title(f"{base_key}\nr={best_corr:.2f}")

            for idx in range(len(source_targets), n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].axis("off")

            plt.suptitle(f"PCA Across Layers (SOURCE) - {scheme.label}", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "cross_layer_pca_source.png", dpi=150)
            plt.close()


def plot_position_comparison(
    base_key: str,
    position_targets: dict[str, str],
    linear_probe_results: dict[str, LinearProbeResult],
    pca_results: dict[str, PCAResult],
    schemes: list[ColoringScheme],
    output_dir: Path,
):
    """Generate comparison plots for dest vs source positions."""
    if len(position_targets) < 2:
        return

    # Comparison of R² scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison bar chart
    ax = axes[0]
    positions = list(position_targets.keys())
    r2_scores = [linear_probe_results[position_targets[p]].r2_mean for p in positions]
    colors = ["#4CAF50" if p == "dest" else "#F44336" for p in positions]
    ax.bar(positions, r2_scores, color=colors, alpha=0.8)
    ax.set_ylabel("R² Score")
    ax.set_title(f"{base_key}\nLinear Probe R² by Position")
    ax.set_ylim(-0.1, 1.1)
    for i, (pos, r2) in enumerate(zip(positions, r2_scores)):
        ax.text(i, max(r2, 0) + 0.05, f"{r2:.3f}", ha="center", fontsize=11)

    # PCA correlation comparison
    ax = axes[1]
    for pos in positions:
        target_key = position_targets[pos]
        if target_key in pca_results:
            pca_result = pca_results[target_key]
            corrs = [c[1] for c in pca_result.pc_correlations[:10]]
            color = "#4CAF50" if pos == "dest" else "#F44336"
            ax.plot(range(len(corrs)), [abs(c) for c in corrs],
                   marker='o', label=pos, color=color, alpha=0.8)
    ax.set_xlabel("PC Index (sorted by correlation)")
    ax.set_ylabel("|Correlation with log(horizon)|")
    ax.set_title("PCA Correlation by Position")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.suptitle(f"{base_key} - Position Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "position_comparison.png", dpi=150)
    plt.close()

    # Side-by-side PCA scatter for each coloring scheme
    for scheme in schemes[:3]:  # Just first 3 schemes for comparison
        fig, axes = plt.subplots(1, len(positions), figsize=(6 * len(positions), 5))
        if len(positions) == 1:
            axes = [axes]

        for idx, pos in enumerate(positions):
            target_key = position_targets[pos]
            if target_key in pca_results:
                pca_result = pca_results[target_key]
                X_pca = pca_result.transformed
                best_pc_idx = pca_result.pc_correlations[0][0]

                ax = axes[idx]
                _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme)
                ax.set_xlabel("PC0")
                ax.set_ylabel(f"PC{best_pc_idx}")
                ax.set_title(f"{pos.upper()}")

        plt.suptitle(f"{base_key} - {scheme.label}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{scheme.name}.png", dpi=150)
        plt.close()


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
    """Generate all plots with organized directory structure.

    Structure:
        plots/
        ├── linear_probe_summary.png
        ├── linear_probe_scatter.png
        └── targets/
            └── L19_mlp_out/
                ├── position_comparison.png
                ├── comparison_*.png
                └── by_position/
                    ├── dest/
                    │   ├── pca/
                    │   ├── embeddings/
                    │   └── 3d/
                    └── source/
                        └── ...
    """
    output_dir = config.output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = get_coloring_schemes(data)
    logger.info(f"Using {len(schemes)} coloring schemes: {[s.name for s in schemes]}")

    # Summary plots in root
    logger.info("Generating summary plots...")
    plot_linear_probe_summary(data, linear_probe_results, output_dir)

    # Cross-layer comparison plots
    logger.info("Generating cross-layer plots...")
    plot_cross_layer_summary(linear_probe_results, pca_results, schemes, output_dir)

    # Group targets by base (layer+component)
    target_groups = _group_targets_by_base(list(pca_results.keys()))

    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    for base_key, position_targets in target_groups.items():
        logger.info(f"  Processing {base_key}...")
        base_dir = targets_dir / base_key
        base_dir.mkdir(parents=True, exist_ok=True)

        # Generate comparison plots at base level
        plot_position_comparison(
            base_key, position_targets, linear_probe_results,
            pca_results, schemes, base_dir
        )

        # Generate per-position plots
        by_position_dir = base_dir / "by_position"
        by_position_dir.mkdir(parents=True, exist_ok=True)

        for pos, target_key in position_targets.items():
            logger.info(f"    Plotting {pos}...")
            pos_dir = by_position_dir / pos
            pos_dir.mkdir(parents=True, exist_ok=True)

            plot_target_pca(target_key, pca_results[target_key], schemes, pos_dir)

            if target_key in embedding_results:
                plot_target_embeddings(target_key, embedding_results[target_key], schemes, pos_dir)

            plot_target_3d(target_key, pca_results[target_key], schemes, pos_dir)

    logger.info(f"All plots saved to {output_dir}")
