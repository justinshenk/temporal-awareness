"""Plotting functions for geometric visualization.

Memory-optimized implementation:
- Process plots in batches with explicit cleanup
- For per-target plots, clear after each target
- Use compact coloring scheme storage
"""

import gc
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
from .geo_viz_config import (
    GeoVizConfig,
    ACTIVATION_DTYPE,
    PLOT_GC_INTERVAL,
    MAX_TRAJECTORY_SAMPLES,
)
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
    horizons_months = np.array([get_time_horizon_months(s) for s in data.samples], dtype=ACTIVATION_DTYPE)
    horizons = np.array([_months_to_years(m) for m in horizons_months], dtype=ACTIVATION_DTYPE)

    # Time scale categories (weeks, months, years, decades)
    time_scales = np.array([_get_time_scale(m) for m in horizons_months], dtype=np.int8)
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
    horizons_weeks = (horizons_months * (30.44 / 7)).astype(ACTIVATION_DTYPE)
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
    horizons_decades = (horizons / 10.0).astype(ACTIVATION_DTYPE)
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
        chosen_times = np.array([_months_to_years(c.chosen_time_months) for c in data.choices], dtype=ACTIVATION_DTYPE)
        schemes.append(
            ColoringScheme(
                name="chosen_time",
                label="Chosen Delivery Time (years)",
                values=chosen_times,
                use_log=True,
            )
        )

        chosen_rewards = np.array([c.chosen_reward for c in data.choices], dtype=ACTIVATION_DTYPE)
        schemes.append(
            ColoringScheme(
                name="chosen_reward",
                label="Chosen Reward",
                values=chosen_rewards,
                use_log=False,
            )
        )

        chose_long = np.array([1 if c.chose_long_term else 0 for c in data.choices], dtype=np.int8)
        schemes.append(
            ColoringScheme(
                name="choice_type",
                label="Choice Type",
                values=chose_long,
                is_categorical=True,
                categories=["Short-term", "Long-term"],
            )
        )

        choice_probs = np.array([c.choice_prob for c in data.choices], dtype=ACTIVATION_DTYPE)
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
    horizons = np.array([_months_to_years(get_time_horizon_months(s)) for s in data.samples], dtype=ACTIVATION_DTYPE)
    log_horizons = np.log10(horizons + 0.1).astype(ACTIVATION_DTYPE)
    buckets = np.digitize(horizons, [1, 5, 10]).astype(np.int8)
    return horizons, log_horizons, buckets


# =============================================================================
# Summary Dashboard and Heatmaps
# =============================================================================


def plot_summary_dashboard(
    linear_probe_results: dict[str, "LinearProbeResult"],
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot summary dashboard heatmaps showing R² scores across layers and positions."""
    # Parse all targets to extract layer, component, position
    target_info = {}
    for key in linear_probe_results.keys():
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]
        position = parts[1]

        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        target_info[key] = {
            "layer": layer,
            "component": component,
            "position": position,
            "r2": linear_probe_results[key].r2_mean,
            "corr": abs(pca_results[key].pc_correlations[0][1]) if key in pca_results else 0,
        }

    if not target_info:
        logger.warning("No valid targets for summary dashboard")
        return

    # Get unique layers, components, positions
    layers = sorted(set(t["layer"] for t in target_info.values()))
    components = sorted(set(t["component"] for t in target_info.values()))
    positions = sorted(set(t["position"] for t in target_info.values()))

    # Order positions: source positions first, then dest
    source_positions = ["time_horizon", "short_term_time", "short_term_reward",
                       "long_term_time", "long_term_reward", "source"]
    dest_positions = ["response", "dest"]
    ordered_positions = [p for p in source_positions if p in positions]
    ordered_positions += [p for p in dest_positions if p in positions]
    ordered_positions += [p for p in positions if p not in ordered_positions]
    positions = ordered_positions

    # Create heatmap for each component
    for component in components:
        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), max(5, len(positions) * 0.5)))

        data = np.full((len(positions), len(layers)), np.nan)
        for key, info in target_info.items():
            if info["component"] == component:
                row_idx = positions.index(info["position"]) if info["position"] in positions else -1
                col_idx = layers.index(info["layer"]) if info["layer"] in layers else -1
                if row_idx >= 0 and col_idx >= 0:
                    data[row_idx, col_idx] = info["r2"]

        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("R² Score", fontsize=10)

        for i in range(len(positions)):
            for j in range(len(layers)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=8)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Position")
        ax.set_title(f"Linear Probe R² - {component}\n(Green=high decodability, Red=low)")

        plt.tight_layout()
        plt.savefig(output_dir / f"dashboard_{component}.png", dpi=150)
        plt.close()

    # Combined dashboard
    n_components = len(components)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components + 2, max(5, len(positions) * 0.5)))
    if n_components == 1:
        axes = [axes]

    for idx, component in enumerate(components):
        ax = axes[idx]

        data = np.full((len(positions), len(layers)), np.nan)
        for key, info in target_info.items():
            if info["component"] == component:
                row_idx = positions.index(info["position"]) if info["position"] in positions else -1
                col_idx = layers.index(info["layer"]) if info["layer"] in layers else -1
                if row_idx >= 0 and col_idx >= 0:
                    data[row_idx, col_idx] = info["r2"]

        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")

        for i in range(len(positions)):
            for j in range(len(layers)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=7)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45)
        if idx == 0:
            ax.set_yticks(range(len(positions)))
            ax.set_yticklabels(positions, fontsize=8)
        else:
            ax.set_yticks([])
        ax.set_title(component, fontsize=10)

    fig.colorbar(im, ax=axes, shrink=0.6, label="R² Score")

    plt.suptitle("Summary Dashboard: Linear Probe R² by Layer, Position, Component", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "dashboard_combined.png", dpi=150)
    plt.close()

    logger.info(f"Saved summary dashboards to {output_dir}")


def plot_trajectory(
    data: "ActivationData",
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot PC1 trajectory across layers for each sample."""
    # Group targets by position, component
    target_info = {}
    for key, pca_result in pca_results.items():
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]
        position = parts[1]

        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        key_id = f"{component}_P{position}"
        if key_id not in target_info:
            target_info[key_id] = {}
        target_info[key_id][layer] = {
            "key": key,
            "pca": pca_result,
            "n_samples": pca_result.transformed.shape[0],
        }

    if not target_info:
        logger.warning("No trajectory data available")
        return

    # Get time horizons
    horizons_months = np.array([get_time_horizon_months(s) for s in data.samples])
    horizons_years = horizons_months / 12.0
    log_horizons = np.log10(horizons_years + 0.1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for key_id, layer_data in target_info.items():
        layers = sorted(layer_data.keys())
        if len(layers) < 2:
            continue

        n_samples = min(layer_data[layer]["n_samples"] for layer in layers)
        if n_samples < 2:
            continue

        log_horizons_subset = log_horizons[:n_samples]
        color_vals = (log_horizons_subset - log_horizons_subset.min()) / (log_horizons_subset.max() - log_horizons_subset.min() + 1e-6)

        trajectories = np.zeros((n_samples, len(layers)), dtype=ACTIVATION_DTYPE)
        for i, layer in enumerate(layers):
            pca_result = layer_data[layer]["pca"]
            best_pc_idx = pca_result.pc_correlations[0][0]
            # Clamp to stored components (may be less than computed due to MAX_STORED_PCA_COMPONENTS)
            n_stored = pca_result.transformed.shape[1]
            best_pc_idx = min(best_pc_idx, n_stored - 1)
            trajectories[:, i] = pca_result.transformed[:n_samples, best_pc_idx]

        # Normalize
        for i in range(len(layers)):
            col = trajectories[:, i]
            std = col.std()
            if std > 1e-10:
                trajectories[:, i] = (col - col.mean()) / std
            else:
                trajectories[:, i] = 0

        fig, ax = plt.subplots(figsize=(12, 6))

        for sample_idx in range(min(n_samples, MAX_TRAJECTORY_SAMPLES)):
            color = plt.cm.plasma(color_vals[sample_idx])
            ax.plot(range(len(layers)), trajectories[sample_idx],
                   color=color, alpha=0.3, linewidth=0.8)

        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(vmin=log_horizons_subset.min(), vmax=log_horizons_subset.max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("log₁₀(time horizon in years)", fontsize=10)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized PC projection")
        ax.set_title(f"Trajectory: {key_id}\n(Best time-correlated PC at each layer, n={n_samples})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"trajectory_{key_id}.png", dpi=150)
        plt.close()

    logger.info(f"Saved trajectory plots to {output_dir}")


def plot_component_decomposition(
    linear_probe_results: dict[str, "LinearProbeResult"],
    pca_results: dict[str, "PCAResult"],
    schemes: list["ColoringScheme"],
    output_dir: Path,
):
    """Plot 2x2 component decomposition showing resid_pre, attn_out, mlp_out, resid_post."""
    target_info = {}
    for key in pca_results.keys():
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]
        position = parts[1]

        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        layer_pos_key = f"L{layer}_P{position}"
        if layer_pos_key not in target_info:
            target_info[layer_pos_key] = {}
        target_info[layer_pos_key][component] = {
            "key": key,
            "pca": pca_results[key],
            "r2": linear_probe_results.get(key, None),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    components_order = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

    scheme = next((s for s in schemes if not s.is_categorical), schemes[0] if schemes else None)
    if scheme is None:
        return

    for layer_pos_key, comp_data in target_info.items():
        available_components = [c for c in components_order if c in comp_data]
        if len(available_components) < 2:
            continue

        n_cols = 2
        n_rows = (len(available_components) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, component in enumerate(available_components):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            pca_result = comp_data[component]["pca"]
            X_pca = pca_result.transformed
            best_pc_idx = pca_result.pc_correlations[0][0]
            best_pc_idx = min(best_pc_idx, X_pca.shape[1] - 1)  # Clamp to stored
            best_corr = pca_result.pc_correlations[0][1]

            r2_info = comp_data[component]["r2"]
            r2_val = r2_info.r2_mean if r2_info else 0

            _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False, n_samples=X_pca.shape[0])
            ax.set_xlabel("PC0")
            ax.set_ylabel(f"PC{best_pc_idx}")
            ax.set_title(f"{component}\nR²={r2_val:.3f}, corr={best_corr:.3f}")

        for idx in range(len(available_components), n_rows * n_cols):
            row, col = idx // 2, idx % 2
            axes[row, col].axis("off")

        plt.suptitle(f"Component Decomposition: {layer_pos_key}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"decomp_{layer_pos_key}.png", dpi=150)
        plt.close()

    logger.info(f"Saved component decomposition plots to {output_dir}")


def plot_component_decomposition_3d(
    linear_probe_results: dict[str, "LinearProbeResult"],
    pca_results: dict[str, "PCAResult"],
    schemes: list["ColoringScheme"],
    output_dir: Path,
):
    """Plot interactive 3D component decomposition showing all 4 components."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not available, skipping 3D component decomposition")
        return

    target_info = {}
    for key in pca_results.keys():
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]
        position = parts[1]

        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        layer_pos_key = f"L{layer}_P{position}"
        if layer_pos_key not in target_info:
            target_info[layer_pos_key] = {}
        target_info[layer_pos_key][component] = {
            "key": key,
            "pca": pca_results[key],
            "r2": linear_probe_results.get(key, None),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    components_order = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
    component_colors = {"resid_pre": "blue", "attn_out": "green", "mlp_out": "orange", "resid_post": "red"}

    # Use first continuous scheme for coloring
    scheme = next((s for s in schemes if not s.is_categorical), schemes[0] if schemes else None)
    if scheme is None:
        return

    for layer_pos_key, comp_data in target_info.items():
        available_components = [c for c in components_order if c in comp_data]
        if len(available_components) < 2:
            continue

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
                   [{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=[c if c in available_components else "" for c in components_order],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        for idx, component in enumerate(components_order):
            if component not in comp_data:
                continue

            row, col = idx // 2 + 1, idx % 2 + 1
            pca_result = comp_data[component]["pca"]
            X_pca = pca_result.transformed
            n_samples = X_pca.shape[0]

            # Get top 3 PCs by correlation, clamped to stored components
            n_stored = X_pca.shape[1]
            top_pcs = [min(c[0], n_stored - 1) for c in pca_result.pc_correlations[:3]]
            if len(top_pcs) < 3:
                top_pcs = list(range(min(3, n_stored)))

            r2_info = comp_data[component]["r2"]
            r2_val = r2_info.r2_mean if r2_info else 0

            # Color by time horizon
            values = scheme.values[:n_samples] if n_samples < len(scheme.values) else scheme.values
            color_vals = np.log10(values + 1) if scheme.use_log else values

            fig.add_trace(
                go.Scatter3d(
                    x=X_pca[:, top_pcs[0]],
                    y=X_pca[:, top_pcs[1]],
                    z=X_pca[:, top_pcs[2]] if len(top_pcs) > 2 else np.zeros(n_samples),
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=color_vals,
                        colorscale="Plasma",
                        opacity=0.7,
                    ),
                    name=f"{component} (R²={r2_val:.3f})",
                    hovertemplate=f"{component}<br>R²={r2_val:.3f}<br>%{{text}}<extra></extra>",
                    text=[f"{scheme.label}: {v:.1f}" for v in values],
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title=f"3D Component Decomposition: {layer_pos_key}",
            height=900,
            width=1200,
            showlegend=True,
        )

        # Update all scenes
        for i in range(1, 5):
            scene_name = f"scene{i}" if i > 1 else "scene"
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title="PC (best)",
                    yaxis_title="PC (2nd)",
                    zaxis_title="PC (3rd)",
                )
            })

        fig.write_html(output_dir / f"decomp_3d_{layer_pos_key}.html")

    logger.info(f"Saved 3D component decomposition plots to {output_dir}")


def plot_direction_alignment(
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot direction alignment heatmaps showing cosine similarity between PC1 directions."""
    directions = {}
    for key, pca_result in pca_results.items():
        parts = key.split("_P")
        if len(parts) != 2:
            continue

        best_pc_idx = pca_result.pc_correlations[0][0]
        best_pc_idx = min(best_pc_idx, pca_result.components.shape[0] - 1)  # Clamp to stored
        direction = pca_result.components[best_pc_idx]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions[key] = direction

    if len(directions) < 2:
        logger.warning("Not enough targets for direction alignment")
        return

    keys = sorted(directions.keys())
    n = len(keys)

    similarity = np.zeros((n, n), dtype=ACTIVATION_DTYPE)
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            similarity[i, j] = abs(np.dot(directions[key_i], directions[key_j]))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.3), max(8, n * 0.25)))
    im = ax.imshow(similarity, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    labels = []
    for key in keys:
        parts = key.split("_P")
        base = parts[0].replace("_", "")
        pos = parts[1][:4] if len(parts) > 1 else ""
        labels.append(f"{base[:6]}_{pos}")

    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=6)

    plt.colorbar(im, ax=ax, shrink=0.6, label="|cosine similarity|")
    ax.set_title("Direction Alignment: PC1 Cosine Similarity Across Targets")

    plt.tight_layout()
    plt.savefig(output_dir / "direction_alignment.png", dpi=150)
    plt.close()

    # Per-position alignment plots
    position_groups = {}
    for key in keys:
        parts = key.split("_P")
        if len(parts) == 2:
            pos = parts[1]
            if pos not in position_groups:
                position_groups[pos] = []
            position_groups[pos].append(key)

    for pos, pos_keys in position_groups.items():
        if len(pos_keys) < 2:
            continue

        n_pos = len(pos_keys)
        sim_pos = np.zeros((n_pos, n_pos), dtype=ACTIVATION_DTYPE)
        for i, key_i in enumerate(pos_keys):
            for j, key_j in enumerate(pos_keys):
                sim_pos[i, j] = abs(np.dot(directions[key_i], directions[key_j]))

        fig, ax = plt.subplots(figsize=(max(6, n_pos * 0.5), max(5, n_pos * 0.4)))
        im = ax.imshow(sim_pos, cmap="viridis", vmin=0, vmax=1, aspect="auto")

        labels_pos = [k.split("_P")[0] for k in pos_keys]
        ax.set_xticks(range(n_pos))
        ax.set_xticklabels(labels_pos, fontsize=8, rotation=45)
        ax.set_yticks(range(n_pos))
        ax.set_yticklabels(labels_pos, fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8, label="|cos sim|")
        ax.set_title(f"Direction Alignment: Position={pos}")

        plt.tight_layout()
        plt.savefig(output_dir / f"direction_alignment_{pos}.png", dpi=150)
        plt.close()

    logger.info(f"Saved direction alignment plots to {output_dir}")


def plot_decision_boundary(
    data: "ActivationData",
    linear_probe_results: dict[str, "LinearProbeResult"],
    output_dir: Path,
):
    """Plot decision boundary accuracy across layers."""
    if not data.choices:
        logger.warning("No choice data for decision boundary plot")
        return

    chose_long = np.array([1 if c.chose_long_term else 0 for c in data.choices], dtype=np.int8)

    if len(set(chose_long)) < 2:
        logger.warning("All samples have same choice, skipping decision boundary")
        return

    layer_accuracy = {}
    for key, result in linear_probe_results.items():
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]
        position = parts[1]

        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        predictions = result.predictions
        n_samples = len(predictions)
        target_chose_long = chose_long[:n_samples] if n_samples < len(chose_long) else chose_long
        median_pred = np.median(predictions)
        predicted_choice = (predictions > median_pred).astype(np.int8)
        accuracy = float((predicted_choice == target_chose_long).mean())

        layer_key = f"L{layer}_{component}_P{position}"
        layer_accuracy[layer_key] = {
            "layer": layer,
            "component": component,
            "position": position,
            "accuracy": accuracy,
        }

    layers_set = sorted(set(info["layer"] for info in layer_accuracy.values()))

    dest_positions = {"response", "dest"}
    src_positions = {"time_horizon", "short_term_time", "short_term_reward",
                    "long_term_time", "long_term_reward", "source"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (pos_type, pos_set) in zip(axes, [("Dest", dest_positions), ("Source", src_positions)]):
        layer_acc = {layer: [] for layer in layers_set}
        for key, info in layer_accuracy.items():
            if info["position"] in pos_set:
                layer_acc[info["layer"]].append(info["accuracy"])

        layers_with_data = [l for l in layers_set if layer_acc[l]]
        if not layers_with_data:
            ax.text(0.5, 0.5, f"No {pos_type} data", transform=ax.transAxes, ha="center")
            continue

        mean_acc = [np.mean(layer_acc[l]) for l in layers_with_data]
        std_acc = [np.std(layer_acc[l]) for l in layers_with_data]

        ax.errorbar(layers_with_data, mean_acc, yerr=std_acc,
                   marker="o", linewidth=2, capsize=5, color="#2196F3" if pos_type == "Dest" else "#F44336")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Choice Prediction Accuracy")
        ax.set_title(f"Decision Boundary: {pos_type} Positions")
        ax.set_ylim(0.3, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Linear Probe Accuracy for Predicting Choice (Short vs Long Term)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "decision_boundary.png", dpi=150)
    plt.close()

    logger.info(f"Saved decision boundary plot to {output_dir}")


def plot_scree(
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot scree plots showing variance explained by each PC."""
    output_dir.mkdir(parents=True, exist_ok=True)

    key_targets = []
    for key in pca_results.keys():
        if "response" in key or "dest" in key:
            key_targets.append(key)

    for key in pca_results.keys():
        if key not in key_targets:
            key_targets.append(key)
        if len(key_targets) >= 12:
            break

    for target_key in key_targets:
        pca_result = pca_results[target_key]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        n_show = min(20, len(pca_result.explained_variance))
        ax.bar(range(n_show), pca_result.explained_variance[:n_show], color="#2196F3", alpha=0.7)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Variance Explained")

        ax = axes[1]
        cumsum = np.cumsum(pca_result.explained_variance[:n_show])
        ax.plot(range(n_show), cumsum, marker="o", color="#4CAF50", linewidth=2)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(0.95, color="orange", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        ax.set_title("Cumulative Variance Explained")
        ax.legend()
        ax.set_ylim(0, 1.05)

        plt.suptitle(f"Scree Plot: {target_key}", fontsize=11)
        plt.tight_layout()

        safe_name = target_key.replace("/", "_")
        plt.savefig(output_dir / f"scree_{safe_name}.png", dpi=150)
        plt.close()

    # Summary scree
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for idx, target_key in enumerate(key_targets[:8]):
        pca_result = pca_results[target_key]
        cumsum = np.cumsum(pca_result.explained_variance[:20])
        parts = target_key.split("_P")
        label = f"{parts[0][-8:]}_P{parts[1][:4]}" if len(parts) == 2 else target_key[:15]
        ax.plot(range(len(cumsum)), cumsum, marker=".", color=colors[idx % len(colors)],
               linewidth=1.5, alpha=0.8, label=label)

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Scree Comparison: Key Targets")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scree_comparison.png", dpi=150)
    plt.close()

    logger.info(f"Saved scree plots to {output_dir}")


# =============================================================================
# Summary Plots
# =============================================================================


def plot_linear_probe_summary(
    data: ActivationData,
    results: dict[str, LinearProbeResult],
    output_dir: Path,
    top_n: int = 30,
):
    """Plot linear probe summary bar chart showing top N targets."""
    horizons, log_horizons, _ = _get_horizons(data)

    # Sort and take top N
    all_targets = sorted(results.keys(), key=lambda k: results[k].r2_mean, reverse=True)
    targets = all_targets[:top_n]
    r2_values = [results[t].r2_mean for t in targets]
    r2_stds = [results[t].r2_std for t in targets]

    colors = ["#4CAF50" if "Pdest" in t or "Presponse" in t else "#F44336" for t in targets]

    fig, ax = plt.subplots(figsize=(12, max(6, len(targets) * 0.25)))

    ax.barh(range(len(targets)), r2_values, xerr=r2_stds, color=colors, alpha=0.8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel("R² Score (5-fold CV)")
    ax.set_title(f"Linear Probe for Time Horizon Decoding (Top {len(targets)})\n(Green=Dest, Red=Source)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-0.1, 1.1)
    ax.invert_yaxis()  # Best at top

    for i, (t, r2) in enumerate(zip(targets, r2_values)):
        ax.text(max(r2, 0) + 0.02, i, f"{r2:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe_summary.png", dpi=150)
    plt.close()

    # Scatter plots for top 16 only
    scatter_targets = all_targets[:16]
    n_targets = len(scatter_targets)
    n_cols = min(4, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, target_key in enumerate(scatter_targets):
        result = results[target_key]
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        n_samples = len(result.predictions)
        target_log_horizons = log_horizons[:n_samples] if n_samples < len(log_horizons) else log_horizons

        ax.scatter(
            target_log_horizons,
            result.predictions,
            c=target_log_horizons,
            cmap=CMAP_GRADIENT,
            s=10,
            alpha=0.6,
        )
        ax.plot(
            [target_log_horizons.min(), target_log_horizons.max()],
            [target_log_horizons.min(), target_log_horizons.max()],
            "r--",
            alpha=0.5,
        )
        ax.set_xlabel("Actual log₁₀(years)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target_key}\nR²={result.r2_mean:.3f}", fontsize=9)

    for idx in range(len(scatter_targets), n_rows * n_cols):
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
    n_samples: int | None = None,
) -> None:
    """Scatter plot with a coloring scheme."""
    values = scheme.values
    if n_samples is not None and n_samples < len(values):
        values = values[:n_samples]

    if scheme.is_categorical:
        if scheme.name == "time_scale":
            colors = TIME_SCALE_COLORS
        else:
            colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

        categories = scheme.categories or [str(i) for i in range(len(colors))]
        unique_vals = sorted(set(int(v) for v in values))

        for val in unique_vals:
            color = colors[val % len(colors)]
            label = categories[val] if val < len(categories) else str(val)
            mask = values == val
            if mask.sum() > 0:
                ax.scatter(x[mask], y[mask], c=color, s=15, alpha=0.7, label=label)
        ax.legend(markerscale=2, fontsize=8)
    else:
        vmin = float(values.min())
        vmax = float(values.max())

        if scheme.use_log:
            vmin = max(0.01, vmin)
            vmax = max(vmin * 1.1, vmax)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmax <= vmin:
                vmax = vmin + 1
            norm = None

        sc = ax.scatter(x, y, c=values, cmap=CMAP_GRADIENT, s=15, alpha=0.7, norm=norm)
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
    # Clamp to stored components
    best_pc_idx = min(best_pc_idx, X_pca.shape[1] - 1)

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
    n_samples = X_pca.shape[0]
    _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], schemes[0], n_samples=n_samples)
    ax.set_xlabel(f"PC0 ({pca_result.explained_variance[0]*100:.1f}%)")
    ax.set_ylabel(f"PC{best_pc_idx} (r={best_corr:.2f})")
    ax.set_title("PC0 vs Best Time-Correlated PC")

    ax = axes[1, 1]
    _scatter_with_scheme(ax, X_pca[:, best_pc_idx], schemes[0].values[:n_samples], schemes[0], n_samples=n_samples)
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
        _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, n_samples=n_samples)
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
    """Generate PCA 2D embedding plots for a single target."""
    emb_dir = target_dir / "pca_2d"
    emb_dir.mkdir(parents=True, exist_ok=True)

    coords = embedding_result.pca_embedding
    if coords is None or len(coords) == 0:
        return

    # One plot per coloring scheme
    for scheme in schemes:
        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter_with_scheme(ax, coords[:, 0], coords[:, 1], scheme, n_samples=coords.shape[0])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA 2D - {scheme.label}")
        plt.tight_layout()
        plt.savefig(emb_dir / f"{scheme.name}.png", dpi=150)
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
    n_samples = X_pca.shape[0]
    n_stored = X_pca.shape[1]
    top_pcs = [min(c[0], n_stored - 1) for c in pca_result.pc_correlations[:3]]
    corrs = [pca_result.pc_correlations[i][1] for i in range(min(3, len(pca_result.pc_correlations)))]

    for scheme in schemes:
        values = scheme.values[:n_samples] if n_samples < len(scheme.values) else scheme.values

        if scheme.is_categorical:
            if scheme.name == "time_scale":
                color_map = {0: "purple", 1: "blue", 2: "green", 3: "orange"}
            else:
                color_map = {0: "blue", 1: "red", 2: "green", 3: "orange"}
            colors = [color_map.get(int(v), "gray") for v in values]
            hover_text = [scheme.categories[int(v)] if scheme.categories and int(v) < len(scheme.categories) else str(int(v)) for v in values]
            marker = dict(size=4, color=colors, opacity=0.8)
        else:
            color_vals = np.log10(values + 1) if scheme.use_log else values
            hover_text = [f"{scheme.label}: {v:.1f}" for v in values]
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


def _is_absolute_position(pos: str) -> bool:
    """Check if position is an absolute index (e.g., 86, 145)."""
    return pos.isdigit()


def _parse_target_key(target_key: str) -> tuple[str, str, str]:
    """Parse target key into (base_key, position, position_type)."""
    source_positions = {"time_horizon", "short_term_time", "short_term_reward",
                       "long_term_time", "long_term_reward", "source"}
    dest_positions = {"response", "dest"}

    p_idx = target_key.rfind("_P")
    if p_idx == -1:
        return target_key, "unknown", "unknown"

    base_key = target_key[:p_idx]
    position = target_key[p_idx + 2:]

    if position in source_positions:
        pos_type = "src"
    elif position in dest_positions:
        pos_type = "dst"
    elif _is_absolute_position(position):
        pos_type = "absolute"
    else:
        pos_type = "unknown"

    return base_key, position, pos_type


def _group_targets_by_base(target_keys: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    """Group target keys by their base (layer+component) and position type."""
    groups = {}
    for key in target_keys:
        base, pos, pos_type = _parse_target_key(key)
        if base not in groups:
            groups[base] = {"dst": {}, "src": {}, "absolute": {}}
        if pos_type in groups[base]:
            groups[base][pos_type][pos] = key
    return groups


def _get_pos_color(pos: str) -> str:
    """Get color for a position name."""
    dest_positions = {"response", "dest"}
    if pos in dest_positions:
        return "#4CAF50"
    elif _is_absolute_position(pos):
        idx = int(pos)
        if idx >= 140:
            return "#9C27B0"
        else:
            return "#2196F3"
    else:
        return "#F44336"


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
    """Generate all plots with memory-efficient processing.

    Folder structure:
        plots/
        ├── 01_dashboard/           # Summary heatmaps
        ├── 02_linear_probe/        # Linear probe summary
        ├── 03_decision_boundary/   # Choice prediction
        ├── 04_trajectories/        # PC1 across layers
        ├── 05_direction_alignment/ # Cosine similarity
        ├── 06_scree/               # Variance explained
        ├── 07_component_decomp/    # 2x2 component plots (2D)
        ├── 08_component_decomp_3d/ # Interactive 3D component plots
        └── 09_targets/             # Per-target plots
            └── {base_key}/
                └── {pos_type}/
                    ├── pca/
                    ├── pca_2d/
                    └── 3d/
    """
    output_dir = config.output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = get_coloring_schemes(data)
    logger.info(f"Using {len(schemes)} coloring schemes: {[s.name for s in schemes]}")

    # Priority 1: Summary dashboard
    dashboard_dir = output_dir / "01_dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating summary dashboard heatmaps...")
    plot_summary_dashboard(linear_probe_results, pca_results, dashboard_dir)
    gc.collect()

    # Priority 2: Linear probe summary
    linear_dir = output_dir / "02_linear_probe"
    linear_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating linear probe summary...")
    plot_linear_probe_summary(data, linear_probe_results, linear_dir)
    gc.collect()

    # Priority 3: Decision boundary
    boundary_dir = output_dir / "03_decision_boundary"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating decision boundary plot...")
    plot_decision_boundary(data, linear_probe_results, boundary_dir)
    gc.collect()

    # Priority 4: Trajectory plots
    traj_dir = output_dir / "04_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating trajectory plots...")
    plot_trajectory(data, pca_results, traj_dir)
    gc.collect()

    # Priority 5: Direction alignment
    align_dir = output_dir / "05_direction_alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating direction alignment plots...")
    plot_direction_alignment(pca_results, align_dir)
    gc.collect()

    # Priority 6: Scree plots
    scree_dir = output_dir / "06_scree"
    scree_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating scree plots...")
    plot_scree(pca_results, scree_dir)
    gc.collect()

    # Priority 7: Component decomposition (2D)
    decomp_dir = output_dir / "07_component_decomp"
    decomp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating component decomposition plots...")
    plot_component_decomposition(linear_probe_results, pca_results, schemes, decomp_dir)
    gc.collect()

    # Priority 8: Component decomposition (3D)
    decomp_3d_dir = output_dir / "08_component_decomp_3d"
    decomp_3d_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating 3D component decomposition plots...")
    plot_component_decomposition_3d(linear_probe_results, pca_results, schemes, decomp_3d_dir)
    gc.collect()

    # Per-target plots (process in batches)
    logger.info("Generating per-target plots...")
    targets_dir = output_dir / "09_targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    target_groups = _group_targets_by_base(list(pca_results.keys()))
    n_groups = len(target_groups)

    for group_idx, (base_key, pos_type_targets) in enumerate(target_groups.items()):
        if group_idx % 10 == 0:
            logger.info(f"  Processing group {group_idx}/{n_groups}: {base_key}")

        base_dir = targets_dir / base_key
        base_dir.mkdir(parents=True, exist_ok=True)

        # Process each position type
        for pos_type in ["dst", "src", "absolute"]:
            positions = pos_type_targets.get(pos_type, {})
            if not positions:
                continue

            type_dir = base_dir / pos_type
            type_dir.mkdir(parents=True, exist_ok=True)

            for pos, target_key in positions.items():
                if target_key not in pca_results:
                    continue

                plot_target_pca(target_key, pca_results[target_key], schemes, type_dir)
                if target_key in embedding_results:
                    plot_target_embeddings(target_key, embedding_results[target_key], schemes, type_dir)
                plot_target_3d(target_key, pca_results[target_key], schemes, type_dir)

        # GC after each group
        if group_idx % PLOT_GC_INTERVAL == 0:
            gc.collect()

    logger.info(f"All plots saved to {output_dir}")
