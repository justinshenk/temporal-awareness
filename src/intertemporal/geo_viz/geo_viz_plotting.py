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

from .geo_viz_analysis import EmbeddingResult, LinearProbeResult, NoHorizonProjectionResult, PCAResult
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
                    # Clamp R² to [0, 1] for display - negative values indicate poor fit
                    data[row_idx, col_idx] = np.clip(info["r2"], 0.0, 1.0)

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

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
                    # Clamp R² to [0, 1] for display - negative values indicate poor fit
                    data[row_idx, col_idx] = np.clip(info["r2"], 0.0, 1.0)

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

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

    # Add colorbar with proper positioning outside the plot
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="R² Score", pad=0.02)

    plt.suptitle("Summary Dashboard: Linear Probe R² by Layer, Position, Component", fontsize=12)
    # Use tight_layout with rect to leave space for colorbar on right
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
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
# Cross-Position Similarity Plots
# =============================================================================


def plot_cross_position_similarity(
    cross_position_results: dict[str, "CrossPositionSimilarityResult"],
    output_dir: Path,
):
    """Plot cross-position cosine similarity heatmaps.

    Shows how PC0 directions at source positions correlate with
    PC0 directions at destination positions across layers.
    """
    from .geo_viz_analysis import CrossPositionSimilarityResult

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cross_position_results:
        logger.warning("No cross-position similarity results to plot")
        return

    # Group by component
    component_results = {}
    for lc_key, result in cross_position_results.items():
        component = result.component
        if component not in component_results:
            component_results[component] = {}
        component_results[component][result.layer] = result

    # 1. Summary plot: layer x component heatmap of best similarities
    components = sorted(component_results.keys())
    all_layers = sorted(set(r.layer for r in cross_position_results.values()))

    if len(all_layers) > 1 and len(components) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.5), max(4, len(components) * 0.5)))

        data = np.zeros((len(components), len(all_layers)))
        for i, comp in enumerate(components):
            for j, layer in enumerate(all_layers):
                key = f"L{layer}_{comp}"
                if key in cross_position_results:
                    data[i, j] = cross_position_results[key].best_similarity

        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Best |cos similarity|", fontsize=10)

        for i in range(len(components)):
            for j in range(len(all_layers)):
                val = data[i, j]
                if val > 0:
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_xticks(range(len(all_layers)))
        ax.set_xticklabels([f"L{l}" for l in all_layers], fontsize=9)
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Component")
        ax.set_title("Cross-Position Similarity: Source to Dest PC0 Direction\n(High = temporal direction similar to decision direction)")

        plt.tight_layout()
        plt.savefig(output_dir / "cross_position_summary.png", dpi=150)
        plt.close()

    # 2. Per-component: layer trajectory of similarity
    for component, layer_results in component_results.items():
        layers = sorted(layer_results.keys())
        if len(layers) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        best_sims = [layer_results[l].best_similarity for l in layers]
        mean_sims = [layer_results[l].mean_similarity for l in layers]

        ax.plot(layers, best_sims, marker="o", linewidth=2, label="Best pair", color="#2196F3")
        ax.plot(layers, mean_sims, marker="s", linewidth=2, label="Mean", color="#4CAF50", alpha=0.7)

        ax.set_xlabel("Layer")
        ax.set_ylabel("|Cosine Similarity|")
        ax.set_title(f"Cross-Position Similarity: {component}\n(Source PC0 vs Dest PC0)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"cross_position_{component}.png", dpi=150)
        plt.close()

    # 3. Per layer-component detailed matrices
    for lc_key, result in cross_position_results.items():
        if result.similarity_matrix.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(4, len(result.dest_positions) * 1.5),
                                        max(3, len(result.source_positions) * 0.8)))

        im = ax.imshow(result.similarity_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|cos sim|")

        for i in range(len(result.source_positions)):
            for j in range(len(result.dest_positions)):
                val = result.similarity_matrix[i, j]
                if val > 0:
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_xticks(range(len(result.dest_positions)))
        ax.set_xticklabels(result.dest_positions, fontsize=9)
        ax.set_yticks(range(len(result.source_positions)))
        ax.set_yticklabels(result.source_positions, fontsize=9)
        ax.set_xlabel("Destination Position")
        ax.set_ylabel("Source Position")
        ax.set_title(f"{lc_key}: Source vs Dest PC0 Direction Similarity")

        plt.tight_layout()
        plt.savefig(output_dir / f"cross_position_{lc_key}_matrix.png", dpi=150)
        plt.close()

    logger.info(f"Saved cross-position similarity plots to {output_dir}")


# =============================================================================
# Continuous Time Probe Plots
# =============================================================================


def plot_continuous_time_probe(
    continuous_time_results: dict[str, "ContinuousTimeProbeResult"],
    output_dir: Path,
    top_n: int = 20,
):
    """Plot continuous time horizon probe results for source positions.

    Shows how well we can predict the raw time_horizon_months value
    from activations at different source positions and layers.
    """
    from .geo_viz_analysis import ContinuousTimeProbeResult

    output_dir.mkdir(parents=True, exist_ok=True)

    if not continuous_time_results:
        logger.warning("No continuous time probe results to plot")
        return

    # 1. Summary bar chart of top R² scores
    sorted_results = sorted(
        continuous_time_results.items(),
        key=lambda x: x[1].r2_mean,
        reverse=True
    )

    targets = [k for k, _ in sorted_results[:top_n]]
    r2_values = [continuous_time_results[t].r2_mean for t in targets]
    r2_stds = [continuous_time_results[t].r2_std for t in targets]

    fig, ax = plt.subplots(figsize=(12, max(6, len(targets) * 0.25)))

    colors = []
    for t in targets:
        if "time_horizon" in t:
            colors.append("#4CAF50")  # Green for time_horizon position
        elif "short_term" in t:
            colors.append("#2196F3")  # Blue for short-term positions
        elif "long_term" in t:
            colors.append("#FF9800")  # Orange for long-term positions
        else:
            colors.append("#9C27B0")  # Purple for source aggregate

    ax.barh(range(len(targets)), r2_values, xerr=r2_stds, color=colors, alpha=0.8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel("R² Score (5-fold CV)")
    ax.set_title(f"Continuous Time Horizon Probe at Source Positions (Top {len(targets)})\n"
                 f"(Green=time_horizon, Blue=short_term, Orange=long_term, Purple=source)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-0.1, 1.1)
    ax.invert_yaxis()

    for i, (t, r2) in enumerate(zip(targets, r2_values)):
        ax.text(max(r2, 0) + 0.02, i, f"{r2:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "continuous_time_probe_summary.png", dpi=150)
    plt.close()

    # 2. Layer trajectory by position type
    position_results = {}
    for key, result in continuous_time_results.items():
        # Parse target key
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        position = parts[1]
        base = parts[0]

        import re
        layer_match = re.match(r"L(\d+)_(.+)", base)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        component = layer_match.group(2)

        pos_comp_key = f"{position}_{component}"
        if pos_comp_key not in position_results:
            position_results[pos_comp_key] = {}
        position_results[pos_comp_key][layer] = result.r2_mean

    # Plot trajectories grouped by component
    components = set()
    positions_set = set()
    for key in position_results.keys():
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            positions_set.add(parts[0])
            components.add(parts[1])

    for component in sorted(components):
        fig, ax = plt.subplots(figsize=(10, 5))

        position_colors = {
            "time_horizon": "#4CAF50",
            "short_term_time": "#2196F3",
            "short_term_reward": "#03A9F4",
            "long_term_time": "#FF9800",
            "long_term_reward": "#FFC107",
            "source": "#9C27B0",
        }

        for position in sorted(positions_set):
            key = f"{position}_{component}"
            if key not in position_results:
                continue

            layer_data = position_results[key]
            layers = sorted(layer_data.keys())
            r2s = [layer_data[l] for l in layers]

            color = position_colors.get(position, "#607D8B")
            ax.plot(layers, r2s, marker="o", linewidth=2, label=position, color=color)

        ax.set_xlabel("Layer")
        ax.set_ylabel("R² Score")
        ax.set_title(f"Continuous Time Horizon Probe: {component}\n(Source positions only)")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"continuous_time_probe_{component}_trajectory.png", dpi=150)
        plt.close()

    # 3. Scatter plots for top targets
    scatter_targets = [k for k, _ in sorted_results[:9]]
    n_targets = len(scatter_targets)
    if n_targets > 0:
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, target_key in enumerate(scatter_targets):
            result = continuous_time_results[target_key]
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Plot predictions vs actuals (both in months, log scale)
            actuals_log = np.log10(result.actuals + 1)
            predictions_log = np.log10(result.predictions + 1)

            ax.scatter(
                actuals_log,
                predictions_log,
                c=actuals_log,
                cmap=CMAP_GRADIENT,
                s=10,
                alpha=0.6,
            )

            # Perfect prediction line
            min_val = min(actuals_log.min(), predictions_log.min())
            max_val = max(actuals_log.max(), predictions_log.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

            ax.set_xlabel("Actual log₁₀(months+1)")
            ax.set_ylabel("Predicted log₁₀(months+1)")
            ax.set_title(f"{target_key}\nR²={result.r2_mean:.3f}", fontsize=9)

        # Hide unused axes
        for idx in range(len(scatter_targets), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "continuous_time_probe_scatter.png", dpi=150)
        plt.close()

    logger.info(f"Saved continuous time probe plots to {output_dir}")


# =============================================================================
# Logit Lens Plots
# =============================================================================


def plot_logit_lens(
    logit_lens_result: "LogitLensResult",
    output_dir: Path,
):
    """Generate logit lens analysis plots.

    Creates:
    1. Line plot: layer vs mean logit difference (shows when model "crystallizes" answer)
    2. Heatmap: samples x layers with logit difference values
    3. Cosine similarity between normalized residual stream and logit direction per layer

    Expected pattern:
    - Near-zero alignment in early layers (L0-L17)
    - Rising alignment L19-L24 as circuit activates
    - High alignment by L28+ as model commits
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = logit_lens_result.layers
    n_layers = logit_lens_result.n_layers
    n_samples = logit_lens_result.n_samples
    logit_diffs = logit_lens_result.logit_diffs  # [n_layers, n_samples]
    cosine_sims = logit_lens_result.cosine_sims  # [n_layers, n_samples]

    # 1. Line plot: layer vs mean logit difference
    fig, ax = plt.subplots(figsize=(12, 6))

    mean_logit_diff = logit_diffs.mean(axis=1)
    std_logit_diff = logit_diffs.std(axis=1)

    ax.plot(layers, mean_logit_diff, marker="o", linewidth=2, color="#2196F3", label="Mean logit diff")
    ax.fill_between(
        layers,
        mean_logit_diff - std_logit_diff,
        mean_logit_diff + std_logit_diff,
        alpha=0.2,
        color="#2196F3",
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Annotate key layers
    key_layers = [0, 12, 19, 21, 24, 28, n_layers - 1] if n_layers > 24 else layers
    for l in key_layers:
        if l < len(layers):
            idx = layers.index(l) if l in layers else -1
            if idx >= 0:
                ax.annotate(
                    f"L{l}",
                    xy=(l, mean_logit_diff[idx]),
                    xytext=(l, mean_logit_diff[idx] + std_logit_diff[idx] * 0.5),
                    fontsize=8,
                    ha="center",
                )

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Logit({logit_lens_result.token_a_str}) - Logit({logit_lens_result.token_b_str})")
    ax.set_title(
        f"Logit Lens: When Does the Model Commit to Its Answer?\n"
        f"(Token '{logit_lens_result.token_a_str}' vs '{logit_lens_result.token_b_str}', n={n_samples})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_trajectory.png", dpi=150)
    plt.close()

    # 2. Heatmap: samples x layers
    fig, ax = plt.subplots(figsize=(14, 8))

    # Limit samples for visualization
    max_samples_heatmap = min(200, n_samples)
    sample_indices = np.linspace(0, n_samples - 1, max_samples_heatmap, dtype=int)

    heatmap_data = logit_diffs[:, sample_indices].T  # [n_samples, n_layers]

    # Determine color scale (symmetric around zero)
    vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
    vmin = -vmax

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Logit({logit_lens_result.token_a_str}) - Logit({logit_lens_result.token_b_str})")

    # Set layer ticks
    layer_tick_step = max(1, n_layers // 10)
    ax.set_xticks(range(0, n_layers, layer_tick_step))
    ax.set_xticklabels([f"L{layers[i]}" for i in range(0, n_layers, layer_tick_step)])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Sample")
    ax.set_title(
        f"Logit Lens Heatmap: Per-Sample Evolution\n"
        f"(Blue = '{logit_lens_result.token_b_str}' favored, Red = '{logit_lens_result.token_a_str}' favored)"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_heatmap.png", dpi=150)
    plt.close()

    # 3. Cosine similarity plot
    fig, ax = plt.subplots(figsize=(12, 6))

    mean_cos_sim = cosine_sims.mean(axis=1)
    std_cos_sim = cosine_sims.std(axis=1)

    ax.plot(layers, mean_cos_sim, marker="o", linewidth=2, color="#4CAF50", label="Mean cosine sim")
    ax.fill_between(
        layers,
        mean_cos_sim - std_cos_sim,
        mean_cos_sim + std_cos_sim,
        alpha=0.2,
        color="#4CAF50",
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity with Logit Direction")
    ax.set_title(
        f"Residual Stream Alignment with Logit Direction\n"
        f"(Direction: {logit_lens_result.token_a_str} - {logit_lens_result.token_b_str}, n={n_samples})"
    )
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_cosine_sim.png", dpi=150)
    plt.close()

    # 4. Combined plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Logit difference trajectory
    ax = axes[0, 0]
    ax.plot(layers, mean_logit_diff, marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(
        layers,
        mean_logit_diff - std_logit_diff,
        mean_logit_diff + std_logit_diff,
        alpha=0.2,
        color="#2196F3",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Logit Difference")
    ax.set_title("Mean Logit Difference by Layer")
    ax.grid(True, alpha=0.3)

    # Top-right: Cosine similarity trajectory
    ax = axes[0, 1]
    ax.plot(layers, mean_cos_sim, marker="o", linewidth=2, color="#4CAF50")
    ax.fill_between(
        layers,
        mean_cos_sim - std_cos_sim,
        mean_cos_sim + std_cos_sim,
        alpha=0.2,
        color="#4CAF50",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Residual-Logit Direction Alignment")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Heatmap
    ax = axes[1, 0]
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(0, n_layers, layer_tick_step))
    ax.set_xticklabels([f"L{layers[i]}" for i in range(0, n_layers, layer_tick_step)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sample")
    ax.set_title("Per-Sample Logit Difference")

    # Bottom-right: Histogram of final layer logit diffs
    ax = axes[1, 1]
    final_logit_diffs = logit_diffs[-1, :]  # Last layer
    ax.hist(final_logit_diffs, bins=50, color="#9C27B0", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Decision boundary")
    ax.axvline(
        final_logit_diffs.mean(),
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {final_logit_diffs.mean():.2f}",
    )
    ax.set_xlabel(f"Logit({logit_lens_result.token_a_str}) - Logit({logit_lens_result.token_b_str})")
    ax.set_ylabel("Count")
    ax.set_title(f"Final Layer (L{layers[-1]}) Logit Difference Distribution")
    ax.legend()

    plt.suptitle(
        f"Logit Lens Analysis: '{logit_lens_result.token_a_str}' vs '{logit_lens_result.token_b_str}' (n={n_samples})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "logit_lens_summary.png", dpi=150)
    plt.close()

    logger.info(f"Saved logit lens plots to {output_dir}")


# =============================================================================
# No-Horizon Projection Plots
# =============================================================================


def plot_no_horizon_projection(
    no_horizon_results: dict[str, NoHorizonProjectionResult],
    output_dir: Path,
    top_n: int = 20,
):
    """Plot no-horizon projection analysis results.

    Shows where samples WITHOUT time horizons project in PCA space
    fitted on samples WITH time horizons. Tests the "default bias" hypothesis:
    do no-horizon samples cluster with short-horizon or long-horizon samples?

    Generates:
    1. Summary: bias_ratio heatmap across layers/components
    2. Summary: bar chart of distances to short vs long centroids
    3. Per-target: 2D scatter with horizon samples colored, no-horizon as stars
    4. Distribution: histogram of per-sample distances
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not no_horizon_results:
        logger.warning("No no-horizon projection results to plot")
        return

    # Parse results to extract layer, component, position
    result_info = {}
    for key, result in no_horizon_results.items():
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

        result_info[key] = {
            "layer": layer,
            "component": component,
            "position": position,
            "result": result,
        }

    if not result_info:
        logger.warning("No valid no-horizon results after parsing")
        return

    # =========================================================================
    # 1. Summary heatmap: bias_ratio by layer and component
    # =========================================================================
    positions = sorted(set(info["position"] for info in result_info.values()))

    # Group by position for separate heatmaps
    for position in positions:
        pos_results = {k: v for k, v in result_info.items() if v["position"] == position}
        if len(pos_results) < 2:
            continue

        pos_layers = sorted(set(info["layer"] for info in pos_results.values()))
        pos_components = sorted(set(info["component"] for info in pos_results.values()))

        if len(pos_layers) < 1 or len(pos_components) < 1:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(pos_layers) * 0.8), max(4, len(pos_components) * 0.6)))

        heatmap_data = np.full((len(pos_components), len(pos_layers)), np.nan)
        for key, info in pos_results.items():
            row_idx = pos_components.index(info["component"])
            col_idx = pos_layers.index(info["layer"])
            heatmap_data[row_idx, col_idx] = info["result"].bias_ratio

        # bias_ratio > 1 means closer to short, < 1 means closer to long
        # Use diverging colormap centered at 1
        vmax = max(2.0, np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 2.0)
        vmin = min(0.5, np.nanmin(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 0.5)

        im = ax.imshow(heatmap_data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Bias Ratio (d_long / d_short)\n>1 = closer to short, <1 = closer to long", fontsize=9)

        for i in range(len(pos_components)):
            for j in range(len(pos_layers)):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val - 1) > 0.3 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_xticks(range(len(pos_layers)))
        ax.set_xticklabels([f"L{layer}" for layer in pos_layers], fontsize=9)
        ax.set_yticks(range(len(pos_components)))
        ax.set_yticklabels(pos_components, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Component")
        ax.set_title(f"No-Horizon Bias Ratio - Position: {position}\n"
                     f"(Red = closer to short-horizon, Blue = closer to long-horizon)")

        plt.tight_layout()
        plt.savefig(output_dir / f"no_horizon_bias_heatmap_{position}.png", dpi=150)
        plt.close()

    # =========================================================================
    # 2. Summary bar chart: top targets by bias
    # =========================================================================
    sorted_results = sorted(
        result_info.items(),
        key=lambda x: abs(x[1]["result"].bias_ratio - 1),
        reverse=True
    )

    targets = [k for k, _ in sorted_results[:top_n]]
    bias_values = [result_info[t]["result"].bias_ratio for t in targets]
    dist_short = [result_info[t]["result"].dist_to_short for t in targets]
    dist_long = [result_info[t]["result"].dist_to_long for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(targets) * 0.25)))

    # Left: bias ratio
    ax = axes[0]
    colors = ["#F44336" if b > 1 else "#2196F3" for b in bias_values]
    ax.barh(range(len(targets)), bias_values, color=colors, alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="No bias")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_xlabel("Bias Ratio (d_long / d_short)")
    ax.set_title("No-Horizon Default Bias\n(Red = short bias, Blue = long bias)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    for i, (t, b) in enumerate(zip(targets, bias_values)):
        ax.text(b + 0.02, i, f"{b:.2f}", va="center", fontsize=7)

    # Right: stacked distances
    ax = axes[1]
    y_pos = range(len(targets))
    ax.barh(y_pos, dist_short, color="#F44336", alpha=0.7, label="Dist to short centroid")
    ax.barh(y_pos, dist_long, left=dist_short, color="#2196F3", alpha=0.7, label="Dist to long centroid")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_xlabel("Distance in PC Space")
    ax.set_title("Distance to Short vs Long Centroids")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "no_horizon_bias_summary.png", dpi=150)
    plt.close()

    # =========================================================================
    # 3. Per-target scatter plots (top targets)
    # =========================================================================
    scatter_targets = [k for k, _ in sorted_results[:12]]
    n_targets = len(scatter_targets)

    if n_targets > 0:
        n_cols = min(4, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, target_key in enumerate(scatter_targets):
            result = result_info[target_key]["result"]
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Plot horizon samples colored by time horizon
            horizon_proj = result.horizon_projected
            no_horizon_proj = result.no_horizon_projected
            horizon_values = result.horizon_values_months

            # Use log scale for coloring
            log_horizons = np.log10(horizon_values + 1)

            # Plot horizon samples (circles)
            ax.scatter(
                horizon_proj[:, 0],
                horizon_proj[:, 1],
                c=log_horizons,
                cmap=CMAP_GRADIENT,
                s=20,
                alpha=0.6,
                label="With horizon"
            )

            # Plot no-horizon samples (stars)
            ax.scatter(
                no_horizon_proj[:, 0],
                no_horizon_proj[:, 1],
                c="black",
                marker="*",
                s=80,
                alpha=0.9,
                label="No horizon",
                edgecolors="white",
                linewidths=0.5
            )

            # Plot centroids
            ax.scatter(
                result.short_horizon_centroid[0],
                result.short_horizon_centroid[1],
                c="red",
                marker="X",
                s=150,
                edgecolors="white",
                linewidths=2,
                label="Short centroid",
                zorder=10
            )
            ax.scatter(
                result.long_horizon_centroid[0],
                result.long_horizon_centroid[1],
                c="blue",
                marker="X",
                s=150,
                edgecolors="white",
                linewidths=2,
                label="Long centroid",
                zorder=10
            )
            ax.scatter(
                result.no_horizon_centroid[0],
                result.no_horizon_centroid[1],
                c="black",
                marker="D",
                s=100,
                edgecolors="white",
                linewidths=2,
                label="No-horizon centroid",
                zorder=10
            )

            # Draw arrow from no-horizon centroid to show bias direction
            # Arrow points toward the closer centroid
            if result.dist_to_short < result.dist_to_long:
                target_centroid = result.short_horizon_centroid
                arrow_color = "red"
            else:
                target_centroid = result.long_horizon_centroid
                arrow_color = "blue"

            ax.annotate(
                "",
                xy=(target_centroid[0], target_centroid[1]),
                xytext=(result.no_horizon_centroid[0], result.no_horizon_centroid[1]),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, alpha=0.7)
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            bias_str = "SHORT" if result.bias_ratio > 1 else "LONG"
            ax.set_title(f"{target_key}\nBias: {bias_str} ({result.bias_ratio:.2f})", fontsize=9)

            if idx == 0:
                ax.legend(fontsize=6, loc="upper right")

        # Hide unused axes
        for idx in range(len(scatter_targets), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "no_horizon_scatter.png", dpi=150)
        plt.close()

    # =========================================================================
    # 4. Distribution histogram
    # =========================================================================
    # Aggregate all per-sample distances
    all_dist_short = []
    all_dist_long = []
    for info in result_info.values():
        result = info["result"]
        all_dist_short.extend(result.no_horizon_dist_to_short.tolist())
        all_dist_long.extend(result.no_horizon_dist_to_long.tolist())

    if all_dist_short and all_dist_long:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: overlaid histograms
        ax = axes[0]
        ax.hist(all_dist_short, bins=30, alpha=0.6, color="#F44336", label="Dist to short centroid")
        ax.hist(all_dist_long, bins=30, alpha=0.6, color="#2196F3", label="Dist to long centroid")
        ax.axvline(np.mean(all_dist_short), color="#F44336", linestyle="--", linewidth=2, label=f"Mean short: {np.mean(all_dist_short):.2f}")
        ax.axvline(np.mean(all_dist_long), color="#2196F3", linestyle="--", linewidth=2, label=f"Mean long: {np.mean(all_dist_long):.2f}")
        ax.set_xlabel("Distance in PC Space")
        ax.set_ylabel("Count")
        ax.set_title("No-Horizon Sample Distances to Centroids")
        ax.legend(fontsize=8)

        # Right: difference histogram
        ax = axes[1]
        diff = np.array(all_dist_long) - np.array(all_dist_short)
        ax.hist(diff, bins=30, alpha=0.7, color="#9C27B0")
        ax.axvline(0, color="black", linestyle="--", linewidth=2, label="No bias")
        ax.axvline(np.mean(diff), color="orange", linestyle="-", linewidth=2, label=f"Mean: {np.mean(diff):.2f}")
        ax.set_xlabel("Distance Difference (d_long - d_short)")
        ax.set_ylabel("Count")
        ax.set_title("Bias Distribution\n(>0 = closer to short, <0 = closer to long)")
        ax.legend()

        # Add text annotation
        pct_short_bias = (diff > 0).sum() / len(diff) * 100
        ax.text(0.02, 0.98, f"{pct_short_bias:.1f}% closer to short",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "no_horizon_distribution.png", dpi=150)
        plt.close()

    # =========================================================================
    # 5. Layer trajectory of bias
    # =========================================================================
    # Group by component and position, plot bias across layers
    grouped = {}
    for key, info in result_info.items():
        group_key = f"{info['component']}_{info['position']}"
        if group_key not in grouped:
            grouped[group_key] = {}
        grouped[group_key][info['layer']] = info['result'].bias_ratio

    for group_key, layer_data in grouped.items():
        if len(layer_data) < 3:
            continue

        layers_sorted = sorted(layer_data.keys())
        bias_values_layer = [layer_data[layer] for layer in layers_sorted]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(layers_sorted, bias_values_layer, marker="o", linewidth=2, color="#9C27B0")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="No bias")
        ax.fill_between(layers_sorted, 1.0, bias_values_layer,
                       where=[b > 1 for b in bias_values_layer],
                       alpha=0.3, color="#F44336", label="Short bias")
        ax.fill_between(layers_sorted, 1.0, bias_values_layer,
                       where=[b <= 1 for b in bias_values_layer],
                       alpha=0.3, color="#2196F3", label="Long bias")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Bias Ratio (d_long / d_short)")
        ax.set_title(f"No-Horizon Bias Across Layers: {group_key}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_key = group_key.replace("/", "_").replace("\\", "_")
        plt.savefig(output_dir / f"no_horizon_trajectory_{safe_key}.png", dpi=150)
        plt.close()

    logger.info(f"Saved no-horizon projection plots to {output_dir}")


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
    cross_position_results: dict | None = None,
    continuous_time_results: dict | None = None,
    no_horizon_results: dict[str, NoHorizonProjectionResult] | None = None,
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
        ├── 09_targets/             # Per-target plots
        │   └── {base_key}/
        │       └── {pos_type}/
        │           ├── pca/
        │           ├── pca_2d/
        │           └── 3d/
        ├── 10_cross_position/      # Cross-position similarity (optional)
        ├── 11_continuous_time/     # Continuous time probe (optional)
        └── 12_no_horizon/          # No-horizon projection analysis (optional)

    Args:
        data: Activation data
        linear_probe_results: Linear probe results
        pca_results: PCA results
        embedding_results: Embedding results
        config: Pipeline config
        cross_position_results: Optional cross-position similarity results
        continuous_time_results: Optional continuous time probe results
        no_horizon_results: Optional no-horizon projection results
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

    # Optional: Cross-position similarity plots
    if cross_position_results:
        cross_pos_dir = output_dir / "10_cross_position"
        cross_pos_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating cross-position similarity plots...")
        plot_cross_position_similarity(cross_position_results, cross_pos_dir)
        gc.collect()

    # Optional: Continuous time probe plots
    if continuous_time_results:
        continuous_dir = output_dir / "11_continuous_time"
        continuous_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating continuous time probe plots...")
        plot_continuous_time_probe(continuous_time_results, continuous_dir)
        gc.collect()

    # Optional: No-horizon projection plots
    if no_horizon_results:
        no_horizon_dir = output_dir / "12_no_horizon"
        no_horizon_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating no-horizon projection plots...")
        plot_no_horizon_projection(no_horizon_results, no_horizon_dir)
        gc.collect()

    logger.info(f"All plots saved to {output_dir}")
