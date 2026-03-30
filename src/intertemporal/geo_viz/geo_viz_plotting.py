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
# Summary Dashboard and Heatmaps
# =============================================================================


def plot_summary_dashboard(
    linear_probe_results: dict[str, "LinearProbeResult"],
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot summary dashboard heatmaps showing R² scores across layers and positions.

    Creates 4 heatmaps (one per component) with:
    - Rows = positions
    - Columns = layers
    - Cell color = R² score (separation/decodability)
    """
    # Parse all targets to extract layer, component, position
    target_info = {}
    for key in linear_probe_results.keys():
        # Parse key like "L19_mlp_out_Presponse"
        parts = key.split("_P")
        if len(parts) != 2:
            continue
        base = parts[0]  # "L19_mlp_out"
        position = parts[1]  # "response"

        # Parse layer and component from base
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

        # Build data matrix
        data = np.full((len(positions), len(layers)), np.nan)
        for key, info in target_info.items():
            if info["component"] == component:
                row_idx = positions.index(info["position"]) if info["position"] in positions else -1
                col_idx = layers.index(info["layer"]) if info["layer"] in layers else -1
                if row_idx >= 0 and col_idx >= 0:
                    data[row_idx, col_idx] = info["r2"]

        # Plot heatmap
        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("R² Score", fontsize=10)

        # Add text annotations
        for i in range(len(positions)):
            for j in range(len(layers)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=8)

        # Labels
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

    # Combined dashboard with all components
    n_components = len(components)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components + 2, max(5, len(positions) * 0.5)))
    if n_components == 1:
        axes = [axes]

    for idx, component in enumerate(components):
        ax = axes[idx]

        # Build data matrix
        data = np.full((len(positions), len(layers)), np.nan)
        for key, info in target_info.items():
            if info["component"] == component:
                row_idx = positions.index(info["position"]) if info["position"] in positions else -1
                col_idx = layers.index(info["layer"]) if info["layer"] in layers else -1
                if row_idx >= 0 and col_idx >= 0:
                    data[row_idx, col_idx] = info["r2"]

        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")

        # Annotations
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

    # Add single colorbar
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
    """Plot PC1 trajectory across layers for each sample.

    Shows how the time-correlated PC1 projection evolves across layers.
    Each line = one sample, colored by time horizon.
    Handles positions with different sample counts by using the minimum common count.
    """
    # Group targets by position, component - need consistent layers
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

    # Get time horizons for coloring (full set)
    horizons_months = np.array([get_time_horizon_months(s) for s in data.samples])
    horizons_years = horizons_months / 12.0
    log_horizons = np.log10(horizons_years + 0.1)

    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for key_id, layer_data in target_info.items():
        layers = sorted(layer_data.keys())
        if len(layers) < 2:
            continue

        # Find minimum sample count across all layers for this trajectory
        n_samples = min(layer_data[layer]["n_samples"] for layer in layers)
        if n_samples < 2:
            logger.warning(f"  Skipping trajectory {key_id}: too few samples")
            continue

        # Use subset of horizons matching the sample count
        log_horizons_subset = log_horizons[:n_samples]
        color_vals = (log_horizons_subset - log_horizons_subset.min()) / (log_horizons_subset.max() - log_horizons_subset.min() + 1e-6)

        # Extract PC1 projection for best correlated PC at each layer
        trajectories = np.zeros((n_samples, len(layers)))
        for i, layer in enumerate(layers):
            pca_result = layer_data[layer]["pca"]
            best_pc_idx = pca_result.pc_correlations[0][0]
            # Use only the first n_samples
            trajectories[:, i] = pca_result.transformed[:n_samples, best_pc_idx]

        # Normalize trajectories
        for i in range(len(layers)):
            col = trajectories[:, i]
            std = col.std()
            if std > 1e-10:
                trajectories[:, i] = (col - col.mean()) / std
            else:
                trajectories[:, i] = 0

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each sample's trajectory
        for sample_idx in range(min(n_samples, 200)):  # Limit samples for clarity
            color = plt.cm.plasma(color_vals[sample_idx])
            ax.plot(range(len(layers)), trajectories[sample_idx],
                   color=color, alpha=0.3, linewidth=0.8)

        # Add colorbar
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
        plt.savefig(traj_dir / f"trajectory_{key_id}.png", dpi=150)
        plt.close()

    logger.info(f"Saved trajectory plots to {traj_dir}")


def plot_component_decomposition(
    linear_probe_results: dict[str, "LinearProbeResult"],
    pca_results: dict[str, "PCAResult"],
    schemes: list["ColoringScheme"],
    output_dir: Path,
):
    """Plot 2x2 component decomposition showing resid_pre, attn_out, mlp_out, resid_post.

    For key layer/position combinations, shows how information flows through components.
    """
    # Parse targets
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

    decomp_dir = output_dir / "component_decomposition"
    decomp_dir.mkdir(parents=True, exist_ok=True)

    components_order = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

    # Use first non-categorical scheme
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
            best_corr = pca_result.pc_correlations[0][1]

            r2_info = comp_data[component]["r2"]
            r2_val = r2_info.r2_mean if r2_info else 0

            _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False, n_samples=X_pca.shape[0])
            ax.set_xlabel("PC0")
            ax.set_ylabel(f"PC{best_pc_idx}")
            ax.set_title(f"{component}\nR²={r2_val:.3f}, corr={best_corr:.3f}")

        # Hide unused axes
        for idx in range(len(available_components), n_rows * n_cols):
            row, col = idx // 2, idx % 2
            axes[row, col].axis("off")

        plt.suptitle(f"Component Decomposition: {layer_pos_key}", fontsize=12)
        plt.tight_layout()
        plt.savefig(decomp_dir / f"decomp_{layer_pos_key}.png", dpi=150)
        plt.close()

    logger.info(f"Saved component decomposition plots to {decomp_dir}")


def plot_direction_alignment(
    pca_results: dict[str, "PCAResult"],
    output_dir: Path,
):
    """Plot direction alignment heatmaps showing cosine similarity between PC1 directions.

    Shows how the time-encoding direction aligns across layers and positions.
    """
    # Parse targets and extract PC1 directions
    directions = {}
    for key, pca_result in pca_results.items():
        parts = key.split("_P")
        if len(parts) != 2:
            continue

        # Get direction of best correlated PC
        best_pc_idx = pca_result.pc_correlations[0][0]
        direction = pca_result.components[best_pc_idx]  # d_model dimensional
        direction = direction / (np.linalg.norm(direction) + 1e-10)  # Normalize
        directions[key] = direction

    if len(directions) < 2:
        logger.warning("Not enough targets for direction alignment")
        return

    keys = sorted(directions.keys())
    n = len(keys)

    # Compute cosine similarity matrix
    similarity = np.zeros((n, n))
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            similarity[i, j] = abs(np.dot(directions[key_i], directions[key_j]))

    # Plot full heatmap
    fig, ax = plt.subplots(figsize=(max(10, n * 0.3), max(8, n * 0.25)))
    im = ax.imshow(similarity, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    # Labels (abbreviated)
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

    # Create per-position alignment plots
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
        sim_pos = np.zeros((n_pos, n_pos))
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
    """Plot decision boundary accuracy across layers.

    Shows how well a linear classifier can separate short-term vs long-term choices.
    """
    if not data.choices:
        logger.warning("No choice data for decision boundary plot")
        return

    # Get binary labels
    chose_long = np.array([1 if c.chose_long_term else 0 for c in data.choices])

    # Skip if all same choice
    if len(set(chose_long)) < 2:
        logger.warning("All samples have same choice, skipping decision boundary")
        return

    # Parse targets by layer
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

        # Use predictions to compute choice accuracy
        # Higher prediction = longer horizon = should choose long-term
        predictions = result.predictions
        n_samples = len(predictions)
        # Handle different sample counts (some positions don't exist in all prompts)
        target_chose_long = chose_long[:n_samples] if n_samples < len(chose_long) else chose_long
        median_pred = np.median(predictions)
        predicted_choice = (predictions > median_pred).astype(int)
        accuracy = (predicted_choice == target_chose_long).mean()

        layer_key = f"L{layer}_{component}_P{position}"
        layer_accuracy[layer_key] = {
            "layer": layer,
            "component": component,
            "position": position,
            "accuracy": accuracy,
        }

    # Plot by layer (aggregated across components and positions)
    layers_set = sorted(set(info["layer"] for info in layer_accuracy.values()))

    # Group by position type
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
    """Plot scree plots showing variance explained by each PC.

    Shows which PCs capture the most variance at each layer/position.
    """
    scree_dir = output_dir / "scree"
    scree_dir.mkdir(parents=True, exist_ok=True)

    # Individual scree plots for key targets
    key_targets = []
    for key in pca_results.keys():
        if "response" in key or "dest" in key:
            key_targets.append(key)

    # Also include some source targets
    for key in pca_results.keys():
        if key not in key_targets:
            key_targets.append(key)
        if len(key_targets) >= 12:
            break

    for target_key in key_targets:
        pca_result = pca_results[target_key]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scree plot
        ax = axes[0]
        n_show = min(20, len(pca_result.explained_variance))
        ax.bar(range(n_show), pca_result.explained_variance[:n_show], color="#2196F3", alpha=0.7)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Variance Explained")

        # Cumulative variance
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

        # Clean filename
        safe_name = target_key.replace("/", "_")
        plt.savefig(scree_dir / f"scree_{safe_name}.png", dpi=150)
        plt.close()

    # Summary scree comparing key targets
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for idx, target_key in enumerate(key_targets[:8]):
        pca_result = pca_results[target_key]
        cumsum = np.cumsum(pca_result.explained_variance[:20])
        # Abbreviated label
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

    logger.info(f"Saved scree plots to {scree_dir}")


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

        # Handle different sample counts (some positions don't exist in all prompts)
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
    n_samples: int | None = None,
) -> None:
    """Scatter plot with a coloring scheme.

    Args:
        ax: Matplotlib axes
        x: X coordinates
        y: Y coordinates
        scheme: Coloring scheme with values for all samples
        add_colorbar: Whether to add colorbar
        n_samples: Number of samples to use (slice scheme.values if different)
    """
    # Slice scheme values to match number of samples if needed
    values = scheme.values
    if n_samples is not None and n_samples < len(values):
        values = values[:n_samples]

    if scheme.is_categorical:
        # Use time scale colors for time_scale scheme, otherwise default colors
        if scheme.name == "time_scale":
            colors = TIME_SCALE_COLORS
        else:
            colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]  # Extend if needed

        categories = scheme.categories or [str(i) for i in range(len(colors))]
        unique_vals = sorted(set(values.astype(int)))

        for val in unique_vals:
            color = colors[val % len(colors)]
            label = categories[val] if val < len(categories) else str(val)
            mask = values == val
            if mask.sum() > 0:
                ax.scatter(x[mask], y[mask], c=color, s=15, alpha=0.7, label=label)
        ax.legend(markerscale=2, fontsize=8)
    else:
        vmin = values.min()
        vmax = values.max()

        # Handle edge cases for log normalization
        if scheme.use_log:
            vmin = max(0.01, vmin)  # Ensure positive for log
            vmax = max(vmin * 1.1, vmax)  # Ensure vmax > vmin
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            # Ensure vmax > vmin for linear norm
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
            _scatter_with_scheme(ax, coords[:, 0], coords[:, 1], scheme, n_samples=coords.shape[0])
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
            _scatter_with_scheme(ax, coords[:, 0], coords[:, 1], scheme, add_colorbar=False, n_samples=coords.shape[0])
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


def _is_absolute_position(pos: str) -> bool:
    """Check if position is an absolute index (e.g., 86, 145)."""
    return pos.isdigit()


def _parse_target_key(target_key: str) -> tuple[str, str, str]:
    """Parse target key into (base_key, position, position_type).

    E.g., "L19_mlp_out_Pdest" -> ("L19_mlp_out", "dest", "dst")
          "L19_mlp_out_Pshort_term_time" -> ("L19_mlp_out", "short_term_time", "src")
          "L19_mlp_out_P86" -> ("L19_mlp_out", "86", "absolute")

    Returns:
        (base_key, position_name, position_type) where position_type is "dst", "src", or "absolute"
    """
    # Named positions that are in source (prompt)
    source_positions = {"time_horizon", "short_term_time", "short_term_reward",
                       "long_term_time", "long_term_reward", "source"}
    # Named positions that are in dest (response)
    dest_positions = {"response", "dest"}

    # Find the _P marker
    p_idx = target_key.rfind("_P")
    if p_idx == -1:
        return target_key, "unknown", "unknown"

    base_key = target_key[:p_idx]
    position = target_key[p_idx + 2:]  # Skip "_P"

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
    """Group target keys by their base (layer+component) and position type.

    Returns: {base_key: {"dst": {pos: key}, "src": {pos: key}, "absolute": {pos: key}}}
    """
    groups = {}
    for key in target_keys:
        base, pos, pos_type = _parse_target_key(key)
        if base not in groups:
            groups[base] = {"dst": {}, "src": {}, "absolute": {}}
        if pos_type in groups[base]:
            groups[base][pos_type][pos] = key
    return groups


def plot_cross_layer_summary(
    linear_probe_results: dict[str, LinearProbeResult],
    pca_results: dict[str, PCAResult],
    schemes: list[ColoringScheme],
    output_dir: Path,
):
    """Generate cross-layer comparison plots showing all layers/positions together."""
    # Parse all targets and group by position type (dst/src/absolute)
    targets_by_pos = {"dest": [], "source": [], "absolute": []}
    for key in linear_probe_results.keys():
        base, pos, pos_type = _parse_target_key(key)
        # Map to legacy names for this function
        if pos_type == "dst":
            targets_by_pos["dest"].append((key, base))
        elif pos_type == "src":
            targets_by_pos["source"].append((key, base))
        elif pos_type == "absolute":
            targets_by_pos["absolute"].append((key, base))

    # Sort by layer number
    def extract_layer(item):
        match = re.search(r"L(\d+)", item[1])
        return int(match.group(1)) if match else 0

    for pos in targets_by_pos:
        targets_by_pos[pos].sort(key=extract_layer)

    # Cross-layer R² comparison
    n_pos_types = sum(1 for pos in targets_by_pos.values() if pos)
    fig, axes = plt.subplots(1, max(2, n_pos_types), figsize=(6 * max(2, n_pos_types), 6))
    if n_pos_types <= 2:
        axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes]

    pos_colors = {"dest": "#4CAF50", "source": "#F44336", "absolute": "#2196F3"}
    ax_idx = 0
    for pos, targets in targets_by_pos.items():
        if not targets:
            continue
        ax = axes[ax_idx] if ax_idx < len(axes) else axes[-1]
        ax_idx += 1

        labels = [t[1] for t in targets]  # base keys
        r2_scores = [linear_probe_results[t[0]].r2_mean for t in targets]
        colors = [pos_colors.get(pos, "#666666")] * len(targets)

        ax.barh(range(len(labels)), r2_scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("R² Score")
        ax.set_title(f"{pos.upper()} Positions\nAcross Layers")
        ax.set_xlim(-0.1, 1.1)
        ax.axvline(0, color="black", linewidth=0.5)

        for i, r2 in enumerate(r2_scores):
            ax.text(max(r2, 0) + 0.02, i, f"{r2:.3f}", va="center", fontsize=8)

    # Hide unused axes
    for idx in range(ax_idx, len(axes)):
        axes[idx].axis("off")

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

                _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False, n_samples=X_pca.shape[0])
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

                    _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False, n_samples=X_pca.shape[0])
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

        # Same for absolute positions if available
        absolute_targets = targets_by_pos.get("absolute", [])
        if absolute_targets:
            n_targets = len(absolute_targets)
            n_cols = min(4, n_targets)
            n_rows = (n_targets + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
            axes = np.atleast_2d(axes)

            for idx, (target_key, base_key) in enumerate(absolute_targets):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                if target_key in pca_results:
                    pca_result = pca_results[target_key]
                    X_pca = pca_result.transformed
                    best_pc_idx = pca_result.pc_correlations[0][0]
                    best_corr = pca_result.pc_correlations[0][1]

                    _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, add_colorbar=False, n_samples=X_pca.shape[0])
                    ax.set_xlabel("PC0")
                    ax.set_ylabel(f"PC{best_pc_idx}")
                    ax.set_title(f"{base_key}\nr={best_corr:.2f}")

            for idx in range(len(absolute_targets), n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].axis("off")

            plt.suptitle(f"PCA Across Layers (ABSOLUTE POSITIONS) - {scheme.label}", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "cross_layer_pca_absolute.png", dpi=150)
            plt.close()


def _get_pos_color(pos: str) -> str:
    """Get color for a position name (green for dest/response, red for source, blue for absolute)."""
    dest_positions = {"response", "dest"}
    if pos in dest_positions:
        return "#4CAF50"  # Green for dest
    elif _is_absolute_position(pos):
        # Absolute positions: use a gradient from blue (low idx) to purple (high idx)
        idx = int(pos)
        if idx >= 140:  # Near response positions
            return "#9C27B0"  # Purple for high indices (likely dest area)
        else:
            return "#2196F3"  # Blue for lower indices (likely source area)
    else:
        return "#F44336"  # Red for named source positions


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
    colors = [_get_pos_color(p) for p in positions]
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
            color = _get_pos_color(pos)
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
                _scatter_with_scheme(ax, X_pca[:, 0], X_pca[:, best_pc_idx], scheme, n_samples=X_pca.shape[0])
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
        ├── dashboard_*.png           # Summary heatmaps by component
        ├── linear_probe_summary.png
        ├── cross_layer_*.png
        ├── decision_boundary.png
        ├── direction_alignment*.png
        ├── scree_comparison.png
        ├── trajectories/             # PC projection across layers
        ├── scree/                    # Individual scree plots
        ├── component_decomposition/  # 2x2 component breakdowns
        └── targets/
            └── L19_mlp_out/
                ├── position_comparison.png
                ├── comparison_*.png
                ├── dst/              # Aggregated dest position plots
                ├── src/              # Aggregated source position plots
                └── by_position/      # Per named position (when available)
                    ├── short_term_time/
                    ├── long_term_reward/
                    └── response/
    """
    output_dir = config.output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = get_coloring_schemes(data)
    logger.info(f"Using {len(schemes)} coloring schemes: {[s.name for s in schemes]}")

    # Priority 1: Summary dashboard heatmaps
    logger.info("Generating summary dashboard heatmaps...")
    plot_summary_dashboard(linear_probe_results, pca_results, output_dir)

    # Priority 2: Trajectory plots
    logger.info("Generating trajectory plots...")
    plot_trajectory(data, pca_results, output_dir)

    # Priority 3: Component decomposition 2x2 grids
    logger.info("Generating component decomposition plots...")
    plot_component_decomposition(linear_probe_results, pca_results, schemes, output_dir)

    # Priority 4: Direction alignment heatmaps
    logger.info("Generating direction alignment plots...")
    plot_direction_alignment(pca_results, output_dir)

    # Priority 5: Decision boundary accuracy
    logger.info("Generating decision boundary plot...")
    plot_decision_boundary(data, linear_probe_results, output_dir)

    # Priority 6: Scree plots
    logger.info("Generating scree plots...")
    plot_scree(pca_results, output_dir)

    # Original summary plots
    logger.info("Generating linear probe summary...")
    plot_linear_probe_summary(data, linear_probe_results, output_dir)

    # Cross-layer comparison plots
    logger.info("Generating cross-layer plots...")
    plot_cross_layer_summary(linear_probe_results, pca_results, schemes, output_dir)

    # Group targets by base (layer+component) and position type
    target_groups = _group_targets_by_base(list(pca_results.keys()))

    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    for base_key, pos_type_targets in target_groups.items():
        logger.info(f"  Processing {base_key}...")
        base_dir = targets_dir / base_key
        base_dir.mkdir(parents=True, exist_ok=True)

        # Flatten for comparison plots (combine dst and src)
        all_positions = {}
        for pos_type, positions in pos_type_targets.items():
            all_positions.update(positions)

        # Generate comparison plots at base level
        if len(all_positions) > 1:
            plot_position_comparison(
                base_key, all_positions, linear_probe_results,
                pca_results, schemes, base_dir
            )

        # Generate plots for each position type (dst/, src/, absolute/)
        for pos_type in ["dst", "src", "absolute"]:
            positions = pos_type_targets.get(pos_type, {})
            if not positions:
                continue

            # Create dst/, src/, or absolute/ directory
            type_dir = base_dir / pos_type
            type_dir.mkdir(parents=True, exist_ok=True)

            # If only one position of this type (e.g., just "dest"), put plots directly in type_dir
            if len(positions) == 1:
                pos, target_key = list(positions.items())[0]
                logger.info(f"    Plotting {pos_type}/{pos}...")

                plot_target_pca(target_key, pca_results[target_key], schemes, type_dir)
                if target_key in embedding_results:
                    plot_target_embeddings(target_key, embedding_results[target_key], schemes, type_dir)
                plot_target_3d(target_key, pca_results[target_key], schemes, type_dir)
            else:
                # Multiple positions of this type - use by_position/
                by_position_dir = base_dir / "by_position"
                by_position_dir.mkdir(parents=True, exist_ok=True)

                for pos, target_key in positions.items():
                    logger.info(f"    Plotting by_position/{pos}...")
                    pos_dir = by_position_dir / pos
                    pos_dir.mkdir(parents=True, exist_ok=True)

                    plot_target_pca(target_key, pca_results[target_key], schemes, pos_dir)
                    if target_key in embedding_results:
                        plot_target_embeddings(target_key, embedding_results[target_key], schemes, pos_dir)
                    plot_target_3d(target_key, pca_results[target_key], schemes, pos_dir)

    logger.info(f"All plots saved to {output_dir}")
