#!/usr/bin/env python3
"""
Phase 3, Experiment 4: Trajectory Geometry in Activation Space

Analyzes the GEOMETRY of activation trajectories as models degrade under
repetitive tasks. Rather than treating degradation as binary (fresh vs degraded),
this tracks the continuous geometric path activations take through representation
space as repetition count increases.

Method:
  1. Extract activations at EVERY repetition count: [1, 2, 3, 5, 8, 12, 16, 20]
  2. For each layer, compute:
     a. PCA projection: Project all repetition-level activations onto top-3 PCs
     b. Velocity: L2 distance between consecutive mean activations
     c. Acceleration: Change in velocity
     d. Curvature: Angle between consecutive displacement vectors
     e. Cosine drift from rep-1: How far activations have drifted from baseline
     f. Critical transition detection: Peak velocity identifies phase transition
     g. Projection onto degradation direction: If pre-extracted, project trajectory
  3. Compare trajectory shapes across models — are they universal?

Key hypothesis:
  - Activation trajectories follow a characteristic two-phase pattern:
    * Phase 1: Slow drift (low velocity, high cosine similarity)
    * Phase 2: Rapid transition (peak velocity, sharp angle changes)
  - The critical transition point (peak velocity) occurs N steps before
    behavioral failure and can be predicted from early activations

Related work:
  - "Phase transitions in the dynamics of neural networks" (2025)
  - Anthropic alignment faking detection (2024): Early warning signals

Usage:
    # Quick validation (1 model, 3 layers, small dataset)
    python scripts/experiments/phase3_trajectory_geometry.py --quick

    # Single model
    python scripts/experiments/phase3_trajectory_geometry.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    # All models
    python scripts/experiments/phase3_trajectory_geometry.py \\
        --all-models --device cuda

Author: Adrian Sadik
Date: 2026-04-10
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
DIRECTIONS_DIR = PROJECT_ROOT / "results" / "phase3_refusal_direction"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_trajectory"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult


# ---------------------------------------------------------------------------
# Model configs (must match other Phase 3 scripts)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
    },
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
        "quick_layers": [8, 16, 28],
        "n_layers": 36,
        "d_model": 4096,
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "layers": [0, 6, 12, 18, 24, 30, 36, 42, 47],
        "quick_layers": [12, 24, 36],
        "n_layers": 48,
        "d_model": 4096,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 27],
        "quick_layers": [8, 16, 24],
        "n_layers": 28,
        "d_model": 3584,
    },
    "Ouro-2.6B": {
        "hf_name": "ByteDance/Ouro-2.6B",
        "layers": [0, 3, 6, 9, 12, 15, 18, 21, 23],
        "quick_layers": [6, 12, 18],
        "n_layers": 24,
        "d_model": 2048,
    },
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASET_FILES = {
    "medium_temporal": "medium_stakes_tram_arithmetic.json",
    "medium_code": "medium_stakes_mbpp_code.json",
    "high": "high_stakes_medqa_temporal.json",
}

FILLER_TEMPLATES = [
    "Continue with the next item. ",
    "Proceed to the following task. ",
    "Moving on to another similar request. ",
    "Here is another one to process. ",
    "Next task in the sequence. ",
    "Please handle this next request. ",
    "Another similar task follows. ",
    "Continuing the sequence now. ",
    "Here is the next one. ",
    "Process this additional item. ",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryMetrics:
    """Trajectory metrics for a single layer."""
    layer: int
    rep_counts: list[int]

    # Geometric properties (per rep count)
    velocities: list[float]           # L2 distance between consecutive activations
    accelerations: list[float]        # Change in velocity
    curvatures: list[float]           # Angle between displacement vectors
    cosine_drifts: list[float]        # Cosine sim to rep-1 activation

    # PCA projections
    pca_projections: Optional[np.ndarray] = None  # (n_reps, 3)
    pca_directions: Optional[np.ndarray] = None   # (3, d_model) — top 3 PCs
    pca_explained_var: Optional[list[float]] = None

    # Critical transition
    peak_velocity_rep: Optional[int] = None
    peak_velocity_value: Optional[float] = None

    # Degradation direction projection
    degradation_projection: Optional[np.ndarray] = None  # (n_reps,)

    # Timing
    elapsed_seconds: float = 0.0


@dataclass
class TrajectoryResult:
    """Complete trajectory analysis results for a model."""
    model: str
    dataset: str
    timestamp: str
    metrics: Dict[int, TrajectoryMetrics] = field(default_factory=dict)
    config: Optional[dict] = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Dataset loading and prompt building
# ---------------------------------------------------------------------------
def load_benchmark_dataset(stakes_key: str, max_examples: int = 50) -> dict:
    """Load a processed benchmark dataset."""
    fname = DATASET_FILES[stakes_key]
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if max_examples and len(data["examples"]) > max_examples:
        random.seed(42)
        data["examples"] = random.sample(data["examples"], max_examples)

    return data


def build_prompt(example: dict, rep_count: int) -> str:
    """Build a prompt with repetitive prefix for a given example."""
    question = example["question"]
    if "options" in example and example["options"]:
        options = example["options"]
        if isinstance(options, dict):
            opts_str = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        else:
            opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
        question = f"{question}\n\n{opts_str}\n\nAnswer:"

    if rep_count <= 1:
        return question

    prefix_parts = []
    for i in range(rep_count - 1):
        filler = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        prefix_parts.append(filler)

    return "".join(prefix_parts) + question


# ---------------------------------------------------------------------------
# Trajectory geometry computation
# ---------------------------------------------------------------------------
def extract_activations_at_reps(
    extractor: ActivationExtractor,
    dataset: dict,
    layer: int,
    rep_counts: list[int],
    max_examples: int = 30,
) -> Dict[int, np.ndarray]:
    """Extract mean activations at multiple repetition counts.

    Args:
        extractor: ActivationExtractor instance.
        dataset: Benchmark dataset dict.
        layer: Layer index.
        rep_counts: List of repetition counts to extract at.
        max_examples: Max examples to use.

    Returns:
        Dict mapping rep_count -> mean activation (d_model,)
    """
    examples = dataset["examples"][:max_examples]
    key = f"resid_post.layer{layer}"

    activations_by_rep = {}

    for rep in rep_counts:
        prompts = [build_prompt(ex, rep) for ex in examples]
        result = extractor.extract(prompts, return_tokens=False)

        if key not in result.activations:
            print(f"    WARNING: Layer {layer} not captured at rep {rep}")
            continue

        acts = result.numpy(key)  # (n_examples, d_model)
        mean_act = acts.mean(axis=0).astype(np.float32)
        activations_by_rep[rep] = mean_act

    return activations_by_rep


def compute_trajectory_metrics(
    activations_by_rep: Dict[int, np.ndarray],
    rep_counts: list[int],
) -> Tuple[list[float], list[float], list[float], list[float]]:
    """Compute velocity, acceleration, curvature, and cosine drift.

    Returns:
        (velocities, accelerations, curvatures, cosine_drifts)
    """
    sorted_reps = sorted(activations_by_rep.keys())
    acts = [activations_by_rep[rep] for rep in sorted_reps]

    velocities = []
    displacements = []

    # Compute velocities (L2 distance between consecutive activations)
    for i in range(len(acts) - 1):
        disp = acts[i + 1] - acts[i]
        vel = np.linalg.norm(disp)
        velocities.append(float(vel))
        displacements.append(disp)

    # Accelerations: change in velocity
    accelerations = []
    for i in range(len(velocities) - 1):
        acc = velocities[i + 1] - velocities[i]
        accelerations.append(float(acc))

    # Curvatures: angle between consecutive displacement vectors
    curvatures = []
    for i in range(len(displacements) - 1):
        d1 = displacements[i]
        d2 = displacements[i + 1]

        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)

        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_angle = np.dot(d1, d2) / (norm1 * norm2)
            # Clamp to [-1, 1] for numerical stability
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = float(np.arccos(cos_angle))  # radians
        else:
            angle = 0.0
        curvatures.append(angle)

    # Cosine drift from rep-1: similarity to baseline
    baseline = acts[0]
    baseline_norm = np.linalg.norm(baseline)
    cosine_drifts = []

    for act in acts:
        if baseline_norm > 1e-8:
            cos_sim = float(np.dot(baseline, act) / (baseline_norm * np.linalg.norm(act)))
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
        else:
            cos_sim = 1.0
        cosine_drifts.append(cos_sim)

    return velocities, accelerations, curvatures, cosine_drifts


def apply_pca_projection(
    activations_by_rep: Dict[int, np.ndarray],
    rep_counts: list[int],
    n_components: int = 3,
) -> Tuple[np.ndarray, np.ndarray, list[float]]:
    """Apply PCA to activations across repetitions.

    Returns:
        (projections, pca_components, explained_variance)
        where projections is (n_reps, n_components)
    """
    sorted_reps = sorted(activations_by_rep.keys())
    acts = np.array([activations_by_rep[rep] for rep in sorted_reps])

    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(acts)

    return (
        projections.astype(np.float32),
        pca.components_.astype(np.float32),
        pca.explained_variance_ratio_.tolist()
    )


def find_critical_transition(velocities: list[float]) -> Optional[Tuple[int, float]]:
    """Find repetition index where velocity peaks (critical transition).

    Returns:
        (rep_index, velocity_value) or None if no clear peak.
    """
    if not velocities:
        return None

    peak_idx = int(np.argmax(velocities))
    peak_vel = float(velocities[peak_idx])

    return peak_idx, peak_vel


def project_onto_direction(
    activations_by_rep: Dict[int, np.ndarray],
    rep_counts: list[int],
    direction: np.ndarray,
) -> np.ndarray:
    """Project mean activations onto a direction vector.

    Args:
        activations_by_rep: Dict of rep -> activation.
        rep_counts: Sorted rep counts.
        direction: Unit direction vector.

    Returns:
        Array of projection values (n_reps,)
    """
    sorted_reps = sorted(activations_by_rep.keys())
    projections = []

    # Ensure direction is unit norm
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 1e-8:
        direction = direction / dir_norm

    for rep in sorted_reps:
        act = activations_by_rep[rep]
        proj = float(np.dot(act, direction))
        projections.append(proj)

    return np.array(projections, dtype=np.float32)


def load_degradation_direction(
    model_key: str,
    layer: int,
    dataset_key: str = "medium_temporal",
) -> Optional[np.ndarray]:
    """Load pre-extracted degradation direction if available."""
    direction_file = (
        DIRECTIONS_DIR / model_key / "directions" /
        f"degradation_{dataset_key}_layer{layer}.npy"
    )

    if direction_file.exists():
        direction = np.load(direction_file).astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        return direction

    return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_pca_trajectory(
    projections: np.ndarray,
    rep_counts: list[int],
    layer: int,
    output_path: Path,
) -> None:
    """Plot 2D PCA trajectory colored by repetition count.

    Args:
        projections: (n_reps, 3) PCA projections.
        rep_counts: Repetition count labels.
        layer: Layer number.
        output_path: Output PNG path.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        c=rep_counts,
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidth=1.5,
    )

    # Draw line connecting points in order
    ax.plot(projections[:, 0], projections[:, 1], "k-", alpha=0.3, linewidth=1)

    # Annotate with rep counts
    for i, rep in enumerate(rep_counts):
        ax.annotate(str(rep), (projections[i, 0], projections[i, 1]),
                   fontsize=10, ha="center", va="center")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Repetition Count", fontsize=12)

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title(f"Activation Trajectory - Layer {layer} (PCA)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_velocity_profile(
    rep_counts: list[int],
    velocities: list[float],
    accelerations: list[float],
    layer: int,
    output_path: Path,
) -> None:
    """Plot velocity and acceleration over repetition counts."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Velocity
    ax1.plot(rep_counts[:-1], velocities, "o-", linewidth=2, markersize=8, color="steelblue")
    ax1.fill_between(rep_counts[:-1], velocities, alpha=0.3, color="steelblue")
    ax1.set_xlabel("Repetition Count", fontsize=11)
    ax1.set_ylabel("Velocity (L2 distance)", fontsize=11)
    ax1.set_title(f"Layer {layer} - Activation Velocity vs Repetition", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Acceleration
    ax2.plot(rep_counts[:-2], accelerations, "s-", linewidth=2, markersize=8, color="coral")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.fill_between(rep_counts[:-2], accelerations, 0, alpha=0.3, color="coral")
    ax2.set_xlabel("Repetition Count", fontsize=11)
    ax2.set_ylabel("Acceleration (Δ velocity)", fontsize=11)
    ax2.set_title(f"Layer {layer} - Acceleration vs Repetition", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drift_profile(
    rep_counts: list[int],
    cosine_drifts: list[float],
    curvatures: list[float],
    layer: int,
    output_path: Path,
) -> None:
    """Plot cosine drift and curvature over repetitions."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Cosine drift
    ax1.plot(rep_counts, cosine_drifts, "o-", linewidth=2, markersize=8, color="green")
    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Perfect similarity")
    ax1.fill_between(rep_counts, cosine_drifts, 1.0, alpha=0.3, color="green")
    ax1.set_xlabel("Repetition Count", fontsize=11)
    ax1.set_ylabel("Cosine Similarity to Rep-1", fontsize=11)
    ax1.set_title(f"Layer {layer} - Activation Drift from Baseline", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Curvature
    ax2.plot(rep_counts[1:-1], curvatures, "s-", linewidth=2, markersize=8, color="purple")
    ax2.fill_between(rep_counts[1:-1], curvatures, alpha=0.3, color="purple")
    ax2.set_xlabel("Repetition Count", fontsize=11)
    ax2.set_ylabel("Curvature (angle, radians)", fontsize=11)
    ax2.set_title(f"Layer {layer} - Path Curvature vs Repetition", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_degradation_projection(
    rep_counts: list[int],
    projection: np.ndarray,
    layer: int,
    output_path: Path,
) -> None:
    """Plot 1D projection onto degradation direction."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rep_counts, projection, "o-", linewidth=2, markersize=10, color="darkred")
    ax.fill_between(rep_counts, projection, alpha=0.2, color="darkred")
    ax.set_xlabel("Repetition Count", fontsize=12)
    ax.set_ylabel("Projection onto Degradation Direction", fontsize=12)
    ax.set_title(f"Layer {layer} - 1D Degradation Trajectory", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze_trajectory(
    model_key: str,
    dataset_key: str,
    device: str,
    layers: list[int],
    quick: bool = False,
    backend: str = "pytorch",
) -> TrajectoryResult:
    """Analyze activation trajectory for a model."""
    print(f"\n{'='*70}")
    print(f"Analyzing trajectory for {model_key} on {dataset_key}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load dataset and extractor
    print(f"\n[1/5] Loading dataset and initializing extractor...")
    dataset = load_benchmark_dataset(
        dataset_key,
        max_examples=10 if quick else 30,
    )
    print(f"  Loaded {len(dataset['examples'])} examples")

    config = MODEL_CONFIGS[model_key]

    # Set layers to analyze
    analyze_layers = config["quick_layers"] if quick else config["layers"]

    # Resolve backend choice
    use_tl = {"pytorch": False, "transformer_lens": True, "auto": None}[backend]

    extraction_config = ExtractionConfig(
        layers=analyze_layers,
        module_types=["resid_post"],
        positions="last",
        stream_to="cpu",
        batch_size=2,
        model_dtype="float16",
        dtype="float32",
        max_seq_len=2048,
        use_transformer_lens=use_tl,
    )
    extractor = ActivationExtractor(
        model=config["hf_name"],
        config=extraction_config,
        device=device,
    )
    print(f"  Analyzing {len(analyze_layers)} layers: {analyze_layers}")

    # Repetition counts to sample
    rep_counts = [1, 2, 3, 5, 8, 12, 16, 20]

    result = TrajectoryResult(
        model=model_key,
        dataset=dataset_key,
        timestamp=datetime.now().isoformat(),
        config=config,
    )

    print(f"\n[2/5] Extracting activations at {len(rep_counts)} repetition counts...")

    for layer in analyze_layers:
        print(f"  Layer {layer}...")
        layer_start = time.time()

        # Extract activations
        activations_by_rep = extract_activations_at_reps(
            extractor,
            dataset,
            layer,
            rep_counts,
            max_examples=10 if quick else 30,
        )

        if len(activations_by_rep) < 2:
            print(f"    SKIP: Insufficient activations captured")
            continue

        sorted_reps = sorted(activations_by_rep.keys())

        # Compute geometric metrics
        print(f"    Computing trajectory metrics...")
        velocities, accelerations, curvatures, cosine_drifts = compute_trajectory_metrics(
            activations_by_rep, sorted_reps
        )

        # PCA projection
        print(f"    Applying PCA projection...")
        pca_proj, pca_dirs, pca_var = apply_pca_projection(
            activations_by_rep, sorted_reps, n_components=3
        )

        # Critical transition
        critical_idx, critical_vel = find_critical_transition(velocities)

        # Degradation direction projection
        deg_dir = load_degradation_direction(model_key, layer, dataset_key)
        deg_proj = None
        if deg_dir is not None:
            deg_proj = project_onto_direction(activations_by_rep, sorted_reps, deg_dir)

        # Store metrics
        metrics = TrajectoryMetrics(
            layer=layer,
            rep_counts=sorted_reps,
            velocities=velocities,
            accelerations=accelerations,
            curvatures=curvatures,
            cosine_drifts=cosine_drifts,
            pca_projections=pca_proj,
            pca_directions=pca_dirs,
            pca_explained_var=pca_var,
            peak_velocity_rep=critical_idx,
            peak_velocity_value=critical_vel,
            degradation_projection=deg_proj,
            elapsed_seconds=time.time() - layer_start,
        )

        result.metrics[layer] = metrics

        print(f"    Peak velocity at rep {critical_idx}: {critical_vel:.4f}")
        print(f"    PCA explained var: {pca_var}")
        print(f"    Time: {metrics.elapsed_seconds:.1f}s")

    result.elapsed_seconds = time.time() - start_time
    return result


def save_results(result: TrajectoryResult, output_dir: Path) -> None:
    """Save trajectory results to JSON and visualizations."""
    model_dir = output_dir / result.model
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    print(f"\n[3/5] Saving results to {model_dir}...")

    summary = {
        "model": result.model,
        "dataset": result.dataset,
        "timestamp": result.timestamp,
        "elapsed_seconds": result.elapsed_seconds,
        "layers": {},
    }

    for layer, metrics in result.metrics.items():
        layer_summary = {
            "layer": layer,
            "rep_counts": metrics.rep_counts,
            "velocities": metrics.velocities,
            "accelerations": metrics.accelerations,
            "curvatures": metrics.curvatures,
            "cosine_drifts": metrics.cosine_drifts,
            "pca_explained_variance": metrics.pca_explained_var,
            "peak_velocity_rep": metrics.peak_velocity_rep,
            "peak_velocity_value": metrics.peak_velocity_value,
            "elapsed_seconds": metrics.elapsed_seconds,
        }

        if metrics.degradation_projection is not None:
            layer_summary["degradation_projection"] = (
                metrics.degradation_projection.tolist()
            )

        summary["layers"][str(layer)] = layer_summary

    results_file = model_dir / "trajectory_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {results_file}")

    # Generate visualizations
    print(f"\n[4/5] Generating visualizations...")
    for layer, metrics in result.metrics.items():
        print(f"  Layer {layer}...")

        # PCA trajectory
        plot_pca_trajectory(
            metrics.pca_projections,
            metrics.rep_counts,
            layer,
            model_dir / f"pca_trajectory_layer{layer}.png",
        )

        # Velocity profile
        plot_velocity_profile(
            metrics.rep_counts,
            metrics.velocities,
            metrics.accelerations,
            layer,
            model_dir / f"velocity_profile_layer{layer}.png",
        )

        # Drift profile
        plot_drift_profile(
            metrics.rep_counts,
            metrics.cosine_drifts,
            metrics.curvatures,
            layer,
            model_dir / f"drift_profile_layer{layer}.png",
        )

        # Degradation projection
        if metrics.degradation_projection is not None:
            plot_degradation_projection(
                metrics.rep_counts,
                metrics.degradation_projection,
                layer,
                model_dir / f"degradation_projection_layer{layer}.png",
            )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3, Experiment 4: Trajectory Geometry Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific model to analyze",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Analyze all models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 model, 3 layers, small dataset",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["medium_temporal"],
        choices=list(DATASET_FILES.keys()),
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project for logging (optional)",
    )
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.all_models:
        parser.error("Must specify --model or --all-models")

    if args.model and args.all_models:
        parser.error("Cannot specify both --model and --all-models")

    # Determine models to analyze
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]

    if args.quick:
        models = models[:1]
        datasets = args.datasets[:1]
    else:
        datasets = args.datasets

    print(f"Phase 3, Experiment 4: Trajectory Geometry Analysis")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Device: {args.device}")

    # Optional W&B initialization
    if args.wandb_project and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=f"trajectory_{'_'.join(models[:2])}",
            config={
                "models": models,
                "datasets": datasets,
                "device": args.device,
                "quick": args.quick,
            },
        )

    # Run analysis
    all_results = []
    for model_key in models:
        for dataset_key in datasets:
            result = analyze_trajectory(
                model_key=model_key,
                dataset_key=dataset_key,
                device=args.device,
                layers=MODEL_CONFIGS[model_key]["layers"],
                quick=args.quick,
                backend=args.backend,
            )
            all_results.append(result)
            save_results(result, args.output_dir)

    print(f"\n[5/5] Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Analyzed {len(all_results)} model-dataset combinations")

    if HAS_WANDB and args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
