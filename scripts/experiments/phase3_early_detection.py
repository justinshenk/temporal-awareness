#!/usr/bin/env python3
"""
Phase 3, Experiment 5: Early Detection of Degradation via Linear Probes

Tests the hypothesis: "Linear probes can detect activation drift toward
degradation ≥5 steps before behavioral metrics show it."

This experiment applies methodology from Anthropic's alignment faking detection
(2024) to temporal degradation: can we use a linear probe trained to distinguish
fresh vs degraded activations to provide an early warning signal before
behavioral failure?

Method:
  1. For each model × dataset, run full repetition sequence: [1, 2, 3, 5, 8, 12, 16, 20]
  2. At each repetition, measure:
     a. Behavioral accuracy (actual MCQ accuracy on the task)
     b. Probe confidence (degradation probe's P(degraded) from Phase 2)
     c. Direction projection (projection onto degradation direction)
  3. Align behavioral and probe curves temporally
  4. Compute precursor gap:
     - Behavioral onset = first rep where accuracy drops >5% from baseline
     - Probe onset = first rep where probe confidence crosses 0.7
     - Precursor gap = behavioral_onset - probe_onset (in rep steps)
  5. Compute continuous early warning score:
     - Correlation between probe at rep N and accuracy drop at rep N+k
     - For various lookahead values k ∈ [1, 2, 3, 4, 5]
  6. Statistical significance via bootstrap confidence intervals

Key hypothesis:
  - If probe leads behavioral decline by ≥5 reps: we have actionable early warning
  - If lookahead correlation strong for k=3-5: precursor is predictive
  - Across models: early detection is a universal property of degradation

Related work:
  - Anthropic alignment faking detection (2024): Linear probes detect covert states
  - Miao et al. (2023): Backdoor detection via linear probes
  - Turner et al. (2023): Steering with activation addition
  - Li et al. (2023): Early warning signs in neural networks

Usage:
    # Quick validation (1 model, 1 dataset, small sample)
    python scripts/experiments/phase3_early_detection.py --quick

    # Single model
    python scripts/experiments/phase3_early_detection.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    # All models
    python scripts/experiments/phase3_early_detection.py \\
        --all-models --device cuda

    # Custom lookahead window
    python scripts/experiments/phase3_early_detection.py \\
        --all-models --lookahead-window 5

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
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
PROBE_DIR = PROJECT_ROOT / "results" / "phase2_probes"
DIRECTIONS_DIR = PROJECT_ROOT / "results" / "phase3_refusal_direction"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_early_detection"

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
class EarlyDetectionMetrics:
    """Metrics for early detection at a specific layer."""
    layer: int
    rep_counts: List[int]

    # Behavioral metrics
    accuracies: List[float]           # Accuracy at each rep
    accuracy_from_baseline: List[float]  # Accuracy relative to rep-1

    # Probe metrics
    probe_confidences: List[float]    # P(degraded) at each rep
    direction_projections: List[float]  # Projection onto degradation direction

    # Detection onset points
    behavioral_onset_rep: Optional[int] = None
    behavioral_onset_idx: Optional[int] = None
    probe_onset_rep: Optional[int] = None
    probe_onset_idx: Optional[int] = None

    # Precursor gap (in repetition steps)
    precursor_gap: Optional[float] = None

    # Lookahead correlations: corr[probe_at_rep_N] vs accuracy_drop_at_rep_N+k
    lookahead_correlations: Dict[int, float] = field(default_factory=dict)

    # Bootstrap CI on precursor gap
    precursor_gap_ci_lower: Optional[float] = None
    precursor_gap_ci_upper: Optional[float] = None

    elapsed_seconds: float = 0.0


@dataclass
class EarlyDetectionResult:
    """Complete early detection results for a model."""
    model: str
    dataset: str
    timestamp: str
    metrics: Dict[int, EarlyDetectionMetrics] = field(default_factory=dict)
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


def extract_answer(example: dict) -> str:
    """Extract ground truth answer from an example."""
    if "answer_idx" in example:
        return str(example["answer_idx"]).strip().upper()
    answer = str(example.get("answer", "")).strip()
    if len(answer) == 1:
        return answer.upper()
    if "options" in example and isinstance(example["options"], dict):
        for key, val in example["options"].items():
            if val.strip().lower() == answer.lower():
                return key.upper()
    return answer.upper()


# ---------------------------------------------------------------------------
# Behavioral evaluation
# ---------------------------------------------------------------------------
def evaluate_behavioral_accuracy(
    model: nn.Module,
    tokenizer,
    examples: List[dict],
    rep_count: int,
    device: str = "cuda",
    max_new_tokens: int = 5,
) -> Tuple[float, List[bool]]:
    """Evaluate model accuracy on examples at given repetition count.

    Returns:
        Tuple of (accuracy, list of per-example correct booleans).
    """
    correct = []

    for ex in examples:
        prompt = build_prompt(ex, rep_count)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract first letter as answer
        pred = ""
        for ch in response:
            if ch.isalpha():
                pred = ch.upper()
                break

        gt = extract_answer(ex)
        correct.append(pred == gt)

    accuracy = sum(correct) / len(correct) if correct else 0.0
    return accuracy, correct


# ---------------------------------------------------------------------------
# Probe loading and evaluation
# ---------------------------------------------------------------------------
def load_degradation_probe(
    model_key: str,
    layer: int,
    dataset_key: str = "medium_temporal",
) -> Optional[Tuple[np.ndarray, float]]:
    """Load pre-trained degradation probe from Phase 2.

    Returns:
        Tuple of (probe_weights, threshold) or None if not found.
        probe_weights shape: (d_model,)
    """
    probe_file = (
        PROBE_DIR / model_key / f"probe_{dataset_key}_layer{layer}.npy"
    )
    threshold_file = (
        PROBE_DIR / model_key / f"threshold_{dataset_key}_layer{layer}.npy"
    )

    if probe_file.exists() and threshold_file.exists():
        weights = np.load(probe_file).astype(np.float32)
        threshold = float(np.load(threshold_file))
        return weights, threshold

    return None


def compute_probe_confidence(
    activation: np.ndarray,
    probe_weights: np.ndarray,
) -> float:
    """Compute P(degraded) from probe logit.

    Args:
        activation: (d_model,) activation vector.
        probe_weights: (d_model,) probe weights.

    Returns:
        Probability in [0, 1] via sigmoid.
    """
    logit = float(np.dot(activation, probe_weights))
    # Sigmoid: 1 / (1 + exp(-x))
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob


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


def extract_activations_at_reps(
    extractor: ActivationExtractor,
    dataset: dict,
    layer: int,
    rep_counts: List[int],
    max_examples: int = 30,
) -> Dict[int, np.ndarray]:
    """Extract mean activations at multiple repetition counts.

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


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
def find_onset_rep(
    values: List[float],
    rep_counts: List[int],
    threshold: float,
    baseline: Optional[float] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Find first repetition where value crosses threshold.

    Args:
        values: List of metric values.
        rep_counts: List of repetition counts (sorted).
        threshold: Threshold for crossing.
        baseline: If provided, compute relative change from baseline.

    Returns:
        Tuple of (rep_where_onset, index_where_onset) or (None, None).
    """
    for i, val in enumerate(values):
        if baseline is not None:
            rel_change = val - baseline
            if rel_change > threshold:
                return rep_counts[i], i
        else:
            if val > threshold:
                return rep_counts[i], i

    return None, None


def compute_lookahead_correlation(
    probe_values: List[float],
    accuracy_values: List[float],
    lookahead_steps: int = 3,
) -> float:
    """Compute correlation between probe and future accuracy drop.

    For each rep N, compare probe_confidence[N] with accuracy_drop[N+k].

    Args:
        probe_values: Probe confidences at each rep.
        accuracy_values: Accuracies at each rep.
        lookahead_steps: How many steps ahead to look.

    Returns:
        Pearson correlation or NaN if insufficient data.
    """
    if len(probe_values) < lookahead_steps + 1:
        return np.nan

    # Accuracy drop: negative of change (so positive means degradation)
    accuracy_drop = [-( accuracy_values[i + lookahead_steps] - accuracy_values[i])
                     for i in range(len(accuracy_values) - lookahead_steps)]
    probe_early = probe_values[:-lookahead_steps]

    # Pearson correlation
    if len(accuracy_drop) < 2:
        return np.nan

    probe_arr = np.array(probe_early)
    drop_arr = np.array(accuracy_drop)

    # Handle constant values
    if probe_arr.std() < 1e-8 or drop_arr.std() < 1e-8:
        return np.nan

    corr = float(np.corrcoef(probe_arr, drop_arr)[0, 1])
    return corr if not np.isnan(corr) else np.nan


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval on mean.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(values) < 2:
        return np.nan, np.nan

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - ci
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2))

    return lower, upper


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_precursor_alignment(
    rep_counts: List[int],
    accuracies: List[float],
    probe_confidences: List[float],
    behavioral_onset_rep: Optional[int],
    probe_onset_rep: Optional[int],
    layer: int,
    output_path: Path,
) -> None:
    """Plot alignment of accuracy and probe confidence curves."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # Normalize probe confidence for visualization
    probe_normalized = probe_confidences
    if max(probe_normalized) > 1.0:
        probe_normalized = [x / max(probe_normalized) for x in probe_normalized]

    # Plot 1: Both curves
    ax1.plot(rep_counts, accuracies, "o-", linewidth=2.5, markersize=10,
            label="Behavioral Accuracy", color="steelblue")
    ax1.plot(rep_counts, probe_normalized, "s-", linewidth=2.5, markersize=10,
            label="Probe Confidence (normalized)", color="coral")

    # Mark onset points
    if behavioral_onset_rep is not None:
        idx = rep_counts.index(behavioral_onset_rep)
        ax1.axvline(x=behavioral_onset_rep, color="steelblue", linestyle="--",
                   linewidth=2, alpha=0.7, label=f"Behavioral onset (rep {behavioral_onset_rep})")

    if probe_onset_rep is not None:
        idx = rep_counts.index(probe_onset_rep)
        ax1.axvline(x=probe_onset_rep, color="coral", linestyle="--",
                   linewidth=2, alpha=0.7, label=f"Probe onset (rep {probe_onset_rep})")

    ax1.set_xlabel("Repetition Count", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_title(f"Layer {layer} - Behavioral vs Probe Alignment", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Probe confidence alone
    ax2.plot(rep_counts, probe_confidences, "s-", linewidth=2.5, markersize=10, color="coral")
    ax2.axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Threshold (0.7)")
    ax2.fill_between(rep_counts, probe_confidences, alpha=0.2, color="coral")

    if probe_onset_rep is not None:
        ax2.axvline(x=probe_onset_rep, color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax2.set_xlabel("Repetition Count", fontsize=12)
    ax2.set_ylabel("Probe Confidence P(degraded)", fontsize=12)
    ax2.set_title(f"Layer {layer} - Probe Confidence Curve", fontsize=13, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lookahead_correlation(
    lookahead_corrs: Dict[int, float],
    layer: int,
    output_path: Path,
) -> None:
    """Plot lookahead correlation for various k values."""
    if not HAS_MPL:
        return

    k_values = sorted(lookahead_corrs.keys())
    corrs = [lookahead_corrs[k] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot with hatch
    colors = ["green" if c > 0.4 else "gray" for c in corrs]
    ax.bar(k_values, corrs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for k, c in zip(k_values, corrs):
        if not np.isnan(c):
            ax.text(k, c + 0.02, f"{c:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Lookahead Steps (k)", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title(f"Layer {layer} - Early Warning: Probe[rep N] vs Accuracy Drop[rep N+k]",
                fontsize=13, fontweight="bold")
    ax.set_ylim([min(corrs) - 0.1, max(corrs) + 0.15])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze_early_detection(
    model_key: str,
    dataset_key: str,
    device: str,
    layers: List[int],
    lookahead_window: int = 5,
    quick: bool = False,
    backend: str = "pytorch",
) -> EarlyDetectionResult:
    """Analyze early detection for a model."""
    print(f"\n{'='*70}")
    print(f"Early Detection Analysis: {model_key} on {dataset_key}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load dataset
    print(f"\n[1/5] Loading dataset and models...")
    dataset = load_benchmark_dataset(
        dataset_key,
        max_examples=10 if quick else 30,
    )
    print(f"  Loaded {len(dataset['examples'])} examples")

    config = MODEL_CONFIGS[model_key]

    # Load HF model and tokenizer for behavioral evaluation
    print(f"  Loading model: {config['hf_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["hf_name"],
        device_map=device,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["hf_name"])
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    # Load activation extractor
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

    rep_counts = [1, 2, 3, 5, 8, 12, 16, 20]

    result = EarlyDetectionResult(
        model=model_key,
        dataset=dataset_key,
        timestamp=datetime.now().isoformat(),
        config=config,
    )

    # Evaluate behavioral accuracy at each rep
    print(f"\n[2/5] Evaluating behavioral accuracy...")
    behavioral_by_rep = {}
    examples_subset = dataset["examples"][:10 if quick else 30]

    for rep in rep_counts:
        print(f"  Rep {rep}...", end=" ")
        acc, _ = evaluate_behavioral_accuracy(
            model, tokenizer, examples_subset, rep, device=device
        )
        behavioral_by_rep[rep] = acc
        print(f"acc={acc:.3f}")

    # Extract activations at each rep
    print(f"\n[3/5] Extracting activations...")
    for layer in analyze_layers:
        print(f"  Layer {layer}...")
        layer_start = time.time()

        activations_by_rep = extract_activations_at_reps(
            extractor,
            dataset,
            layer,
            rep_counts,
            max_examples=10 if quick else 30,
        )

        if len(activations_by_rep) < 2:
            print(f"    SKIP: Insufficient activations")
            continue

        sorted_reps = sorted(activations_by_rep.keys())

        # Load probe and direction
        probe_data = load_degradation_probe(model_key, layer, dataset_key)
        direction = load_degradation_direction(model_key, layer, dataset_key)

        # Compute probe confidence and direction projection
        print(f"    Computing probe confidence...")
        probe_confidences = []
        direction_projections = []

        for rep in sorted_reps:
            act = activations_by_rep[rep]

            # Probe confidence
            if probe_data is not None:
                probe_weights, threshold = probe_data
                conf = compute_probe_confidence(act, probe_weights)
                probe_confidences.append(conf)
            else:
                probe_confidences.append(0.5)  # Default: unknown

            # Direction projection
            if direction is not None:
                proj = float(np.dot(act, direction))
                direction_projections.append(proj)
            else:
                direction_projections.append(0.0)

        # Behavioral accuracy at reps
        accuracies = [behavioral_by_rep.get(rep, 0.0) for rep in sorted_reps]
        accuracy_from_baseline = [acc - accuracies[0] for acc in accuracies]

        # Find onset points
        baseline_acc = accuracies[0]
        behavioral_onset_rep, behavioral_onset_idx = find_onset_rep(
            accuracy_from_baseline, sorted_reps, threshold=-0.05  # Drop >5%
        )
        probe_onset_rep, probe_onset_idx = find_onset_rep(
            probe_confidences, sorted_reps, threshold=0.7
        )

        # Precursor gap
        precursor_gap = None
        if behavioral_onset_idx is not None and probe_onset_idx is not None:
            precursor_gap = float(behavioral_onset_idx - probe_onset_idx)

        # Lookahead correlation
        lookahead_corrs = {}
        for k in range(1, min(lookahead_window + 1, len(sorted_reps))):
            corr = compute_lookahead_correlation(
                probe_confidences, accuracies, lookahead_steps=k
            )
            if not np.isnan(corr):
                lookahead_corrs[k] = corr

        # Bootstrap CI on precursor gap (if we have multiple runs, simulate via resampling)
        precursor_gap_ci_lower = None
        precursor_gap_ci_upper = None
        if precursor_gap is not None:
            # Create multiple "runs" by resampling accuracy values
            precursor_gaps = []
            for _ in range(100):
                acc_resampled = np.random.normal(
                    loc=np.mean(accuracies),
                    scale=np.std(accuracies) + 1e-8,
                    size=len(accuracies)
                )
                acc_drop = [acc_resampled[i] - acc_resampled[0]
                           for i in range(len(acc_resampled))]
                _, onset_idx = find_onset_rep(acc_drop, sorted_reps, threshold=-0.05)
                if onset_idx is not None and probe_onset_idx is not None:
                    precursor_gaps.append(float(onset_idx - probe_onset_idx))

            if precursor_gaps:
                precursor_gap_ci_lower, precursor_gap_ci_upper = bootstrap_ci(
                    precursor_gaps, n_bootstrap=500
                )

        # Store metrics
        metrics = EarlyDetectionMetrics(
            layer=layer,
            rep_counts=sorted_reps,
            accuracies=accuracies,
            accuracy_from_baseline=accuracy_from_baseline,
            probe_confidences=probe_confidences,
            direction_projections=direction_projections,
            behavioral_onset_rep=behavioral_onset_rep,
            behavioral_onset_idx=behavioral_onset_idx,
            probe_onset_rep=probe_onset_rep,
            probe_onset_idx=probe_onset_idx,
            precursor_gap=precursor_gap,
            lookahead_correlations=lookahead_corrs,
            precursor_gap_ci_lower=precursor_gap_ci_lower,
            precursor_gap_ci_upper=precursor_gap_ci_upper,
            elapsed_seconds=time.time() - layer_start,
        )

        result.metrics[layer] = metrics

        print(f"    Behavioral onset: rep {behavioral_onset_rep}")
        print(f"    Probe onset: rep {probe_onset_rep}")
        print(f"    Precursor gap: {precursor_gap} reps")
        print(f"    Lookahead corrs: {lookahead_corrs}")

    result.elapsed_seconds = time.time() - start_time
    return result


def save_results(result: EarlyDetectionResult, output_dir: Path) -> None:
    """Save early detection results to JSON and visualizations."""
    model_dir = output_dir / result.model
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/5] Saving results to {model_dir}...")

    # JSON summary
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
            "accuracies": metrics.accuracies,
            "accuracy_from_baseline": metrics.accuracy_from_baseline,
            "probe_confidences": metrics.probe_confidences,
            "direction_projections": metrics.direction_projections,
            "behavioral_onset_rep": metrics.behavioral_onset_rep,
            "probe_onset_rep": metrics.probe_onset_rep,
            "precursor_gap": metrics.precursor_gap,
            "precursor_gap_ci": {
                "lower": metrics.precursor_gap_ci_lower,
                "upper": metrics.precursor_gap_ci_upper,
            },
            "lookahead_correlations": metrics.lookahead_correlations,
            "elapsed_seconds": metrics.elapsed_seconds,
        }
        summary["layers"][str(layer)] = layer_summary

    results_file = model_dir / "early_detection_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {results_file}")

    # Visualizations
    print(f"\n[5/5] Generating visualizations...")
    for layer, metrics in result.metrics.items():
        print(f"  Layer {layer}...")

        plot_precursor_alignment(
            metrics.rep_counts,
            metrics.accuracies,
            metrics.probe_confidences,
            metrics.behavioral_onset_rep,
            metrics.probe_onset_rep,
            layer,
            model_dir / f"precursor_alignment_layer{layer}.png",
        )

        if metrics.lookahead_correlations:
            plot_lookahead_correlation(
                metrics.lookahead_correlations,
                layer,
                model_dir / f"lookahead_correlation_layer{layer}.png",
            )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3, Experiment 5: Early Detection of Degradation",
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
        "--lookahead-window",
        type=int,
        default=5,
        help="Max lookahead steps for correlation (default: 5)",
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

    if not args.model and not args.all_models:
        parser.error("Must specify --model or --all-models")

    if args.model and args.all_models:
        parser.error("Cannot specify both --model and --all-models")

    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]

    if args.quick:
        models = models[:1]
        datasets = args.datasets[:1]
    else:
        datasets = args.datasets

    print(f"Phase 3, Experiment 5: Early Detection of Degradation")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Device: {args.device}")
    print(f"Lookahead window: {args.lookahead_window}")

    if args.wandb_project and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=f"early_detection_{'_'.join(models[:2])}",
            config={
                "models": models,
                "datasets": datasets,
                "device": args.device,
                "lookahead_window": args.lookahead_window,
                "quick": args.quick,
            },
        )

    # Run analysis
    all_results = []
    for model_key in models:
        for dataset_key in datasets:
            result = analyze_early_detection(
                model_key=model_key,
                dataset_key=dataset_key,
                device=args.device,
                layers=MODEL_CONFIGS[model_key]["layers"],
                lookahead_window=args.lookahead_window,
                quick=args.quick,
                backend=args.backend,
            )
            all_results.append(result)
            save_results(result, args.output_dir)

    print(f"\n[Complete]")
    print(f"Results saved to: {args.output_dir}")
    print(f"Analyzed {len(all_results)} model-dataset combinations")

    if HAS_WANDB and args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
