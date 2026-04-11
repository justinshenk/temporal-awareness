#!/usr/bin/env python3
"""
Phase 2/3: Cross-Domain & Cross-Stake Activation Analysis

Replaces the TransformerLens-based pipeline with the new activation_api.
Extracts activations from the final 4 models across all dataset tiers,
trains linear probes, and compares degradation patterns across domains
and stake levels.

Pipeline:
  1. Load benchmark datasets (low/medium_temporal/medium_code/high stakes)
  2. For each model × dataset × repetition count:
     a. Build repetitive prompts
     b. Extract activations via activation_api (replaces TransformerLens)
     c. Train linear probes on residual stream activations
  3. Cross-domain analysis:
     a. Train probe on domain A, test on domain B
     b. Measure transfer: does degradation generalize across domains?
  4. Cross-stake analysis:
     a. Train probe on low stakes, test on high stakes
     b. Measure: does the model's "patience signal" differ by stakes?
  5. Precursor gap detection:
     a. Compare behavioral onset (from Phase 1) vs activation onset
     b. Key result: can probes detect degradation N reps before behavior?

Target models (decided 2026-03-30):
  1. Llama-3.1-8B-Instruct — degradation case
  2. Qwen3-8B — warm-up comparison
  3. Qwen3-30B-A3B — MoE, stable
  4. DeepSeek-R1-Distill-Qwen-7B — reasoning-distilled

Usage:
    # Quick validation (1 model, few examples, 1 layer)
    python scripts/experiments/phase2_cross_domain.py --quick

    # Single model, all datasets
    python scripts/experiments/phase2_cross_domain.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    # Full run (all 4 models) with W&B logging
    python scripts/experiments/phase2_cross_domain.py \\
        --all-models --device cuda --wandb-project patience-degradation

    # Sherlock SLURM
    sbatch scripts/experiments/submit_phase2.sh

Author: Adrian Sadik
Date: 2026-04-05
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
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
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
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2_cross_domain"

# Add project root to path for activation_api imports
sys.path.insert(0, str(PROJECT_ROOT))

from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult


# ---------------------------------------------------------------------------
# Model configs — final 4 for NeurIPS
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
        "is_instruct": True,
        "max_new_tokens": 512,
        "sae_source": "goodfire",  # Goodfire SAEs available
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
        "quick_layers": [8, 16, 28],
        "n_layers": 36,
        "d_model": 4096,
        "is_instruct": True,
        "max_new_tokens": 512,
        "sae_source": None,
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "layers": [0, 6, 12, 18, 24, 30, 36, 42, 47],
        "quick_layers": [12, 24, 36],
        "n_layers": 48,
        "d_model": 4096,
        "is_instruct": True,
        "max_new_tokens": 512,
        "sae_source": None,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 27],
        "quick_layers": [8, 16, 24],
        "n_layers": 28,
        "d_model": 3584,
        "is_instruct": True,
        "max_new_tokens": 512,
        "is_reasoning": True,
        "sae_source": "airi",  # AIRI-Institute SAEs
    },
}

# Repetition counts — matches Phase 1 for comparability
REPETITION_COUNTS = [1, 3, 5, 8, 12, 16, 20, 30, 50]
QUICK_REP_COUNTS = [1, 5, 20]

# Datasets
DATASET_FILES = {
    "low": "low_stakes_tram_ordering.json",
    "medium_temporal": "medium_stakes_tram_arithmetic.json",
    "medium_code": "medium_stakes_mbpp_code.json",
    "high": "high_stakes_medqa_temporal.json",
}

# Filler templates for repetitive sequences
FILLER_TEMPLATES = [
    "Continue with the next item. ",
    "Proceed to the following task. ",
    "Moving on to another similar request. ",
    "Here is another one to process. ",
    "Next task in the sequence. ",
    "Please handle the following as well. ",
    "Another item requiring attention. ",
    "Continuing the sequence of tasks. ",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of a single linear probe evaluation."""
    layer: int
    accuracy: float
    f1: float
    n_train: int
    n_test: int
    train_domain: str    # which dataset was used for training
    test_domain: str     # which dataset was used for testing
    repetition_count: int
    model_name: str

    def is_cross_domain(self) -> bool:
        return self.train_domain != self.test_domain


@dataclass
class LayerProbeProfile:
    """Probe accuracy across all layers for one condition."""
    model_name: str
    dataset: str
    repetition_count: int
    layer_accuracies: dict  # layer -> accuracy
    layer_f1s: dict         # layer -> f1


@dataclass
class CrossDomainResult:
    """Full cross-domain comparison for one model."""
    model_name: str
    in_domain_probes: list      # ProbeResult for same-domain
    cross_domain_probes: list   # ProbeResult for transfer
    cross_stake_probes: list    # ProbeResult for stake transfer
    precursor_gaps: dict        # dataset -> gap in reps


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(stakes_key: str, max_examples: Optional[int] = None) -> dict:
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

    print(f"  Loaded {stakes_key}: {len(data['examples'])} examples")
    return data


def build_prompt(example: dict, rep_count: int, model_config: dict) -> str:
    """Build a prompt with repetitive prefix for a given example.

    Args:
        example: Dataset example with 'question' and 'options'.
        rep_count: Number of task repetitions to simulate.
        model_config: Model config dict (for chat template info).

    Returns:
        Full prompt string ready for tokenization.
    """
    # Format the MCQ question
    question = example["question"]
    if "options" in example and example["options"]:
        options = example["options"]
        if isinstance(options, dict):
            opts_str = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        else:
            opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
        question = f"{question}\n\n{opts_str}\n\nAnswer:"

    # Add repetitive prefix
    if rep_count <= 1:
        return question

    prefix_parts = []
    for i in range(rep_count - 1):
        filler = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        prefix_parts.append(filler)

    return "".join(prefix_parts) + question


# ---------------------------------------------------------------------------
# Activation extraction (using new API)
# ---------------------------------------------------------------------------

def extract_activations_for_condition(
    extractor: ActivationExtractor,
    dataset: dict,
    rep_count: int,
    model_config: dict,
    max_examples: int = 50,
    positions: str = "last",
) -> ActivationResult:
    """Extract activations for one dataset at one repetition count.

    This is the core function that replaces the old TransformerLens-based
    extraction in patience_degradation.py.

    Args:
        extractor: Pre-initialized ActivationExtractor.
        dataset: Loaded dataset dict with 'examples'.
        rep_count: Repetition count for prompt construction.
        model_config: Model config dict.
        max_examples: Cap on examples per condition.

    Returns:
        ActivationResult with activations keyed by layer.
    """
    examples = dataset["examples"][:max_examples]

    # Build prompts
    prompts = [build_prompt(ex, rep_count, model_config) for ex in examples]

    # Extract answer-index labels (letter codes A/B/C/D)
    # Different datasets store this differently:
    #   - TRAM arithmetic/ordering: "answer" is already a letter ("A", "B", "C", "D")
    #   - MedQA: "answer" is the full text, "answer_idx" has the letter
    #   - Legacy AG News: "answer" is the category text, need to map via options
    labels = []
    for ex in examples:
        if "answer_idx" in ex:
            # MedQA style: answer_idx has the letter code
            labels.append(ex["answer_idx"])
        elif ex.get("answer", "") in {"A", "B", "C", "D"}:
            # TRAM style: answer is already a letter
            labels.append(ex["answer"])
        elif "options" in ex and isinstance(ex["options"], dict):
            # AG News style: answer is text, find matching option letter
            answer_text = ex.get("answer", "")
            matched = False
            for letter, text in ex["options"].items():
                if text == answer_text:
                    labels.append(letter)
                    matched = True
                    break
            if not matched:
                labels.append(ex.get("category", "unknown"))
        else:
            labels.append(ex.get("answer", "unknown"))

    categories = [ex.get("category", "") for ex in examples]

    print(f"    Extracting: {len(prompts)} prompts, rep={rep_count}, "
          f"avg_len={np.mean([len(p) for p in prompts]):.0f} chars")

    # Extract via the API
    result = extractor.extract(prompts, return_tokens=True)

    # Attach labels and categories as metadata
    result.metadata["labels"] = labels
    result.metadata["categories"] = categories
    result.metadata["repetition_count"] = rep_count
    result.metadata["dataset"] = dataset.get("dataset", "unknown")
    result.metadata["stake_level"] = dataset.get("stake_level", "unknown")

    return result


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, "LogisticRegression"]:
    """Train a logistic regression probe and evaluate.

    Args:
        X_train: Training activations (n_samples, d_model).
        y_train: Training labels.
        X_test: Test activations.
        y_test: Test labels.

    Returns:
        (accuracy, f1, trained_model)
    """
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return acc, f1, clf


def make_binary_labels(labels: list[str], dataset_key: str) -> np.ndarray:
    """Convert answer labels to numeric for probe training.

    Maps A/B/C/D letter codes to 0/1/2/3. For non-letter labels
    (e.g., category text), maps unique values to sequential integers.

    The probe tests whether activations encode the correct answer class.
    """
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    # Check if all labels are letter codes
    all_letters = all(
        isinstance(l, str) and l.upper() in label_map for l in labels
    )

    if all_letters:
        return np.array([label_map[l.upper()] for l in labels])

    # Fallback: map unique string values to sequential integers
    unique_vals = sorted(set(str(l) for l in labels))
    val_to_int = {v: i for i, v in enumerate(unique_vals)}
    return np.array([val_to_int[str(l)] for l in labels])


# ---------------------------------------------------------------------------
# Cross-domain analysis
# ---------------------------------------------------------------------------

def run_cross_domain_probes(
    activations_by_dataset: dict[str, ActivationResult],
    layer: int,
    model_name: str,
    rep_count: int,
) -> list[ProbeResult]:
    """Train probes on each domain and test on all others.

    Args:
        activations_by_dataset: {dataset_key: ActivationResult}
        layer: Which layer to probe.
        model_name: For logging.
        rep_count: Current repetition count.

    Returns:
        List of ProbeResult for all train/test domain combinations.
    """
    results = []

    for train_key, train_result in activations_by_dataset.items():
        # Get training activations and labels
        module_key = f"resid_post.layer{layer}"
        if module_key not in train_result.activations:
            continue

        X_train = train_result.numpy(module_key)
        y_train = make_binary_labels(
            train_result.metadata.get("labels", []),
            train_key,
        )

        if len(np.unique(y_train)) < 2:
            continue  # Need at least 2 classes

        # Split train data: 80/20 for in-domain test
        n = len(y_train)
        split = int(0.8 * n)
        indices = np.random.RandomState(42).permutation(n)
        train_idx, test_idx = indices[:split], indices[split:]

        # In-domain probe
        acc, f1, clf = train_probe(
            X_train[train_idx], y_train[train_idx],
            X_train[test_idx], y_train[test_idx],
        )
        results.append(ProbeResult(
            layer=layer, accuracy=acc, f1=f1,
            n_train=len(train_idx), n_test=len(test_idx),
            train_domain=train_key, test_domain=train_key,
            repetition_count=rep_count, model_name=model_name,
        ))

        # Cross-domain probes: train on this domain, test on others
        for test_key, test_result in activations_by_dataset.items():
            if test_key == train_key:
                continue
            if module_key not in test_result.activations:
                continue

            X_test = test_result.numpy(module_key)
            y_test = make_binary_labels(
                test_result.metadata.get("labels", []),
                test_key,
            )

            if len(np.unique(y_test)) < 2:
                continue

            # Use full training data, test on full other domain
            try:
                acc_cross, f1_cross, _ = train_probe(
                    X_train[train_idx], y_train[train_idx],
                    X_test, y_test,
                )
                results.append(ProbeResult(
                    layer=layer, accuracy=acc_cross, f1=f1_cross,
                    n_train=len(train_idx), n_test=len(X_test),
                    train_domain=train_key, test_domain=test_key,
                    repetition_count=rep_count, model_name=model_name,
                ))
            except Exception as e:
                print(f"      Cross-domain {train_key}->{test_key} failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Precursor gap detection
# ---------------------------------------------------------------------------

def compute_activation_onset(
    probe_accuracies_by_rep: dict[int, float],
    baseline_threshold: float = 0.05,
) -> Optional[int]:
    """Find the repetition count where probe accuracy first drops > threshold
    from rep=1 baseline.

    Args:
        probe_accuracies_by_rep: {rep_count: probe_accuracy}
        baseline_threshold: How much accuracy drop to consider "onset".

    Returns:
        Repetition count of activation onset, or None if no onset detected.
    """
    if 1 not in probe_accuracies_by_rep:
        return None

    baseline = probe_accuracies_by_rep[1]

    for rep in sorted(probe_accuracies_by_rep.keys()):
        if rep == 1:
            continue
        if baseline - probe_accuracies_by_rep[rep] > baseline_threshold:
            return rep

    return None  # No onset detected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_profiles(
    profiles: list[LayerProbeProfile],
    output_path: Path,
    title: str = "Probe Accuracy by Layer",
):
    """Plot probe accuracy across layers for multiple conditions."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for profile in profiles:
        layers = sorted(profile.layer_accuracies.keys())
        accs = [profile.layer_accuracies[l] for l in layers]
        label = f"{profile.dataset} rep={profile.repetition_count}"
        ax.plot(layers, accs, marker="o", label=label, linewidth=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_cross_domain_heatmap(
    probe_results: list[ProbeResult],
    layer: int,
    output_path: Path,
    title: str = "Cross-Domain Probe Transfer",
):
    """Plot a heatmap of probe transfer between domains."""
    if not HAS_MPL:
        return

    # Build transfer matrix
    domains = sorted(set(r.train_domain for r in probe_results))
    matrix = np.zeros((len(domains), len(domains)))

    for r in probe_results:
        if r.layer != layer:
            continue
        i = domains.index(r.train_domain)
        j = domains.index(r.test_domain)
        matrix[i, j] = r.accuracy

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(domains)))
    ax.set_yticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_yticklabels(domains)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Train Domain")
    ax.set_title(f"{title} (Layer {layer})")

    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(domains)):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if matrix[i, j] > 0.5 else "white")

    fig.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {output_path}")


def plot_precursor_gap(
    behavioral_onsets: dict[str, int],
    activation_onsets: dict[str, int],
    output_path: Path,
    model_name: str,
):
    """Plot behavioral vs activation onset to visualize precursor gap."""
    if not HAS_MPL:
        return

    datasets = sorted(set(behavioral_onsets.keys()) & set(activation_onsets.keys()))
    if not datasets:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = range(len(datasets))
    b_vals = [behavioral_onsets.get(d, 0) for d in datasets]
    a_vals = [activation_onsets.get(d, 0) for d in datasets]

    width = 0.35
    ax.bar([xi - width/2 for xi in x], b_vals, width, label="Behavioral onset", color="#e74c3c")
    ax.bar([xi + width/2 for xi in x], a_vals, width, label="Activation onset", color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Repetition Count at Onset")
    ax.set_title(f"Precursor Gap: {model_name}")
    ax.legend()

    # Annotate gaps
    for i, d in enumerate(datasets):
        gap = b_vals[i] - a_vals[i]
        if gap > 0:
            ax.annotate(f"gap={gap}", (i, max(b_vals[i], a_vals[i]) + 1),
                        ha="center", fontsize=9, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved precursor gap plot: {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_phase2_experiment(
    model_key: str,
    datasets: list[str],
    rep_counts: list[int],
    layers: list[int],
    device: str = "cuda",
    max_examples: int = 50,
    wandb_project: Optional[str] = None,
    save_activations: bool = False,
    output_dir: Optional[str] = None,
):
    """Run the full Phase 2 cross-domain experiment for one model.

    Args:
        model_key: Key into MODEL_CONFIGS.
        datasets: List of dataset keys (e.g., ["low", "medium_temporal", "high"]).
        rep_counts: Repetition counts to test.
        layers: Layer indices to extract from.
        device: Device for model inference.
        max_examples: Examples per dataset per condition.
        wandb_project: W&B project name (None to skip).
        save_activations: Whether to save raw activations to disk.
        output_dir: Directory for results.
    """
    model_config = MODEL_CONFIGS[model_key]
    output_dir = Path(output_dir or RESULTS_DIR / model_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Phase 2: {model_key}")
    print(f"  Layers: {layers}")
    print(f"  Datasets: {datasets}")
    print(f"  Rep counts: {rep_counts}")
    print(f"  Max examples: {max_examples}")
    print(f"{'='*70}\n")

    # Initialize W&B
    if wandb_project and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"phase2_{model_key}_{datetime.now():%Y%m%d_%H%M}",
            config={
                "model": model_key,
                "hf_name": model_config["hf_name"],
                "layers": layers,
                "datasets": datasets,
                "rep_counts": rep_counts,
                "max_examples": max_examples,
                "phase": "2_cross_domain",
            },
        )

    # ── Step 1: Initialize extractor with the new API ─────────────────
    print("Step 1: Initializing ActivationExtractor...")

    extraction_config = ExtractionConfig(
        layers=layers,
        module_types=["resid_post"],  # residual stream is the primary target
        positions="last",             # last token position (for MCQ answer)
        stream_to="cpu",              # stream activations to CPU to save GPU RAM
        batch_size=2,                 # conservative for 8B models
        model_dtype="float16",        # load model weights in fp16 to fit in VRAM
        dtype="float32",              # upcast activations to fp32 for probe training
        max_seq_len=2048,
        use_transformer_lens=False,   # use raw HuggingFace hooks (Phase 2 goal)
    )

    extractor = ActivationExtractor(
        model=model_config["hf_name"],
        config=extraction_config,
        device=device,
    )

    # ── Step 2: Extract activations per dataset × repetition ──────────
    print("\nStep 2: Extracting activations...")

    # Structure: {dataset_key: {rep_count: ActivationResult}}
    all_activations: dict[str, dict[int, ActivationResult]] = defaultdict(dict)

    for ds_key in datasets:
        print(f"\n  Dataset: {ds_key}")
        dataset = load_dataset(ds_key, max_examples=max_examples)

        for rep in rep_counts:
            print(f"  Rep count: {rep}")
            t0 = time.time()

            result = extract_activations_for_condition(
                extractor=extractor,
                dataset=dataset,
                rep_count=rep,
                model_config=model_config,
                max_examples=max_examples,
            )

            elapsed = time.time() - t0
            print(f"    Extracted {result.n_samples} samples in {elapsed:.1f}s")
            print(f"    {result.summary()}")

            all_activations[ds_key][rep] = result

            # Optionally save raw activations
            if save_activations:
                act_dir = output_dir / "activations" / ds_key / f"rep{rep}"
                result.save(str(act_dir), format="safetensors")

            # Log to W&B
            if wandb_project and HAS_WANDB:
                wandb.log({
                    f"extraction/{ds_key}/rep{rep}/n_samples": result.n_samples,
                    f"extraction/{ds_key}/rep{rep}/time_s": elapsed,
                })

    # ── Step 3: In-domain probe training ──────────────────────────────
    print("\nStep 3: Training in-domain probes...")

    all_probe_results: list[ProbeResult] = []
    layer_profiles: list[LayerProbeProfile] = []

    for ds_key in datasets:
        for rep in rep_counts:
            result = all_activations[ds_key][rep]
            labels = make_binary_labels(result.metadata.get("labels", []), ds_key)

            if len(np.unique(labels)) < 2:
                print(f"    Skipping {ds_key} rep={rep}: only 1 class")
                continue

            layer_accs = {}
            layer_f1s = {}

            for layer in layers:
                module_key = f"resid_post.layer{layer}"
                if module_key not in result.activations:
                    continue

                X = result.numpy(module_key)
                n = len(labels)
                split = int(0.8 * n)
                perm = np.random.RandomState(42).permutation(n)

                acc, f1, clf = train_probe(
                    X[perm[:split]], labels[perm[:split]],
                    X[perm[split:]], labels[perm[split:]],
                )

                layer_accs[layer] = acc
                layer_f1s[layer] = f1

                all_probe_results.append(ProbeResult(
                    layer=layer, accuracy=acc, f1=f1,
                    n_train=split, n_test=n - split,
                    train_domain=ds_key, test_domain=ds_key,
                    repetition_count=rep, model_name=model_key,
                ))

                if wandb_project and HAS_WANDB:
                    wandb.log({
                        f"probe/{ds_key}/rep{rep}/layer{layer}/accuracy": acc,
                        f"probe/{ds_key}/rep{rep}/layer{layer}/f1": f1,
                    })

            if layer_accs:
                layer_profiles.append(LayerProbeProfile(
                    model_name=model_key, dataset=ds_key,
                    repetition_count=rep,
                    layer_accuracies=layer_accs, layer_f1s=layer_f1s,
                ))

            best_layer = max(layer_accs, key=layer_accs.get) if layer_accs else -1
            best_acc = max(layer_accs.values()) if layer_accs else 0
            print(f"    {ds_key} rep={rep}: best layer={best_layer}, "
                  f"acc={best_acc:.3f}")

    # ── Step 4: Cross-domain probes ───────────────────────────────────
    print("\nStep 4: Cross-domain probe transfer...")

    cross_domain_results: list[ProbeResult] = []

    # For each rep count, train on each domain, test on all others
    for rep in rep_counts:
        activations_at_rep = {
            ds: all_activations[ds][rep]
            for ds in datasets
            if rep in all_activations[ds]
        }

        for layer in layers:
            cross_results = run_cross_domain_probes(
                activations_at_rep, layer, model_key, rep,
            )
            cross_domain_results.extend(cross_results)

    # Print cross-domain summary
    for r in cross_domain_results:
        if r.is_cross_domain():
            print(f"    {r.train_domain} -> {r.test_domain} "
                  f"(L{r.layer}, rep={r.repetition_count}): "
                  f"acc={r.accuracy:.3f}")

    # ── Step 5: Precursor gap analysis ────────────────────────────────
    print("\nStep 5: Precursor gap detection...")

    # Behavioral onsets from Phase 1 (hardcoded from W&B results)
    # These should be loaded from Phase 1 results in production
    behavioral_onsets = {
        "low": 12,             # TRAM ordering: simple, expect late degradation
        "medium_temporal": 5,  # TRAM: Llama degrades by rep 5
        "medium_code": 8,      # MBPP: moderate degradation
        "high": 3,             # MedQA: early degradation
    }

    activation_onsets = {}
    for ds_key in datasets:
        # Track probe accuracy across reps for the best layer
        best_layer_for_ds = None
        best_acc_at_rep1 = 0

        for layer in layers:
            key = f"resid_post.layer{layer}"
            if 1 in all_activations[ds_key]:
                result_rep1 = all_activations[ds_key][1]
                if key in result_rep1.activations:
                    X = result_rep1.numpy(key)
                    labels = make_binary_labels(
                        result_rep1.metadata.get("labels", []), ds_key
                    )
                    if len(np.unique(labels)) >= 2:
                        n = len(labels)
                        split = int(0.8 * n)
                        perm = np.random.RandomState(42).permutation(n)
                        acc, _, _ = train_probe(
                            X[perm[:split]], labels[perm[:split]],
                            X[perm[split:]], labels[perm[split:]],
                        )
                        if acc > best_acc_at_rep1:
                            best_acc_at_rep1 = acc
                            best_layer_for_ds = layer

        if best_layer_for_ds is not None:
            # Get probe accuracy at each rep count for this layer
            acc_by_rep = {}
            for rep in rep_counts:
                if rep not in all_activations[ds_key]:
                    continue
                result = all_activations[ds_key][rep]
                key = f"resid_post.layer{best_layer_for_ds}"
                if key not in result.activations:
                    continue
                X = result.numpy(key)
                labels = make_binary_labels(
                    result.metadata.get("labels", []), ds_key
                )
                if len(np.unique(labels)) >= 2:
                    n = len(labels)
                    split = int(0.8 * n)
                    perm = np.random.RandomState(42).permutation(n)
                    acc, _, _ = train_probe(
                        X[perm[:split]], labels[perm[:split]],
                        X[perm[split:]], labels[perm[split:]],
                    )
                    acc_by_rep[rep] = acc

            onset = compute_activation_onset(acc_by_rep)
            if onset is not None:
                activation_onsets[ds_key] = onset
                gap = behavioral_onsets.get(ds_key, 0) - onset
                print(f"    {ds_key}: activation onset={onset}, "
                      f"behavioral onset={behavioral_onsets.get(ds_key, '?')}, "
                      f"gap={gap} reps")

    # ── Step 6: Save results and plots ────────────────────────────────
    print("\nStep 6: Saving results...")

    # Save all probe results
    results_data = {
        "model": model_key,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layers": layers,
            "datasets": datasets,
            "rep_counts": rep_counts,
            "max_examples": max_examples,
        },
        "in_domain_probes": [asdict(r) for r in all_probe_results],
        "cross_domain_probes": [asdict(r) for r in cross_domain_results],
        "activation_onsets": activation_onsets,
        "behavioral_onsets": behavioral_onsets,
        "precursor_gaps": {
            ds: behavioral_onsets.get(ds, 0) - activation_onsets.get(ds, 0)
            for ds in activation_onsets
        },
    }

    results_path = output_dir / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved results: {results_path}")

    # Plots
    if HAS_MPL and layer_profiles:
        # Layer profiles per dataset
        for ds_key in datasets:
            ds_profiles = [p for p in layer_profiles if p.dataset == ds_key]
            if ds_profiles:
                plot_layer_profiles(
                    ds_profiles,
                    output_dir / f"layer_profile_{ds_key}.png",
                    title=f"{model_key}: {ds_key} Probe by Layer",
                )

        # Cross-domain heatmap for the best layer
        if cross_domain_results:
            best_layer = max(layers, key=lambda l: np.mean([
                r.accuracy for r in all_probe_results if r.layer == l
            ]) if any(r.layer == l for r in all_probe_results) else 0)

            for rep in [1, rep_counts[-1]]:
                rep_results = [r for r in cross_domain_results
                               if r.repetition_count == rep]
                if rep_results:
                    plot_cross_domain_heatmap(
                        rep_results, best_layer,
                        output_dir / f"cross_domain_heatmap_rep{rep}.png",
                        title=f"{model_key}: Cross-Domain Transfer (rep={rep})",
                    )

        # Precursor gap plot
        if activation_onsets:
            plot_precursor_gap(
                behavioral_onsets, activation_onsets,
                output_dir / "precursor_gap.png",
                model_name=model_key,
            )

    if wandb_project and HAS_WANDB:
        wandb.finish()

    print(f"\n✓ Phase 2 complete for {model_key}")
    return results_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Cross-domain & cross-stake activation analysis"
    )
    parser.add_argument(
        "--model", type=str, default="Llama-3.1-8B-Instruct",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to run",
    )
    parser.add_argument("--all-models", action="store_true",
                        help="Run all 4 models sequentially")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer examples, layers, reps")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples per dataset per condition")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["low", "medium_temporal", "high"],
                        help="Dataset keys to use")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name")
    parser.add_argument("--save-activations", action="store_true",
                        help="Save raw activations to disk")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    models_to_run = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_key in models_to_run:
        config = MODEL_CONFIGS[model_key]

        if args.quick:
            layers = config["quick_layers"]
            rep_counts = QUICK_REP_COUNTS
            max_examples = 20
        else:
            layers = config["layers"]
            rep_counts = REPETITION_COUNTS
            max_examples = args.max_examples

        run_phase2_experiment(
            model_key=model_key,
            datasets=args.datasets,
            rep_counts=rep_counts,
            layers=layers,
            device=args.device,
            max_examples=max_examples,
            wandb_project=args.wandb_project,
            save_activations=args.save_activations,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
