#!/usr/bin/env python3
"""
Phase 3, Experiment 5: Attention Head Analysis of Degradation

Analyzes which attention heads change behavior during degradation —
the circuit-level mechanism linking to Lost in the Middle.

Method:
  1. For each model, extract attention patterns at early (rep=1) and late
     (rep=15) repetitions using model forward pass with output_attentions=True
  2. For each attention head at each layer, compute:
     a. Attention entropy: Shannon entropy of attention distribution. Higher
        entropy = more diffuse attention. Track change from fresh to degraded.
     b. Attention to task tokens: Fraction of attention to final 20% of sequence
        (where the actual question is) vs. first 80% (repetitive prefix)
     c. Head degradation correlation: correlation between that head's entropy
        change and overall accuracy degradation across examples
  3. Identify degradation heads: heads whose attention patterns shift most
     between fresh and degraded states
  4. Classify heads:
     - Task heads: high attention to question tokens, low entropy
     - Context heads: distribute attention across full context
     - Degradation heads: entropy increases most during degradation
  5. Optional: ablate top-3 degradation heads at late reps, test if zeroing
     them restores accuracy (circuit-level causality test)

Key hypothesis:
  - Task heads are disrupted during degradation (attention diffuses, Lost in Middle)
  - Degradation heads show systematic entropy increase (top 10-20 heads per layer)
  - Ablating degradation heads partially recovers accuracy

Related work:
  - Lost in the Middle (Liu et al., 2023): models attend less to middle content
  - Circuit analysis (Vig & Belinkov, 2019): identifying functional heads
  - Attn head dissection (Voita et al., 2019): head importance via gradient saliency
  - Sparse Autoencoders (Cunningham et al., 2023): finding task-relevant features

Target models: Final 5 models

Usage:
    # Quick validation (1 model, 3 layers)
    python scripts/experiments/phase3_attention_analysis.py --quick

    # Single model
    python scripts/experiments/phase3_attention_analysis.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    # All models
    python scripts/experiments/phase3_attention_analysis.py \\
        --all-models --device cuda

    # Sherlock SLURM
    sbatch scripts/experiments/submit_phase3_attention.sh

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
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.stats import entropy, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_attention"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
    },
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
        "quick_layers": [8, 16, 28],
        "n_layers": 36,
        "d_model": 4096,
        "n_heads": 32,
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "layers": [0, 6, 12, 18, 24, 30, 36, 42, 47],
        "quick_layers": [12, 24, 36],
        "n_layers": 48,
        "d_model": 4096,
        "n_heads": 32,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 27],
        "quick_layers": [8, 16, 24],
        "n_layers": 28,
        "d_model": 3584,
        "n_heads": 28,
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
class AttentionHeadMetrics:
    """Metrics for a single attention head."""
    layer: int
    head_idx: int
    entropy_fresh: float           # Entropy at rep=1
    entropy_degraded: float        # Entropy at rep=15
    entropy_delta: float           # degraded - fresh
    task_attention_fresh: float    # Fraction to task tokens at rep=1
    task_attention_degraded: float # Fraction to task tokens at rep=15
    task_attention_delta: float    # degraded - fresh
    degradation_correlation: Optional[float] = None  # Corr with overall accuracy drop
    head_class: Optional[str] = None  # "task", "context", "degradation", "other"


@dataclass
class LayerAttentionSummary:
    """Summary of all heads at one layer."""
    layer: int
    model: str
    n_heads: int
    mean_entropy_delta: float
    max_entropy_delta: float
    n_degradation_heads: int  # heads with |entropy_delta| > threshold


@dataclass
class AttentionAnalysisResult:
    """Complete attention analysis for one model."""
    model: str
    dataset: str
    n_examples: int
    rep_counts: list[int]
    layer_summaries: list[dict]
    head_metrics: list[dict]
    top_degradation_heads: list[dict]  # Top 10 heads per layer
    ablation_results: Optional[list] = None


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
    """Build a prompt with repetitive prefix."""
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
    """Extract ground truth answer."""
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
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_patterns(
    model_key: str,
    examples: list[dict],
    rep_counts: list[int],
    device: str = "cuda",
) -> Tuple[dict, dict]:
    """
    Extract attention patterns from a model for given examples and repetitions.

    Args:
        model_key: Key in MODEL_CONFIGS
        examples: List of examples to process
        rep_counts: List of repetition counts (e.g., [1, 15])
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        attention_fresh: {layer: {"entropy": (n_examples, n_heads), "task_attn": (n_examples, n_heads)}}
        attention_degraded: {layer: {"entropy": (n_examples, n_heads), "task_attn": (n_examples, n_heads)}}
    """
    if not HAS_TORCH:
        raise ImportError("torch required")
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required")

    print(f"\n  Extracting attention patterns from {model_key}")
    print(f"  Examples: {len(examples)}, Repetitions: {rep_counts}")

    config = MODEL_CONFIGS[model_key]
    hf_name = config["hf_name"]
    layers_to_extract = config["layers"]
    n_heads = config["n_heads"]

    # Load model and tokenizer
    print(f"  Loading model {hf_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map=device,
        torch_dtype=torch.float16,
        output_attentions=True,
    )
    model.eval()

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Storage for per-example, per-head metrics (NOT raw attention matrices,
    # since each example has a different seq_len and np.stack would fail).
    # metrics_by_rep[rep][layer] = list of dicts with per-head entropy & task_attn
    metrics_by_rep = {}

    # Process each repetition count
    for rep_count in rep_counts:
        print(f"  Processing rep_count={rep_count}...")
        metrics_storage = {}  # {layer: [{"entropy": (n_heads,), "task_attn": (n_heads,)}, ...]}

        # Process each example
        with torch.no_grad():
            for ex_idx, example in enumerate(examples):
                if (ex_idx + 1) % max(1, len(examples) // 5) == 0:
                    print(f"    Example {ex_idx + 1}/{len(examples)}")

                # Build prompt for this repetition
                prompt = build_prompt(example, rep_count)

                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs, output_attentions=True)

                # Extract attention from specified layers
                # outputs.attentions is a tuple of (batch, n_heads, seq_len, seq_len)
                # per layer for all layers
                attention_tuple = outputs.attentions

                for layer_idx, attn_tensor in enumerate(attention_tuple):
                    # Only extract from specified layers
                    if layer_idx not in layers_to_extract:
                        continue

                    # Move to CPU and convert to numpy
                    attn_np = attn_tensor[0].cpu().detach().float().numpy()
                    # Shape: (n_heads, seq_len, seq_len)
                    cur_n_heads, cur_seq_len, _ = attn_np.shape

                    if layer_idx not in metrics_storage:
                        metrics_storage[layer_idx] = []

                    # Compute per-head entropy for this example
                    head_entropies = np.zeros(cur_n_heads)
                    head_task_attn = np.zeros(cur_n_heads)
                    task_start = int(cur_seq_len * 0.8)  # last 20% is task tokens

                    for h in range(cur_n_heads):
                        attn = attn_np[h]  # (seq_len, seq_len)
                        # Mean entropy across query positions
                        eps = 1e-10
                        row_entropies = []
                        for i in range(cur_seq_len):
                            dist = attn[i, :] + eps
                            if HAS_SCIPY:
                                row_entropies.append(entropy(dist))
                            else:
                                row_entropies.append(-np.sum(dist * np.log(dist)))
                        head_entropies[h] = float(np.mean(row_entropies))

                        # Task token attention fraction
                        task_attention = attn[:, task_start:].sum()
                        total_attention = attn.sum()
                        head_task_attn[h] = float(task_attention / (total_attention + eps))

                    metrics_storage[layer_idx].append({
                        "entropy": head_entropies,
                        "task_attn": head_task_attn,
                        "seq_len": cur_seq_len,
                    })

        metrics_by_rep[rep_count] = metrics_storage

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Organize by fresh/degraded based on rep_counts order
    fresh_rep = rep_counts[0]
    degraded_rep = rep_counts[1] if len(rep_counts) > 1 else rep_counts[0]

    # Convert per-example metrics into stacked arrays for downstream:
    # {layer: (n_examples, n_heads)} for entropy and task_attn
    attention_fresh = {}
    attention_degraded = {}

    for layer_idx in metrics_by_rep.get(fresh_rep, {}):
        fresh_items = metrics_by_rep[fresh_rep][layer_idx]
        n_examples = len(fresh_items)
        n_heads = len(fresh_items[0]["entropy"])
        ent_arr = np.zeros((n_examples, n_heads))
        task_arr = np.zeros((n_examples, n_heads))
        for i, item in enumerate(fresh_items):
            ent_arr[i] = item["entropy"]
            task_arr[i] = item["task_attn"]
        attention_fresh[layer_idx] = {
            "entropy": ent_arr,
            "task_attn": task_arr,
        }

    for layer_idx in metrics_by_rep.get(degraded_rep, {}):
        deg_items = metrics_by_rep[degraded_rep][layer_idx]
        n_examples = len(deg_items)
        n_heads = len(deg_items[0]["entropy"])
        ent_arr = np.zeros((n_examples, n_heads))
        task_arr = np.zeros((n_examples, n_heads))
        for i, item in enumerate(deg_items):
            ent_arr[i] = item["entropy"]
            task_arr[i] = item["task_attn"]
        attention_degraded[layer_idx] = {
            "entropy": ent_arr,
            "task_attn": task_arr,
        }

    return attention_fresh, attention_degraded


# ---------------------------------------------------------------------------
# Attention metrics computation
# ---------------------------------------------------------------------------

def compute_attention_entropy(attention_matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute Shannon entropy of attention distributions.

    Args:
        attention_matrix: shape (n_examples, n_heads, seq_len, seq_len)
                         each slice [e, h, i, :] is an attention distribution

    Returns:
        entropies: shape (n_examples, n_heads) - entropy per head per example
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")

    n_examples, n_heads, seq_len, _ = attention_matrix.shape
    entropies = np.zeros((n_examples, n_heads))

    for e in range(n_examples):
        for h in range(n_heads):
            attn = attention_matrix[e, h, :, :]  # (seq_len, seq_len)
            # Average entropy across query positions
            head_entropies = []
            for i in range(seq_len):
                attn_dist = attn[i, :] + eps
                h_val = entropy(attn_dist)
                head_entropies.append(h_val)
            entropies[e, h] = np.mean(head_entropies)

    return entropies


def compute_task_attention_fraction(attention_matrix: np.ndarray,
                                    seq_len: int,
                                    task_token_fraction: float = 0.2) -> np.ndarray:
    """
    Compute fraction of attention going to task tokens (final task_token_fraction%).

    Args:
        attention_matrix: shape (n_examples, n_heads, seq_len, seq_len)
        seq_len: sequence length
        task_token_fraction: fraction of sequence that is task (default 0.2)

    Returns:
        fractions: shape (n_examples, n_heads)
    """
    n_examples, n_heads, _, _ = attention_matrix.shape

    # Task tokens are in the final task_token_fraction of the sequence
    task_start = int(seq_len * (1 - task_token_fraction))
    task_end = seq_len

    fractions = np.zeros((n_examples, n_heads))

    for e in range(n_examples):
        for h in range(n_heads):
            attn = attention_matrix[e, h, :, :]  # (seq_len, seq_len)
            # Sum attention to task tokens (columns)
            task_attention = attn[:, task_start:task_end].sum()
            total_attention = attn.sum()
            fractions[e, h] = task_attention / (total_attention + 1e-10)

    return fractions


def analyze_attention_heads(
    model_key: str,
    attention_fresh: dict,
    attention_degraded: dict,
    examples: list[dict],
    layers: list[int],
) -> Tuple[list[AttentionHeadMetrics], list[LayerAttentionSummary]]:
    """
    Compute attention metrics for all heads.

    Returns:
        head_metrics: list of AttentionHeadMetrics for each head
        layer_summaries: list of LayerAttentionSummary for each layer
    """
    config = MODEL_CONFIGS[model_key]
    n_heads = config["n_heads"]

    head_metrics = []
    layer_summaries = []

    for layer in layers:
        print(f"  Layer {layer}: computing head metrics...")

        # New format: each layer maps to a dict with pre-computed metrics
        # {"entropy": np.array(n_examples, n_heads), "task_attn": np.array(n_examples, n_heads)}
        layer_fresh = attention_fresh[layer]
        layer_degraded = attention_degraded[layer]

        entropies_fresh = layer_fresh["entropy"]      # (n_examples, n_heads)
        entropies_degraded = layer_degraded["entropy"]
        task_attn_fresh = layer_fresh["task_attn"]     # (n_examples, n_heads)
        task_attn_degraded = layer_degraded["task_attn"]

        # Per-head analysis
        for head_idx in range(n_heads):
            entropy_f = entropies_fresh[:, head_idx].mean()
            entropy_d = entropies_degraded[:, head_idx].mean()
            entropy_delta = entropy_d - entropy_f

            task_f = task_attn_fresh[:, head_idx].mean()
            task_d = task_attn_degraded[:, head_idx].mean()
            task_delta = task_d - task_f

            metric = AttentionHeadMetrics(
                layer=layer,
                head_idx=head_idx,
                entropy_fresh=float(entropy_f),
                entropy_degraded=float(entropy_d),
                entropy_delta=float(entropy_delta),
                task_attention_fresh=float(task_f),
                task_attention_degraded=float(task_d),
                task_attention_delta=float(task_delta),
            )

            # Classify head
            if entropy_delta > 0.1 and task_delta < -0.1:
                metric.head_class = "degradation"
            elif task_f > 0.5:
                metric.head_class = "task"
            elif entropy_f < 2.0:
                metric.head_class = "context"
            else:
                metric.head_class = "other"

            head_metrics.append(metric)

        # Layer summary
        entropy_deltas = entropies_degraded.mean(axis=0) - entropies_fresh.mean(axis=0)
        summary = LayerAttentionSummary(
            layer=layer,
            model=model_key,
            n_heads=n_heads,
            mean_entropy_delta=float(entropy_deltas.mean()),
            max_entropy_delta=float(entropy_deltas.max()),
            n_degradation_heads=int((np.abs(entropy_deltas) > 0.1).sum()),
        )
        layer_summaries.append(summary)

    return head_metrics, layer_summaries


# ---------------------------------------------------------------------------
# Top degradation heads identification
# ---------------------------------------------------------------------------

def identify_top_degradation_heads(
    head_metrics: list[AttentionHeadMetrics],
    top_k: int = 10,
) -> dict[int, list[dict]]:
    """
    Identify top-k degradation heads per layer by entropy delta.

    Returns:
        {layer: [{"head_idx": h, "entropy_delta": d, ...}, ...]}
    """
    by_layer = defaultdict(list)

    for metric in head_metrics:
        by_layer[metric.layer].append({
            "head_idx": metric.head_idx,
            "entropy_delta": metric.entropy_delta,
            "task_attention_delta": metric.task_attention_delta,
            "head_class": metric.head_class,
            "entropy_fresh": metric.entropy_fresh,
            "entropy_degraded": metric.entropy_degraded,
        })

    # Sort by |entropy_delta| and keep top-k
    top_degradation = {}
    for layer, heads in by_layer.items():
        heads_sorted = sorted(heads, key=lambda x: abs(x["entropy_delta"]), reverse=True)
        top_degradation[layer] = heads_sorted[:top_k]

    return top_degradation


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_entropy_heatmap(head_metrics: list[AttentionHeadMetrics],
                         model_key: str,
                         output_path: Path) -> None:
    """Plot heatmap of entropy changes per head per layer."""
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available for {output_path}")
        return

    config = MODEL_CONFIGS[model_key]
    n_heads = config["n_heads"]
    layers = sorted(set(m.layer for m in head_metrics))

    # Build matrix
    matrix = np.full((len(layers), n_heads), np.nan)
    for metric in head_metrics:
        layer_idx = layers.index(metric.layer)
        matrix[layer_idx, metric.head_idx] = metric.entropy_delta

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                ax=ax, cbar_kws={"label": "Entropy Delta (degraded - fresh)"})
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_title(f"{model_key}: Attention Entropy Change During Degradation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_task_attention_heatmap(head_metrics: list[AttentionHeadMetrics],
                                model_key: str,
                                output_path: Path) -> None:
    """Plot task attention fraction per head per layer."""
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available for {output_path}")
        return

    config = MODEL_CONFIGS[model_key]
    n_heads = config["n_heads"]
    layers = sorted(set(m.layer for m in head_metrics))

    matrix = np.full((len(layers), n_heads), np.nan)
    for metric in head_metrics:
        layer_idx = layers.index(metric.layer)
        matrix[layer_idx, metric.head_idx] = metric.task_attention_delta

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(matrix, cmap="coolwarm", center=0, vmin=-0.3, vmax=0.3,
                ax=ax, cbar_kws={"label": "Task Attention Delta"})
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_title(f"{model_key}: Task Attention Change During Degradation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_degradation_heads_profile(top_degradation: dict[int, list[dict]],
                                   model_key: str,
                                   output_path: Path) -> None:
    """Plot profiles of top degradation heads."""
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available for {output_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    layers = sorted(top_degradation.keys())[:4]  # Show first 4 layers

    for ax_idx, layer in enumerate(layers):
        heads = top_degradation[layer][:5]  # Top 5 per layer

        head_idxs = [h["head_idx"] for h in heads]
        entropy_deltas = [h["entropy_delta"] for h in heads]

        ax = axes[ax_idx]
        bars = ax.barh(range(len(head_idxs)), entropy_deltas, color="steelblue")
        ax.set_yticks(range(len(head_idxs)))
        ax.set_yticklabels([f"H{h}" for h in head_idxs])
        ax.set_xlabel("Entropy Delta")
        ax.set_title(f"Layer {layer} - Top Degradation Heads")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    fig.suptitle(f"{model_key}: Degradation Head Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention analysis of degradation mechanism")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 model, 3 layers)")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()),
                        help="Single model to analyze")
    parser.add_argument("--all-models", action="store_true",
                        help="Analyze all models")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--datasets", nargs="+", default=["medium_temporal"],
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PHASE 3, EXPERIMENT 5: Attention Head Analysis of Degradation")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")

    if not HAS_TORCH:
        print("ERROR: torch not available")
        return 1

    # Determine models
    if args.quick:
        models_to_use = ["Llama-3.1-8B-Instruct"]
    elif args.model:
        models_to_use = [args.model]
    elif args.all_models:
        models_to_use = list(MODEL_CONFIGS.keys())
    else:
        models_to_use = list(MODEL_CONFIGS.keys())

    print(f"Models: {models_to_use}")

    if args.wandb_project and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"attention-{args.model}",
        )

    all_results = []

    for model_key in models_to_use:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING {model_key}")
        print(f"{'=' * 80}")

        config = MODEL_CONFIGS[model_key]
        layers = config["quick_layers"] if args.quick else config["layers"]

        for dataset_key in args.datasets:
            print(f"\nDataset: {dataset_key}")

            dataset = load_benchmark_dataset(dataset_key, max_examples=20 if args.quick else 50)
            examples = dataset["examples"]

            # Extract attention patterns
            attention_fresh, attention_degraded = extract_attention_patterns(
                model_key, examples, rep_counts=[1, 15], device=args.device
            )

            # Analyze attention heads
            head_metrics, layer_summaries = analyze_attention_heads(
                model_key, attention_fresh, attention_degraded, examples, layers
            )

            # Identify top degradation heads
            top_degradation = identify_top_degradation_heads(head_metrics, top_k=10)

            # Create result object
            result = AttentionAnalysisResult(
                model=model_key,
                dataset=dataset_key,
                n_examples=len(examples),
                rep_counts=[1, 15],
                layer_summaries=[asdict(s) for s in layer_summaries],
                head_metrics=[asdict(m) for m in head_metrics],
                top_degradation_heads=top_degradation,
            )
            all_results.append(result)

            # Create output directory for model
            model_output_dir = args.output_dir / model_key
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            results_path = model_output_dir / f"attention_results_{dataset_key}.json"
            with open(results_path, "w") as f:
                json.dump(asdict(result), f, indent=2, cls=NumpyEncoder)
            print(f"  Saved: {results_path}")

            # Generate plots
            plot_entropy_heatmap(
                head_metrics, model_key,
                model_output_dir / f"entropy_change_heatmap_{dataset_key}.png"
            )
            plot_task_attention_heatmap(
                head_metrics, model_key,
                model_output_dir / f"task_attention_fraction_{dataset_key}.png"
            )
            plot_degradation_heads_profile(
                top_degradation, model_key,
                model_output_dir / f"degradation_heads_profile_{dataset_key}.png"
            )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Models analyzed: {len(models_to_use)}")
    print(f"Total results: {len(all_results)}")

    if args.wandb_project and HAS_WANDB:
        wandb.log({
            "n_models": len(models_to_use),
            "n_results": len(all_results),
        })
        wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
