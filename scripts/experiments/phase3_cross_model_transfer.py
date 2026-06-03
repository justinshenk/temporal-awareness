#!/usr/bin/env python3
"""
Phase 3, Experiment 4: Cross-Model Transfer of Degradation Direction

Tests whether the degradation direction is UNIVERSAL across architectures
or model-specific. If the direction is universal, it suggests a fundamental
mechanism in LLMs; if specific, it suggests architecture-dependent learning.

Method:
  1. Load pre-extracted degradation direction vectors from each model
     (from results/phase3_refusal_direction/{model}/directions/)
  2. For models with same d_model (Llama-3.1-8B-Instruct, Llama-3.1-8B,
     Qwen3-8B, Qwen3-30B-A3B all d_model=4096):
     a. Direct cosine similarity between degradation directions at matching layers
     b. Cross-model probe transfer: train probe on Model A, test on Model B
        (same dataset, same repetition levels)
     c. Direction injection transfer: inject Model A's direction into Model B,
        measure accuracy impact (reuses causal patching hooks)
  3. For DeepSeek (d_model=3584 vs 4096):
     a. Use CKA (Centered Kernel Alignment) for dimension-agnostic comparison
     b. Train linear projection from DeepSeek space to Llama space
     c. Test if projected directions transfer
  4. Base vs Instruct Llama: does base model have the same degradation direction?
     If NOT, proves RLHF creates the degradation mechanism.

Key hypothesis:
  - HIGH cosine similarity (>0.6) across different architectures → universal mechanism
  - HIGH cross-model probe transfer (>0.65 acc) → shared representation
  - Successful direction injection across models → direction is generalizable
  - No degradation direction in base Llama → RLHF-specific learning

Related work:
  - Turner et al. (2023): Activation addition generalizes across models
  - Meng et al. (2023): ROME changes generalize to related models
  - Huang et al. (2024): Mechanistic features are shared across architectures
  - Li et al. (2024): RLHF-specific mechanisms in instruction-tuned models

Target models: All 5 (Llama-3.1-8B-Instruct, Llama-3.1-8B, Qwen3-8B,
              Qwen3-30B-A3B, DeepSeek-R1-Distill-Qwen-7B)

Usage:
    # Quick validation (2 models, 3 layers)
    python scripts/experiments/phase3_cross_model_transfer.py --quick

    # Single pairwise comparison
    python scripts/experiments/phase3_cross_model_transfer.py \\
        --model-pair Llama-3.1-8B-Instruct Qwen3-8B --device cuda

    # All pairwise
    python scripts/experiments/phase3_cross_model_transfer.py \\
        --all-pairs --device cuda

    # Base vs Instruct comparison
    python scripts/experiments/phase3_cross_model_transfer.py \\
        --base-vs-instruct --device cuda

    # Sherlock SLURM
    sbatch scripts/experiments/submit_phase3_cross_model.sh

Author: Adrian Sadik
Date: 2026-04-10
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
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
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
DIRECTIONS_DIR = PROJECT_ROOT / "results" / "phase3_refusal_direction"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_cross_model_transfer"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        "d_model": 2048,
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
    "Ouro-2.6B": {
        "hf_name": "ByteDance/Ouro-2.6B",
        "layers": [0, 3, 6, 9, 12, 15, 18, 21, 23],
        "quick_layers": [6, 12, 18],
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 16,
    },
}


# ---------------------------------------------------------------------------
# Datasets (matching other Phase 3 experiments)
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
class CosineSimilarityResult:
    """Cosine similarity between two directions."""
    model_a: str
    model_b: str
    layer: int
    similarity: float
    direction_a_norm: float
    direction_b_norm: float


@dataclass
class ProbeTransferResult:
    """Result of cross-model probe transfer."""
    source_model: str
    target_model: str
    dataset: str
    train_layer: int
    test_layer: int
    train_accuracy: float
    test_accuracy: float
    n_examples: int


@dataclass
class CKAResult:
    """Centered Kernel Alignment between two activation spaces."""
    model_a: str
    model_b: str
    layer_a: int
    layer_b: int
    cka_similarity: float


@dataclass
class DirectionInjectionResult:
    """Result of injecting one model's direction into another."""
    source_model: str
    target_model: str
    layer: int
    dataset: str
    baseline_accuracy: float
    injected_accuracy: float
    accuracy_delta: float
    injection_strength: float
    n_examples: int


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
# Direction loading
# ---------------------------------------------------------------------------

def load_direction(model_key: str, layer: int,
                   dataset_key: str = "medium_temporal",
                   source_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load pre-extracted degradation direction vector.

    Looks in: results/phase3_refusal_direction/{model}/directions/
    """
    if source_dir is None:
        source_dir = DIRECTIONS_DIR

    model_dir = source_dir / model_key / "directions"
    fname = f"degradation_L{layer}_{dataset_key}.npy"
    path = model_dir / fname

    if path.exists():
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Cosine similarity: direct direction comparison
# ---------------------------------------------------------------------------

def compute_cosine_similarity_matrix(
    models: list[str],
    layers_per_model: dict[str, list[int]],
    dataset_key: str = "medium_temporal",
) -> Tuple[dict, np.ndarray]:
    """
    Compute pairwise cosine similarity between degradation directions.
    Returns only for shared dimensions (d_model >= 4096).
    """
    results = []
    sim_matrix_data = {}

    models_same_dim = [m for m in models if MODEL_CONFIGS[m]["d_model"] >= 4096]
    n = len(models_same_dim)
    sim_matrix = np.full((n, n), np.nan)

    for i, model_a in enumerate(models_same_dim):
        for j, model_b in enumerate(models_same_dim):
            if i >= j:
                continue  # upper triangle only

            layer_a = layers_per_model[model_a][len(layers_per_model[model_a]) // 2]
            layer_b = layers_per_model[model_b][len(layers_per_model[model_b]) // 2]

            dir_a = load_direction(model_a, layer_a, dataset_key)
            dir_b = load_direction(model_b, layer_b, dataset_key)

            if dir_a is None or dir_b is None:
                print(f"  WARNING: Missing direction for {model_a} L{layer_a} or {model_b} L{layer_b}")
                continue

            # Normalize
            dir_a_norm = dir_a / (np.linalg.norm(dir_a) + 1e-8)
            dir_b_norm = dir_b / (np.linalg.norm(dir_b) + 1e-8)

            sim = float(np.dot(dir_a_norm, dir_b_norm))
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

            result = CosineSimilarityResult(
                model_a=model_a,
                model_b=model_b,
                layer=layer_a,
                similarity=sim,
                direction_a_norm=float(np.linalg.norm(dir_a)),
                direction_b_norm=float(np.linalg.norm(dir_b)),
            )
            results.append(result)
            print(f"  {model_a} <-> {model_b}: cosine_sim={sim:.4f}")

    return {
        "results": [asdict(r) for r in results],
        "models": models_same_dim,
        "sim_matrix": sim_matrix,
    }, sim_matrix


# ---------------------------------------------------------------------------
# CKA: dimension-agnostic comparison for DeepSeek
# ---------------------------------------------------------------------------

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Centered Kernel Alignment (linear variant).
    X, Y: (n_samples, d) activation matrices
    Returns: CKA score in [0, 1]
    """
    # Assume X, Y are already centered (mean=0)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K_XX = X @ X.T
    K_YY = Y @ Y.T
    K_XY = X @ Y.T

    # CKA = ||K_XY||_F^2 / (||K_XX||_F * ||K_YY||_F)
    numerator = np.sum(K_XY ** 2)
    denominator = np.sqrt(np.sum(K_XX ** 2) * np.sum(K_YY ** 2))

    if denominator < 1e-8:
        return 0.0
    return numerator / denominator


def compare_with_cka(
    model_a: str,
    model_b: str,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """
    Use CKA-like approach: treat directions as "features" and compute alignment.
    Simple version: normalize both to unit norm and compute cosine.
    """
    dir_a_norm = direction_a / (np.linalg.norm(direction_a) + 1e-8)
    dir_b_norm = direction_b / (np.linalg.norm(direction_b) + 1e-8)

    # For CKA-style: project to common dimension or use cosine
    return float(np.dot(dir_a_norm, dir_b_norm))


# ---------------------------------------------------------------------------
# Cross-model probe transfer
# ---------------------------------------------------------------------------

def extract_activations_at_layer(
    model_key: str,
    layer: int,
    examples: list[dict],
    rep_counts: list[int],
    backend: str = "pytorch",
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract activations from a model at a given layer using ActivationExtractor.
    Returns: (activation_matrix, rep_labels, example_ids)
    """
    if not HAS_TORCH:
        raise ImportError("torch required")

    print(f"  Extracting activations from {model_key} L{layer}")
    print(f"    {len(examples)} examples × {len(rep_counts)} repetition levels")

    # Build prompts for all (example, rep_count) pairs
    prompts = []
    rep_labels = []
    example_ids = []

    for i, ex in enumerate(examples):
        ex_id = ex.get("id", f"ex_{i}")
        for rep in rep_counts:
            prompt = build_prompt(ex, rep)
            prompts.append(prompt)
            rep_labels.append(rep)
            example_ids.append(ex_id)

    # Initialize extractor for this model
    # Resolve backend choice
    use_tl = {"pytorch": False, "transformer_lens": True, "auto": None}[backend]

    extraction_config = ExtractionConfig(
        layers=[layer],  # just the one layer we need
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
        model=MODEL_CONFIGS[model_key]["hf_name"],
        config=extraction_config,
        device="cuda",
    )

    # Extract activations
    result = extractor.extract(prompts, return_tokens=False)
    acts = result.numpy(f"resid_post.layer{layer}")  # (n_prompts, d_model)

    # Clean up to free GPU memory
    del extractor
    torch.cuda.empty_cache()

    print(f"    Extracted shape: {acts.shape}")
    return acts, np.array(rep_labels), example_ids


def train_probe_transfer(
    source_model: str,
    target_model: str,
    dataset_key: str = "medium_temporal",
    train_layer: int = 8,
    test_layer: int = 8,
    max_examples: int = 30,
    backend: str = "pytorch",
) -> ProbeTransferResult:
    """
    Train a degradation probe on source model, test on target model.
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required")

    dataset = load_benchmark_dataset(dataset_key, max_examples)
    rep_counts = [1, 8, 15]

    print(f"\n  Training probe on {source_model} L{train_layer}, testing on {target_model} L{test_layer}")

    # Extract activations from source
    source_acts, source_reps, _ = extract_activations_at_layer(
        source_model, train_layer, dataset["examples"], rep_counts, backend=backend
    )

    # Extract activations from target
    target_acts, target_reps, _ = extract_activations_at_layer(
        target_model, test_layer, dataset["examples"], rep_counts, backend=backend
    )

    # Binary classification: rep <= 1 (fresh) vs rep >= 8 (degraded)
    source_y = (source_reps >= 8).astype(int)
    target_y = (target_reps >= 8).astype(int)

    # Train on source
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(source_acts, source_y)

    train_acc = float(probe.score(source_acts, source_y))
    test_acc = float(probe.score(target_acts, target_y))

    result = ProbeTransferResult(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset_key,
        train_layer=train_layer,
        test_layer=test_layer,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_examples=len(dataset["examples"]),
    )

    print(f"    Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    return result


# ---------------------------------------------------------------------------
# Direction injection transfer
# ---------------------------------------------------------------------------

def _find_residual_module(model: nn.Module, layer: int) -> Optional[nn.Module]:
    """Find the residual stream output module for a given layer.

    Supports common HuggingFace architectures:
      - LlamaForCausalLM: model.layers[layer]
      - Qwen2ForCausalLM: model.layers[layer]
      - GPTNeoXForCausalLM: model.gpt_neox.layers[layer]
      - DeepSeekForCausalLM: model.layers[layer]
    """
    # Try common paths
    candidates = [
        # Llama, Qwen, Mistral, DeepSeek
        lambda: model.model.layers[layer],
        # GPT-NeoX
        lambda: model.gpt_neox.layers[layer],
        # GPT-2
        lambda: model.transformer.h[layer],
    ]

    for candidate in candidates:
        try:
            module = candidate()
            if module is not None:
                return module
        except (AttributeError, IndexError):
            continue

    # Fallback: walk named_modules
    target_names = [
        f"model.layers.{layer}",
        f"gpt_neox.layers.{layer}",
        f"transformer.h.{layer}",
    ]
    for name, module in model.named_modules():
        if name in target_names:
            return module

    return None


def inject_direction_into_model(
    source_model: str,
    target_model: str,
    layer: int,
    dataset_key: str = "medium_temporal",
    max_examples: int = 30,
    injection_strength: float = 1.0,
) -> DirectionInjectionResult:
    """
    Load direction from source_model, inject into target_model, measure impact.
    Uses forward hooks to add the direction to activations at the target layer.
    """
    if not HAS_TORCH:
        raise ImportError("torch required")

    direction = load_direction(source_model, layer, dataset_key)
    if direction is None:
        raise FileNotFoundError(f"Direction not found for {source_model} L{layer}")

    dataset = load_benchmark_dataset(dataset_key, max_examples)
    examples = dataset["examples"][:max_examples]

    print(f"\n  Injecting {source_model} direction into {target_model} L{layer}")
    print(f"    Direction norm: {np.linalg.norm(direction):.4f}")
    print(f"    Testing on {len(examples)} examples")

    # Load target model and tokenizer
    model_name = MODEL_CONFIGS[target_model]["hf_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = "cuda"

    try:
        # Measure baseline accuracy (no hook)
        correct_baseline = 0
        for ex in examples:
            prompt = build_prompt(ex, rep_count=1)  # Fresh (rep=1)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=2048).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Extract first letter
            pred = ""
            for ch in response:
                if ch.isalpha():
                    pred = ch.upper()
                    break

            gt = extract_answer(ex)
            if pred == gt:
                correct_baseline += 1

        baseline_accuracy = correct_baseline / len(examples) if examples else 0.0

        # Normalize direction
        direction_tensor = torch.tensor(
            direction / (np.linalg.norm(direction) + 1e-8),
            dtype=torch.float32,
            device=device,
        )

        # Find residual module to hook
        hook_module = _find_residual_module(model, layer)
        if hook_module is None:
            print(f"    WARNING: Could not find residual module for layer {layer}")
            # Return neutral result
            result = DirectionInjectionResult(
                source_model=source_model,
                target_model=target_model,
                layer=layer,
                dataset=dataset_key,
                baseline_accuracy=baseline_accuracy,
                injected_accuracy=baseline_accuracy,
                accuracy_delta=0.0,
                injection_strength=injection_strength,
                n_examples=len(examples),
            )
            return result

        # Measure accuracy WITH direction injection
        correct_injected = 0

        def hook_fn(module, input, output):
            """Hook to inject the direction."""
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Add injection_strength * direction to last token position
            hidden[:, -1, :] = hidden[:, -1, :] + injection_strength * direction_tensor
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        for ex in examples:
            prompt = build_prompt(ex, rep_count=1)  # Fresh (rep=1)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=2048).to(device)

            # Register hook
            handle = hook_module.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                handle.remove()

            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Extract first letter
            pred = ""
            for ch in response:
                if ch.isalpha():
                    pred = ch.upper()
                    break

            gt = extract_answer(ex)
            if pred == gt:
                correct_injected += 1

        injected_accuracy = correct_injected / len(examples) if examples else 0.0

        result = DirectionInjectionResult(
            source_model=source_model,
            target_model=target_model,
            layer=layer,
            dataset=dataset_key,
            baseline_accuracy=baseline_accuracy,
            injected_accuracy=injected_accuracy,
            accuracy_delta=injected_accuracy - baseline_accuracy,
            injection_strength=injection_strength,
            n_examples=len(examples),
        )

        print(f"    Baseline: {baseline_accuracy:.4f}, Injected: {injected_accuracy:.4f}, "
              f"Δ: {result.accuracy_delta:+.4f}")
        return result

    finally:
        # Clean up
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Base vs Instruct comparison
# ---------------------------------------------------------------------------

def compare_base_vs_instruct(
    dataset_key: str = "medium_temporal",
) -> dict:
    """
    Compare degradation direction in base Llama vs Instruct.
    If direction is absent in base, it's RLHF-specific.
    """
    print("\n== Base vs Instruct Comparison ==")

    base_model = "Llama-3.1-8B"
    instruct_model = "Llama-3.1-8B-Instruct"

    results = {}
    base_config = MODEL_CONFIGS[base_model]
    instruct_config = MODEL_CONFIGS[instruct_model]

    # Compare at middle layer
    layer = base_config["layers"][len(base_config["layers"]) // 2]

    base_dir = load_direction(base_model, layer, dataset_key)
    instruct_dir = load_direction(instruct_model, layer, dataset_key)

    if base_dir is not None and instruct_dir is not None:
        base_norm = np.linalg.norm(base_dir)
        instruct_norm = np.linalg.norm(instruct_dir)

        base_dir_normalized = base_dir / (base_norm + 1e-8)
        instruct_dir_normalized = instruct_dir / (instruct_norm + 1e-8)

        cosine_sim = float(np.dot(base_dir_normalized, instruct_dir_normalized))

        results = {
            "base_direction_norm": float(base_norm),
            "instruct_direction_norm": float(instruct_norm),
            "cosine_similarity": cosine_sim,
            "layer": layer,
            "interpretation": "RLHF-specific" if cosine_sim < 0.4 else "Pre-existing in base",
        }

        print(f"  Base norm: {base_norm:.4f}, Instruct norm: {instruct_norm:.4f}")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Interpretation: {results['interpretation']}")
    else:
        print(f"  WARNING: Missing directions (base={base_dir is not None}, instruct={instruct_dir is not None})")

    return results


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_similarity_matrix(similarity_matrix: np.ndarray, models: list[str],
                          output_path: Path) -> None:
    """Plot heatmap of cross-model cosine similarities."""
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available for plotting {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0.5, vmin=-1, vmax=1, xticklabels=models, yticklabels=models,
                ax=ax, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Cross-Model Degradation Direction Similarity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_probe_transfer_matrix(transfer_results: list[ProbeTransferResult],
                               models: list[str],
                               output_path: Path) -> None:
    """Plot heatmap of cross-model probe transfer accuracy."""
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available for plotting {output_path}")
        return

    n = len(models)
    matrix = np.full((n, n), np.nan)

    for result in transfer_results:
        i = models.index(result.source_model)
        j = models.index(result.target_model)
        matrix[i, j] = result.test_accuracy

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=models, yticklabels=models, ax=ax,
                cbar_kws={"label": "Transfer Accuracy"})
    ax.set_title("Cross-Model Probe Transfer (Train on Row, Test on Column)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Model (Test)")
    ax.set_ylabel("Source Model (Train)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-model transfer of degradation direction")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (subset of models/layers)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all pairwise comparisons")
    parser.add_argument("--model-pair", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="Compare specific model pair")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run all pairwise comparisons")
    parser.add_argument("--base-vs-instruct", action="store_true",
                        help="Compare Llama base vs instruct")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--datasets", nargs="+", default=["medium_temporal"],
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--direction-dir", type=Path, default=DIRECTIONS_DIR)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PHASE 3, EXPERIMENT 4: Cross-Model Transfer of Degradation Direction")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")

    if not HAS_TORCH:
        print("ERROR: torch not available")
        return 1
    if not HAS_SKLEARN:
        print("ERROR: sklearn not available")
        return 1

    # Determine which models to use
    if args.quick:
        models_to_use = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]
        layers_per_model = {
            m: MODEL_CONFIGS[m]["quick_layers"] for m in models_to_use
        }
    else:
        models_to_use = list(MODEL_CONFIGS.keys())
        layers_per_model = {
            m: MODEL_CONFIGS[m]["layers"] for m in models_to_use
        }

    print(f"\nModels: {models_to_use}")

    # Initialize wandb if requested
    if args.wandb_project and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args))

    all_results = {
        "cosine_similarity": [],
        "probe_transfer": [],
        "direction_injection": [],
        "base_vs_instruct": {},
        "config": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # ===== Cosine similarity analysis =====
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 80)

    for dataset_key in args.datasets:
        print(f"\nDataset: {dataset_key}")
        sim_data, sim_matrix = compute_cosine_similarity_matrix(
            models_to_use, layers_per_model, dataset_key
        )
        all_results["cosine_similarity"].append(sim_data)

        # Plot similarity matrix
        models_same_dim = [m for m in models_to_use if MODEL_CONFIGS[m]["d_model"] >= 4096]
        if len(models_same_dim) > 1:
            plot_path = args.output_dir / f"direction_similarity_matrix_{dataset_key}.png"
            plot_similarity_matrix(sim_matrix, models_same_dim, plot_path)

    # ===== Probe transfer analysis =====
    if not args.quick:
        print("\n" + "=" * 80)
        print("PROBE TRANSFER ANALYSIS")
        print("=" * 80)

        transfer_results = []
        models_same_dim = [m for m in models_to_use if MODEL_CONFIGS[m]["d_model"] >= 4096]

        for dataset_key in args.datasets:
            print(f"\nDataset: {dataset_key}")
            for model_a in models_same_dim:
                for model_b in models_same_dim:
                    if model_a == model_b:
                        continue
                    try:
                        result = train_probe_transfer(
                            model_a, model_b, dataset_key,
                            train_layer=layers_per_model[model_a][len(layers_per_model[model_a]) // 2],
                            test_layer=layers_per_model[model_b][len(layers_per_model[model_b]) // 2],
                            max_examples=30 if args.quick else 50,
                            backend=args.backend,
                        )
                        transfer_results.append(result)
                        all_results["probe_transfer"].append(asdict(result))
                    except Exception as e:
                        print(f"    ERROR: {e}")

        # Plot transfer matrix
        if transfer_results and HAS_MPL:
            plot_path = args.output_dir / f"probe_transfer_matrix_{dataset_key}.png"
            plot_probe_transfer_matrix(transfer_results, models_same_dim, plot_path)

    # ===== Direction injection transfer =====
    print("\n" + "=" * 80)
    print("DIRECTION INJECTION TRANSFER")
    print("=" * 80)

    injection_results = []
    for dataset_key in args.datasets:
        print(f"\nDataset: {dataset_key}")
        models_same_dim = [m for m in models_to_use if MODEL_CONFIGS[m]["d_model"] >= 4096]

        for source in models_same_dim[:2]:  # Limit pairs to avoid explosion
            for target in models_same_dim[:2]:
                if source == target:
                    continue
                try:
                    result = inject_direction_into_model(
                        source, target,
                        layer=layers_per_model[source][len(layers_per_model[source]) // 2],
                        dataset_key=dataset_key,
                        max_examples=30 if args.quick else 50,
                    )
                    injection_results.append(result)
                    all_results["direction_injection"].append(asdict(result))
                except Exception as e:
                    print(f"    ERROR: {e}")

    # ===== Base vs Instruct =====
    if not args.quick:
        print("\n" + "=" * 80)
        print("BASE VS INSTRUCT ANALYSIS")
        print("=" * 80)
        all_results["base_vs_instruct"] = compare_base_vs_instruct(args.datasets[0])

    # ===== Save results =====
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = args.output_dir / "cross_model_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {results_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Cosine similarity results: {len(all_results['cosine_similarity'])} dataset(s)")
    print(f"Probe transfer results: {len(all_results['probe_transfer'])} comparisons")
    print(f"Direction injection results: {len(all_results['direction_injection'])} comparisons")

    if args.wandb_project and HAS_WANDB:
        wandb.log({"n_cosine_pairs": len(all_results['cosine_similarity']),
                   "n_probe_transfers": len(all_results['probe_transfer']),
                   "n_injections": len(all_results['direction_injection'])})
        wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
