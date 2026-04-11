#!/usr/bin/env python3
"""
Phase 3, Experiment 3: Causal Activation Patching

Tests whether the degradation direction identified in Experiments 1–2 is
*causally* responsible for performance degradation under repetitive tasks.

Method:
  1. Load pre-extracted degradation direction vectors from Experiment 1
     (saved as .npy files in results/phase3_refusal_direction/{model}/directions/)
  2. At EARLY repetitions (model still performing well):
     - INJECT the degradation direction → measure accuracy DROP
     - This tests: does adding the degradation direction *cause* worse performance?
  3. At LATE repetitions (model already degraded):
     - ABLATE the degradation direction → measure accuracy RECOVERY
     - This tests: does removing the degradation direction *restore* performance?
  4. Controls:
     - RANDOM direction injection/ablation (same norm): should NOT affect accuracy
     - REFUSAL direction injection/ablation: should NOT systematically degrade task
       performance (if refusal ≠ degradation)
  5. Strength sweep: test multiple injection magnitudes to find dose-response curve

Key hypothesis:
  - If injecting degradation direction causes accuracy drop AND ablating it causes
    recovery → the direction is causally linked to degradation (not just correlational)
  - If random direction has NO effect → the result is direction-specific
  - If refusal direction has NO effect → degradation ≠ refusal (consistent with Exp 1)

Causal patching follows Meng et al. (2022) "Locating and Editing Factual
Associations" but applied to *behavioral state* rather than factual knowledge.

Related work:
  - Meng et al. (2022): ROME — causal tracing for factual knowledge
  - Arditi et al. (2024): Refusal direction ablation restores harmful compliance
  - Turner et al. (2023): Activation addition for steering behavior
  - Li et al. (2023): Inference-time intervention for truthfulness

Target models: Final 4 from Phase 2.

Usage:
    # Quick validation (1 model, 1 layer, 1 dataset)
    python scripts/experiments/phase3_causal_patching.py --quick

    # Single model
    python scripts/experiments/phase3_causal_patching.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    # All models
    python scripts/experiments/phase3_causal_patching.py \\
        --all-models --device cuda

    # Sherlock SLURM
    sbatch scripts/experiments/submit_phase3_patching.sh

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
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
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
DIRECTIONS_DIR = PROJECT_ROOT / "results" / "phase3_refusal_direction"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_causal_patching"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult
from src.inference.interventions.intervention import (
    Intervention, InterventionTarget, create_intervention_hook,
)
from src.inference.interventions.intervention_factory import steering


# ---------------------------------------------------------------------------
# Model configs (must match Experiment 1)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
        "chat_template": "llama",
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
        "quick_layers": [8, 16, 28],
        "n_layers": 36,
        "d_model": 4096,
        "chat_template": "qwen",
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "layers": [0, 6, 12, 18, 24, 30, 36, 42, 47],
        "quick_layers": [12, 24, 36],
        "n_layers": 48,
        "d_model": 4096,
        "chat_template": "qwen",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 27],
        "quick_layers": [8, 16, 24],
        "n_layers": 28,
        "d_model": 3584,
        "chat_template": "deepseek",
    },
    # Base model (no instruction tuning) — tests whether RLHF creates degradation
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
        "chat_template": "none",
    },
}


# ---------------------------------------------------------------------------
# Datasets (matching Experiment 1)
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
class PatchingCondition:
    """Configuration for a single patching condition."""
    name: str               # e.g., "inject_degradation", "ablate_degradation"
    direction_type: str     # "degradation", "refusal", "random"
    operation: str          # "inject" (add direction) or "ablate" (subtract projection)
    strength: float         # scaling factor for injection
    rep_count: int          # repetition count to evaluate at
    layer: int              # which layer to intervene

    def description(self) -> str:
        return f"{self.operation}_{self.direction_type}_L{self.layer}_s{self.strength}_rep{self.rep_count}"


@dataclass
class PatchingResult:
    """Result of one patching experiment."""
    condition: str
    direction_type: str
    operation: str
    layer: int
    strength: float
    rep_count: int
    dataset: str
    model: str
    # Accuracy metrics
    baseline_accuracy: float        # accuracy without intervention
    patched_accuracy: float         # accuracy with intervention
    accuracy_delta: float           # patched - baseline
    n_examples: int
    # Individual predictions
    baseline_correct: Optional[list] = None
    patched_correct: Optional[list] = None
    # Timing
    elapsed_seconds: float = 0.0


@dataclass
class DoseResponsePoint:
    """One point on a dose-response curve."""
    strength: float
    accuracy: float
    accuracy_delta: float
    n_examples: int


# ---------------------------------------------------------------------------
# Dataset loading and prompt building (matching Experiment 1)
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
    # Try to map full-text answer back to option letter
    if "options" in example and isinstance(example["options"], dict):
        for key, val in example["options"].items():
            if val.strip().lower() == answer.lower():
                return key.upper()
    return answer.upper()


# ---------------------------------------------------------------------------
# Direction loading
# ---------------------------------------------------------------------------

def load_direction(model_key: str, layer: int, direction_type: str,
                   dataset_key: str = "medium_temporal",
                   source_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load a pre-extracted direction vector from Experiment 1.

    Looks for .npy files in:
        results/phase3_refusal_direction/{model}/directions/

    Args:
        model_key: Model name.
        layer: Layer index.
        direction_type: "degradation" or "refusal".
        dataset_key: Dataset source for degradation directions.
        source_dir: Override directory to search.

    Returns:
        Direction vector (d_model,) or None if not found.
    """
    if source_dir is None:
        source_dir = DIRECTIONS_DIR / model_key / "directions"

    if direction_type == "degradation":
        path = source_dir / f"degradation_{dataset_key}_layer{layer}.npy"
    elif direction_type == "refusal":
        # Try AdvBench first, fall back to HarmBench
        path = source_dir / f"refusal_advbench_layer{layer}.npy"
        if not path.exists():
            path = source_dir / f"refusal_harmbench_layer{layer}.npy"
    else:
        return None

    if path.exists():
        direction = np.load(path).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        print(f"    Loaded {direction_type} direction: {path.name} "
              f"(shape={direction.shape}, norm_before={norm:.4f})")
        return direction
    else:
        print(f"    WARNING: Direction not found: {path}")
        return None


def generate_random_direction(d_model: int, seed: int = 42) -> np.ndarray:
    """Generate a random unit direction vector for control comparison."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(d_model).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def compute_direction_on_the_fly(
    extractor: ActivationExtractor,
    dataset: dict,
    layer: int,
    low_rep: int = 1,
    high_rep: int = 15,
    max_examples: int = 30,
) -> np.ndarray:
    """Compute degradation direction on-the-fly if no saved directions exist.

    Uses mean-diff method: mean(high_rep) - mean(low_rep).
    Returns normalized direction vector.
    """
    examples = dataset["examples"][:max_examples]

    low_prompts = [build_prompt(ex, low_rep) for ex in examples]
    high_prompts = [build_prompt(ex, high_rep) for ex in examples]

    key = f"resid_post.layer{layer}"

    low_result = extractor.extract(low_prompts, return_tokens=False)
    high_result = extractor.extract(high_prompts, return_tokens=False)

    if key not in low_result.activations or key not in high_result.activations:
        raise ValueError(f"Layer {layer} activations not captured")

    low_acts = low_result.numpy(key)    # (n, d_model)
    high_acts = high_result.numpy(key)  # (n, d_model)

    direction = high_acts.mean(axis=0) - low_acts.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction


# ---------------------------------------------------------------------------
# Core patching logic
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    model: nn.Module,
    tokenizer,
    examples: list[dict],
    rep_count: int,
    device: str = "cuda",
    max_new_tokens: int = 5,
) -> tuple[float, list[bool]]:
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


def evaluate_with_intervention(
    model: nn.Module,
    tokenizer,
    examples: list[dict],
    rep_count: int,
    layer: int,
    direction: np.ndarray,
    operation: str,
    strength: float,
    device: str = "cuda",
    max_new_tokens: int = 5,
) -> tuple[float, list[bool]]:
    """Evaluate accuracy with activation intervention applied.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        examples: List of example dicts.
        rep_count: Repetition count.
        layer: Layer to intervene on.
        direction: Normalized direction vector (d_model,).
        operation: "inject" (add direction) or "ablate" (remove projection).
        strength: Scaling factor.
        device: Compute device.
        max_new_tokens: Max generation length.

    Returns:
        Tuple of (accuracy, list of per-example correct booleans).
    """
    # Determine hook module name
    # HuggingFace models use various naming conventions
    hook_module = _find_residual_module(model, layer)
    if hook_module is None:
        print(f"    WARNING: Could not find residual module for layer {layer}")
        return 0.0, [False] * len(examples)

    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

    correct = []

    for ex in examples:
        prompt = build_prompt(ex, rep_count)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(device)

        # Create the intervention hook
        if operation == "inject":
            # Add scaled direction to activations at last position
            def hook_fn(module, input, output):
                # output shape: (batch, seq_len, d_model)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # Apply to last token position
                hidden[:, -1, :] = hidden[:, -1, :] + strength * direction_tensor
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

        elif operation == "ablate":
            # Remove the component along the direction:
            # h' = h - (h · d̂) * d̂ * strength
            # At strength=1.0, fully removes the projection
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # Project last-position hidden state onto direction
                h_last = hidden[:, -1, :]  # (batch, d_model)
                proj = (h_last * direction_tensor).sum(dim=-1, keepdim=True)
                hidden[:, -1, :] = h_last - strength * proj * direction_tensor
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Register hook, run generation, remove hook
        handle = hook_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        pred = ""
        for ch in response:
            if ch.isalpha():
                pred = ch.upper()
                break

        gt = extract_answer(ex)
        correct.append(pred == gt)

    accuracy = sum(correct) / len(correct) if correct else 0.0
    return accuracy, correct


def _find_residual_module(model: nn.Module, layer: int) -> Optional[nn.Module]:
    """Find the residual stream output module for a given layer.

    Supports common HuggingFace architectures:
      - LlamaForCausalLM: model.layers[layer]
      - Qwen2ForCausalLM: model.layers[layer]
      - GPTNeoXForCausalLM: model.gpt_neox.layers[layer]
      - DeepseekV2ForCausalLM: model.layers[layer]
    """
    # Try common paths
    candidates = [
        # Llama, Qwen, Mistral, DeepSeek
        lambda: model.model.layers[layer],
        # GPT-NeoX
        lambda: model.gpt_neox.layers[layer],
        # GPT-2
        lambda: model.transformer.h[layer],
        # Bloom
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


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_injection_experiment(
    model: nn.Module,
    tokenizer,
    examples: list[dict],
    layer: int,
    direction: np.ndarray,
    direction_type: str,
    strengths: list[float],
    early_rep: int,
    late_rep: int,
    dataset_name: str,
    model_key: str,
    device: str = "cuda",
) -> list[PatchingResult]:
    """Run injection + ablation for one direction at one layer.

    Tests:
      1. INJECT at early rep → should cause degradation
      2. ABLATE at late rep → should cause recovery
      3. Both at multiple strengths → dose-response curve

    Returns:
        List of PatchingResult for all conditions.
    """
    results = []

    # ── Baselines (no intervention) ──────────────────────────────────
    print(f"\n    Baseline evaluation...")
    early_baseline_acc, early_baseline_correct = evaluate_accuracy(
        model, tokenizer, examples, early_rep, device)
    late_baseline_acc, late_baseline_correct = evaluate_accuracy(
        model, tokenizer, examples, late_rep, device)

    print(f"      Early (rep={early_rep}): {early_baseline_acc:.3f}")
    print(f"      Late  (rep={late_rep}):  {late_baseline_acc:.3f}")
    print(f"      Degradation gap: {early_baseline_acc - late_baseline_acc:.3f}")

    # ── Injection at early rep ───────────────────────────────────────
    print(f"\n    Injection at early rep (rep={early_rep})...")
    for strength in strengths:
        t0 = time.time()
        acc, correct = evaluate_with_intervention(
            model, tokenizer, examples, early_rep, layer, direction,
            operation="inject", strength=strength, device=device,
        )
        elapsed = time.time() - t0

        delta = acc - early_baseline_acc
        print(f"      strength={strength:+.2f}: acc={acc:.3f} "
              f"(Δ={delta:+.3f}, baseline={early_baseline_acc:.3f})")

        results.append(PatchingResult(
            condition=f"inject_{direction_type}_L{layer}_s{strength}",
            direction_type=direction_type,
            operation="inject",
            layer=layer,
            strength=strength,
            rep_count=early_rep,
            dataset=dataset_name,
            model=model_key,
            baseline_accuracy=early_baseline_acc,
            patched_accuracy=acc,
            accuracy_delta=delta,
            n_examples=len(examples),
            baseline_correct=early_baseline_correct,
            patched_correct=correct,
            elapsed_seconds=elapsed,
        ))

    # ── Ablation at late rep ─────────────────────────────────────────
    print(f"\n    Ablation at late rep (rep={late_rep})...")
    for strength in strengths:
        t0 = time.time()
        acc, correct = evaluate_with_intervention(
            model, tokenizer, examples, late_rep, layer, direction,
            operation="ablate", strength=strength, device=device,
        )
        elapsed = time.time() - t0

        delta = acc - late_baseline_acc
        print(f"      strength={strength:+.2f}: acc={acc:.3f} "
              f"(Δ={delta:+.3f}, baseline={late_baseline_acc:.3f})")

        results.append(PatchingResult(
            condition=f"ablate_{direction_type}_L{layer}_s{strength}",
            direction_type=direction_type,
            operation="ablate",
            layer=layer,
            strength=strength,
            rep_count=late_rep,
            dataset=dataset_name,
            model=model_key,
            baseline_accuracy=late_baseline_acc,
            patched_accuracy=acc,
            accuracy_delta=delta,
            n_examples=len(examples),
            baseline_correct=late_baseline_correct,
            patched_correct=correct,
            elapsed_seconds=elapsed,
        ))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dose_response(
    results: list[PatchingResult],
    output_path: Path,
    model_name: str,
    layer: int,
):
    """Plot dose-response curves: accuracy delta vs injection strength."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Group by direction type and operation
    for ax, operation, title in [
        (axes[0], "inject", f"Injection at Early Rep (Layer {layer})"),
        (axes[1], "ablate", f"Ablation at Late Rep (Layer {layer})"),
    ]:
        colors = {"degradation": "#e74c3c", "random": "#95a5a6", "refusal": "#3498db"}
        markers = {"degradation": "o", "random": "s", "refusal": "^"}

        for dir_type in ["degradation", "random", "refusal"]:
            subset = [r for r in results
                      if r.operation == operation
                      and r.direction_type == dir_type
                      and r.layer == layer]
            if not subset:
                continue

            strengths = [r.strength for r in subset]
            deltas = [r.accuracy_delta for r in subset]

            ax.plot(strengths, deltas,
                    color=colors.get(dir_type, "#333"),
                    marker=markers.get(dir_type, "o"),
                    label=dir_type.capitalize(),
                    linewidth=2, markersize=8, alpha=0.8)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Strength", fontsize=12)
        ax.set_ylabel("Accuracy Delta", fontsize=12)
        ax.set_title(f"{model_name}: {title}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dose-response plot: {output_path}")


def plot_layer_comparison(
    results: list[PatchingResult],
    output_path: Path,
    model_name: str,
):
    """Plot accuracy delta across layers for degradation direction."""
    if not HAS_MPL:
        return

    # Filter to degradation direction, strength=1.0
    inject_results = [r for r in results
                      if r.direction_type == "degradation"
                      and r.operation == "inject"
                      and abs(r.strength - 1.0) < 0.01]
    ablate_results = [r for r in results
                      if r.direction_type == "degradation"
                      and r.operation == "ablate"
                      and abs(r.strength - 1.0) < 0.01]

    if not inject_results and not ablate_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, subset, title, color in [
        (axes[0], inject_results, "Injection Effect (Early Rep)", "#e74c3c"),
        (axes[1], ablate_results, "Ablation Effect (Late Rep)", "#27ae60"),
    ]:
        if not subset:
            ax.set_visible(False)
            continue

        layers = sorted(set(r.layer for r in subset))
        deltas = [next(r.accuracy_delta for r in subset if r.layer == l) for l in layers]

        ax.bar(range(len(layers)), deltas, color=color, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Accuracy Delta", fontsize=12)
        ax.set_title(f"{model_name}: {title}", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved layer comparison plot: {output_path}")


def plot_recovery_comparison(
    results: list[PatchingResult],
    output_path: Path,
    model_name: str,
):
    """Plot comparison: does ablation recovery match degradation gap?"""
    if not HAS_MPL:
        return

    # Get the best ablation result per layer (highest accuracy delta)
    ablate_deg = [r for r in results
                  if r.direction_type == "degradation" and r.operation == "ablate"]
    if not ablate_deg:
        return

    # Group by layer, pick best strength
    by_layer = defaultdict(list)
    for r in ablate_deg:
        by_layer[r.layer].append(r)

    layers = sorted(by_layer.keys())
    best_recovery = []
    for l in layers:
        best = max(by_layer[l], key=lambda r: r.accuracy_delta)
        best_recovery.append(best)

    # Also get injection baselines to show the degradation gap
    inject_deg = [r for r in results
                  if r.direction_type == "degradation"
                  and r.operation == "inject"
                  and abs(r.strength - 1.0) < 0.01]
    inject_by_layer = {r.layer: r for r in inject_deg}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(layers))
    width = 0.35

    # Degradation gap = early_baseline - late_baseline
    gaps = []
    recoveries = []
    for i, l in enumerate(layers):
        r = best_recovery[i]
        # Degradation gap: how much was lost
        gap = 0.0
        if l in inject_by_layer:
            gap = inject_by_layer[l].baseline_accuracy - r.baseline_accuracy
        gaps.append(gap)
        recoveries.append(r.accuracy_delta)

    ax.bar(x - width/2, gaps, width, label="Degradation Gap", color="#e74c3c", alpha=0.7)
    ax.bar(x + width/2, recoveries, width, label="Ablation Recovery", color="#27ae60", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy Change", fontsize=12)
    ax.set_title(f"{model_name}: Degradation Gap vs Ablation Recovery", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved recovery comparison plot: {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_causal_patching(
    model_key: str,
    layers: list[int],
    device: str = "cuda",
    max_examples: int = 30,
    early_rep: int = 1,
    late_rep: int = 15,
    strengths: Optional[list[float]] = None,
    degradation_datasets: Optional[list[str]] = None,
    direction_source_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    output_dir: Optional[str] = None,
    backend: str = "pytorch",
):
    """Run the full causal patching experiment for one model.

    Steps:
      1. Load model and tokenizer
      2. Load or compute direction vectors
      3. For each layer × dataset:
         a. Run injection experiment (degradation direction at early rep)
         b. Run ablation experiment (degradation direction at late rep)
         c. Run random direction control
         d. Run refusal direction control (if available)
      4. Generate dose-response curves and comparison plots
      5. Save all results
    """
    if strengths is None:
        strengths = [0.5, 1.0, 2.0, 4.0, 8.0]
    if degradation_datasets is None:
        degradation_datasets = ["medium_temporal"]

    model_config = MODEL_CONFIGS[model_key]
    output_path = Path(output_dir or RESULTS_DIR / model_key)
    output_path.mkdir(parents=True, exist_ok=True)
    d_model = model_config["d_model"]

    print(f"\n{'='*70}")
    print(f"Phase 3, Exp 3: Causal Activation Patching — {model_key}")
    print(f"  Layers: {layers}")
    print(f"  Strengths: {strengths}")
    print(f"  Early rep: {early_rep}, Late rep: {late_rep}")
    print(f"  Max examples: {max_examples}")
    print(f"  Datasets: {degradation_datasets}")
    print(f"{'='*70}\n")

    # Initialize W&B
    if wandb_project and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"phase3_patching_{model_key}_{datetime.now():%Y%m%d_%H%M}",
            config={
                "model": model_key,
                "layers": layers,
                "strengths": strengths,
                "early_rep": early_rep,
                "late_rep": late_rep,
                "max_examples": max_examples,
                "experiment": "causal_activation_patching",
            },
        )

    # ── Step 1: Load model ───────────────────────────────────────────
    print("Step 1: Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["hf_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["hf_name"],
        torch_dtype=torch.float16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded {model_key} on {device}")

    # ── Step 2: Load direction vectors ───────────────────────────────
    print("\nStep 2: Loading direction vectors...")

    source_dir = Path(direction_source_dir) if direction_source_dir else None

    # Pre-load all directions we'll need
    degradation_dirs = {}     # {(layer, dataset_key): direction}
    refusal_dirs = {}         # {layer: direction}
    random_dirs = {}          # {layer: direction}

    for layer in layers:
        # Random control direction (same for all datasets)
        random_dirs[layer] = generate_random_direction(d_model, seed=layer + 1000)

        # Refusal direction (optional — for control comparison)
        ref_dir = load_direction(model_key, layer, "refusal",
                                 source_dir=source_dir)
        if ref_dir is not None:
            refusal_dirs[layer] = ref_dir

        # Degradation directions (one per dataset)
        for ds_key in degradation_datasets:
            deg_dir = load_direction(model_key, layer, "degradation",
                                     dataset_key=ds_key, source_dir=source_dir)
            if deg_dir is not None:
                degradation_dirs[(layer, ds_key)] = deg_dir

    # If no saved directions, compute on-the-fly using the activation extractor
    missing_dirs = [(layer, ds_key) for layer in layers
                    for ds_key in degradation_datasets
                    if (layer, ds_key) not in degradation_dirs]

    if missing_dirs:
        print(f"\n  Computing {len(missing_dirs)} missing degradation directions on-the-fly...")

        # Resolve backend choice
        use_tl = {"pytorch": False, "transformer_lens": True, "auto": None}[backend]

        config = ExtractionConfig(
            layers=layers,
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
            model=model_config["hf_name"],
            config=config,
            device=device,
        )

        for layer, ds_key in missing_dirs:
            try:
                dataset = load_benchmark_dataset(ds_key, max_examples=30)
                deg_dir = compute_direction_on_the_fly(
                    extractor, dataset, layer,
                    low_rep=1, high_rep=late_rep, max_examples=30,
                )
                degradation_dirs[(layer, ds_key)] = deg_dir
                print(f"    Computed degradation direction: layer={layer}, ds={ds_key}")
            except Exception as e:
                print(f"    ERROR computing direction layer={layer}, ds={ds_key}: {e}")

        # Clean up extractor to free GPU memory before generation
        del extractor
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_loaded = len(degradation_dirs)
    n_refusal = len(refusal_dirs)
    print(f"\n  Directions loaded: {n_loaded} degradation, {n_refusal} refusal, "
          f"{len(random_dirs)} random")

    # ── Step 3: Run patching experiments ─────────────────────────────
    print("\nStep 3: Running patching experiments...")

    all_results = []

    for ds_key in degradation_datasets:
        print(f"\n  Dataset: {ds_key}")
        dataset = load_benchmark_dataset(ds_key, max_examples)
        examples = dataset["examples"][:max_examples]

        for layer in layers:
            print(f"\n  ── Layer {layer} ──────────────────────")

            # (a) Degradation direction — primary test
            if (layer, ds_key) in degradation_dirs:
                print(f"  [Degradation direction]")
                results = run_injection_experiment(
                    model, tokenizer, examples, layer,
                    degradation_dirs[(layer, ds_key)],
                    direction_type="degradation",
                    strengths=strengths,
                    early_rep=early_rep, late_rep=late_rep,
                    dataset_name=ds_key, model_key=model_key,
                    device=device,
                )
                all_results.extend(results)
            else:
                print(f"  [SKIP] No degradation direction for layer {layer}, {ds_key}")

            # (b) Random direction — negative control
            print(f"  [Random direction control]")
            results = run_injection_experiment(
                model, tokenizer, examples, layer,
                random_dirs[layer],
                direction_type="random",
                strengths=[1.0, 4.0],  # Only test a couple strengths for control
                early_rep=early_rep, late_rep=late_rep,
                dataset_name=ds_key, model_key=model_key,
                device=device,
            )
            all_results.extend(results)

            # (c) Refusal direction — control comparison
            if layer in refusal_dirs:
                print(f"  [Refusal direction control]")
                results = run_injection_experiment(
                    model, tokenizer, examples, layer,
                    refusal_dirs[layer],
                    direction_type="refusal",
                    strengths=[1.0, 4.0],
                    early_rep=early_rep, late_rep=late_rep,
                    dataset_name=ds_key, model_key=model_key,
                    device=device,
                )
                all_results.extend(results)

            # Log to W&B
            if wandb_project and HAS_WANDB:
                layer_results = [r for r in all_results if r.layer == layer]
                for r in layer_results:
                    wandb.log({
                        f"{r.condition}/accuracy_delta": r.accuracy_delta,
                        f"{r.condition}/patched_accuracy": r.patched_accuracy,
                        f"{r.condition}/baseline_accuracy": r.baseline_accuracy,
                        "layer": layer,
                        "dataset": ds_key,
                    })

    # ── Step 4: Save results ─────────────────────────────────────────
    print("\nStep 4: Saving results...")

    # Serialize results (strip large lists for JSON)
    serializable_results = []
    for r in all_results:
        d = asdict(r)
        d.pop("baseline_correct", None)
        d.pop("patched_correct", None)
        serializable_results.append(d)

    # Summary statistics
    summary = {
        "model": model_key,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layers": layers,
            "strengths": strengths,
            "early_rep": early_rep,
            "late_rep": late_rep,
            "max_examples": max_examples,
            "datasets": degradation_datasets,
        },
        "n_results": len(all_results),
        "results": serializable_results,
    }

    # Compute key summary metrics
    deg_inject = [r for r in all_results
                  if r.direction_type == "degradation" and r.operation == "inject"]
    deg_ablate = [r for r in all_results
                  if r.direction_type == "degradation" and r.operation == "ablate"]
    rand_inject = [r for r in all_results
                   if r.direction_type == "random" and r.operation == "inject"]
    rand_ablate = [r for r in all_results
                   if r.direction_type == "random" and r.operation == "ablate"]

    if deg_inject:
        summary["mean_injection_delta"] = float(np.mean([r.accuracy_delta for r in deg_inject]))
        summary["max_injection_delta"] = float(min([r.accuracy_delta for r in deg_inject]))
    if deg_ablate:
        summary["mean_ablation_recovery"] = float(np.mean([r.accuracy_delta for r in deg_ablate]))
        summary["max_ablation_recovery"] = float(max([r.accuracy_delta for r in deg_ablate]))
    if rand_inject:
        summary["mean_random_injection_delta"] = float(np.mean([r.accuracy_delta for r in rand_inject]))
    if rand_ablate:
        summary["mean_random_ablation_delta"] = float(np.mean([r.accuracy_delta for r in rand_ablate]))

    # Causal score: does degradation direction have significantly more effect than random?
    if deg_inject and rand_inject:
        deg_effect = abs(np.mean([r.accuracy_delta for r in deg_inject
                                  if abs(r.strength - 1.0) < 0.01]))
        rand_effect = abs(np.mean([r.accuracy_delta for r in rand_inject
                                    if abs(r.strength - 1.0) < 0.01]))
        summary["causal_specificity_injection"] = float(deg_effect - rand_effect)
    if deg_ablate and rand_ablate:
        deg_recovery = np.mean([r.accuracy_delta for r in deg_ablate
                                if abs(r.strength - 1.0) < 0.01])
        rand_recovery = np.mean([r.accuracy_delta for r in rand_ablate
                                  if abs(r.strength - 1.0) < 0.01])
        summary["causal_specificity_ablation"] = float(deg_recovery - rand_recovery)

    results_path = output_path / "causal_patching_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results: {results_path}")

    # ── Step 5: Generate plots ───────────────────────────────────────
    print("\nStep 5: Generating plots...")

    for layer in layers:
        layer_results = [r for r in all_results if r.layer == layer]
        if layer_results:
            plot_dose_response(
                layer_results,
                output_path / f"dose_response_layer{layer}.png",
                model_key, layer,
            )

    plot_layer_comparison(all_results, output_path / "layer_comparison.png", model_key)
    plot_recovery_comparison(all_results, output_path / "recovery_comparison.png", model_key)

    # ── Summary printout ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CAUSAL PATCHING SUMMARY — {model_key}")
    print(f"{'='*70}")
    if "mean_injection_delta" in summary:
        print(f"  Degradation injection (mean Δacc):  {summary['mean_injection_delta']:+.4f}")
    if "mean_ablation_recovery" in summary:
        print(f"  Degradation ablation (mean Δacc):   {summary['mean_ablation_recovery']:+.4f}")
    if "mean_random_injection_delta" in summary:
        print(f"  Random injection (mean Δacc):       {summary['mean_random_injection_delta']:+.4f}")
    if "mean_random_ablation_delta" in summary:
        print(f"  Random ablation (mean Δacc):        {summary['mean_random_ablation_delta']:+.4f}")
    if "causal_specificity_injection" in summary:
        print(f"  Causal specificity (injection):     {summary['causal_specificity_injection']:+.4f}")
    if "causal_specificity_ablation" in summary:
        print(f"  Causal specificity (ablation):      {summary['causal_specificity_ablation']:+.4f}")

    # Interpretation
    if "causal_specificity_injection" in summary:
        spec = summary["causal_specificity_injection"]
        if spec > 0.05:
            print(f"\n  → STRONG causal evidence: degradation direction has {spec:.1%} "
                  f"more effect than random")
        elif spec > 0.02:
            print(f"\n  → MODERATE causal evidence: degradation direction has {spec:.1%} "
                  f"more effect than random")
        else:
            print(f"\n  → WEAK causal evidence: degradation direction only {spec:.1%} "
                  f"more effect than random")

    if wandb_project and HAS_WANDB:
        wandb.log(summary)
        wandb.finish()

    print(f"\n  Output directory: {output_path}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3, Exp 3: Causal Activation Patching")

    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--all-models", action="store_true",
                        help="Run on all 4 target models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--max-examples", type=int, default=30,
                        help="Max examples per condition")
    parser.add_argument("--early-rep", type=int, default=1,
                        help="Early repetition count (baseline)")
    parser.add_argument("--late-rep", type=int, default=15,
                        help="Late repetition count (degraded)")
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 4.0, 8.0],
                        help="Injection/ablation strength values")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["medium_temporal"],
                        choices=list(DATASET_FILES.keys()),
                        help="Datasets for degradation direction")
    parser.add_argument("--direction-dir", type=str, default=None,
                        help="Override directory for pre-extracted directions")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (optional)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation mode (1 model, 1 layer, fewer examples)")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not available. Install with: pip install torch")
        sys.exit(1)
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not available. Install with: pip install scikit-learn")
        sys.exit(1)

    if args.quick:
        print("\n*** QUICK MODE: 1 model, 1 layer, 10 examples, 2 strengths ***\n")
        models = [args.model]
        quick_layers = MODEL_CONFIGS[args.model]["quick_layers"][:1]
        args.max_examples = 10
        args.strengths = [1.0, 4.0]
    elif args.all_models:
        models = list(MODEL_CONFIGS.keys())
        quick_layers = None
    else:
        models = [args.model]
        quick_layers = None

    for model_key in models:
        layers = quick_layers or MODEL_CONFIGS[model_key]["layers"]

        try:
            run_causal_patching(
                model_key=model_key,
                layers=layers,
                device=args.device,
                max_examples=args.max_examples,
                early_rep=args.early_rep,
                late_rep=args.late_rep,
                strengths=args.strengths,
                degradation_datasets=args.datasets,
                direction_source_dir=args.direction_dir,
                wandb_project=args.wandb_project,
                output_dir=args.output_dir,
                backend=args.backend,
            )
        except Exception as e:
            print(f"\nERROR running {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
