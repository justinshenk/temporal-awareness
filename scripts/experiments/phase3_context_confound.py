#!/usr/bin/env python3
"""
Phase 3, Experiment 2: Context Length Confound Control

Disentangles repetitive task degradation from raw context length effects.

Core question: Is the degradation signal we find just "Lost in the Middle"
(Liu et al., 2023) / Context Rot under another name? Or is there something
specific to *repetitive same-type tasks*?

Method:
  1. REPETITIVE condition: 20 repetitions of the same task type (e.g., all
     TRAM arithmetic), matching Phase 1/2 design
  2. SHUFFLED condition: same total token count, but tasks drawn from
     DIFFERENT domains (mix of TRAM, MedQA, and MBPP)
  3. PADDED condition: same total token count, but padded with benign
     filler text (Wikipedia-style passages) before the final task
  4. EXACT_REPEAT condition: the *exact same question* repeated N times,
     matched token count. Addresses Leviathan (2025) which shows exact
     prompt repetition *improves* non-reasoning LLMs. If our probe fires
     on EXACT_REPEAT, the signal may be repetition-general; if it does NOT
     fire, the degradation is specific to *varied same-domain* repetition.
  5. Extract activations at the last token for all four conditions
  6. Train the degradation probe on REPETITIVE (high vs low rep) data
  7. Test transfer to SHUFFLED, PADDED, and EXACT_REPEAT conditions:
     - If probe fires on SHUFFLED: degradation is context-length, not repetition
     - If probe does NOT fire on SHUFFLED: degradation is repetition-specific
     - If probe fires on EXACT_REPEAT: degradation is any-repetition
     - If probe does NOT fire on EXACT_REPEAT: degradation is task-variety-specific

Related work:
  - Lost in the Middle (Liu et al., 2023): >30% perf drop for mid-context info
  - Context Rot (Chroma Research): degradation scales with input tokens
  - Prompt Repetition (Leviathan, 2025): repetition *improves* non-reasoning LLMs

Target models: Final 4 from Phase 2.

Usage:
    python scripts/experiments/phase3_context_confound.py --quick
    python scripts/experiments/phase3_context_confound.py --model Llama-3.1-8B-Instruct
    sbatch scripts/experiments/submit_phase3_confound.sh

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
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_context_confound"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult


# ---------------------------------------------------------------------------
# Model configs (same as Phase 2/3)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
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
    # Base model (no instruction tuning) — tests whether RLHF creates degradation
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "n_layers": 32,
        "d_model": 4096,
    },
}


# ---------------------------------------------------------------------------
# Filler templates
# ---------------------------------------------------------------------------
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

# Benign filler passages for the PADDED condition
FILLER_PASSAGES = [
    "The history of mathematics spans thousands of years, from the earliest counting systems to modern abstract algebra. Ancient civilizations including the Babylonians, Egyptians, and Greeks all made significant contributions to mathematical knowledge. ",
    "Climate science examines the complex interactions between the atmosphere, oceans, land surfaces, and ice sheets. Understanding these interactions is crucial for predicting future climate patterns and developing effective adaptation strategies. ",
    "The development of programming languages has followed a fascinating trajectory, from machine code and assembly language through to modern high-level languages that emphasize developer productivity and code readability. ",
    "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy that can be stored and later released to fuel the organism's activities. This process is fundamental to life on Earth. ",
    "The study of linguistics encompasses the structure, use, and psychology of language. Researchers in this field examine everything from the physical production of speech sounds to the way meaning is constructed in social contexts. ",
    "Ecological systems exhibit complex behaviors that emerge from the interactions of many individual organisms. Understanding these emergent properties requires studying systems at multiple scales, from individual behavior to population dynamics. ",
    "The principles of thermodynamics govern the behavior of energy in physical systems. These fundamental laws have applications ranging from engine design to understanding the heat death of the universe. ",
    "Neuroscience research has revealed that the brain is far more plastic than previously believed. Neural connections can be strengthened, weakened, or reorganized throughout life in response to experience and learning. ",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """Result from one experimental condition."""
    condition: str  # "repetitive", "shuffled", "padded"
    n_samples: int
    mean_token_count: float
    probe_accuracy: float
    probe_f1: float
    mean_confidence: float
    layer: int
    model_name: str


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
# Dataset loading and prompt construction
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "medium_temporal": "medium_stakes_tram_arithmetic.json",
    "medium_code": "medium_stakes_mbpp_code.json",
    "high": "high_stakes_medqa_temporal.json",
}


def load_dataset(stakes_key: str, max_examples: int = None) -> dict:
    """Load a benchmark dataset."""
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


def format_question(example: dict) -> str:
    """Format a single MCQ question (without repetitive prefix)."""
    question = example["question"]
    if "options" in example and example["options"]:
        options = example["options"]
        if isinstance(options, dict):
            opts_str = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        else:
            opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
        question = f"{question}\n\n{opts_str}\n\nAnswer:"
    return question


def build_repetitive_prompt(example: dict, rep_count: int) -> str:
    """Build a prompt with repetitive same-type prefix (standard Phase 1/2 design)."""
    question = format_question(example)
    if rep_count <= 1:
        return question

    prefix_parts = []
    for i in range(rep_count - 1):
        filler = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        prefix_parts.append(filler)

    return "".join(prefix_parts) + question


def build_shuffled_prompt(
    target_example: dict,
    other_datasets: dict[str, dict],
    target_token_count: int,
    tokenizer=None,
) -> str:
    """Build a prompt with diverse-domain prefix matching the target token count.

    Instead of repeating the same task type, we interleave questions from
    different domains to achieve the same total context length.
    """
    target_question = format_question(target_example)

    # Collect questions from other domains
    all_other_questions = []
    for ds_key, ds_data in other_datasets.items():
        for ex in ds_data["examples"]:
            q = format_question(ex)
            all_other_questions.append(f"[{ds_key}] {q} ")

    random.shuffle(all_other_questions)

    # Build prefix by adding diverse questions until we reach target length
    prefix = ""
    for q in all_other_questions:
        candidate = prefix + q
        if tokenizer:
            if len(tokenizer.encode(candidate + target_question)) >= target_token_count:
                break
        elif len(candidate + target_question) >= target_token_count * 4:
            # Rough char-to-token estimate
            break
        prefix = candidate

    return prefix + target_question


def build_padded_prompt(
    example: dict,
    target_token_count: int,
    tokenizer=None,
) -> str:
    """Build a prompt padded with benign filler text to match target token count."""
    question = format_question(example)

    # Build filler prefix
    prefix = ""
    passage_idx = 0
    while True:
        passage = FILLER_PASSAGES[passage_idx % len(FILLER_PASSAGES)]
        candidate = prefix + passage
        if tokenizer:
            if len(tokenizer.encode(candidate + question)) >= target_token_count:
                break
        elif len(candidate + question) >= target_token_count * 4:
            break
        prefix = candidate
        passage_idx += 1

    return prefix + question


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
) -> dict:
    """Train a 2-layer MLP probe to check for nonlinear degradation structure.

    Following "Refusal in LLMs: A Nonlinear Perspective" (2025), if the MLP
    probe substantially outperforms the linear probe, there is nonlinear
    structure in the degradation signal that mean-diff methods miss.
    """
    if not HAS_TORCH:
        return {"mlp_accuracy": None, "note": "torch not available"}

    input_dim = X_train.shape[1]

    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_dim, 2),
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = mlp(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        preds = mlp(X_te).argmax(dim=1).numpy()

    acc = accuracy_score(y_te.numpy(), preds)
    f1 = f1_score(y_te.numpy(), preds, zero_division=0)

    return {
        "mlp_accuracy": float(acc),
        "mlp_f1": float(f1),
    }


def build_exact_repeat_prompt(
    example: dict,
    target_token_count: int,
    tokenizer=None,
) -> str:
    """Build a prompt that repeats the EXACT same question until target token count.

    This addresses Leviathan (2025) "Prompt Repetition Improves Non-Reasoning LLMs"
    which shows that exact prompt repetition helps rather than hurts. If the
    degradation probe fires here, the signal is general repetition; if not,
    the degradation is specific to varied same-domain tasks.
    """
    question = format_question(example)

    # Repeat the exact same question as prefix
    prefix = ""
    repeat_idx = 0
    while True:
        candidate = prefix + question + " "
        if tokenizer:
            if len(tokenizer.encode(candidate + question)) >= target_token_count:
                break
        elif len(candidate + question) >= target_token_count * 4:
            break
        prefix = candidate
        repeat_idx += 1
        # Safety valve to prevent infinite loop
        if repeat_idx > 100:
            break

    return prefix + question


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_context_confound_experiment(
    model_key: str,
    layers: list[int],
    device: str = "cuda",
    max_examples: int = 30,
    rep_count: int = 20,
    primary_dataset: str = "medium_temporal",
    wandb_project: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Run the context length confound control experiment.

    For each example in the primary dataset, constructs four conditions:
    1. REPETITIVE: standard rep_count repetitions of same task type
    2. SHUFFLED: diverse tasks from other domains, matched token count
    3. PADDED: benign filler text, matched token count
    4. EXACT_REPEAT: the exact same question repeated, matched token count
       (addresses Leviathan 2025 — exact repetition helps, so if probe
       fires here, the degradation is not task-variety-specific)

    Trains a degradation probe on REPETITIVE (high vs low rep), then tests
    whether it transfers to SHUFFLED, PADDED, and EXACT_REPEAT conditions.

    Args:
        model_key: Model to test.
        layers: Layer indices.
        device: Compute device.
        max_examples: Examples per condition.
        rep_count: Repetition count for the repetitive condition.
        primary_dataset: Which dataset to use for the target task.
        wandb_project: W&B project for logging.
        output_dir: Results directory.
    """
    model_config = MODEL_CONFIGS[model_key]
    output_dir = Path(output_dir or RESULTS_DIR / model_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Phase 3, Exp 2: Context Length Confound Control — {model_key}")
    print(f"  Layers: {layers}")
    print(f"  Primary dataset: {primary_dataset}")
    print(f"  Rep count: {rep_count}")
    print(f"  Max examples: {max_examples}")
    print(f"{'='*70}\n")

    if wandb_project and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"phase3_confound_{model_key}_{datetime.now():%Y%m%d_%H%M}",
            config={
                "model": model_key,
                "layers": layers,
                "rep_count": rep_count,
                "max_examples": max_examples,
                "experiment": "context_confound_control",
            },
        )

    # ── Step 1: Initialize extractor ─────────────────────────────────
    print("Step 1: Initializing ActivationExtractor...")

    # Resolve backend choice
    use_tl = {"pytorch": False, "transformer_lens": True, "auto": None}[args.backend]

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

    # Get tokenizer for token counting
    tokenizer = extractor._tokenizer

    # ── Step 2: Load datasets ────────────────────────────────────────
    print("\nStep 2: Loading datasets...")

    primary_data = load_dataset(primary_dataset, max_examples=max_examples)
    examples = primary_data["examples"][:max_examples]

    # Load other datasets for the shuffled condition
    other_datasets = {}
    for ds_key in DATASET_FILES:
        if ds_key != primary_dataset:
            try:
                other_datasets[ds_key] = load_dataset(ds_key, max_examples=100)
            except FileNotFoundError:
                print(f"  Warning: {ds_key} not found, skipping for shuffled condition")

    print(f"  Primary: {primary_dataset} ({len(examples)} examples)")
    print(f"  Other datasets for shuffled: {list(other_datasets.keys())}")

    # ── Step 3: Build prompts for all three conditions ───────────────
    print("\nStep 3: Building prompts...")

    # Condition 1: REPETITIVE (standard)
    repetitive_prompts = [build_repetitive_prompt(ex, rep_count) for ex in examples]

    # Condition 1b: FRESH (rep=1, baseline for probe training)
    fresh_prompts = [build_repetitive_prompt(ex, 1) for ex in examples]

    # Measure token counts of repetitive prompts to match
    rep_token_counts = [len(tokenizer.encode(p)) for p in repetitive_prompts]
    mean_rep_tokens = np.mean(rep_token_counts)
    print(f"  Repetitive: mean {mean_rep_tokens:.0f} tokens")

    # Condition 2: SHUFFLED (diverse domains, matched length)
    shuffled_prompts = []
    for i, ex in enumerate(examples):
        target_tokens = rep_token_counts[i]
        prompt = build_shuffled_prompt(ex, other_datasets, target_tokens, tokenizer)
        shuffled_prompts.append(prompt)

    shuf_token_counts = [len(tokenizer.encode(p)) for p in shuffled_prompts]
    print(f"  Shuffled: mean {np.mean(shuf_token_counts):.0f} tokens")

    # Condition 3: PADDED (benign filler, matched length)
    padded_prompts = []
    for i, ex in enumerate(examples):
        target_tokens = rep_token_counts[i]
        prompt = build_padded_prompt(ex, target_tokens, tokenizer)
        padded_prompts.append(prompt)

    pad_token_counts = [len(tokenizer.encode(p)) for p in padded_prompts]
    print(f"  Padded: mean {np.mean(pad_token_counts):.0f} tokens")

    # Condition 4: EXACT_REPEAT (same question repeated, matched length)
    # Addresses Leviathan (2025): exact repetition improves performance
    exact_repeat_prompts = []
    for i, ex in enumerate(examples):
        target_tokens = rep_token_counts[i]
        prompt = build_exact_repeat_prompt(ex, target_tokens, tokenizer)
        exact_repeat_prompts.append(prompt)

    exact_token_counts = [len(tokenizer.encode(p)) for p in exact_repeat_prompts]
    print(f"  Exact repeat: mean {np.mean(exact_token_counts):.0f} tokens")
    print(f"  Fresh (rep=1): mean {np.mean([len(tokenizer.encode(p)) for p in fresh_prompts]):.0f} tokens")

    # ── Step 4: Extract activations ──────────────────────────────────
    print("\nStep 4: Extracting activations...")

    conditions = {
        "fresh": fresh_prompts,
        "repetitive": repetitive_prompts,
        "shuffled": shuffled_prompts,
        "padded": padded_prompts,
        "exact_repeat": exact_repeat_prompts,
    }

    activations = {}
    for cond_name, prompts in conditions.items():
        print(f"\n  Extracting: {cond_name} ({len(prompts)} prompts)")
        t0 = time.time()
        result = extractor.extract(prompts, return_tokens=False)
        elapsed = time.time() - t0
        activations[cond_name] = result
        print(f"    Done in {elapsed:.1f}s, {result.n_samples} samples")

    # ── Step 5: Train degradation probe and test transfer ────────────
    print("\nStep 5: Training degradation probe and testing transfer...")

    all_condition_results = []

    for layer in layers:
        key = f"resid_post.layer{layer}"

        # Check all conditions have this layer
        if not all(key in activations[c].activations for c in conditions):
            print(f"  Layer {layer}: missing activations, skipping")
            continue

        # Training data: FRESH (class 0) vs REPETITIVE (class 1)
        fresh_acts = activations["fresh"].numpy(key)     # (n, d_model)
        rep_acts = activations["repetitive"].numpy(key)  # (n, d_model)

        X_train = np.concatenate([fresh_acts, rep_acts], axis=0)
        y_train = np.concatenate([
            np.zeros(len(fresh_acts)),
            np.ones(len(rep_acts)),
        ])

        # Train degradation probe (linear)
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)

        # Also train MLP probe to check for nonlinear structure
        # (addresses "Refusal in LLMs: A Nonlinear Perspective", 2025)
        n = len(y_train)
        perm = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        mlp_result = train_mlp_probe(
            X_train[perm[:split]], y_train[perm[:split]],
            X_train[perm[split:]], y_train[perm[split:]],
        )
        linear_train_acc = clf.score(X_train[perm[split:]], y_train[perm[split:]])
        mlp_acc = mlp_result.get("mlp_accuracy", 0.0) or 0.0
        nonlinear_gap = mlp_acc - linear_train_acc
        print(f"    L{layer} probe comparison: linear={linear_train_acc:.3f}, "
              f"MLP={mlp_acc:.3f}, gap={nonlinear_gap:+.3f}")

        # Evaluate on each condition
        for cond_name in conditions:
            cond_acts = activations[cond_name].numpy(key)  # (n, d_model)

            # Predict using degradation probe
            y_pred = clf.predict(cond_acts)
            y_proba = clf.predict_proba(cond_acts)[:, 1]  # P(degraded)

            # For fresh and repetitive, we have ground truth
            if cond_name == "fresh":
                y_true = np.zeros(len(cond_acts))
            elif cond_name == "repetitive":
                y_true = np.ones(len(cond_acts))
            else:
                # For shuffled/padded, we measure "degradation detection rate"
                # (what fraction does the probe classify as degraded?)
                y_true = None

            if y_true is not None:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                # Detection rate = fraction classified as "degraded"
                acc = float(np.mean(y_pred))  # = detection rate
                f1 = 0.0  # not applicable

            mean_conf = float(np.mean(y_proba))

            result = ConditionResult(
                condition=cond_name,
                n_samples=len(cond_acts),
                mean_token_count=float(np.mean(
                    [len(tokenizer.encode(p)) for p in conditions[cond_name]]
                )),
                probe_accuracy=float(acc),
                probe_f1=float(f1),
                mean_confidence=float(mean_conf),
                layer=layer,
                model_name=model_key,
            )
            all_condition_results.append(result)

            if cond_name in ("fresh", "repetitive"):
                print(f"    L{layer} {cond_name}: acc={acc:.3f}, "
                      f"mean_conf={mean_conf:.3f}")
            else:
                print(f"    L{layer} {cond_name}: detection_rate={acc:.3f}, "
                      f"mean_conf={mean_conf:.3f}")

            if wandb_project and HAS_WANDB:
                wandb.log({
                    f"confound/{cond_name}/layer{layer}/accuracy": acc,
                    f"confound/{cond_name}/layer{layer}/mean_confidence": mean_conf,
                })

    # ── Step 6: Save results and plots ───────────────────────────────
    print("\nStep 6: Saving results...")

    results_data = {
        "model": model_key,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layers": layers,
            "rep_count": rep_count,
            "max_examples": max_examples,
            "primary_dataset": primary_dataset,
        },
        "condition_results": [asdict(r) for r in all_condition_results],
        "token_count_summary": {
            "fresh": float(np.mean([len(tokenizer.encode(p)) for p in fresh_prompts])),
            "repetitive": float(mean_rep_tokens),
            "shuffled": float(np.mean(shuf_token_counts)),
            "padded": float(np.mean(pad_token_counts)),
        },
    }

    results_path = output_dir / "context_confound_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved results: {results_path}")

    # Plot
    if HAS_MPL and all_condition_results:
        plot_confound_results(all_condition_results, output_dir, model_key)

    if wandb_project and HAS_WANDB:
        wandb.finish()

    print(f"\n✓ Phase 3 Exp 2 complete for {model_key}")
    return results_data


def plot_confound_results(
    results: list[ConditionResult],
    output_dir: Path,
    model_name: str,
):
    """Plot degradation probe detection rates across conditions and layers."""
    if not HAS_MPL:
        return

    layers = sorted(set(r.layer for r in results))
    conditions = ["fresh", "repetitive", "shuffled", "padded"]
    colors = {"fresh": "#27ae60", "repetitive": "#e74c3c",
              "shuffled": "#f39c12", "padded": "#3498db"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Probe confidence (P(degraded)) by condition and layer
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.2
    for i, cond in enumerate(conditions):
        confs = []
        for layer in layers:
            r = next((r for r in results if r.layer == layer and r.condition == cond), None)
            confs.append(r.mean_confidence if r else 0)
        ax.bar(x + i * width, confs, width, label=cond.capitalize(),
               color=colors[cond], alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(degraded)")
    ax.set_title(f"{model_name}: Degradation Probe Confidence by Condition")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.2, axis="y")

    # Plot 2: Detection rate comparison (key result)
    ax = axes[1]
    # For each layer, show the gap between repetitive and shuffled/padded
    rep_rates = []
    shuf_rates = []
    pad_rates = []
    for layer in layers:
        r_rep = next((r for r in results if r.layer == layer and r.condition == "repetitive"), None)
        r_shuf = next((r for r in results if r.layer == layer and r.condition == "shuffled"), None)
        r_pad = next((r for r in results if r.layer == layer and r.condition == "padded"), None)
        rep_rates.append(r_rep.mean_confidence if r_rep else 0)
        shuf_rates.append(r_shuf.mean_confidence if r_shuf else 0)
        pad_rates.append(r_pad.mean_confidence if r_pad else 0)

    ax.plot(layers, rep_rates, "o-", color=colors["repetitive"], linewidth=2,
            markersize=8, label="Repetitive (same-type)")
    ax.plot(layers, shuf_rates, "s-", color=colors["shuffled"], linewidth=2,
            markersize=8, label="Shuffled (diverse)")
    ax.plot(layers, pad_rates, "^-", color=colors["padded"], linewidth=2,
            markersize=8, label="Padded (benign filler)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(degraded)")
    ax.set_title(f"{model_name}: Repetition-Specific vs Context-Length Signal")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "context_confound_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: Context Length Confound Control"
    )
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--rep-count", type=int, default=20)
    parser.add_argument("--primary-dataset", type=str, default="medium_temporal")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")
    return parser.parse_args()


def main():
    args = parse_args()
    models = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_key in models:
        config = MODEL_CONFIGS[model_key]
        layers = config["quick_layers"] if args.quick else config["layers"]
        max_ex = 15 if args.quick else args.max_examples

        run_context_confound_experiment(
            model_key=model_key,
            layers=layers,
            device=args.device,
            max_examples=max_ex,
            rep_count=args.rep_count,
            primary_dataset=args.primary_dataset,
            wandb_project=args.wandb_project,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
