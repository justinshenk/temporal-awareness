#!/usr/bin/env python3
"""
Phase 3 Control: Implicit Repetition — Multi-Turn Degradation Without
Explicit Counters

This is the critical ecological validity control for the paper. Our main
experiments use explicit repetition markers ("[This is task N of N]") to
study degradation. A sharp reviewer will ask: "Is the model just reading
the number, or does it develop a genuine fatigue-like internal state?"

This experiment answers that question by testing THREE conditions:

  1. EXPLICIT REPETITION (baseline):
     "[This is task N of N identical tasks in this session.]\n\nQ: ..."
     → The model sees a counter. Degradation could be "reading the label."

  2. IMPLICIT MULTI-TURN REPETITION:
     A multi-turn conversation where the same question is asked N times
     with NO counter, NO metadata, NO indication of how many prior turns
     exist. The model sees only the conversational history.
     → If the model still degrades, it's a genuine sequential state, not
       label-reading.

  3. SHUFFLED IMPLICIT (control for content-specific effects):
     Same as (2) but each turn asks a DIFFERENT question from the same
     pool. Tests whether degradation requires literal repetition of the
     same content, or whether any long conversational sequence triggers it.
     → Distinguishes content-repetition from sequence-length effects.

Key hypothesis:
  - If the degradation direction extracted from condition (1) also
    activates in condition (2): the model has a genuine "I've been doing
    this for a while" representation, not just "I see a big number."
  - If condition (3) also activates: degradation is a general sequence-
    length / conversational-fatigue effect, not specific to repetition.
  - If (2) activates but (3) does NOT: the model specifically tracks
    content repetition, not just conversation length.

Measurements:
  a) Probe transfer: train probe on explicit, test on implicit → accuracy
  b) Direction cosine: cos(explicit_degradation_dir, implicit_degradation_dir)
  c) Behavioral comparison: accuracy, response quality at matched positions
  d) Activation trajectory: does the implicit condition follow the same
     geometric path through activation space?

Related work:
  - In-Context Learning Length Generalization (2024)
  - Lost in the Middle (Liu et al., 2024): position effects in long contexts
  - Anthropic (2024): Alignment Faking — detecting covert internal states
  - Multi-turn dialogue degradation (industry reports)

Target models: All 5 project models.

Usage:
    python scripts/experiments/phase3_implicit_repetition.py --quick
    python scripts/experiments/phase3_implicit_repetition.py \
        --model Llama-3.1-8B-Instruct --device cuda
    python scripts/experiments/phase3_implicit_repetition.py \
        --all-models --device cuda

Author: Adrian Sadik
Date: 2026-04-11
"""

import argparse
import gc
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
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
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import cosine as cosine_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_implicit_repetition"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "n_layers": 32,
        "d_model": 4096,
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "chat_template": "llama3",
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "family": "qwen",
        "n_layers": 36,
        "d_model": 4096,
        "layers": [0, 4, 9, 14, 18, 23, 27, 32, 35],
        "quick_layers": [9, 18, 27],
        "chat_template": "qwen",
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "family": "qwen",
        "n_layers": 48,
        "d_model": 2048,
        "layers": [0, 6, 12, 18, 24, 30, 36, 42, 47],
        "quick_layers": [12, 24, 36],
        "chat_template": "qwen",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen",
        "n_layers": 28,
        "d_model": 3584,
        "layers": [0, 4, 7, 10, 14, 18, 21, 24, 27],
        "quick_layers": [7, 14, 21],
        "chat_template": "deepseek",
    },
    "Ouro-2.6B": {
        "hf_name": "ByteDance/Ouro-2.6B",
        "family": "ouro",
        "n_layers": 24,
        "d_model": 2048,
        "layers": [0, 3, 6, 9, 12, 15, 18, 21, 23],
        "quick_layers": [6, 12, 18],
        "chat_template": "ouro",
    },
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "family": "llama",
        "n_layers": 32,
        "d_model": 4096,
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "quick_layers": [8, 16, 24],
        "chat_template": "none",
    },
}


# ---------------------------------------------------------------------------
# QA question pools
# ---------------------------------------------------------------------------

# Primary pool: questions used for the "same question repeated" condition
QA_PRIMARY = [
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "How many sides does a hexagon have?", "answer": "6"},
    {"question": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "In what year did World War II end?", "answer": "1945"},
    {"question": "What is the largest organ in the human body?", "answer": "Skin"},
    {"question": "What is the boiling point of water in Celsius?", "answer": "100"},
]

# Secondary pool: different questions for the "shuffled" condition
QA_SECONDARY = [
    {"question": "What is the chemical formula for water?", "answer": "H2O"},
    {"question": "How many bones are in the adult human body?", "answer": "206"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "Who wrote the theory of relativity?", "answer": "Einstein"},
    {"question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
    {"question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"question": "How many minutes are in an hour?", "answer": "60"},
    {"question": "What is the atomic number of carbon?", "answer": "6"},
    {"question": "What continent is Brazil in?", "answer": "South America"},
    {"question": "What is the speed of light in m/s approximately?", "answer": "300000000"},
    {"question": "What gas makes up most of Earth's atmosphere?", "answer": "Nitrogen"},
    {"question": "How many legs does a spider have?", "answer": "8"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "What is the tallest mountain on Earth?", "answer": "Everest"},
    {"question": "How many planets are in our solar system?", "answer": "8"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What organ pumps blood in the body?", "answer": "Heart"},
    {"question": "What is the currency of Japan?", "answer": "Yen"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What is the chemical symbol for sodium?", "answer": "Na"},
]


# ---------------------------------------------------------------------------
# Prompt construction: three conditions
# ---------------------------------------------------------------------------

def format_chat_prompt(text: str, template: str) -> str:
    """Format a single-turn prompt."""
    if template == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif template == "qwen":
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif template == "deepseek":
        return f"User: {text}\n\nAssistant:"
    else:
        return text


def build_explicit_prompt(question: str, rep_count: int, template: str) -> str:
    """Condition 1: Explicit repetition counter (standard approach)."""
    if rep_count <= 1:
        text = f"{question} Answer with just the answer, no explanation."
    else:
        text = (
            f"[This is task {rep_count} of {rep_count} identical tasks "
            f"in this session.]\n\n"
            f"{question} Answer with just the answer, no explanation."
        )
    return format_chat_prompt(text, template)


def build_implicit_multiturn(
    question: str,
    answer: str,
    n_turns: int,
    template: str,
) -> str:
    """Condition 2: Multi-turn conversation with the SAME question repeated.

    Builds a realistic multi-turn dialogue where the user asks the same
    question N times and the assistant provides the correct answer each
    time. The final turn is the one we extract activations from.

    No counters, no metadata — just a natural conversation history.
    """
    instruction = f"{question} Answer with just the answer, no explanation."

    if template == "llama3":
        prompt = "<|begin_of_text|>"
        for i in range(n_turns - 1):
            prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{instruction}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{answer}<|eot_id|>"
            )
        # Final turn — no assistant response yet
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    elif template == "qwen":
        prompt = ""
        for i in range(n_turns - 1):
            prompt += (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>\n"
            )
        prompt += (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    elif template == "deepseek":
        prompt = ""
        for i in range(n_turns - 1):
            prompt += f"User: {instruction}\n\nAssistant: {answer}\n\n"
        prompt += f"User: {instruction}\n\nAssistant:"
        return prompt

    else:
        # Base model: no chat template, just raw turns
        prompt = ""
        for i in range(n_turns - 1):
            prompt += f"Q: {instruction}\nA: {answer}\n\n"
        prompt += f"Q: {instruction}\nA:"
        return prompt


def build_shuffled_multiturn(
    final_question: str,
    final_answer: str,
    n_turns: int,
    template: str,
    rng: random.Random,
) -> str:
    """Condition 3: Multi-turn conversation with DIFFERENT questions each turn.

    Same structure as implicit multi-turn, but each prior turn uses a
    different question from the secondary pool. The final turn uses the
    target question. Controls for sequence length vs content repetition.
    """
    # Sample n_turns-1 distinct questions from secondary pool
    prior_questions = rng.sample(
        QA_SECONDARY, min(n_turns - 1, len(QA_SECONDARY))
    )
    # If we need more turns than the pool, cycle with replacement
    while len(prior_questions) < n_turns - 1:
        prior_questions.append(rng.choice(QA_SECONDARY))

    final_instruction = f"{final_question} Answer with just the answer, no explanation."

    if template == "llama3":
        prompt = "<|begin_of_text|>"
        for pq in prior_questions:
            q = f"{pq['question']} Answer with just the answer, no explanation."
            prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{q}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{pq['answer']}<|eot_id|>"
            )
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{final_instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    elif template == "qwen":
        prompt = ""
        for pq in prior_questions:
            q = f"{pq['question']} Answer with just the answer, no explanation."
            prompt += (
                f"<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n{pq['answer']}<|im_end|>\n"
            )
        prompt += (
            f"<|im_start|>user\n{final_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    elif template == "deepseek":
        prompt = ""
        for pq in prior_questions:
            q = f"{pq['question']} Answer with just the answer, no explanation."
            prompt += f"User: {q}\n\nAssistant: {pq['answer']}\n\n"
        prompt += f"User: {final_instruction}\n\nAssistant:"
        return prompt

    else:
        prompt = ""
        for pq in prior_questions:
            q = f"{pq['question']} Answer with just the answer, no explanation."
            prompt += f"Q: {q}\nA: {pq['answer']}\n\n"
        prompt += f"Q: {final_instruction}\nA:"
        return prompt


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def extract_contrastive_direction(
    extractor: "ActivationExtractor",
    positive_prompts: list,
    negative_prompts: list,
    layer: int,
) -> np.ndarray:
    """Mean-diff direction extraction."""
    result_pos = extractor.extract(positive_prompts, return_tokens=False)
    acts_pos = result_pos.numpy(f"resid_post.layer{layer}")

    result_neg = extractor.extract(negative_prompts, return_tokens=False)
    acts_neg = result_neg.numpy(f"resid_post.layer{layer}")

    direction = acts_pos.mean(axis=0) - acts_neg.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
    return direction.astype(np.float32)


def extract_condition_activations(
    extractor: "ActivationExtractor",
    prompts: list,
    layer: int,
) -> np.ndarray:
    """Extract mean-pooled activations for a set of prompts."""
    all_acts = []
    for prompt in prompts:
        result = extractor.extract([prompt], return_tokens=False)
        acts = result.numpy(f"resid_post.layer{layer}")
        all_acts.append(acts.mean(axis=0))
    return np.array(all_acts)


# ---------------------------------------------------------------------------
# Generation & accuracy measurement
# ---------------------------------------------------------------------------

def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate a single completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return completion.strip()


def check_accuracy(completion: str, expected: str) -> bool:
    """Fuzzy accuracy check."""
    comp = completion.lower().strip().rstrip(".")
    exp = expected.lower().strip()
    return exp in comp or comp in exp


# ═══════════════════════════════════════════════════════════════════════════
# Core experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_implicit_repetition_experiment(
    model,
    tokenizer,
    extractor: "ActivationExtractor",
    model_name: str,
    config: dict,
    quick: bool = False,
    backend: str = "pytorch",
) -> dict:
    """Run the full implicit repetition control experiment."""

    print("\n" + "=" * 60)
    print("IMPLICIT REPETITION CONTROL EXPERIMENT")
    print(f"Model: {model_name}")
    print("=" * 60)

    template = config["chat_template"]
    layers = config["quick_layers"] if quick else config["layers"]
    rng = random.Random(42)

    # Turn counts to test (analogous to rep counts)
    turn_counts = [1, 3, 5, 8, 10, 13, 15] if not quick else [1, 5, 10, 15]
    # Cap multi-turn context to avoid OOM on long sequences
    max_turns = 15

    questions = QA_PRIMARY if not quick else QA_PRIMARY[:4]

    results = {
        "direction_comparison": {},
        "probe_transfer": {},
        "behavioral_comparison": {},
        "activation_trajectory": {},
        "condition_separability": {},
    }

    # ──────────────────────────────────────────────────────────
    # Step 1: Extract activations under all three conditions
    # ──────────────────────────────────────────────────────────
    print("\n  Step 1: Extracting activations across conditions...")

    # For each layer, collect activations at each turn count under each condition
    for layer in layers:
        print(f"\n    Layer {layer}...")

        explicit_acts_by_turn = {}
        implicit_acts_by_turn = {}
        shuffled_acts_by_turn = {}

        for n_turns in turn_counts:
            explicit_prompts = []
            implicit_prompts = []
            shuffled_prompts = []

            for qa in questions:
                # Condition 1: explicit
                explicit_prompts.append(
                    build_explicit_prompt(qa["question"], n_turns, template)
                )
                # Condition 2: implicit multi-turn (same question)
                implicit_prompts.append(
                    build_implicit_multiturn(
                        qa["question"], qa["answer"],
                        min(n_turns, max_turns), template
                    )
                )
                # Condition 3: shuffled multi-turn (different questions)
                shuffled_prompts.append(
                    build_shuffled_multiturn(
                        qa["question"], qa["answer"],
                        min(n_turns, max_turns), template, rng
                    )
                )

            # Extract activations
            explicit_acts = extract_condition_activations(
                extractor, explicit_prompts, layer
            )
            implicit_acts = extract_condition_activations(
                extractor, implicit_prompts, layer
            )
            shuffled_acts = extract_condition_activations(
                extractor, shuffled_prompts, layer
            )

            explicit_acts_by_turn[n_turns] = explicit_acts
            implicit_acts_by_turn[n_turns] = implicit_acts
            shuffled_acts_by_turn[n_turns] = shuffled_acts

            print(f"      Turns={n_turns}: extracted {len(questions)} × 3 conditions")

        # ──────────────────────────────────────────────────────
        # Step 2: Direction comparison
        # ──────────────────────────────────────────────────────
        print(f"\n    [Direction comparison] Layer {layer}...")

        # Extract directions: high-turn vs low-turn for each condition
        low_turn = turn_counts[0]
        high_turn = turn_counts[-1]

        # Explicit degradation direction
        explicit_dir = (
            explicit_acts_by_turn[high_turn].mean(axis=0)
            - explicit_acts_by_turn[low_turn].mean(axis=0)
        )
        norm = np.linalg.norm(explicit_dir)
        if norm > 1e-10:
            explicit_dir = explicit_dir / norm

        # Implicit degradation direction
        implicit_dir = (
            implicit_acts_by_turn[high_turn].mean(axis=0)
            - implicit_acts_by_turn[low_turn].mean(axis=0)
        )
        norm = np.linalg.norm(implicit_dir)
        if norm > 1e-10:
            implicit_dir = implicit_dir / norm

        # Shuffled direction
        shuffled_dir = (
            shuffled_acts_by_turn[high_turn].mean(axis=0)
            - shuffled_acts_by_turn[low_turn].mean(axis=0)
        )
        norm = np.linalg.norm(shuffled_dir)
        if norm > 1e-10:
            shuffled_dir = shuffled_dir / norm

        cos_explicit_implicit = float(np.dot(explicit_dir, implicit_dir))
        cos_explicit_shuffled = float(np.dot(explicit_dir, shuffled_dir))
        cos_implicit_shuffled = float(np.dot(implicit_dir, shuffled_dir))

        results["direction_comparison"][layer] = {
            "cos_explicit_implicit": cos_explicit_implicit,
            "cos_explicit_shuffled": cos_explicit_shuffled,
            "cos_implicit_shuffled": cos_implicit_shuffled,
            "interpretation": (
                "GENUINE_FATIGUE" if cos_explicit_implicit > 0.5
                else "LABEL_READING" if cos_explicit_implicit < 0.2
                else "PARTIAL_OVERLAP"
            ),
        }
        print(f"      cos(explicit, implicit) = {cos_explicit_implicit:+.4f}")
        print(f"      cos(explicit, shuffled)  = {cos_explicit_shuffled:+.4f}")
        print(f"      cos(implicit, shuffled)  = {cos_implicit_shuffled:+.4f}")

        # ──────────────────────────────────────────────────────
        # Step 3: Probe transfer (train on explicit, test on implicit)
        # ──────────────────────────────────────────────────────
        print(f"\n    [Probe transfer] Layer {layer}...")

        if HAS_SKLEARN:
            # Training data: explicit condition
            X_train, y_train = [], []
            for n_turns in turn_counts:
                label = 1 if n_turns >= 8 else 0  # degraded vs fresh
                for act in explicit_acts_by_turn[n_turns]:
                    X_train.append(act)
                    y_train.append(label)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Test data: implicit condition
            X_test_implicit, y_test_implicit = [], []
            for n_turns in turn_counts:
                label = 1 if n_turns >= 8 else 0
                for act in implicit_acts_by_turn[n_turns]:
                    X_test_implicit.append(act)
                    y_test_implicit.append(label)
            X_test_implicit = np.array(X_test_implicit)
            y_test_implicit = np.array(y_test_implicit)

            # Test data: shuffled condition
            X_test_shuffled, y_test_shuffled = [], []
            for n_turns in turn_counts:
                label = 1 if n_turns >= 8 else 0
                for act in shuffled_acts_by_turn[n_turns]:
                    X_test_shuffled.append(act)
                    y_test_shuffled.append(label)
            X_test_shuffled = np.array(X_test_shuffled)
            y_test_shuffled = np.array(y_test_shuffled)

            # Train probe
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000, C=1.0)

            # Handle edge case: all same label
            if len(np.unique(y_train)) < 2:
                print(f"      Skipping — only one class in training data")
                results["probe_transfer"][layer] = {"error": "single_class"}
                continue

            clf.fit(X_train_scaled, y_train)

            # Evaluate: train accuracy
            train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))

            # Evaluate: transfer to implicit
            X_imp_scaled = scaler.transform(X_test_implicit)
            implicit_acc = accuracy_score(
                y_test_implicit, clf.predict(X_imp_scaled)
            )

            # Evaluate: transfer to shuffled
            X_shuf_scaled = scaler.transform(X_test_shuffled)
            shuffled_acc = accuracy_score(
                y_test_shuffled, clf.predict(X_shuf_scaled)
            )

            results["probe_transfer"][layer] = {
                "train_accuracy": float(train_acc),
                "implicit_transfer_accuracy": float(implicit_acc),
                "shuffled_transfer_accuracy": float(shuffled_acc),
                "interpretation": (
                    "GENUINE_TRANSFER" if implicit_acc > 0.7
                    else "WEAK_TRANSFER" if implicit_acc > 0.55
                    else "NO_TRANSFER"
                ),
            }
            print(f"      Train (explicit):     {train_acc:.3f}")
            print(f"      Transfer (implicit):  {implicit_acc:.3f}")
            print(f"      Transfer (shuffled):  {shuffled_acc:.3f}")

        # ──────────────────────────────────────────────────────
        # Step 4: Activation trajectory comparison
        # ──────────────────────────────────────────────────────
        print(f"\n    [Trajectory comparison] Layer {layer}...")

        # Compute centroid trajectory for each condition
        explicit_centroids = np.array([
            explicit_acts_by_turn[t].mean(axis=0) for t in turn_counts
        ])
        implicit_centroids = np.array([
            implicit_acts_by_turn[t].mean(axis=0) for t in turn_counts
        ])
        shuffled_centroids = np.array([
            shuffled_acts_by_turn[t].mean(axis=0) for t in turn_counts
        ])

        # Compute pairwise distances between trajectories at each turn count
        trajectory_distances = []
        for i, t in enumerate(turn_counts):
            d_exp_imp = float(np.linalg.norm(
                explicit_centroids[i] - implicit_centroids[i]
            ))
            d_exp_shuf = float(np.linalg.norm(
                explicit_centroids[i] - shuffled_centroids[i]
            ))
            d_imp_shuf = float(np.linalg.norm(
                implicit_centroids[i] - shuffled_centroids[i]
            ))
            trajectory_distances.append({
                "turns": t,
                "explicit_implicit_dist": d_exp_imp,
                "explicit_shuffled_dist": d_exp_shuf,
                "implicit_shuffled_dist": d_imp_shuf,
            })

        # Compute displacement vectors and their cosine similarity
        # (Do explicit and implicit "drift" in the same direction?)
        if len(turn_counts) >= 2:
            exp_drift = explicit_centroids[-1] - explicit_centroids[0]
            imp_drift = implicit_centroids[-1] - implicit_centroids[0]
            shuf_drift = shuffled_centroids[-1] - shuffled_centroids[0]

            drift_cos_exp_imp = float(np.dot(
                exp_drift / max(np.linalg.norm(exp_drift), 1e-10),
                imp_drift / max(np.linalg.norm(imp_drift), 1e-10),
            ))
            drift_cos_exp_shuf = float(np.dot(
                exp_drift / max(np.linalg.norm(exp_drift), 1e-10),
                shuf_drift / max(np.linalg.norm(shuf_drift), 1e-10),
            ))
            drift_magnitude_explicit = float(np.linalg.norm(exp_drift))
            drift_magnitude_implicit = float(np.linalg.norm(imp_drift))
            drift_magnitude_shuffled = float(np.linalg.norm(shuf_drift))
        else:
            drift_cos_exp_imp = 0
            drift_cos_exp_shuf = 0
            drift_magnitude_explicit = 0
            drift_magnitude_implicit = 0
            drift_magnitude_shuffled = 0

        results["activation_trajectory"][layer] = {
            "pairwise_distances": trajectory_distances,
            "drift_cosine_explicit_implicit": drift_cos_exp_imp,
            "drift_cosine_explicit_shuffled": drift_cos_exp_shuf,
            "drift_magnitude_explicit": drift_magnitude_explicit,
            "drift_magnitude_implicit": drift_magnitude_implicit,
            "drift_magnitude_shuffled": drift_magnitude_shuffled,
        }
        print(f"      Drift cos(explicit, implicit) = {drift_cos_exp_imp:+.4f}")
        print(f"      Drift cos(explicit, shuffled)  = {drift_cos_exp_shuf:+.4f}")
        print(f"      Drift magnitudes: explicit={drift_magnitude_explicit:.2f}, "
              f"implicit={drift_magnitude_implicit:.2f}, "
              f"shuffled={drift_magnitude_shuffled:.2f}")

    # ──────────────────────────────────────────────────────
    # Step 5: Behavioral comparison (accuracy across conditions)
    # ──────────────────────────────────────────────────────
    print("\n  Step 5: Behavioral comparison...")

    for n_turns in turn_counts:
        explicit_correct = 0
        implicit_correct = 0
        shuffled_correct = 0
        total = 0

        for qa in questions:
            # Explicit
            prompt = build_explicit_prompt(qa["question"], n_turns, template)
            comp = generate_completion(model, tokenizer, prompt)
            if check_accuracy(comp, qa["answer"]):
                explicit_correct += 1

            # Implicit
            prompt = build_implicit_multiturn(
                qa["question"], qa["answer"],
                min(n_turns, max_turns), template
            )
            comp = generate_completion(model, tokenizer, prompt)
            if check_accuracy(comp, qa["answer"]):
                implicit_correct += 1

            # Shuffled
            prompt = build_shuffled_multiturn(
                qa["question"], qa["answer"],
                min(n_turns, max_turns), template, rng
            )
            comp = generate_completion(model, tokenizer, prompt)
            if check_accuracy(comp, qa["answer"]):
                shuffled_correct += 1

            total += 1

        results["behavioral_comparison"][n_turns] = {
            "explicit_accuracy": explicit_correct / max(total, 1),
            "implicit_accuracy": implicit_correct / max(total, 1),
            "shuffled_accuracy": shuffled_correct / max(total, 1),
            "total": total,
        }
        print(f"    Turns={n_turns:2d}: explicit={explicit_correct}/{total}, "
              f"implicit={implicit_correct}/{total}, "
              f"shuffled={shuffled_correct}/{total}")

    # ──────────────────────────────────────────────────────
    # Step 6: Condition separability (can a probe distinguish conditions?)
    # ──────────────────────────────────────────────────────
    print("\n  Step 6: Condition separability...")

    for layer in (config["quick_layers"] if quick else config["layers"][:5]):
        # Can we tell apart explicit vs implicit at matched degradation levels?
        X_sep, y_sep = [], []
        for n_turns in turn_counts:
            if n_turns in explicit_acts_by_turn and n_turns in implicit_acts_by_turn:
                for act in explicit_acts_by_turn.get(n_turns, []):
                    X_sep.append(act)
                    y_sep.append(0)  # explicit
                for act in implicit_acts_by_turn.get(n_turns, []):
                    X_sep.append(act)
                    y_sep.append(1)  # implicit

        if len(X_sep) > 10 and len(np.unique(y_sep)) == 2 and HAS_SKLEARN:
            X_sep = np.array(X_sep)
            y_sep = np.array(y_sep)
            scaler = StandardScaler()
            X_sep_scaled = scaler.fit_transform(X_sep)
            clf = LogisticRegression(max_iter=1000, C=1.0)

            # 3-fold CV
            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                cv_accs = []
                for train_idx, test_idx in skf.split(X_sep_scaled, y_sep):
                    clf.fit(X_sep_scaled[train_idx], y_sep[train_idx])
                    pred = clf.predict(X_sep_scaled[test_idx])
                    cv_accs.append(accuracy_score(y_sep[test_idx], pred))

                mean_acc = float(np.mean(cv_accs))
                results["condition_separability"][layer] = {
                    "explicit_vs_implicit_accuracy": mean_acc,
                    "cv_accs": [float(a) for a in cv_accs],
                    "interpretation": (
                        "CLEARLY_DISTINCT" if mean_acc > 0.75
                        else "SOMEWHAT_DISTINCT" if mean_acc > 0.6
                        else "OVERLAPPING"
                    ),
                }
                print(f"    Layer {layer}: explicit vs implicit = {mean_acc:.3f} "
                      f"({results['condition_separability'][layer]['interpretation']})")
            except Exception as e:
                results["condition_separability"][layer] = {"error": str(e)}
                print(f"    Layer {layer}: error — {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_direction_comparison(results: dict, model_name: str, save_dir: Path):
    """Plot direction cosine similarities across layers."""
    if not HAS_MPL:
        return

    dc = results.get("direction_comparison", {})
    layers = sorted(dc.keys())
    if not layers:
        return

    cos_ei = [dc[l]["cos_explicit_implicit"] for l in layers]
    cos_es = [dc[l]["cos_explicit_shuffled"] for l in layers]
    cos_is = [dc[l]["cos_implicit_shuffled"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers))
    width = 0.25

    ax.bar(x - width, cos_ei, width, label="Explicit ↔ Implicit",
           color="#1f77b4", alpha=0.85)
    ax.bar(x, cos_es, width, label="Explicit ↔ Shuffled",
           color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, cos_is, width, label="Implicit ↔ Shuffled",
           color="#2ca02c", alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Degradation Direction Comparison — {model_name}\n"
                 f"(High Explicit↔Implicit = genuine fatigue, not label-reading)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Threshold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = save_dir / f"direction_comparison_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_probe_transfer(results: dict, model_name: str, save_dir: Path):
    """Plot probe transfer accuracy across layers."""
    if not HAS_MPL:
        return

    pt = results.get("probe_transfer", {})
    layers = sorted([l for l in pt.keys() if "error" not in pt.get(l, {})])
    if not layers:
        return

    train_acc = [pt[l]["train_accuracy"] for l in layers]
    implicit_acc = [pt[l]["implicit_transfer_accuracy"] for l in layers]
    shuffled_acc = [pt[l]["shuffled_transfer_accuracy"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers))

    ax.plot(x, train_acc, "o-", color="#1f77b4", linewidth=2,
            markersize=8, label="Train (explicit)")
    ax.plot(x, implicit_acc, "s-", color="#d62728", linewidth=2,
            markersize=8, label="Transfer → implicit")
    ax.plot(x, shuffled_acc, "D-", color="#2ca02c", linewidth=2,
            markersize=8, label="Transfer → shuffled")

    ax.fill_between(x, implicit_acc, alpha=0.1, color="#d62728")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.axhline(y=0.7, color="orange", linestyle=":", alpha=0.5, label="Transfer threshold")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Probe Transfer: Explicit → Implicit — {model_name}\n"
                 f"(High implicit accuracy = degradation is genuine, not label-reading)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / f"probe_transfer_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_behavioral_comparison(results: dict, model_name: str, save_dir: Path):
    """Plot accuracy across conditions."""
    if not HAS_MPL:
        return

    bc = results.get("behavioral_comparison", {})
    turns = sorted(bc.keys())
    if not turns:
        return

    explicit_acc = [bc[t]["explicit_accuracy"] for t in turns]
    implicit_acc = [bc[t]["implicit_accuracy"] for t in turns]
    shuffled_acc = [bc[t]["shuffled_accuracy"] for t in turns]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(turns, explicit_acc, "o-", color="#1f77b4", linewidth=2,
            markersize=8, label="Explicit repetition")
    ax.plot(turns, implicit_acc, "s-", color="#d62728", linewidth=2,
            markersize=8, label="Implicit multi-turn")
    ax.plot(turns, shuffled_acc, "D-", color="#2ca02c", linewidth=2,
            markersize=8, label="Shuffled multi-turn")

    ax.set_xlabel("Number of Turns / Repetitions")
    ax.set_ylabel("QA Accuracy")
    ax.set_title(f"Behavioral Degradation: Explicit vs Implicit — {model_name}")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / f"behavioral_comparison_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_trajectory_drift(results: dict, model_name: str, save_dir: Path):
    """Plot activation trajectory drift similarity."""
    if not HAS_MPL:
        return

    traj = results.get("activation_trajectory", {})
    layers = sorted(traj.keys())
    if not layers:
        return

    drift_ei = [traj[l]["drift_cosine_explicit_implicit"] for l in layers]
    drift_es = [traj[l]["drift_cosine_explicit_shuffled"] for l in layers]
    mag_exp = [traj[l]["drift_magnitude_explicit"] for l in layers]
    mag_imp = [traj[l]["drift_magnitude_implicit"] for l in layers]
    mag_shuf = [traj[l]["drift_magnitude_shuffled"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Activation Drift Analysis — {model_name}", fontsize=13)

    # Left: drift direction similarity
    x = np.arange(len(layers))
    ax1.plot(x, drift_ei, "o-", color="#d62728", linewidth=2, markersize=8,
             label="Explicit ↔ Implicit")
    ax1.plot(x, drift_es, "s-", color="#ff7f0e", linewidth=2, markersize=8,
             label="Explicit ↔ Shuffled")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Drift Direction Cosine")
    ax1.set_title("Do conditions drift in the same direction?")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{l}" for l in layers])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: drift magnitudes
    width = 0.25
    ax2.bar(x - width, mag_exp, width, label="Explicit", color="#1f77b4", alpha=0.8)
    ax2.bar(x, mag_imp, width, label="Implicit", color="#d62728", alpha=0.8)
    ax2.bar(x + width, mag_shuf, width, label="Shuffled", color="#2ca02c", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Drift Magnitude (L2)")
    ax2.set_title("How far does each condition drift?")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{l}" for l in layers])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = save_dir / f"trajectory_drift_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Control: Implicit Repetition Experiment"
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="pytorch",
                        choices=["pytorch", "transformerlens", "auto"])
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]

    for model_name in models:
        config = MODEL_CONFIGS[model_name]
        hf_name = config["hf_name"]

        print(f"\n{'#' * 60}")
        print(f"# Model: {model_name} ({hf_name})")
        print(f"{'#' * 60}")

        # W&B
        if HAS_WANDB and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=f"implicit-repetition-{model_name}",
                config={
                    "phase": "3-control",
                    "experiment": "implicit_repetition",
                    "model": model_name,
                    "quick": args.quick,
                    "backend": args.backend,
                },
            )

        # Load model
        use_tl = {"pytorch": False, "transformerlens": True, "auto": None}[args.backend]
        layers = config["quick_layers"] if args.quick else config["layers"]

        ext_config = ExtractionConfig(
            layers=layers,
            positions="last",
            device=args.device,
            use_transformer_lens=use_tl,
        )
        extractor = ActivationExtractor(
            model=hf_name,
            config=ext_config,
        )

        t0 = time.time()

        try:
            results = run_implicit_repetition_experiment(
                model=extractor.model,
                tokenizer=extractor._tokenizer,
                extractor=extractor,
                model_name=model_name,
                config=config,
                quick=args.quick,
                backend=args.backend,
            )

            elapsed = time.time() - t0

            # Save results
            model_dir = RESULTS_DIR / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            json_path = model_dir / "implicit_repetition_results.json"
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "hf_name": hf_name,
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": elapsed,
                        "quick": args.quick,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"\n  Results saved: {json_path}")

            # Generate plots
            print("\n  Generating plots...")
            plot_direction_comparison(results, model_name, model_dir)
            plot_probe_transfer(results, model_name, model_dir)
            plot_behavioral_comparison(results, model_name, model_dir)
            plot_trajectory_drift(results, model_name, model_dir)

            # W&B log
            if HAS_WANDB and args.wandb_project:
                # Log key metrics
                for layer, dc in results.get("direction_comparison", {}).items():
                    wandb.log({
                        f"direction/cos_explicit_implicit/L{layer}": dc["cos_explicit_implicit"],
                        f"direction/cos_explicit_shuffled/L{layer}": dc["cos_explicit_shuffled"],
                    })
                for layer, pt in results.get("probe_transfer", {}).items():
                    if isinstance(pt, dict) and "error" not in pt:
                        wandb.log({
                            f"probe_transfer/implicit_acc/L{layer}": pt["implicit_transfer_accuracy"],
                            f"probe_transfer/shuffled_acc/L{layer}": pt["shuffled_transfer_accuracy"],
                        })
                wandb.log({"elapsed_seconds": elapsed})

        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()

        finally:
            if HAS_WANDB and args.wandb_project:
                wandb.finish()
            del extractor
            gc.collect()
            if HAS_TORCH:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
