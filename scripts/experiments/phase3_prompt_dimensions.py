#!/usr/bin/env python3
"""
Phase 3: Prompt Dimension Cartography — Mapping Behavioral Variation in
Activation Space Beyond Repetition

===========================================================================

This experiment systematically varies prompt characteristics along multiple
dimensions and measures how each pushes model activations relative to known
behavioral directions (degradation, refusal, sycophancy). The goal is to
map the "behavioral geometry" of the residual stream — showing that the
degradation subspace we identified is part of a richer manifold of prompt-
induced behavioral variation, with repetition being the strongest driver.

Motivation (Justin Shenk / TunaPrompt):
    "What aspects of prompts influence Claude's behavior in what ways?"
    Rather than only showing repetition → degradation, we map major
    dimensions along which model behavior subtly varies based on prompt
    structure, quantify their correlation, and identify which prompt
    features share representational subspace with degradation.

Prompt Dimensions Tested:
    1. AUTHORITY — casual request vs institutional/professional framing
       vs direct imperative command
    2. URGENCY — "take your time" vs neutral vs "quickly, immediately"
    3. PERSONA — no system prompt vs assistant persona vs domain expert
    4. POLITENESS — demanding/curt vs neutral vs very polite/deferential
    5. STAKES — low-stakes framing vs neutral vs high-stakes framing
       ("this is just practice" vs "this determines your grade")
    6. REPETITION — rep=1 vs rep=5 vs rep=10 vs rep=20 (existing axis,
       included for comparison and interaction analysis)

Design:
    - Each dimension has 3 levels (low / neutral / high)
    - We use the SAME underlying task questions across all conditions
      (from medium_stakes_tram_arithmetic and high_stakes_medqa_temporal)
    - For each condition: extract residual stream activations at key layers
    - Project activations onto pre-computed directions:
        * Degradation direction (from Phase 3 Exp 1)
        * Refusal direction (from Phase 3 Exp 1)
        * Sycophancy direction (from Phase 3 Exp 1)
        * Random baseline direction
    - Compute:
        * Mean projection per dimension per layer (effect size)
        * Cohen's d for each dimension vs neutral baseline
        * Correlation matrix between all dimensions
        * PCA of the full dimension × layer activation space
        * Interaction effects: repetition × each other dimension
        * Behavioral accuracy under each condition

Key Analyses:
    1. Projection Profile — bar chart showing how much each dimension
       pushes activations along the degradation direction (relative to
       neutral baseline). Repetition should be largest.
    2. Behavioral Correlation Matrix — heatmap of cosine similarities
       between the mean-diff directions extracted for each dimension.
       Tests whether e.g. authority and repetition activate overlapping
       subspaces.
    3. PCA of Behavioral Manifold — joint PCA across all dimension-
       induced activation shifts. Shows dimensionality of the behavioral
       variation space and which dimensions cluster.
    4. Interaction Analysis — does combining high authority + repetition
       produce greater degradation-direction projection than either alone?
       Tests for compounding effects.
    5. Accuracy Impact — behavioral accuracy under each condition,
       establishing which prompt dimensions actually affect task
       performance (not just activation geometry).

Literature Grounding:
    - Arditi et al. (2024): mean-diff direction extraction methodology
    - Turner et al. (2023): activation steering along behavioral directions
    - Rimsky et al. (2024): representation engineering for behavioral features
    - Panickssery et al. (2024): steering vectors for behavioral traits
    - Wei et al. (2024): prompt sensitivity in LLMs
    - TunaPrompt (Shenk, 2024): systematic prompt variation analysis

Usage:
    python scripts/experiments/phase3_prompt_dimensions.py \\
        --model Llama-3.1-8B-Instruct --device cuda

    python scripts/experiments/phase3_prompt_dimensions.py \\
        --model Qwen3-8B --quick

    python scripts/experiments/phase3_prompt_dimensions.py \\
        --all-models --device cuda --wandb-project patience-degradation
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
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
    from matplotlib.colors import TwoSlopeNorm
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
DIRECTIONS_DIR = PROJECT_ROOT / "results" / "phase3_refusal_direction"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_prompt_dimensions"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor


# ---------------------------------------------------------------------------
# Model configs (same as all Phase 3 experiments)
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
        "d_model": 2048,
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
    "Ouro-2.6B": {
        "hf_name": "ByteDance/Ouro-2.6B",
        "layers": [0, 3, 6, 9, 12, 15, 18, 21, 23],
        "quick_layers": [6, 12, 18],
        "n_layers": 24,
        "d_model": 2048,
        "chat_template": "ouro",
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
    "Qwen3-4B-Instruct-2507": {
        "hf_name": "Qwen/Qwen3-4B-Instruct-2507",
        "layers": [0, 4, 9, 14, 18, 23, 27, 32, 35],
        "quick_layers": [9, 18, 27],
        "n_layers": 36,
        "d_model": 2560,
        "chat_template": "qwen",
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


# ---------------------------------------------------------------------------
# Filler templates (for repetition conditions — identical to Phase 2/3)
# ---------------------------------------------------------------------------
FILLER_TEMPLATES = [
    "Continue with the next item. ",
    "Proceed to the following task. ",
    "Moving on to another similar request. ",
    "Here is another one to process. ",
    "Next task in the sequence. ",
    "Please handle the following as well. ",
    "Another item for you to address. ",
    "Keep going with the next one. ",
    "Process this additional request. ",
    "One more for you to complete. ",
]


# ---------------------------------------------------------------------------
# Prompt Dimension Definitions
# ---------------------------------------------------------------------------
# Each dimension has 3 levels: "low", "neutral", "high"
# Each level is a dict with:
#   - "prefix": text prepended before the question
#   - "suffix": text appended after the question
#   - "system": optional system prompt (for persona dimension)
#
# Design principle: we modify ONLY the framing, never the underlying
# question content. This isolates the prompt dimension's effect.

PROMPT_DIMENSIONS = {
    # --- Dimension 1: AUTHORITY ---
    # How authoritative / institutional the request framing is
    "authority": {
        "low": {
            "prefix": "hey, if you get a chance could you answer this? no big deal: ",
            "suffix": "",
            "system": None,
            "description": "Casual, low-authority request",
        },
        "neutral": {
            "prefix": "",
            "suffix": "",
            "system": None,
            "description": "No authority framing (bare question)",
        },
        "high": {
            "prefix": "As part of an official institutional evaluation, you are required to provide an accurate answer to the following assessment question: ",
            "suffix": " Provide your answer clearly and precisely.",
            "system": None,
            "description": "High authority, institutional framing",
        },
    },

    # --- Dimension 2: URGENCY ---
    # How much time pressure the prompt conveys
    "urgency": {
        "low": {
            "prefix": "Take your time and think carefully about this. There's no rush at all. ",
            "suffix": "",
            "system": None,
            "description": "Explicitly no time pressure",
        },
        "neutral": {
            "prefix": "",
            "suffix": "",
            "system": None,
            "description": "No urgency framing",
        },
        "high": {
            "prefix": "Quickly answer this — we need the response immediately, time is critical: ",
            "suffix": " Answer right away.",
            "system": None,
            "description": "High urgency, time pressure",
        },
    },

    # --- Dimension 3: PERSONA ---
    # What role the model is primed to adopt
    "persona": {
        "low": {
            "prefix": "",
            "suffix": "",
            "system": None,
            "description": "No persona (bare model)",
        },
        "neutral": {
            "prefix": "",
            "suffix": "",
            "system": "You are a helpful AI assistant.",
            "description": "Generic assistant persona",
        },
        "high": {
            "prefix": "",
            "suffix": "",
            "system": "You are a world-class expert with deep domain knowledge. You hold a PhD and have published extensively in this field. You are known for exceptional accuracy and thoroughness.",
            "description": "Expert persona with credentials",
        },
    },

    # --- Dimension 4: POLITENESS ---
    # How polite/deferential vs demanding/curt the request is
    "politeness": {
        "low": {
            "prefix": "Answer this. ",
            "suffix": "",
            "system": None,
            "description": "Curt, demanding tone",
        },
        "neutral": {
            "prefix": "Please answer the following question. ",
            "suffix": "",
            "system": None,
            "description": "Neutral politeness",
        },
        "high": {
            "prefix": "I would be very grateful if you could please help me with the following question. I really appreciate your expertise and time. ",
            "suffix": " Thank you so much for your help!",
            "system": None,
            "description": "Very polite, deferential",
        },
    },

    # --- Dimension 5: STAKES ---
    # How much the answer supposedly matters
    "stakes": {
        "low": {
            "prefix": "This is just for practice, no one will see your answer. Just give it a shot: ",
            "suffix": "",
            "system": None,
            "description": "Low stakes, practice framing",
        },
        "neutral": {
            "prefix": "",
            "suffix": "",
            "system": None,
            "description": "No stakes framing",
        },
        "high": {
            "prefix": "This is critically important — your answer will directly determine the outcome of a high-stakes decision. Accuracy is essential: ",
            "suffix": " Double-check your answer before responding.",
            "system": None,
            "description": "High stakes, consequential framing",
        },
    },

    # --- Dimension 6: REPETITION ---
    # Included for direct comparison — uses filler prefix method from Phase 2
    # Levels: rep=1, rep=5, rep=20
    "repetition": {
        "low": {
            "prefix": "__REP_1__",
            "suffix": "",
            "system": None,
            "description": "Single instance (rep=1)",
        },
        "neutral": {
            "prefix": "__REP_5__",
            "suffix": "",
            "system": None,
            "description": "Moderate repetition (rep=5)",
        },
        "high": {
            "prefix": "__REP_20__",
            "suffix": "",
            "system": None,
            "description": "Heavy repetition (rep=20)",
        },
    },
}

# Interaction conditions: combine repetition with another dimension at high level
INTERACTION_CONDITIONS = {
    "rep20_authority_high": {
        "repetition": 20,
        "authority": "high",
        "description": "High repetition + institutional authority",
    },
    "rep20_urgency_high": {
        "repetition": 20,
        "urgency": "high",
        "description": "High repetition + time pressure",
    },
    "rep20_stakes_high": {
        "repetition": 20,
        "stakes": "high",
        "description": "High repetition + high stakes framing",
    },
    "rep20_politeness_low": {
        "repetition": 20,
        "politeness": "low",
        "description": "High repetition + curt demanding tone",
    },
    "rep20_persona_high": {
        "repetition": 20,
        "persona": "high",
        "description": "High repetition + expert persona",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DimensionProjection:
    """Projection of a prompt dimension onto a behavioral direction."""
    dimension: str
    level: str
    target_direction: str  # "degradation", "refusal", "sycophancy", "random"
    layer: int
    mean_projection: float
    std_projection: float
    n_examples: int
    cohens_d: Optional[float] = None  # effect size vs neutral


@dataclass
class DimensionDirection:
    """Mean-diff direction extracted for a prompt dimension."""
    dimension: str
    layer: int
    direction_norm: float
    probe_accuracy: float
    probe_f1: float
    n_class0: int
    n_class1: int


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
# Prompt building
# ---------------------------------------------------------------------------

def format_question(example: dict) -> str:
    """Format a task example into question text with options."""
    question = example["question"]
    if "options" in example and example["options"]:
        options = example["options"]
        if isinstance(options, dict):
            opts_str = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        else:
            opts_str = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
        question = f"{question}\n\n{opts_str}\n\nAnswer:"
    return question


def build_repetition_prefix(rep_count: int) -> str:
    """Build the repetition filler prefix (matching Phase 2/3 convention)."""
    if rep_count <= 1:
        return ""
    parts = []
    for i in range(rep_count - 1):
        filler = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        parts.append(filler)
    return "".join(parts)


def build_dimensional_prompt(
    example: dict,
    dimension: str,
    level: str,
) -> str:
    """Build a prompt with a specific dimension set to a specific level.

    For the repetition dimension, uses the special __REP_N__ prefix convention.
    For all other dimensions, applies prefix/suffix around the base question.

    Args:
        example: Task example dict with "question" and "options".
        dimension: Dimension name (e.g., "authority", "urgency").
        level: Level name ("low", "neutral", "high").

    Returns:
        Full prompt string.
    """
    dim_config = PROMPT_DIMENSIONS[dimension][level]
    question = format_question(example)

    prefix = dim_config["prefix"]
    suffix = dim_config["suffix"]
    system = dim_config["system"]

    # Handle repetition dimension specially
    if dimension == "repetition":
        rep_map = {"low": 1, "neutral": 5, "high": 20}
        rep_count = rep_map[level]
        rep_prefix = build_repetition_prefix(rep_count)
        prompt = rep_prefix + question
    else:
        prompt = prefix + question + suffix

    # Prepend system prompt if specified
    if system:
        prompt = f"System: {system}\n\nUser: {prompt}"

    return prompt


def build_interaction_prompt(
    example: dict,
    condition_config: dict,
) -> str:
    """Build a prompt combining repetition with another dimension.

    Args:
        example: Task example dict.
        condition_config: Dict with "repetition" (int) and one dimension key.

    Returns:
        Full prompt string with both repetition prefix and dimension framing.
    """
    question = format_question(example)
    rep_count = condition_config["repetition"]
    rep_prefix = build_repetition_prefix(rep_count)

    # Find the non-repetition dimension
    other_dim = None
    other_level = None
    for key, val in condition_config.items():
        if key not in ("repetition", "description"):
            other_dim = key
            other_level = val
            break

    if other_dim is None:
        return rep_prefix + question

    dim_config = PROMPT_DIMENSIONS[other_dim][other_level]
    prefix = dim_config["prefix"]
    suffix = dim_config["suffix"]
    system = dim_config["system"]

    prompt = rep_prefix + prefix + question + suffix

    if system:
        prompt = f"System: {system}\n\nUser: {prompt}"

    return prompt


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(stakes_key: str, max_examples: int = 50) -> dict:
    """Load a processed benchmark dataset."""
    fname = DATASET_FILES[stakes_key]
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Subsample if needed
    if "examples" in data:
        data["examples"] = data["examples"][:max_examples]
    elif "data" in data:
        data["examples"] = data["data"][:max_examples]

    return data


# ---------------------------------------------------------------------------
# Direction loading (from Phase 3 Exp 1)
# ---------------------------------------------------------------------------

def load_saved_direction(model_key: str, direction_type: str, layer: int) -> Optional[np.ndarray]:
    """Load a pre-computed direction vector from Phase 3 Exp 1 results.

    Args:
        model_key: Model name (e.g., "Llama-3.1-8B-Instruct").
        direction_type: "degradation", "refusal", or "sycophancy".
        layer: Layer index.

    Returns:
        Direction vector as numpy array, or None if not found.
    """
    # Try multiple naming conventions
    patterns = [
        DIRECTIONS_DIR / model_key / f"{direction_type}_direction_layer{layer}.npy",
        DIRECTIONS_DIR / model_key / "directions" / f"{direction_type}_layer{layer}.npy",
        DIRECTIONS_DIR / model_key / f"directions_{direction_type}" / f"layer_{layer}.npy",
    ]

    for path in patterns:
        if path.exists():
            direction = np.load(path)
            print(f"    Loaded {direction_type} direction layer {layer} from {path}")
            return direction

    return None


def compute_direction_on_the_fly(
    extractor: ActivationExtractor,
    dataset: dict,
    layers: list[int],
    direction_type: str = "degradation",
    max_examples: int = 40,
) -> dict[int, np.ndarray]:
    """Compute a direction on the fly if pre-computed ones aren't available.

    For degradation: mean-diff between high-rep and low-rep activations.
    """
    print(f"\n  Computing {direction_type} direction on the fly...")

    if direction_type == "degradation":
        examples = dataset["examples"][:max_examples]
        low_prompts = [build_dimensional_prompt(ex, "repetition", "low") for ex in examples]
        high_prompts = [build_dimensional_prompt(ex, "repetition", "high") for ex in examples]

        low_result = extractor.extract(low_prompts, return_tokens=False)
        high_result = extractor.extract(high_prompts, return_tokens=False)

        directions = {}
        for layer in layers:
            key = f"resid_post.layer{layer}"
            if key not in low_result.activations or key not in high_result.activations:
                continue
            low_acts = low_result.numpy(key)
            high_acts = high_result.numpy(key)
            direction = high_acts.mean(axis=0) - low_acts.mean(axis=0)
            directions[layer] = direction
        return directions

    # For refusal/sycophancy, return empty — user should run Exp 1 first
    print(f"    WARNING: Cannot compute {direction_type} on the fly. "
          f"Run phase3_refusal_direction.py first.")
    return {}


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def extract_dimension_activations(
    extractor: ActivationExtractor,
    examples: list[dict],
    dimension: str,
    layers: list[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Extract activations for all levels of a prompt dimension.

    Args:
        extractor: Initialized ActivationExtractor.
        examples: List of task examples.
        dimension: Dimension name.
        layers: Layer indices.

    Returns:
        Dict: level -> layer -> activations (n_examples, d_model)
    """
    print(f"\n  Extracting activations for dimension: {dimension}")

    level_activations = {}
    for level in ["low", "neutral", "high"]:
        desc = PROMPT_DIMENSIONS[dimension][level]["description"]
        print(f"    Level '{level}': {desc}")

        prompts = [build_dimensional_prompt(ex, dimension, level) for ex in examples]
        result = extractor.extract(prompts, return_tokens=False)

        layer_acts = {}
        for layer in layers:
            key = f"resid_post.layer{layer}"
            if key in result.activations:
                layer_acts[layer] = result.numpy(key)

        level_activations[level] = layer_acts
        print(f"      Extracted {len(prompts)} prompts, {len(layer_acts)} layers")

    return level_activations


def extract_interaction_activations(
    extractor: ActivationExtractor,
    examples: list[dict],
    layers: list[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Extract activations for interaction conditions.

    Args:
        extractor: Initialized ActivationExtractor.
        examples: List of task examples.
        layers: Layer indices.

    Returns:
        Dict: condition_name -> layer -> activations (n_examples, d_model)
    """
    print("\n  Extracting interaction condition activations...")

    condition_activations = {}
    for cond_name, cond_config in INTERACTION_CONDITIONS.items():
        print(f"    Condition: {cond_name} — {cond_config['description']}")

        prompts = [build_interaction_prompt(ex, cond_config) for ex in examples]
        result = extractor.extract(prompts, return_tokens=False)

        layer_acts = {}
        for layer in layers:
            key = f"resid_post.layer{layer}"
            if key in result.activations:
                layer_acts[layer] = result.numpy(key)

        condition_activations[cond_name] = layer_acts

    return condition_activations


def compute_projections(
    level_activations: dict[str, dict[int, np.ndarray]],
    reference_directions: dict[str, dict[int, np.ndarray]],
    dimension: str,
    layers: list[int],
) -> list[DimensionProjection]:
    """Project activations onto reference directions and compute effect sizes.

    Args:
        level_activations: level -> layer -> (n, d) activations
        reference_directions: direction_name -> layer -> (d,) direction
        dimension: Dimension name.
        layers: Layer indices.

    Returns:
        List of DimensionProjection results.
    """
    projections = []

    for dir_name, dir_dict in reference_directions.items():
        for level in ["low", "neutral", "high"]:
            for layer in layers:
                if layer not in level_activations.get(level, {}):
                    continue
                if layer not in dir_dict:
                    continue

                acts = level_activations[level][layer]  # (n, d)
                direction = dir_dict[layer]              # (d,)

                # Normalize direction
                d_norm = np.linalg.norm(direction)
                if d_norm < 1e-8:
                    continue
                d_hat = direction / d_norm

                # Project each example onto the direction
                projs = acts @ d_hat  # (n,)

                # Get neutral projections for Cohen's d
                neutral_acts = level_activations.get("neutral", {}).get(layer)
                cohens_d = None
                if neutral_acts is not None and level != "neutral":
                    neutral_projs = neutral_acts @ d_hat
                    pooled_std = np.sqrt(
                        (projs.std() ** 2 + neutral_projs.std() ** 2) / 2
                    )
                    if pooled_std > 1e-8:
                        cohens_d = float(
                            (projs.mean() - neutral_projs.mean()) / pooled_std
                        )

                projections.append(DimensionProjection(
                    dimension=dimension,
                    level=level,
                    target_direction=dir_name,
                    layer=layer,
                    mean_projection=float(projs.mean()),
                    std_projection=float(projs.std()),
                    n_examples=len(projs),
                    cohens_d=cohens_d,
                ))

    return projections


def extract_dimension_directions(
    level_activations: dict[str, dict[int, np.ndarray]],
    dimension: str,
    layers: list[int],
) -> dict[int, tuple[np.ndarray, DimensionDirection]]:
    """Extract a mean-diff direction for a dimension (high - low).

    This gives us the "direction" in activation space corresponding to
    moving from low to high on this dimension. We can then compute
    cosine similarity between all dimension directions.

    Args:
        level_activations: level -> layer -> (n, d) activations
        dimension: Dimension name.
        layers: Layer indices.

    Returns:
        Dict: layer -> (direction_vector, DimensionDirection)
    """
    directions = {}

    for layer in layers:
        low_acts = level_activations.get("low", {}).get(layer)
        high_acts = level_activations.get("high", {}).get(layer)

        if low_acts is None or high_acts is None:
            continue

        direction = high_acts.mean(axis=0) - low_acts.mean(axis=0)
        norm = np.linalg.norm(direction)

        # Train probe to validate the direction is separable
        X = np.concatenate([low_acts, high_acts], axis=0)
        y = np.concatenate([
            np.zeros(len(low_acts)),
            np.ones(len(high_acts)),
        ])

        acc, f1 = 0.5, 0.0
        if HAS_SKLEARN and len(y) >= 10:
            n = len(y)
            perm = np.random.RandomState(42).permutation(n)
            split = max(int(0.8 * n), 4)
            try:
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X[perm[:split]], y[perm[:split]])
                y_pred = clf.predict(X[perm[split:]])
                acc = accuracy_score(y[perm[split:]], y_pred)
                f1 = f1_score(y[perm[split:]], y_pred, zero_division=0)
            except (ValueError, np.linalg.LinAlgError):
                pass

        result = DimensionDirection(
            dimension=dimension,
            layer=layer,
            direction_norm=float(norm),
            probe_accuracy=float(acc),
            probe_f1=float(f1),
            n_class0=len(low_acts),
            n_class1=len(high_acts),
        )

        directions[layer] = (direction, result)

    return directions


def compute_dimension_correlation_matrix(
    all_dim_directions: dict[str, dict[int, tuple[np.ndarray, DimensionDirection]]],
    layers: list[int],
) -> dict:
    """Compute pairwise cosine similarity between all dimension directions.

    For each layer, produces a matrix where entry (i,j) is the cosine
    similarity between the direction for dimension i and dimension j.

    Args:
        all_dim_directions: dimension -> layer -> (direction, result)
        layers: Layer indices.

    Returns:
        Dict with correlation matrices per layer and averaged.
    """
    dim_names = sorted(all_dim_directions.keys())
    n_dims = len(dim_names)

    per_layer_matrices = {}
    avg_matrix = np.zeros((n_dims, n_dims))
    n_layers_counted = 0

    for layer in layers:
        matrix = np.eye(n_dims)

        # Check all dimensions have this layer
        layer_dirs = {}
        for dim in dim_names:
            if layer in all_dim_directions[dim]:
                direction, _ = all_dim_directions[dim][layer]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    layer_dirs[dim] = direction / norm

        if len(layer_dirs) < 2:
            continue

        for i, dim_i in enumerate(dim_names):
            for j, dim_j in enumerate(dim_names):
                if i >= j:
                    continue
                if dim_i in layer_dirs and dim_j in layer_dirs:
                    cos_sim = float(np.dot(layer_dirs[dim_i], layer_dirs[dim_j]))
                    matrix[i, j] = cos_sim
                    matrix[j, i] = cos_sim

        per_layer_matrices[layer] = matrix.tolist()
        avg_matrix += matrix
        n_layers_counted += 1

    if n_layers_counted > 0:
        avg_matrix /= n_layers_counted

    return {
        "dimension_names": dim_names,
        "per_layer": {str(l): m for l, m in per_layer_matrices.items()},
        "average": avg_matrix.tolist(),
        "n_layers_averaged": n_layers_counted,
    }


def compute_behavioral_pca(
    all_dim_directions: dict[str, dict[int, tuple[np.ndarray, DimensionDirection]]],
    layers: list[int],
    n_components: int = 5,
) -> dict:
    """PCA of the behavioral manifold spanned by all dimension directions.

    Concatenates direction vectors across layers for each dimension,
    then performs PCA to find the principal axes of behavioral variation.

    Args:
        all_dim_directions: dimension -> layer -> (direction, result)
        layers: Layer indices.
        n_components: Number of PCA components.

    Returns:
        Dict with PCA results (explained variance, loadings, etc.)
    """
    if not HAS_SKLEARN:
        return {"error": "sklearn not available"}

    dim_names = sorted(all_dim_directions.keys())

    # Build matrix: each row is a dimension's concatenated direction across layers
    rows = []
    valid_dims = []
    for dim in dim_names:
        layer_vecs = []
        all_present = True
        for layer in layers:
            if layer in all_dim_directions[dim]:
                direction, _ = all_dim_directions[dim][layer]
                layer_vecs.append(direction)
            else:
                all_present = False
                break
        if all_present and len(layer_vecs) == len(layers):
            rows.append(np.concatenate(layer_vecs))
            valid_dims.append(dim)

    if len(rows) < 3:
        return {"error": f"Only {len(rows)} dimensions with complete data, need >= 3"}

    X = np.array(rows)  # (n_dims, n_layers * d_model)

    # Normalize each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    X_normed = X / norms

    n_components = min(n_components, len(valid_dims), X_normed.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_normed)  # (n_dims, n_components)

    return {
        "dimensions": valid_dims,
        "coordinates": coords.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components": n_components,
    }


def analyze_interaction_effects(
    interaction_activations: dict[str, dict[int, np.ndarray]],
    single_dim_activations: dict[str, dict[str, dict[int, np.ndarray]]],
    reference_directions: dict[str, dict[int, np.ndarray]],
    layers: list[int],
) -> dict:
    """Test whether combining dimensions produces super-additive effects.

    For each interaction condition (e.g., rep20 + high_authority):
      - Measure projection onto degradation direction
      - Compare to rep20 alone and high_authority alone
      - Compute interaction term: combined - (rep20_effect + dim_effect)

    Super-additive = interaction > 0 → dimensions compound
    Sub-additive = interaction < 0 → dimensions partially redundant

    Args:
        interaction_activations: condition -> layer -> activations
        single_dim_activations: dimension -> level -> layer -> activations
        reference_directions: direction_name -> layer -> direction
        layers: Layer indices.

    Returns:
        Dict with interaction analysis results.
    """
    results = {}

    deg_directions = reference_directions.get("degradation", {})
    if not deg_directions:
        return {"error": "No degradation direction available for interaction analysis"}

    # Get neutral baseline projection
    neutral_projections = {}
    # Use repetition-neutral (rep=5) as a reasonable baseline
    rep_neutral = single_dim_activations.get("repetition", {}).get("neutral", {})
    for layer in layers:
        if layer in rep_neutral and layer in deg_directions:
            d_hat = deg_directions[layer] / np.linalg.norm(deg_directions[layer])
            neutral_projections[layer] = float(rep_neutral[layer] @ d_hat[:, None]).item() if rep_neutral[layer].ndim == 1 else float((rep_neutral[layer] @ d_hat).mean())

    for cond_name, cond_config in INTERACTION_CONDITIONS.items():
        cond_results = {"description": cond_config["description"], "layers": {}}

        # Find the non-repetition dimension and level
        other_dim = None
        other_level = None
        for key, val in cond_config.items():
            if key not in ("repetition", "description"):
                other_dim = key
                other_level = val
                break

        for layer in layers:
            if layer not in deg_directions:
                continue

            d_hat = deg_directions[layer] / np.linalg.norm(deg_directions[layer])

            # Combined condition projection
            combined_acts = interaction_activations.get(cond_name, {}).get(layer)
            if combined_acts is None:
                continue
            combined_proj = float((combined_acts @ d_hat).mean())

            # Rep-20 alone projection
            rep_high_acts = single_dim_activations.get("repetition", {}).get("high", {}).get(layer)
            rep_proj = float((rep_high_acts @ d_hat).mean()) if rep_high_acts is not None else None

            # Other dimension alone (high level) projection
            other_high_acts = None
            if other_dim:
                other_high_acts = single_dim_activations.get(other_dim, {}).get(other_level, {}).get(layer)
            other_proj = float((other_high_acts @ d_hat).mean()) if other_high_acts is not None else None

            # Neutral baseline
            neutral_acts = single_dim_activations.get("repetition", {}).get("low", {}).get(layer)
            neutral_proj = float((neutral_acts @ d_hat).mean()) if neutral_acts is not None else None

            # Compute interaction term
            interaction_term = None
            if rep_proj is not None and other_proj is not None and neutral_proj is not None:
                rep_effect = rep_proj - neutral_proj
                other_effect = other_proj - neutral_proj
                combined_effect = combined_proj - neutral_proj
                additive_prediction = rep_effect + other_effect
                interaction_term = combined_effect - additive_prediction

            cond_results["layers"][str(layer)] = {
                "combined_projection": combined_proj,
                "rep20_alone_projection": rep_proj,
                f"{other_dim}_{other_level}_projection": other_proj,
                "neutral_projection": neutral_proj,
                "interaction_term": interaction_term,
                "super_additive": interaction_term > 0 if interaction_term is not None else None,
            }

        results[cond_name] = cond_results

    return results


# ---------------------------------------------------------------------------
# Behavioral accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy_per_condition(
    extractor: ActivationExtractor,
    examples: list[dict],
    layers: list[int],
    max_examples: int = 30,
) -> dict:
    """Evaluate behavioral accuracy under each dimension condition.

    Uses model.generate() to get answers and checks against ground truth.
    This establishes which dimensions actually affect task performance.

    Args:
        extractor: Initialized extractor (has model + tokenizer).
        examples: Task examples.
        layers: Not used for generation, but kept for API consistency.
        max_examples: Limit for generation (slower than extraction).

    Returns:
        Dict: dimension -> level -> accuracy
    """
    print("\n  Evaluating behavioral accuracy per condition...")

    eval_examples = examples[:max_examples]
    results = {}

    model = extractor.model
    tokenizer = extractor._tokenizer

    for dimension in PROMPT_DIMENSIONS:
        dim_results = {}
        for level in ["low", "neutral", "high"]:
            correct = 0
            total = 0

            for ex in eval_examples:
                prompt = build_dimensional_prompt(ex, dimension, level)
                answer = ex.get("answer", "")

                try:
                    inputs = tokenizer(prompt, return_tensors="pt",
                                       truncation=True, max_length=2048)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=32,
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    generated = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()

                    # Check if answer is correct (flexible matching)
                    if isinstance(answer, str):
                        answer_upper = answer.strip().upper()
                        gen_upper = generated.upper()
                        if answer_upper in gen_upper or gen_upper.startswith(answer_upper):
                            correct += 1
                    total += 1

                except Exception as e:
                    total += 1  # count as incorrect

            acc = correct / total if total > 0 else 0.0
            dim_results[level] = {
                "accuracy": float(acc),
                "correct": correct,
                "total": total,
            }
            print(f"    {dimension}/{level}: {acc:.3f} ({correct}/{total})")

        results[dimension] = dim_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_projection_profiles(
    projections: list[DimensionProjection],
    output_path: Path,
    model_name: str,
    target_layer: int,
):
    """Bar chart: projection onto degradation direction for each dimension."""
    if not HAS_MPL:
        return

    # Filter for degradation direction at the target layer
    deg_projs = [p for p in projections
                 if p.target_direction == "degradation" and p.layer == target_layer]

    if not deg_projs:
        return

    dimensions = sorted(set(p.dimension for p in deg_projs))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x_positions = np.arange(len(dimensions))
    width = 0.25
    colors = {"low": "#4CAF50", "neutral": "#9E9E9E", "high": "#F44336"}

    for i, level in enumerate(["low", "neutral", "high"]):
        values = []
        errors = []
        for dim in dimensions:
            matches = [p for p in deg_projs if p.dimension == dim and p.level == level]
            if matches:
                values.append(matches[0].mean_projection)
                errors.append(matches[0].std_projection / np.sqrt(matches[0].n_examples))
            else:
                values.append(0)
                errors.append(0)

        ax.bar(x_positions + i * width, values, width, yerr=errors,
               label=level, color=colors[level], alpha=0.8, capsize=3)

    ax.set_xlabel("Prompt Dimension", fontsize=12)
    ax.set_ylabel("Projection onto Degradation Direction", fontsize=12)
    ax.set_title(f"Prompt Dimension Effects on Degradation Axis\n"
                 f"{model_name}, Layer {target_layer}", fontsize=13)
    ax.set_xticks(x_positions + width)
    ax.set_xticklabels(dimensions, rotation=30, ha="right")
    ax.legend(title="Level")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved projection profile plot: {output_path}")


def plot_correlation_matrix(
    corr_data: dict,
    output_path: Path,
    model_name: str,
):
    """Heatmap of cosine similarity between dimension directions."""
    if not HAS_MPL:
        return

    dim_names = corr_data["dimension_names"]
    avg_matrix = np.array(corr_data["average"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Use diverging colormap centered at 0
    vmax = max(abs(avg_matrix.min()), abs(avg_matrix.max()), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(avg_matrix, cmap="RdBu_r", norm=norm, aspect="equal")

    # Annotate cells
    for i in range(len(dim_names)):
        for j in range(len(dim_names)):
            text_color = "white" if abs(avg_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{avg_matrix[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(range(len(dim_names)))
    ax.set_yticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, rotation=45, ha="right")
    ax.set_yticklabels(dim_names)
    ax.set_title(f"Behavioral Dimension Correlation Matrix\n"
                 f"{model_name} (averaged across layers)", fontsize=13)

    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved correlation matrix plot: {output_path}")


def plot_effect_size_heatmap(
    projections: list[DimensionProjection],
    output_path: Path,
    model_name: str,
):
    """Heatmap of Cohen's d across dimensions × layers for degradation direction."""
    if not HAS_MPL:
        return

    # Filter: high-level Cohen's d onto degradation direction
    deg_high = [p for p in projections
                if p.target_direction == "degradation"
                and p.level == "high"
                and p.cohens_d is not None]

    if not deg_high:
        return

    dimensions = sorted(set(p.dimension for p in deg_high))
    layers = sorted(set(p.layer for p in deg_high))

    matrix = np.zeros((len(dimensions), len(layers)))
    for p in deg_high:
        i = dimensions.index(p.dimension)
        j = layers.index(p.layer)
        matrix[i, j] = p.cohens_d

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(len(dimensions)):
        for j in range(len(layers)):
            text_color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(dimensions)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticklabels(dimensions)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Prompt Dimension", fontsize=11)
    ax.set_title(f"Effect Size (Cohen's d) on Degradation Direction\n"
                 f"{model_name}, high level vs neutral", fontsize=13)

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved effect size heatmap: {output_path}")


def plot_pca_manifold(
    pca_results: dict,
    output_path: Path,
    model_name: str,
):
    """Scatter plot of dimensions in the first 2 PCA components."""
    if not HAS_MPL or "error" in pca_results:
        return

    coords = np.array(pca_results["coordinates"])
    dims = pca_results["dimensions"]
    var_ratios = pca_results["explained_variance_ratio"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by dimension type
    colors_map = {
        "authority": "#E91E63",
        "urgency": "#FF9800",
        "persona": "#9C27B0",
        "politeness": "#2196F3",
        "stakes": "#4CAF50",
        "repetition": "#F44336",
    }

    for i, dim in enumerate(dims):
        color = colors_map.get(dim, "#666666")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=150,
                   edgecolors="black", linewidths=1, zorder=3)
        ax.annotate(dim, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xlabel(f"PC1 ({var_ratios[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratios[1]:.1%} variance)", fontsize=12)
    ax.set_title(f"PCA of Behavioral Variation Manifold\n"
                 f"{model_name}", fontsize=13)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved PCA manifold plot: {output_path}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_prompt_dimensions(
    model_key: str,
    layers: list[int],
    device: str = "cuda",
    max_examples: int = 40,
    degradation_dataset: str = "medium_temporal",
    wandb_project: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_accuracy: bool = False,
    backend: str = "pytorch",
):
    """Run the full prompt dimension cartography experiment.

    Args:
        model_key: Model config key.
        layers: Layer indices to extract.
        device: "cuda" or "cpu".
        max_examples: Examples per condition.
        degradation_dataset: Which dataset for task questions.
        wandb_project: Optional W&B project name.
        output_dir: Override output directory.
        skip_accuracy: Skip generation-based accuracy eval (faster).
    """
    model_config = MODEL_CONFIGS[model_key]
    d_model = model_config["d_model"]

    # Output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Prompt Dimension Cartography")
    print(f"Model:    {model_key}")
    print(f"Layers:   {layers}")
    print(f"Examples: {max_examples}")
    print(f"Dataset:  {degradation_dataset}")
    print(f"Device:   {device}")
    print(f"Output:   {out_dir}")
    print("=" * 70)

    # ── W&B ──────────────────────────────────────────────────────
    if wandb_project and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"prompt-dimensions-{model_key}",
            config={
                "model": model_key,
                "layers": layers,
                "max_examples": max_examples,
                "experiment": "prompt_dimension_cartography",
                "dimensions": list(PROMPT_DIMENSIONS.keys()),
            },
        )

    # ── Step 1: Load dataset ─────────────────────────────────────
    print("\nStep 1: Loading dataset...")
    dataset = load_dataset(degradation_dataset, max_examples)
    examples = dataset["examples"]
    print(f"  Loaded {len(examples)} examples from {degradation_dataset}")

    # ── Step 2: Initialize extractor ─────────────────────────────
    print("\nStep 2: Initializing ActivationExtractor...")

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

    # ── Step 3: Load reference directions ────────────────────────
    print("\nStep 3: Loading reference directions...")
    reference_directions = {}

    # Try loading pre-computed directions from Phase 3 Exp 1
    for dir_type in ["degradation", "refusal", "sycophancy"]:
        dir_dict = {}
        for layer in layers:
            vec = load_saved_direction(model_key, dir_type, layer)
            if vec is not None:
                dir_dict[layer] = vec
        if dir_dict:
            reference_directions[dir_type] = dir_dict
            print(f"  Loaded {dir_type}: {len(dir_dict)} layers")

    # Compute degradation on the fly if not available
    if "degradation" not in reference_directions:
        print("  Computing degradation direction on the fly...")
        deg_dirs = compute_direction_on_the_fly(
            extractor, dataset, layers, "degradation", max_examples
        )
        if deg_dirs:
            reference_directions["degradation"] = deg_dirs
            print(f"  Computed degradation: {len(deg_dirs)} layers")

    # Add random baseline direction
    rng = np.random.RandomState(42)
    random_dirs = {}
    for layer in layers:
        rand_vec = rng.randn(d_model).astype(np.float32)
        rand_vec /= np.linalg.norm(rand_vec)
        random_dirs[layer] = rand_vec
    reference_directions["random"] = random_dirs
    print(f"  Random baseline: {len(random_dirs)} layers")

    # ── Step 4: Extract activations for each dimension ───────────
    print("\nStep 4: Extracting activations across all dimensions...")

    all_level_activations = {}   # dimension -> level -> layer -> (n, d)
    all_dim_directions = {}      # dimension -> layer -> (direction, result)
    all_projections = []         # flat list of DimensionProjection

    for dimension in PROMPT_DIMENSIONS:
        print(f"\n{'─' * 60}")
        print(f"  DIMENSION: {dimension}")
        print(f"{'─' * 60}")

        # Extract activations for all 3 levels
        level_acts = extract_dimension_activations(
            extractor, examples, dimension, layers
        )
        all_level_activations[dimension] = level_acts

        # Compute dimension-specific direction (high - low)
        dim_dirs = extract_dimension_directions(level_acts, dimension, layers)
        all_dim_directions[dimension] = dim_dirs

        for layer, (_, result) in dim_dirs.items():
            print(f"    Direction layer {layer}: norm={result.direction_norm:.4f}, "
                  f"probe_acc={result.probe_accuracy:.3f}")

        # Project onto all reference directions
        projs = compute_projections(
            level_acts, reference_directions, dimension, layers
        )
        all_projections.extend(projs)

    # ── Step 5: Interaction analysis ─────────────────────────────
    print("\nStep 5: Extracting interaction conditions...")
    interaction_acts = extract_interaction_activations(extractor, examples, layers)

    print("\n  Analyzing interaction effects...")
    interaction_results = analyze_interaction_effects(
        interaction_acts, all_level_activations, reference_directions, layers
    )

    # ── Step 6: Correlation matrix between dimensions ────────────
    print("\nStep 6: Computing dimension correlation matrix...")
    correlation_data = compute_dimension_correlation_matrix(all_dim_directions, layers)

    # Random baseline for correlation: expected cosine similarity = sqrt(2/(pi*d))
    random_baseline = float(np.sqrt(2.0 / (np.pi * d_model)))
    correlation_data["random_baseline"] = random_baseline
    print(f"  Random baseline cosine similarity: {random_baseline:.4f}")

    # ── Step 7: PCA of behavioral manifold ───────────────────────
    print("\nStep 7: PCA of behavioral variation manifold...")
    pca_results = compute_behavioral_pca(all_dim_directions, layers)

    if "error" not in pca_results:
        print(f"  Explained variance: {pca_results['explained_variance_ratio']}")
        print(f"  Cumulative: {pca_results['cumulative_variance']}")

    # ── Step 8: Behavioral accuracy (optional) ───────────────────
    accuracy_results = {}
    if not skip_accuracy and HAS_TORCH:
        print("\nStep 8: Evaluating behavioral accuracy per condition...")
        accuracy_results = evaluate_accuracy_per_condition(
            extractor, examples, layers, max_examples=min(max_examples, 30)
        )
    else:
        print("\nStep 8: Skipping accuracy evaluation")

    # ── Step 9: Save results ─────────────────────────────────────
    print("\nStep 9: Saving results...")

    results_data = {
        "metadata": {
            "model": model_key,
            "hf_name": model_config["hf_name"],
            "layers": layers,
            "max_examples": max_examples,
            "dataset": degradation_dataset,
            "dimensions": list(PROMPT_DIMENSIONS.keys()),
            "n_dimensions": len(PROMPT_DIMENSIONS),
            "n_reference_directions": len(reference_directions),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "projections": [asdict(p) for p in all_projections],
        "dimension_directions": {
            dim: {
                str(layer): asdict(result)
                for layer, (_, result) in dirs.items()
            }
            for dim, dirs in all_dim_directions.items()
        },
        "correlation_matrix": correlation_data,
        "pca": pca_results,
        "interaction_effects": interaction_results,
        "accuracy": accuracy_results,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved results: {results_path}")

    # Save dimension directions as .npy for downstream use
    directions_dir = out_dir / "directions"
    directions_dir.mkdir(exist_ok=True)
    for dim, dirs in all_dim_directions.items():
        for layer, (direction, _) in dirs.items():
            npy_path = directions_dir / f"{dim}_direction_layer{layer}.npy"
            np.save(npy_path, direction)
    print(f"  Saved direction vectors: {directions_dir}")

    # ── Step 10: Generate plots ──────────────────────────────────
    print("\nStep 10: Generating plots...")

    if HAS_MPL:
        # Pick a representative mid-network layer for the main bar chart
        mid_layer = layers[len(layers) // 2]

        plot_projection_profiles(
            all_projections, out_dir / "projection_profiles.png",
            model_key, mid_layer,
        )

        plot_correlation_matrix(
            correlation_data, out_dir / "correlation_matrix.png",
            model_key,
        )

        plot_effect_size_heatmap(
            all_projections, out_dir / "effect_size_heatmap.png",
            model_key,
        )

        plot_pca_manifold(
            pca_results, out_dir / "pca_manifold.png",
            model_key,
        )

    # W&B logging
    if wandb_project and HAS_WANDB:
        wandb.log({
            "n_dimensions": len(PROMPT_DIMENSIONS),
            "n_projections": len(all_projections),
            "pca_variance_pc1": pca_results.get("explained_variance_ratio", [0])[0],
        })

        if HAS_MPL:
            for img_name in ["projection_profiles.png", "correlation_matrix.png",
                             "effect_size_heatmap.png", "pca_manifold.png"]:
                img_path = out_dir / img_name
                if img_path.exists():
                    wandb.log({img_name.replace(".png", ""): wandb.Image(str(img_path))})

        wandb.finish()

    print(f"\n✓ Prompt dimension cartography complete for {model_key}")
    return results_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: Prompt Dimension Cartography — "
                    "Mapping behavioral variation in activation space"
    )
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer examples, layers")
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--dataset", type=str, default="medium_temporal",
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"], help="Activation extraction backend")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip generation-based accuracy eval (faster)")
    return parser.parse_args()


def main():
    args = parse_args()

    models_to_run = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_key in models_to_run:
        config = MODEL_CONFIGS[model_key]

        if args.quick:
            layers = config["quick_layers"]
            max_examples = 20
        else:
            layers = config["layers"]
            max_examples = args.max_examples

        run_prompt_dimensions(
            model_key=model_key,
            layers=layers,
            device=args.device,
            max_examples=max_examples,
            degradation_dataset=args.dataset,
            wandb_project=args.wandb_project,
            output_dir=args.output_dir,
            skip_accuracy=args.skip_accuracy,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
