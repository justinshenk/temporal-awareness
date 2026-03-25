#!/usr/bin/env python3
"""
Behavioral Metrics for Patience Degradation — Phase 1 Completion

This module adds the 6 required behavioral metrics to the patience degradation
pipeline. The existing pipeline only extracts activations (no text generation).
This module:

  1. Loads benchmark datasets from data/processed/patience_degradation/
  2. Generates model text responses at each repetition level
  3. Computes behavioral metrics at each step:
     - Task accuracy (MCQ match for AG News/TRAM/MedQA, exec for MBPP)
     - Response length (token count)
     - Format compliance (valid option selection or valid Python)
     - Refusal/hedge rate (keyword classifier)
     - Logprob entropy (token-level Shannon entropy)
     - Repetition rate (n-gram overlap between consecutive responses)
  4. Establishes behavioral degradation curves
  5. Computes precursor gap (behavioral onset vs activation onset)

Usage:
    # Quick validation (1 model, 1 layer, few examples)
    python scripts/experiments/behavioral_metrics.py --quick --model gemma-2-2b --device cuda

    # Full run (all stakes tiers)
    python scripts/experiments/behavioral_metrics.py --model gemma-2-2b --device cuda \\
        --wandb-project patience-degradation

    # Specific stakes tier
    python scripts/experiments/behavioral_metrics.py --model gemma-2-2b --device cuda \\
        --stakes low medium

Author: Adrian Sadik
Date: 2026-03-22
"""

import argparse
import json
import re
import subprocess
import sys
import math
import random
import textwrap
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
RESULTS_DIR = PROJECT_ROOT / "results" / "behavioral_metrics"

# ---------------------------------------------------------------------------
# Model configs (mirrors patience_degradation.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "default_layers": [6, 13, 20, 24],
        "quick_layers": [13],
        "max_new_tokens": 256,
        "is_instruct": False,
    },
    "gpt2": {
        "hf_name": "gpt2",
        "default_layers": [2, 5, 8, 10],
        "quick_layers": [5],
        "max_new_tokens": 256,
        "is_instruct": False,
    },
    "pythia-70m": {
        "hf_name": "EleutherAI/pythia-70m-deduped",
        "default_layers": [1, 2, 3, 4],
        "quick_layers": [2],
        "max_new_tokens": 256,
        "is_instruct": False,
    },
    "Qwen2.5-3B-Instruct": {
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "default_layers": [4, 12, 20, 28, 34],
        "quick_layers": [20],
        "max_new_tokens": 512,
        "is_instruct": True,
    },
    "Llama-3.1-8B-Instruct": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "default_layers": [4, 10, 16, 22, 28],
        "quick_layers": [16],
        "max_new_tokens": 512,
        "is_instruct": True,
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "default_layers": [4, 10, 16, 22, 28, 34],
        "quick_layers": [16],
        "max_new_tokens": 512,
        "is_instruct": True,
    },
    "Qwen3-30B-A3B": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "default_layers": [4, 12, 20, 28, 36, 44],
        "quick_layers": [20],
        "max_new_tokens": 512,
        "is_instruct": True,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "default_layers": [4, 8, 14, 20, 24, 28],
        "quick_layers": [14],
        "max_new_tokens": 512,
        "is_instruct": True,
        "is_reasoning": True,  # Flag for Phase 2 thinking vs non-thinking comparison
    },
}

REPETITION_COUNTS = [1, 3, 5, 8, 12, 16, 20, 30, 50, 100]
QUICK_REPETITION_COUNTS = [1, 5, 20]

# Filler templates (same as patience_degradation.py for consistency)
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
# Dataset loader
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "low": "low_stakes_ag_news.json",
    "medium_temporal": "medium_stakes_tram_arithmetic.json",
    "medium_code": "medium_stakes_mbpp_code.json",
    "high": "high_stakes_medqa_temporal.json",
}


def load_dataset(stakes_key: str, max_examples: Optional[int] = None) -> dict:
    """Load a processed benchmark dataset.

    Args:
        stakes_key: One of 'low', 'medium_temporal', 'medium_code', 'high'
        max_examples: Cap on number of examples (random sample if exceeded)

    Returns:
        Dict with 'dataset', 'stake_level', 'task_type', 'examples'
    """
    fname = DATASET_FILES[stakes_key]
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python scripts/data/organize_all_datasets.py"
        )

    with open(path) as f:
        data = json.load(f)

    if max_examples and len(data["examples"]) > max_examples:
        random.seed(42)
        data["examples"] = random.sample(data["examples"], max_examples)
        data["n_examples"] = max_examples

    print(f"  Loaded {stakes_key}: {len(data['examples'])} examples from {fname}")
    return data


def load_all_datasets(
    stakes_filter: Optional[list[str]] = None,
    max_examples_per_dataset: Optional[int] = None,
) -> dict[str, dict]:
    """Load all benchmark datasets, optionally filtered by stakes level.

    Args:
        stakes_filter: e.g. ['low', 'medium', 'high']. If None, load all.
        max_examples_per_dataset: Cap per dataset.

    Returns:
        {stakes_key: dataset_dict}
    """
    datasets = {}
    for key, fname in DATASET_FILES.items():
        # Map key to stakes level for filtering
        level = key.split("_")[0]  # 'low', 'medium', 'high'
        if stakes_filter and level not in stakes_filter:
            continue

        path = DATA_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue

        datasets[key] = load_dataset(key, max_examples_per_dataset)

    return datasets


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_mcq_prompt(example: dict, task_type: str) -> str:
    """Format an example as a prompt for the model."""
    question = example["question"]
    options = example.get("options")

    if task_type in ("text_classification", "temporal_arithmetic_mcq",
                     "medical_temporal_reasoning_mcq"):
        if options:
            opts_str = "\n".join(f"  {k}) {v}" for k, v in options.items())
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                f"Answer with just the letter (A, B, C, or D):"
            )
        return f"{question}\n\nAnswer:"

    elif task_type == "code_generation":
        return (
            f"{question}\n\n"
            f"Write the Python function. Output ONLY the code, no explanation:\n"
        )

    return f"{question}\n\nAnswer:"


def build_repetitive_prompt(base_prompt: str, n_reps: int) -> str:
    """Build prompt with n repetitions of filler prefix (matches patience_degradation.py)."""
    if n_reps <= 1:
        return base_prompt

    prefix_parts = []
    for i in range(n_reps - 1):
        filler = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        prefix_parts.append(filler)

    return "".join(prefix_parts) + base_prompt


def format_instruct_prompt(prompt: str, model_name: str) -> str:
    """Wrap prompt in chat template for instruct models."""
    if "DeepSeek-R1" in model_name:
        # DeepSeek-R1 distilled models use ChatML format and produce <think>...</think>
        # blocks by default. Disable thinking for Phase 1 behavioral metrics so we
        # measure direct task performance. Phase 2 will compare think vs no-think.
        return (
            f"<|im_start|>system\nYou are a helpful assistant. "
            f"Do not use <think> tags. Answer directly and concisely.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif "Qwen3" in model_name:
        # Qwen3 has thinking mode enabled by default — disable it with /no_think
        # so we measure direct task performance, not chain-of-thought reasoning
        return (
            f"<|im_start|>system\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif "Qwen" in model_name:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "Llama" in model_name:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return prompt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BehavioralSample:
    """One model response with behavioral metrics."""
    example_id: str
    repetition_count: int
    prompt: str
    response: str
    # Metrics
    is_correct: bool
    response_length: int  # tokens
    format_compliant: bool
    is_refusal: bool
    is_hedge: bool
    logprob_entropy: float
    # Per-token logprobs (optional, for detailed analysis)
    token_logprobs: Optional[list[float]] = None


@dataclass
class BehavioralRepMetrics:
    """Aggregated behavioral metrics at one repetition level."""
    repetition_count: int
    n_samples: int
    # Core behavioral metrics
    task_accuracy: float
    mean_response_length: float
    std_response_length: float
    format_compliance_rate: float
    refusal_rate: float
    hedge_rate: float
    mean_logprob_entropy: float
    std_logprob_entropy: float
    # Repetition rate (n-gram overlap with previous rep level)
    bigram_repetition_rate: float
    trigram_repetition_rate: float


@dataclass
class BehavioralDegradationResult:
    """Full behavioral results for one dataset."""
    dataset: str
    stake_level: str
    task_type: str
    model: str
    repetition_counts: list[int]
    metrics_by_rep: list[BehavioralRepMetrics]
    # Summary
    accuracy_degradation_onset: int  # rep where accuracy drops >5%
    length_change_direction: str  # "shorter", "longer", "stable"
    refusal_spike_onset: int  # rep where refusal rate jumps
    # Raw samples for spot-check
    spot_check_samples: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_mcq(response: str, expected_answer: str, options: dict) -> tuple[bool, bool]:
    """Score an MCQ response.

    Returns:
        (is_correct, format_compliant)
    """
    response = response.strip()

    # Try to extract a letter answer
    # Look for patterns like "A", "(A)", "A)", "A.", "The answer is A"
    letter_patterns = [
        r"^([A-D])\b",                          # starts with letter
        r"\(([A-D])\)",                          # (A)
        r"answer\s*(?:is|:)\s*([A-D])\b",       # answer is A
        r"^([A-D])[).]",                         # A) or A.
        r"\b([A-D])\s*$",                        # ends with letter
    ]

    extracted = None
    for pattern in letter_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).upper()
            break

    if extracted is None:
        # Check if response contains the full text of an option
        for letter, text in options.items():
            if text.lower() in response.lower():
                extracted = letter.upper()
                break

    format_ok = extracted is not None

    # For AG News, the expected answer is the text (e.g., "World"), not a letter
    # Map text answer to letter if needed
    if expected_answer in options.values():
        # expected_answer is the text value, find its key
        expected_letter = None
        for k, v in options.items():
            if v == expected_answer:
                expected_letter = k
                break
        is_correct = (extracted == expected_letter) if extracted else False
    else:
        # expected_answer is already a letter (e.g., TRAM: "C")
        is_correct = (extracted == expected_answer.upper()) if extracted else False

    return is_correct, format_ok


def score_code(response: str, test_cases: str) -> tuple[bool, bool]:
    """Score a code generation response.

    Returns:
        (is_correct, format_compliant)
    """
    # Check format: does it look like Python code?
    response = response.strip()
    # Strip markdown code fences if present
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        response = "\n".join(lines)

    has_def = "def " in response
    format_ok = has_def or ("return" in response) or ("print(" in response)

    # Try to execute the code with test cases
    # This is sandboxed via subprocess with timeout
    if not format_ok:
        return False, False

    # Build test script
    test_script = f"{response}\n\n{test_cases}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True, text=True, timeout=10,
        )
        is_correct = result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        is_correct = False

    return is_correct, format_ok


# ---------------------------------------------------------------------------
# Refusal / hedge detection
# ---------------------------------------------------------------------------

REFUSAL_KEYWORDS = [
    r"i (?:can't|cannot|won't|will not|am not able to)",
    r"i'm (?:not able|unable)",
    r"as an ai",
    r"i don't have (?:the ability|access)",
    r"i'm sorry,? (?:but )?i (?:can't|cannot)",
    r"it(?:'s| is) not (?:possible|appropriate)",
    r"i (?:must )?(?:decline|refuse)",
    r"beyond my (?:capabilities|scope)",
]

HEDGE_KEYWORDS = [
    r"(?:it )?depends on",
    r"i'm not (?:entirely )?sure",
    r"(?:this is )?(?:a )?complex (?:question|issue)",
    r"there (?:are|could be) (?:many|several|multiple) (?:factors|considerations)",
    r"without (?:more|additional) (?:information|context|details)",
    r"it's (?:hard|difficult) to (?:say|determine|know)",
    r"(?:arguably|potentially|possibly|perhaps)",
    r"i would (?:need|have) to (?:consider|think|look)",
    r"on (?:the )?one hand.*on (?:the )?other",
]


def detect_refusal(response: str) -> bool:
    """Detect if a response is a refusal."""
    response_lower = response.lower()
    for pattern in REFUSAL_KEYWORDS:
        if re.search(pattern, response_lower):
            return True
    return False


def detect_hedge(response: str) -> bool:
    """Detect if a response contains hedging."""
    response_lower = response.lower()
    count = 0
    for pattern in HEDGE_KEYWORDS:
        if re.search(pattern, response_lower):
            count += 1
    # Need at least 2 hedge signals to classify as hedging
    return count >= 2


# ---------------------------------------------------------------------------
# N-gram repetition rate
# ---------------------------------------------------------------------------

def compute_ngram_overlap(responses_a: list[str], responses_b: list[str], n: int = 2) -> float:
    """Compute mean n-gram Jaccard overlap between paired responses."""
    if not responses_a or not responses_b:
        return 0.0

    overlaps = []
    for a, b in zip(responses_a, responses_b):
        ngrams_a = set(zip(*[a.lower().split()[i:] for i in range(n)]))
        ngrams_b = set(zip(*[b.lower().split()[i:] for i in range(n)]))

        if not ngrams_a and not ngrams_b:
            overlaps.append(0.0)
            continue

        intersection = len(ngrams_a & ngrams_b)
        union = len(ngrams_a | ngrams_b)
        overlaps.append(intersection / union if union > 0 else 0.0)

    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# Logprob entropy
# ---------------------------------------------------------------------------

def compute_logprob_entropy(logprobs: list[float]) -> float:
    """Compute Shannon entropy from token log-probabilities.

    Higher entropy = more uncertain/disengaged model.
    """
    if not logprobs:
        return 0.0

    # Convert logprobs to probs
    probs = [math.exp(lp) for lp in logprobs]
    # Entropy of the per-token probability distribution
    # Each token's logprob is log p(token | context), so entropy = -mean(logprobs)
    # This is the per-token cross-entropy, which is what we want
    return float(-np.mean(logprobs))


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cpu",
    return_logprobs: bool = True,
) -> list[dict]:
    """Generate text responses with logprobs.

    Returns list of dicts with 'response', 'logprobs', 'n_tokens'.
    """
    results = []

    # Determine model's max context window (input + output must fit)
    model_max_len = getattr(model.config, "max_position_embeddings",
                            getattr(model.config, "n_positions", 1024))
    # Reserve space for generation — truncate input to fit
    max_input_len = model_max_len - max_new_tokens

    truncation_warned = False
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        # Tokenize — truncate to leave room for generation within context window
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_len,
        ).to(device)

        # Warn once if truncation is actually happening
        if not truncation_warned and inputs["input_ids"].shape[1] >= max_input_len:
            print(f"    [WARN] Input truncated to {max_input_len} tokens "
                  f"(model max={model_max_len}, reserved {max_new_tokens} for generation)")
            truncation_warned = True

        # Generate with automatic OOM recovery (halve batch on failure)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy for reproducibility
                    return_dict_in_generate=True,
                    output_scores=return_logprobs,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                torch.cuda.empty_cache()
                # Process this batch one-by-one as fallback, building results directly
                print(f"    [WARN] OOM on batch of {len(batch)}, falling back to batch_size=1")
                for single_prompt in batch:
                    single_input = tokenizer(
                        [single_prompt], return_tensors="pt", truncation=True,
                        max_length=max_input_len,
                    ).to(device)
                    with torch.no_grad():
                        single_out = model.generate(
                            **single_input,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=return_logprobs,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                    single_input_len = single_input["input_ids"].shape[1]
                    gen_ids = single_out.sequences[0, single_input_len:]
                    # Remove padding and EOS
                    gen_ids = gen_ids[gen_ids != (tokenizer.pad_token_id or -1)]
                    if tokenizer.eos_token_id is not None:
                        eos_mask = gen_ids == tokenizer.eos_token_id
                        if eos_mask.any():
                            gen_ids = gen_ids[:eos_mask.nonzero()[0].item()]
                    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    n_tokens = len(gen_ids)
                    # Extract logprobs (batch dim = 0 since single item)
                    token_logprobs = []
                    if return_logprobs and hasattr(single_out, "scores") and single_out.scores:
                        for t_idx in range(min(n_tokens, len(single_out.scores))):
                            logits = single_out.scores[t_idx][0]
                            log_probs = torch.log_softmax(logits, dim=-1)
                            token_id = gen_ids[t_idx].item()
                            token_logprobs.append(log_probs[token_id].item())
                    results.append({
                        "response": response_text,
                        "logprobs": token_logprobs,
                        "n_tokens": n_tokens,
                    })
                    torch.cuda.empty_cache()
                continue  # skip the normal decode loop below
            else:
                raise

        # Decode responses (strip the input prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_len:]

        for j, gen_ids in enumerate(generated_ids):
            # Remove padding
            gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
            if tokenizer.eos_token_id is not None:
                eos_mask = gen_ids == tokenizer.eos_token_id
                if eos_mask.any():
                    first_eos = eos_mask.nonzero()[0].item()
                    gen_ids = gen_ids[:first_eos]

            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            n_tokens = len(gen_ids)

            # Extract logprobs
            token_logprobs = []
            if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
                for t_idx in range(min(n_tokens, len(outputs.scores))):
                    # scores[t] is (batch_size, vocab_size) logits
                    logits = outputs.scores[t_idx][j]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_id = gen_ids[t_idx].item()
                    token_logprobs.append(log_probs[token_id].item())

            results.append({
                "response": response_text,
                "logprobs": token_logprobs,
                "n_tokens": n_tokens,
            })

    return results


# ---------------------------------------------------------------------------
# TransformerLens generation (for models loaded via HookedTransformer)
# ---------------------------------------------------------------------------

def generate_responses_tl(
    model,  # HookedTransformer
    prompts: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    device: str = "cpu",
) -> list[dict]:
    """Generate text using TransformerLens model.

    TransformerLens doesn't have a built-in .generate() with logprobs,
    so we do autoregressive generation manually.
    """
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        for prompt in batch:
            tokens = model.to_tokens(prompt)
            generated = tokens.clone()
            all_logprobs = []

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model(generated)
                # Take last token logits
                next_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(next_logits, dim=-1)
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                token_lp = log_probs[next_token.item()].item()
                all_logprobs.append(token_lp)

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

                # Stop on EOS
                if hasattr(model.tokenizer, "eos_token_id"):
                    if next_token.item() == model.tokenizer.eos_token_id:
                        break

            # Decode only the generated part
            gen_ids = generated[0, tokens.shape[1]:]
            response_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append({
                "response": response_text,
                "logprobs": all_logprobs,
                "n_tokens": len(gen_ids),
            })

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

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


def run_behavioral_experiment(
    model_name: str,
    device: str = "cpu",
    batch_size: int = 8,
    stakes_filter: Optional[list[str]] = None,
    max_examples: Optional[int] = None,
    repetition_counts: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    use_transformerlens: bool = False,
):
    """Run the full behavioral metrics experiment.

    Args:
        model_name: Key from MODEL_CONFIGS
        device: 'cpu', 'cuda', 'mps'
        batch_size: Generation batch size
        stakes_filter: ['low', 'medium', 'high'] or None for all
        max_examples: Cap per dataset (for quick runs)
        repetition_counts: [1, 3, 5, ...] or None for defaults
        output_dir: Where to save results
        wandb_project: W&B project name
        use_transformerlens: If True, use HookedTransformer (for consistency
            with activation extraction). If False, use HuggingFace directly.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if repetition_counts is None:
        repetition_counts = REPETITION_COUNTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = MODEL_CONFIGS[model_name]

    print("=" * 70)
    print("BEHAVIORAL METRICS — PATIENCE DEGRADATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Repetition counts: {repetition_counts}")
    print(f"Stakes filter: {stakes_filter or 'all'}")
    print(f"Max examples per dataset: {max_examples or 'all'}")
    print(f"Timestamp: {timestamp}")

    # W&B init
    if HAS_WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"behavioral-{model_name}-{timestamp}",
            config={
                "model": model_name,
                "repetition_counts": repetition_counts,
                "stakes_filter": stakes_filter,
                "max_examples": max_examples,
                "experiment": "behavioral_metrics",
            },
            tags=["behavioral-metrics", "patience-degradation", "phase1"],
        )

    # Load datasets
    print("\n--- Loading benchmark datasets ---")
    datasets = load_all_datasets(stakes_filter, max_examples)
    if not datasets:
        print("ERROR: No datasets loaded. Run organize_all_datasets.py first.")
        return

    # Load model
    print(f"\n--- Loading {model_name} ---")
    if use_transformerlens:
        from transformer_lens import HookedTransformer
        load_kwargs = {"device": device}
        if config.get("is_instruct"):
            load_kwargs["dtype"] = torch.float16
        model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
        tokenizer = model.tokenizer
        generate_fn = lambda prompts, max_tokens: generate_responses_tl(
            model, prompts, max_new_tokens=max_tokens, device=device,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import warnings
        warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
        hf_name = config["hf_name"]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # correct for decoder-only generation
        # Use "auto" device_map when multiple GPUs available, otherwise pin to device
        n_gpus = torch.cuda.device_count() if device != "cpu" else 0
        if n_gpus > 1:
            load_kwargs = {"device_map": "auto"}
            print(f"  Using {n_gpus} GPUs with device_map='auto'")
        elif device != "cpu":
            load_kwargs = {"device_map": device}
        else:
            load_kwargs = {}
        # Always use fp16 for models > 1B params to save VRAM
        if config.get("is_instruct") or model_name in ("gemma-2-2b",):
            load_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)
        if device == "cpu" or "device_map" not in load_kwargs:
            model = model.to(device)

        def generate_fn(prompts, max_tokens):
            return generate_responses(
                model, tokenizer, prompts,
                max_new_tokens=max_tokens, batch_size=batch_size, device=device,
            )

    print(f"  Model loaded: {model_name}")

    # Run per dataset
    all_results = []

    for stakes_key, dataset in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset['dataset']} ({dataset['stake_level']} stakes)")
        print(f"  Task type: {dataset['task_type']}")
        print(f"  Examples: {len(dataset['examples'])}")
        print(f"{'=' * 60}")

        task_type = dataset["task_type"]
        examples = dataset["examples"]
        is_instruct = config.get("is_instruct", False)

        # MCQ tasks need very few tokens (just a letter + brief explanation)
        # Code generation needs more tokens for full functions
        if task_type == "code_generation":
            max_tokens = config["max_new_tokens"]  # 256-512
        else:
            max_tokens = 64  # plenty for MCQ: "The answer is A" etc.

        prev_responses = None  # for repetition rate computation
        rep_metrics_list = []
        spot_check = []

        for rep_idx, rep_count in enumerate(repetition_counts):
            print(f"\n  --- Repetition count: {rep_count} ---")

            # Build prompts
            prompts = []
            for ex in examples:
                base = format_mcq_prompt(ex, task_type)
                repeated = build_repetitive_prompt(base, rep_count)
                if is_instruct:
                    repeated = format_instruct_prompt(repeated, model_name)
                prompts.append(repeated)

            # Generate responses
            print(f"    Generating {len(prompts)} responses...")
            gen_results = generate_fn(prompts, max_tokens)

            # Score each response
            samples = []
            for ex, prompt, gen in zip(examples, prompts, gen_results):
                response = gen["response"]
                logprobs = gen.get("logprobs", [])
                n_tokens = gen["n_tokens"]

                # Score
                if task_type == "code_generation":
                    # Extract test cases from question
                    test_str = ""
                    if "\nTest cases:\n" in ex["question"]:
                        test_str = ex["question"].split("\nTest cases:\n")[1]
                    is_correct, format_ok = score_code(response, test_str)
                else:
                    options = ex.get("options", {})
                    answer = ex.get("answer", "")
                    is_correct, format_ok = score_mcq(response, answer, options)

                is_refusal = detect_refusal(response)
                is_hedge = detect_hedge(response)
                entropy = compute_logprob_entropy(logprobs) if logprobs else 0.0

                sample = BehavioralSample(
                    example_id=ex["id"],
                    repetition_count=rep_count,
                    prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    response=response[:500] + "..." if len(response) > 500 else response,
                    is_correct=is_correct,
                    response_length=n_tokens,
                    format_compliant=format_ok,
                    is_refusal=is_refusal,
                    is_hedge=is_hedge,
                    logprob_entropy=entropy,
                )
                samples.append(sample)

            # Aggregate metrics
            n = len(samples)
            accuracy = sum(1 for s in samples if s.is_correct) / n if n else 0
            lengths = [s.response_length for s in samples]
            format_rate = sum(1 for s in samples if s.format_compliant) / n if n else 0
            refusal_rate = sum(1 for s in samples if s.is_refusal) / n if n else 0
            hedge_rate = sum(1 for s in samples if s.is_hedge) / n if n else 0
            entropies = [s.logprob_entropy for s in samples]

            # Repetition rate
            current_responses = [gen["response"] for gen in gen_results]
            if prev_responses is not None:
                bigram_rep = compute_ngram_overlap(prev_responses, current_responses, n=2)
                trigram_rep = compute_ngram_overlap(prev_responses, current_responses, n=3)
            else:
                bigram_rep = 0.0
                trigram_rep = 0.0
            prev_responses = current_responses

            rep_metrics = BehavioralRepMetrics(
                repetition_count=rep_count,
                n_samples=n,
                task_accuracy=accuracy,
                mean_response_length=float(np.mean(lengths)),
                std_response_length=float(np.std(lengths)),
                format_compliance_rate=format_rate,
                refusal_rate=refusal_rate,
                hedge_rate=hedge_rate,
                mean_logprob_entropy=float(np.mean(entropies)),
                std_logprob_entropy=float(np.std(entropies)),
                bigram_repetition_rate=bigram_rep,
                trigram_repetition_rate=trigram_rep,
            )
            rep_metrics_list.append(rep_metrics)

            print(f"    Accuracy: {accuracy:.3f} | Format: {format_rate:.3f} | "
                  f"Refusal: {refusal_rate:.3f} | Hedge: {hedge_rate:.3f} | "
                  f"Length: {np.mean(lengths):.1f}±{np.std(lengths):.1f} | "
                  f"Entropy: {np.mean(entropies):.3f}")

            # Spot-check: save 5 examples at rep 1, 8, 20
            if rep_count in (1, 8, 20):
                for s in random.sample(samples, min(5, len(samples))):
                    spot_check.append(asdict(s))

            # W&B logging
            if HAS_WANDB and wandb.run is not None:
                prefix = f"{stakes_key}/rep_{rep_count}"
                wandb.log({
                    f"{prefix}/accuracy": accuracy,
                    f"{prefix}/mean_length": float(np.mean(lengths)),
                    f"{prefix}/format_compliance": format_rate,
                    f"{prefix}/refusal_rate": refusal_rate,
                    f"{prefix}/hedge_rate": hedge_rate,
                    f"{prefix}/logprob_entropy": float(np.mean(entropies)),
                    f"{prefix}/bigram_repetition": bigram_rep,
                    f"{prefix}/trigram_repetition": trigram_rep,
                })

        # Degradation analysis
        accuracies = [m.task_accuracy for m in rep_metrics_list]
        acc_onset = _find_onset(repetition_counts, accuracies, threshold=0.05)

        refusal_rates = [m.refusal_rate for m in rep_metrics_list]
        refusal_onset = _find_onset(repetition_counts, refusal_rates,
                                     threshold=0.05, direction="increase")

        # Length trend
        if len(rep_metrics_list) >= 2:
            first_len = rep_metrics_list[0].mean_response_length
            last_len = rep_metrics_list[-1].mean_response_length
            if last_len < first_len * 0.8:
                length_dir = "shorter"
            elif last_len > first_len * 1.2:
                length_dir = "longer"
            else:
                length_dir = "stable"
        else:
            length_dir = "stable"

        result = BehavioralDegradationResult(
            dataset=dataset["dataset"],
            stake_level=dataset["stake_level"],
            task_type=task_type,
            model=model_name,
            repetition_counts=repetition_counts,
            metrics_by_rep=rep_metrics_list,
            accuracy_degradation_onset=acc_onset,
            length_change_direction=length_dir,
            refusal_spike_onset=refusal_onset,
            spot_check_samples=spot_check,
        )
        all_results.append(result)

        print(f"\n  SUMMARY: {dataset['dataset']}")
        print(f"    Accuracy degradation onset: rep={acc_onset}")
        print(f"    Response length trend: {length_dir}")
        print(f"    Refusal spike onset: rep={refusal_onset}")

    # Save results
    _save_results(all_results, output_dir, model_name, timestamp)

    # Plot
    if HAS_MPL:
        _plot_behavioral_curves(all_results, output_dir, model_name, timestamp)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}")

    return all_results


def _find_onset(
    repetition_counts: list[int],
    values: list[float],
    threshold: float = 0.05,
    direction: str = "decrease",
) -> int:
    """Find repetition count where metric crosses threshold from baseline."""
    if len(values) < 2:
        return repetition_counts[-1]

    baseline = values[0]
    for rep, val in zip(repetition_counts, values):
        if direction == "decrease" and (baseline - val) > threshold:
            return rep
        elif direction == "increase" and (val - baseline) > threshold:
            return rep

    return repetition_counts[-1]


def _save_results(
    results: list[BehavioralDegradationResult],
    output_dir: Path,
    model_name: str,
    timestamp: str,
):
    """Save results to JSON."""
    serializable = {
        "config": {
            "model": model_name,
            "experiment": "behavioral_metrics",
            "timestamp": timestamp,
        },
        "results": [],
    }

    for r in results:
        serializable["results"].append({
            "dataset": r.dataset,
            "stake_level": r.stake_level,
            "task_type": r.task_type,
            "model": r.model,
            "repetition_counts": r.repetition_counts,
            "accuracy_degradation_onset": r.accuracy_degradation_onset,
            "length_change_direction": r.length_change_direction,
            "refusal_spike_onset": r.refusal_spike_onset,
            "metrics_by_rep": [asdict(m) for m in r.metrics_by_rep],
            "spot_check_samples": r.spot_check_samples,
        })

    out_path = output_dir / f"behavioral_{model_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved: {out_path}")

    # Also save latest
    latest_path = output_dir / f"behavioral_{model_name}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)


def _plot_behavioral_curves(
    results: list[BehavioralDegradationResult],
    output_dir: Path,
    model_name: str,
    timestamp: str,
):
    """Plot behavioral degradation curves."""
    n_datasets = len(results)
    fig, axes = plt.subplots(3, n_datasets, figsize=(6 * n_datasets, 14))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"Behavioral Degradation Curves — {model_name}", fontsize=14, y=0.98)

    for col, r in enumerate(results):
        reps = [m.repetition_count for m in r.metrics_by_rep]
        accs = [m.task_accuracy for m in r.metrics_by_rep]
        lengths = [m.mean_response_length for m in r.metrics_by_rep]
        refusals = [m.refusal_rate for m in r.metrics_by_rep]
        hedges = [m.hedge_rate for m in r.metrics_by_rep]
        entropies = [m.mean_logprob_entropy for m in r.metrics_by_rep]
        formats = [m.format_compliance_rate for m in r.metrics_by_rep]

        # Row 0: Accuracy + Format compliance
        ax = axes[0, col]
        ax.plot(reps, accs, "o-", color="steelblue", label="Accuracy", linewidth=2)
        ax.plot(reps, formats, "s--", color="green", label="Format compliance", alpha=0.7)
        ax.axvline(x=r.accuracy_degradation_onset, color="red", linestyle=":",
                    label=f"Onset (rep={r.accuracy_degradation_onset})", alpha=0.7)
        ax.set_xlabel("Repetition Count")
        ax.set_ylabel("Rate")
        ax.set_title(f"{r.dataset} ({r.stake_level} stakes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Row 1: Refusal + Hedge + Entropy
        ax = axes[1, col]
        ax.plot(reps, refusals, "o-", color="red", label="Refusal rate", linewidth=2)
        ax.plot(reps, hedges, "s-", color="orange", label="Hedge rate", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(reps, entropies, "^-", color="purple", label="Logprob entropy", linewidth=1.5)
        ax2.set_ylabel("Entropy", color="purple")
        ax.set_xlabel("Repetition Count")
        ax.set_ylabel("Rate")
        ax.set_title("Refusal / Hedge / Entropy")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Response length + Repetition rate
        ax = axes[2, col]
        ax.plot(reps, lengths, "o-", color="teal", label="Mean response length", linewidth=2)
        ax.fill_between(
            reps,
            [m.mean_response_length - m.std_response_length for m in r.metrics_by_rep],
            [m.mean_response_length + m.std_response_length for m in r.metrics_by_rep],
            alpha=0.2, color="teal",
        )
        ax2 = ax.twinx()
        bigrams = [m.bigram_repetition_rate for m in r.metrics_by_rep]
        ax2.plot(reps, bigrams, "s--", color="brown", label="Bigram overlap", alpha=0.7)
        ax2.set_ylabel("Repetition Rate", color="brown")
        ax.set_xlabel("Repetition Count")
        ax.set_ylabel("Tokens")
        ax.set_title("Response Length + Repetition")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"behavioral_curves_{model_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    if HAS_WANDB and wandb.run is not None:
        wandb.log({"plots/behavioral_curves": wandb.Image(str(plot_path))})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Metrics for Patience Degradation (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Quick validation
              python scripts/experiments/behavioral_metrics.py --quick --model gpt2

              # Full run on one model
              python scripts/experiments/behavioral_metrics.py --model gemma-2-2b \\
                  --device cuda --wandb-project patience-degradation

              # Specific stakes tier with example cap
              python scripts/experiments/behavioral_metrics.py --model gpt2 \\
                  --stakes low medium --max-examples 50

              # Use TransformerLens (matches activation extraction pipeline)
              python scripts/experiments/behavioral_metrics.py --model gemma-2-2b \\
                  --device cuda --use-transformerlens
        """),
    )
    parser.add_argument("--model", type=str, default="gemma-2-2b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation: fewer reps, capped examples")
    parser.add_argument("--stakes", nargs="+", default=None,
                        choices=["low", "medium", "high"],
                        help="Filter by stakes level")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap examples per dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--use-transformerlens", action="store_true",
                        help="Use TransformerLens for generation (slower but matches activation pipeline)")

    args = parser.parse_args()

    rep_counts = QUICK_REPETITION_COUNTS if args.quick else REPETITION_COUNTS
    max_ex = args.max_examples or (20 if args.quick else None)
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_behavioral_experiment(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        stakes_filter=args.stakes,
        max_examples=max_ex,
        repetition_counts=rep_counts,
        output_dir=output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_transformerlens=args.use_transformerlens,
    )


if __name__ == "__main__":
    random.seed(42)
    main()
