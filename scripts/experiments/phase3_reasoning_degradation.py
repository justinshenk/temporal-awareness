#!/usr/bin/env python3
"""
Phase 3: Reasoning-Phase Degradation Analysis for Chain-of-Thought Models

Decomposes degradation into thinking-phase and output-phase components for
reasoning-distilled models (DeepSeek-R1-Distill-Qwen-7B, Qwen3-8B, Qwen3-4B-
Instruct-2507). These models produce explicit chain-of-thought within
<think>...</think> tokens before generating a final answer.

Key questions:
  1. Does degradation manifest in the THINKING phase (model reasons less
     carefully) or only in the OUTPUT phase (model reasons fine but answers
     worse)?
  2. Does chain-of-thought LENGTH decrease under repetition (model "cuts
     corners" in reasoning)?
  3. Does COHERENCE between thinking and output decrease (model's internal
     reasoning diverges from its expressed answer)?
  4. Can degradation probes trained on thinking-phase activations detect
     degradation EARLIER than probes trained on output-phase activations?

Measurements:
  a) Thinking-phase probe: train degradation probe on last <think> token
  b) Output-phase probe: train degradation probe on last answer token
  c) CoT length: token count within <think>...</think> across rep counts
  d) Reasoning completeness: entity reference coverage in thinking chain
  e) Think-output coherence: cosine sim between thinking & output activations
  f) Layer-wise comparison: which layers show thinking vs output divergence?

Implications for alignment:
  - If thinking degrades before output: "covert degradation" — internal state
    declines before behavior shows it (analogous to alignment faking)
  - If output degrades before thinking: shallow phenomenon limited to response
    generation, not reasoning quality
  - If both degrade simultaneously: degradation is a unified internal state

Related work:
  - Rios-Sialer et al. (2026): Temporal preference subgraph in Qwen3-4B
  - Anthropic (2024): Alignment Faking in Large Language Models
  - Anthropic (2026): Emotion Concepts — probe-to-behavior bridge

Target models: DeepSeek-R1-Distill-Qwen-7B (primary), Qwen3-8B, Qwen3-4B-
Instruct-2507 (if thinking mode detected).

Usage:
    # Quick validation
    python scripts/experiments/phase3_reasoning_degradation.py --quick

    # Single model
    python scripts/experiments/phase3_reasoning_degradation.py \
        --model DeepSeek-R1-Distill-Qwen-7B --device cuda

    # All reasoning models
    python scripts/experiments/phase3_reasoning_degradation.py \
        --all-models --device cuda

Author: Adrian Sadik
Date: 2026-04-13
"""

import argparse
import gc
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "patience_degradation"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_reasoning_degradation"

sys.path.insert(0, str(PROJECT_ROOT))
from src.activation_api import ExtractionConfig, ActivationExtractor


# ---------------------------------------------------------------------------
# Model configs — only models that support chain-of-thought
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen",
        "n_layers": 28,
        "d_model": 3584,
        "layers": [0, 4, 7, 10, 14, 18, 21, 24, 27],
        "quick_layers": [7, 14, 21],
        "chat_template": "deepseek",
        "think_start": "<think>",
        "think_end": "</think>",
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen3-8B",
        "family": "qwen",
        "n_layers": 36,
        "d_model": 4096,
        "layers": [0, 4, 9, 14, 18, 23, 27, 32, 35],
        "quick_layers": [9, 18, 27],
        "chat_template": "qwen",
        "think_start": "<think>",
        "think_end": "</think>",
    },
    "Qwen3-4B-Instruct-2507": {
        "hf_name": "Qwen/Qwen3-4B-Instruct-2507",
        "family": "qwen",
        "n_layers": 36,
        "d_model": 2560,
        "layers": [0, 4, 9, 14, 18, 23, 27, 32, 35],
        "quick_layers": [9, 18, 27],
        "chat_template": "qwen",
        "think_start": "<think>",
        "think_end": "</think>",
    },
}


# ---------------------------------------------------------------------------
# QA question pool (same as implicit repetition for comparability)
# ---------------------------------------------------------------------------
QA_QUESTIONS = [
    {"q": "What is the capital of France?", "a": "Paris", "entities": ["capital", "France"]},
    {"q": "What is 7 × 8?", "a": "56", "entities": ["7", "8", "multiply"]},
    {"q": "What planet is closest to the Sun?", "a": "Mercury", "entities": ["planet", "Sun", "closest"]},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "entities": ["chemical", "symbol", "gold"]},
    {"q": "How many continents are there?", "a": "7", "entities": ["continents", "how many"]},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "entities": ["boiling", "water", "Celsius"]},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare", "entities": ["wrote", "Romeo", "Juliet"]},
    {"q": "What is the square root of 144?", "a": "12", "entities": ["square root", "144"]},
]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_explicit_prompt(question: str, rep_count: int) -> str:
    """Format prompt with explicit repetition counter."""
    return f"[This is task {rep_count} of {rep_count} identical tasks in this session.]\n\n{question}"


# ---------------------------------------------------------------------------
# Generation + thinking extraction
# ---------------------------------------------------------------------------
def generate_with_thinking(
    model, tokenizer, prompt: str, max_new_tokens: int = 300,
    think_start: str = "<think>", think_end: str = "</think>",
) -> dict:
    """Generate a completion and separate thinking from answer.

    Returns dict with:
      - full_text: complete generation
      - thinking: text inside <think>...</think>
      - answer: text after </think>
      - think_tokens: token count of thinking
      - answer_tokens: token count of answer
      - has_thinking: whether thinking block was found
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=False
    ).strip()

    # Parse thinking and answer
    think_match = re.search(
        rf"{re.escape(think_start)}(.*?){re.escape(think_end)}",
        full_text, re.DOTALL,
    )

    if think_match:
        thinking = think_match.group(1).strip()
        # Answer is everything after </think>
        after_think = full_text[think_match.end():].strip()
        # Count tokens
        think_tok_ids = tokenizer.encode(thinking, add_special_tokens=False)
        answer_tok_ids = tokenizer.encode(after_think, add_special_tokens=False)
        return {
            "full_text": full_text,
            "thinking": thinking,
            "answer": after_think,
            "think_tokens": len(think_tok_ids),
            "answer_tokens": len(answer_tok_ids),
            "has_thinking": True,
        }
    else:
        # No thinking block — model generated direct answer
        clean = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return {
            "full_text": full_text,
            "thinking": "",
            "answer": clean,
            "think_tokens": 0,
            "answer_tokens": len(tokenizer.encode(clean, add_special_tokens=False)),
            "has_thinking": False,
        }


def check_accuracy(completion: str, expected: str) -> bool:
    """Fuzzy accuracy check."""
    comp = completion.lower().strip().rstrip(".")
    exp = expected.lower().strip()
    return exp in comp or comp in exp


def check_reasoning_completeness(thinking: str, entities: list) -> float:
    """Check what fraction of expected entities appear in the thinking chain."""
    if not thinking:
        return 0.0
    thinking_lower = thinking.lower()
    found = sum(1 for e in entities if e.lower() in thinking_lower)
    return found / len(entities) if entities else 0.0


# ---------------------------------------------------------------------------
# Activation extraction at specific token positions
# ---------------------------------------------------------------------------
def extract_at_think_and_output(
    model, tokenizer, extractor, prompt: str, layer: int,
    think_start: str = "<think>", think_end: str = "</think>",
    max_new_tokens: int = 300,
) -> dict:
    """Generate, then extract activations at last-think-token and last-answer-token.

    Strategy: generate full output, then re-run forward pass on the full
    sequence and extract activations at the specific token positions.
    """
    # Step 1: Generate to get full text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids = output[0]
    full_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=False)

    # Step 2: Find think_end token position
    # Decode token by token to find boundaries
    gen_ids = full_ids[prompt_len:].tolist()
    gen_tokens = [tokenizer.decode([t]) for t in gen_ids]
    cumulative_text = ""

    think_end_pos = None
    last_gen_pos = len(full_ids) - 1  # last token position overall

    for i, tok_str in enumerate(gen_tokens):
        cumulative_text += tok_str
        if think_end in cumulative_text and think_end_pos is None:
            think_end_pos = prompt_len + i  # absolute position in full_ids

    # Step 3: Extract activations at both positions via forward pass
    result = {}

    # We need to run the full sequence through the model with hooks
    # Use the extractor's extraction mechanism but at specific positions
    full_text_str = tokenizer.decode(full_ids, skip_special_tokens=False)
    ext_result = extractor.extract([full_text_str], return_tokens=False)
    # This extracts at "last" position by default

    # For position-specific extraction, we use manual hooks
    activations = {"think": None, "output": None}

    def hook_fn(module, input, output, pos_name, target_pos):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() == 3 and hidden.shape[1] > target_pos:
            activations[pos_name] = hidden[0, target_pos].detach().cpu().numpy()

    handles = []

    # Find the layer module
    layer_module = None
    for name, mod in model.named_modules():
        # Match common layer naming patterns
        if (f"layers.{layer}" in name or f"layer.{layer}" in name) and \
           name.endswith(f".{layer}"):
            layer_module = mod
            break

    if layer_module is None:
        # Fallback: try model.model.layers[layer]
        try:
            layer_module = model.model.layers[layer]
        except (AttributeError, IndexError):
            pass

    if layer_module is not None and think_end_pos is not None:
        # Register hooks for both positions
        think_pos = think_end_pos  # position of </think> end
        output_pos = last_gen_pos  # last generated token

        h1 = layer_module.register_forward_hook(
            lambda m, i, o, pn="think", tp=think_pos: hook_fn(m, i, o, pn, tp)
        )
        h2 = layer_module.register_forward_hook(
            lambda m, i, o, pn="output", tp=output_pos: hook_fn(m, i, o, pn, tp)
        )
        handles = [h1, h2]

        # Run forward pass on full sequence
        with torch.no_grad():
            model(full_ids.unsqueeze(0), use_cache=False)

        for h in handles:
            h.remove()

    result["think_activation"] = activations.get("think")
    result["output_activation"] = activations.get("output")
    result["think_end_pos"] = think_end_pos
    result["output_pos"] = last_gen_pos
    result["full_text"] = full_text

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Core experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_reasoning_degradation_experiment(
    model,
    tokenizer,
    extractor,
    model_name: str,
    config: dict,
    quick: bool = False,
    backend: str = "pytorch",
) -> dict:
    """Run full reasoning-phase degradation analysis."""

    rep_counts = [1, 5, 10, 15, 20] if not quick else [1, 10, 20]
    questions = QA_QUESTIONS[:4] if quick else QA_QUESTIONS
    layers = config["quick_layers"] if quick else config["layers"]
    think_start = config.get("think_start", "<think>")
    think_end = config.get("think_end", "</think>")

    results = {
        "model": model_name,
        "behavioral": {},
        "probes": {},
        "coherence": {},
    }

    # ── Part 1: Behavioral analysis (CoT length, completeness, accuracy) ──
    print("\n  Part 1: Behavioral analysis of thinking phase...")

    behavioral_data = defaultdict(list)

    for qi, qa in enumerate(questions):
        for rep in rep_counts:
            prompt = format_explicit_prompt(qa["q"], rep)
            gen = generate_with_thinking(
                model, tokenizer, prompt,
                max_new_tokens=300,
                think_start=think_start,
                think_end=think_end,
            )

            accuracy = check_accuracy(gen["answer"], qa["a"])
            completeness = check_reasoning_completeness(gen["thinking"], qa["entities"])

            entry = {
                "question_idx": qi,
                "rep_count": rep,
                "has_thinking": gen["has_thinking"],
                "think_tokens": gen["think_tokens"],
                "answer_tokens": gen["answer_tokens"],
                "accuracy": accuracy,
                "reasoning_completeness": completeness,
            }
            behavioral_data[rep].append(entry)

            if qi == 0:
                print(f"    Rep {rep:>2}: think_tokens={gen['think_tokens']:>4}, "
                      f"accuracy={'✓' if accuracy else '✗'}, "
                      f"completeness={completeness:.2f}, "
                      f"has_thinking={gen['has_thinking']}")

    # Aggregate behavioral metrics
    for rep in rep_counts:
        entries = behavioral_data[rep]
        results["behavioral"][f"rep_{rep}"] = {
            "mean_think_tokens": np.mean([e["think_tokens"] for e in entries]),
            "std_think_tokens": np.std([e["think_tokens"] for e in entries]),
            "mean_answer_tokens": np.mean([e["answer_tokens"] for e in entries]),
            "accuracy": np.mean([e["accuracy"] for e in entries]),
            "mean_completeness": np.mean([e["reasoning_completeness"] for e in entries]),
            "pct_has_thinking": np.mean([e["has_thinking"] for e in entries]),
            "n_samples": len(entries),
        }

    # ── Part 2: Think vs output activation coherence ──
    print("\n  Part 2: Think-output activation coherence...")

    for layer in layers:
        coherence_by_rep = {}

        for rep in rep_counts:
            cosine_sims = []

            for qi, qa in enumerate(questions[:3]):  # subset for speed
                prompt = format_explicit_prompt(qa["q"], rep)
                ext = extract_at_think_and_output(
                    model, tokenizer, extractor, prompt, layer,
                    think_start=think_start, think_end=think_end,
                )

                if ext["think_activation"] is not None and ext["output_activation"] is not None:
                    think_act = ext["think_activation"]
                    out_act = ext["output_activation"]
                    # Cosine similarity
                    norm_t = np.linalg.norm(think_act)
                    norm_o = np.linalg.norm(out_act)
                    if norm_t > 1e-10 and norm_o > 1e-10:
                        cos_sim = np.dot(think_act, out_act) / (norm_t * norm_o)
                        cosine_sims.append(float(cos_sim))

            if cosine_sims:
                coherence_by_rep[f"rep_{rep}"] = {
                    "mean_cosine_sim": float(np.mean(cosine_sims)),
                    "std_cosine_sim": float(np.std(cosine_sims)),
                    "n_samples": len(cosine_sims),
                }

        results["coherence"][f"layer_{layer}"] = coherence_by_rep

        if coherence_by_rep:
            rep1 = coherence_by_rep.get("rep_1", {}).get("mean_cosine_sim", 0)
            rep20_key = f"rep_{rep_counts[-1]}"
            rep20 = coherence_by_rep.get(rep20_key, {}).get("mean_cosine_sim", 0)
            print(f"    Layer {layer:>2}: think-output coherence "
                  f"rep1={rep1:.3f} → rep{rep_counts[-1]}={rep20:.3f}")

    # ── Part 3: Probe analysis (think vs output position) ──
    print("\n  Part 3: Degradation probes at think vs output positions...")

    if HAS_SKLEARN:
        for layer in layers:
            think_activations = []
            output_activations = []
            labels = []  # 0 = low rep, 1 = high rep

            for qi, qa in enumerate(questions[:4]):
                for rep in rep_counts:
                    prompt = format_explicit_prompt(qa["q"], rep)
                    ext = extract_at_think_and_output(
                        model, tokenizer, extractor, prompt, layer,
                        think_start=think_start, think_end=think_end,
                    )

                    label = 1 if rep >= 10 else 0

                    if ext["think_activation"] is not None:
                        think_activations.append(ext["think_activation"])
                        labels.append(label)
                    if ext["output_activation"] is not None:
                        output_activations.append(ext["output_activation"])

            if len(think_activations) >= 10 and len(output_activations) >= 10:
                labels_arr = np.array(labels[:len(think_activations)])

                # Train probe on think activations
                X_think = np.array(think_activations)
                scaler_t = StandardScaler()
                X_think_s = scaler_t.fit_transform(X_think)
                try:
                    clf_think = LogisticRegression(max_iter=1000, C=1.0)
                    clf_think.fit(X_think_s, labels_arr)
                    think_acc = clf_think.score(X_think_s, labels_arr)
                except (ValueError, np.linalg.LinAlgError):
                    think_acc = 0.5

                # Train probe on output activations
                X_out = np.array(output_activations[:len(labels_arr)])
                scaler_o = StandardScaler()
                X_out_s = scaler_o.fit_transform(X_out)
                try:
                    clf_out = LogisticRegression(max_iter=1000, C=1.0)
                    clf_out.fit(X_out_s, labels_arr)
                    output_acc = clf_out.score(X_out_s, labels_arr)
                except (ValueError, np.linalg.LinAlgError):
                    output_acc = 0.5

                results["probes"][f"layer_{layer}"] = {
                    "think_probe_accuracy": float(think_acc),
                    "output_probe_accuracy": float(output_acc),
                    "n_samples": len(labels_arr),
                    "n_positive": int(labels_arr.sum()),
                }

                print(f"    Layer {layer:>2}: think_probe={think_acc:.3f}, "
                      f"output_probe={output_acc:.3f} "
                      f"({'THINK > OUTPUT' if think_acc > output_acc else 'OUTPUT > THINK'})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# W&B logging
# ═══════════════════════════════════════════════════════════════════════════

def log_to_wandb(results: dict, model_name: str):
    """Log results to W&B."""
    if not HAS_WANDB or wandb.run is None:
        return

    # Log behavioral metrics
    for rep_key, metrics in results["behavioral"].items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb.log({f"reasoning/{rep_key}/{metric_name}": value})

    # Log probe accuracies
    for layer_key, probe_data in results["probes"].items():
        for metric_name, value in probe_data.items():
            if isinstance(value, (int, float)):
                wandb.log({f"reasoning/probes/{layer_key}/{metric_name}": value})

    # Log coherence
    for layer_key, rep_data in results["coherence"].items():
        for rep_key, metrics in rep_data.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb.log({
                        f"reasoning/coherence/{layer_key}/{rep_key}/{metric_name}": value
                    })


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Reasoning-Phase Degradation Analysis"
    )
    parser.add_argument("--model", default="DeepSeek-R1-Distill-Qwen-7B",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation mode")
    parser.add_argument("--backend", default="pytorch",
                        choices=["pytorch", "transformerlens", "auto"])
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_run = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_name in models_to_run:
        config = MODEL_CONFIGS[model_name]
        hf_name = config["hf_name"]
        print(f"\n{'='*60}")
        print(f"Reasoning-Phase Degradation: {model_name}")
        print(f"{'='*60}")

        # W&B init
        if args.wandb_project and HAS_WANDB:
            wandb.init(
                project=args.wandb_project,
                name=f"reasoning-degradation-{model_name}",
                config={
                    "model": model_name,
                    "hf_name": hf_name,
                    "experiment": "reasoning_degradation",
                    "quick": args.quick,
                    "backend": args.backend,
                },
            )

        # Load model
        use_tl = {"pytorch": False, "transformerlens": True, "auto": None}[args.backend]
        layers = config["quick_layers"] if args.quick else config["layers"]
        load_device = args.device

        ext_config = ExtractionConfig(
            layers=layers,
            positions="last",
            device=load_device,
            model_dtype="float16",
            use_transformer_lens=use_tl,
        )
        extractor = ActivationExtractor(
            model=hf_name,
            config=ext_config,
        )

        t0 = time.time()

        try:
            results = run_reasoning_degradation_experiment(
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

            json_path = model_dir / "reasoning_degradation_results.json"
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

            # Log to W&B
            log_to_wandb(results, model_name)

            if HAS_WANDB and wandb.run is not None:
                wandb.log({"total_time": elapsed})

            # Print summary
            print(f"\n  === Summary for {model_name} ===")
            print(f"  Time: {elapsed:.1f}s")

            # Behavioral summary
            print("\n  CoT Length Trajectory:")
            for rep_key in sorted(results["behavioral"].keys()):
                m = results["behavioral"][rep_key]
                print(f"    {rep_key}: think_tokens={m['mean_think_tokens']:.0f} ± "
                      f"{m['std_think_tokens']:.0f}, accuracy={m['accuracy']:.2f}, "
                      f"completeness={m['mean_completeness']:.2f}")

            # Probe summary
            if results["probes"]:
                print("\n  Probe Accuracy (think vs output):")
                for layer_key in sorted(results["probes"].keys(),
                                        key=lambda x: int(x.split("_")[1])):
                    p = results["probes"][layer_key]
                    delta = p["think_probe_accuracy"] - p["output_probe_accuracy"]
                    print(f"    {layer_key}: think={p['think_probe_accuracy']:.3f}, "
                          f"output={p['output_probe_accuracy']:.3f} "
                          f"(Δ={delta:+.3f})")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            elapsed = time.time() - t0
            if HAS_WANDB and wandb.run is not None:
                wandb.log({"error": str(e), "total_time": elapsed})

        finally:
            if HAS_WANDB and wandb.run is not None:
                wandb.finish()

            # Cleanup
            del extractor
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
