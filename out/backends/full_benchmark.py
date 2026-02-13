#!/usr/bin/env python3
"""Comprehensive backend benchmark across multiple models."""

import json
import sys
import time
import gc
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import ModelRunner
from src.inference.backends import ModelBackend

# Models to test
# Format: (hf_name, tl_name, mlx_name, display_name, size_category)
# mlx_name can be None if no MLX version available
MODELS = [
    # Small models (<1B) - GPT-2 family (no MLX)
    ("gpt2", "gpt2", None, "GPT-2 (124M)", "small"),
    ("gpt2-medium", "gpt2-medium", None, "GPT-2-Medium (355M)", "small"),
    ("gpt2-large", "gpt2-large", None, "GPT-2-Large (774M)", "small"),
    ("gpt2-xl", "gpt2-xl", None, "GPT-2-XL (1.5B)", "small"),

    # Small models - Pythia family (no MLX)
    ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m", None, "Pythia-160M", "small"),
    ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m", None, "Pythia-410M", "small"),

    # Small models - Qwen family (has MLX)
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", "small"),

    # Medium models (1-3B)
    ("EleutherAI/pythia-1b", "EleutherAI/pythia-1b", None, "Pythia-1B", "medium"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "medium"),
    ("EleutherAI/pythia-2.8b", "EleutherAI/pythia-2.8b", None, "Pythia-2.8B", "medium"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "medium"),

    # Large models (5B+)
    ("EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b", None, "Pythia-6.9B", "large"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "large"),
    ("mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", "mlx-community/Mistral-7B-Instruct-v0.1-4bit", "Mistral-7B", "large"),
]

# Backends to test
BACKENDS = [
    ModelBackend.HUGGINGFACE,
    ModelBackend.PYVENE,
    ModelBackend.TRANSFORMERLENS,
    ModelBackend.MLX,
]

# Test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "def fibonacci(n):",
    "Once upon a time",
]

OUT_DIR = Path("out/backends")


@dataclass
class BackendMetrics:
    """Metrics for a single backend run."""
    backend: str
    model: str
    status: str = "OK"
    error: str = ""

    # Timing
    load_time_ms: float = 0
    forward_time_ms: float = 0
    generate_time_ms: float = 0
    cache_time_ms: float = 0

    # Memory
    memory_gb: float = 0

    # Generation
    generated_texts: dict = field(default_factory=dict)  # prompt -> generated

    # Logits
    logits_shape: tuple = ()
    logits_mean: float = 0
    logits_std: float = 0

    # Top-k predictions at last position (for comparison)
    top_tokens: dict = field(default_factory=dict)  # prompt -> list of top 10000 tokens
    top_probs: dict = field(default_factory=dict)   # prompt -> list of top 10000 probs

    # Full probability distribution for KL divergence
    full_probs: dict = field(default_factory=dict)  # prompt -> full vocab probs (stored as list)

    # Activation cache info
    num_cached: int = 0
    activation_sample: dict = field(default_factory=dict)  # sample activations for comparison


def get_memory_gb():
    """Get current MPS memory usage."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024**3)
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_backend(model_name: str, tl_name: str, mlx_name: str, backend: ModelBackend, prompts: list[str]) -> BackendMetrics:
    """Run comprehensive tests for a single backend."""
    metrics = BackendMetrics(backend=backend.name, model=model_name)

    try:
        clear_memory()

        # Use backend-specific model names
        if backend == ModelBackend.MLX:
            if mlx_name is None:
                metrics.status = "SKIPPED"
                metrics.error = "No MLX model available"
                return metrics
            actual_model = mlx_name
        elif backend == ModelBackend.TRANSFORMERLENS:
            actual_model = tl_name
        else:
            actual_model = model_name

        # Load model
        start = time.perf_counter()
        runner = ModelRunner(actual_model, backend=backend)
        metrics.load_time_ms = (time.perf_counter() - start) * 1000
        metrics.memory_gb = get_memory_gb()

        for prompt in prompts:
            # Forward pass with cache
            start = time.perf_counter()
            logits, cache = runner.run_with_cache(prompt)
            metrics.cache_time_ms += (time.perf_counter() - start) * 1000

            # Store logits info (from last prompt)
            metrics.logits_shape = tuple(logits.shape)
            metrics.logits_mean = float(logits.float().mean())
            metrics.logits_std = float(logits.float().std())

            # Get top 10000 tokens at last position
            last_logits = logits[0, -1, :].float()
            probs = torch.softmax(last_logits, dim=-1)

            # Store full probs for KL divergence (convert to list for JSON)
            metrics.full_probs[prompt] = probs.cpu().numpy().tolist()

            # Get top 10000 (or vocab size if smaller)
            k = min(10000, probs.shape[0])
            top_probs_tensor, top_indices = torch.topk(probs, k)
            metrics.top_tokens[prompt] = top_indices.cpu().tolist()
            metrics.top_probs[prompt] = top_probs_tensor.cpu().tolist()

            # Store cache info
            if cache:
                metrics.num_cached = len(cache)
                # Sample a few activations for comparison
                for name, tensor in list(cache.items())[:3]:
                    if isinstance(tensor, torch.Tensor):
                        # Store mean, std, and a few sample values
                        metrics.activation_sample[name] = {
                            "shape": list(tensor.shape),
                            "mean": float(tensor.float().mean()),
                            "std": float(tensor.float().std()),
                            "sample_values": tensor[0, 0, :5].float().cpu().tolist() if len(tensor.shape) >= 3 else []
                        }

            # Generate text
            start = time.perf_counter()
            generated = runner.generate(prompt, max_new_tokens=30, temperature=0.0)
            metrics.generate_time_ms += (time.perf_counter() - start) * 1000
            metrics.generated_texts[prompt] = generated

        # Forward pass timing (separate from cache)
        start = time.perf_counter()
        input_ids = runner.tokenize(runner._apply_chat_template(prompts[0]))
        with torch.no_grad():
            _ = runner._backend.forward(input_ids)
        metrics.forward_time_ms = (time.perf_counter() - start) * 1000

        del runner
        clear_memory()

    except Exception as e:
        metrics.status = "ERROR"
        metrics.error = str(e)
        traceback.print_exc()
        clear_memory()

    return metrics


def compare_metrics(ref: BackendMetrics, other: BackendMetrics, prompt: str) -> dict:
    """Compare metrics between reference and another backend."""
    comparison = {
        "backend": other.backend,
        "status": other.status,
    }

    if ref.status != "OK" or other.status != "OK":
        return comparison

    # Generation comparison
    ref_gen = ref.generated_texts.get(prompt, "")
    other_gen = other.generated_texts.get(prompt, "")
    comparison["gen_match"] = ref_gen == other_gen
    comparison["ref_gen"] = ref_gen[:100]
    comparison["other_gen"] = other_gen[:100]

    # Top-k comparisons
    ref_tokens = ref.top_tokens.get(prompt, [])
    other_tokens = other.top_tokens.get(prompt, [])

    for k in [1, 5, 10, 20, 100, 1000, 10000]:
        if len(ref_tokens) >= k and len(other_tokens) >= k:
            ref_set = set(ref_tokens[:k])
            other_set = set(other_tokens[:k])
            comparison[f"top{k}_match"] = ref_set == other_set
            comparison[f"top{k}_overlap"] = len(ref_set & other_set) / k
        else:
            comparison[f"top{k}_match"] = None
            comparison[f"top{k}_overlap"] = None

    # Probability comparison
    ref_probs = ref.full_probs.get(prompt)
    other_probs = other.full_probs.get(prompt)

    if ref_probs and other_probs:
        ref_arr = np.array(ref_probs)
        other_arr = np.array(other_probs)

        # Correlation
        comparison["prob_correlation"] = float(np.corrcoef(ref_arr, other_arr)[0, 1])

        # Max and mean difference
        diff = np.abs(ref_arr - other_arr)
        comparison["prob_max_diff"] = float(diff.max())
        comparison["prob_mean_diff"] = float(diff.mean())

        # KL divergence (with epsilon for numerical stability)
        eps = 1e-10
        kl = float(np.sum(ref_arr * (np.log(ref_arr + eps) - np.log(other_arr + eps))))
        comparison["kl_divergence"] = kl

        # Cosine similarity
        comparison["cosine_sim"] = float(
            np.dot(ref_arr, other_arr) / (np.linalg.norm(ref_arr) * np.linalg.norm(other_arr) + eps)
        )

    # Logits comparison
    comparison["logits_mean_diff"] = abs(ref.logits_mean - other.logits_mean)

    return comparison


def test_model(hf_name: str, tl_name: str, mlx_name: str, display_name: str, category: str) -> dict:
    """Test a single model across all backends."""
    print(f"\n{'#'*80}")
    print(f"# Testing: {display_name} ({category})")
    print(f"# HF: {hf_name}")
    print(f"# TL: {tl_name}")
    print(f"# MLX: {mlx_name}")
    print('#'*80)

    results = {
        "model": display_name,
        "hf_name": hf_name,
        "tl_name": tl_name,
        "mlx_name": mlx_name,
        "category": category,
        "backends": {},
        "comparisons": {},
    }

    # Run each backend
    all_metrics = {}
    for backend in BACKENDS:
        print(f"\n{'='*60}")
        print(f"Running {backend.name}...")
        print('='*60)

        metrics = run_backend(hf_name, tl_name, mlx_name, backend, TEST_PROMPTS)
        all_metrics[backend.name] = metrics

        # Store summary
        results["backends"][backend.name] = {
            "status": metrics.status,
            "error": metrics.error,
            "load_time_ms": metrics.load_time_ms,
            "forward_time_ms": metrics.forward_time_ms,
            "generate_time_ms": metrics.generate_time_ms,
            "cache_time_ms": metrics.cache_time_ms,
            "memory_gb": metrics.memory_gb,
            "num_cached": metrics.num_cached,
            "logits_shape": metrics.logits_shape,
            "logits_mean": metrics.logits_mean,
            "generated_texts": metrics.generated_texts,
        }

        if metrics.status == "OK":
            print(f"  Load: {metrics.load_time_ms:.0f}ms")
            print(f"  Memory: {metrics.memory_gb:.2f}GB")
            print(f"  Generate: {metrics.generate_time_ms:.0f}ms")
            print(f"  Cache: {metrics.cache_time_ms:.0f}ms ({metrics.num_cached} activations)")
        else:
            print(f"  ERROR: {metrics.error[:100]}")

    # Compare backends (use HUGGINGFACE as reference)
    ref_backend = "HUGGINGFACE"
    if ref_backend in all_metrics and all_metrics[ref_backend].status == "OK":
        ref_metrics = all_metrics[ref_backend]

        for prompt in TEST_PROMPTS:
            results["comparisons"][prompt] = {}
            for backend_name, metrics in all_metrics.items():
                if backend_name != ref_backend:
                    comparison = compare_metrics(ref_metrics, metrics, prompt)
                    results["comparisons"][prompt][backend_name] = comparison

    return results


def print_summary(all_results: list[dict]):
    """Print a comprehensive summary table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE SUMMARY")
    print("="*120)

    # Performance summary
    print("\n--- Performance (Load/Generate/Memory) ---")
    print(f"{'Model':<25} {'Backend':<15} {'Load(ms)':<10} {'Gen(ms)':<10} {'Mem(GB)':<10} {'Cache':<8}")
    print("-"*88)

    for result in all_results:
        model = result["model"][:24]
        for backend, data in result["backends"].items():
            if data["status"] == "OK":
                print(f"{model:<25} {backend:<15} {data['load_time_ms']:>8.0f}   "
                      f"{data['generate_time_ms']:>8.0f}   {data['memory_gb']:>8.2f}   {data['num_cached']:>6}")
            else:
                print(f"{model:<25} {backend:<15} {'ERROR':<8}   {'-':<8}   {'-':<8}   {'-':<6}")

    # Consistency summary
    print("\n--- Consistency vs HuggingFace ---")
    print(f"{'Model':<25} {'Backend':<12} {'Gen':<5} {'Top1':<5} {'Top5':<5} {'Top100':<6} {'Top1K':<6} {'Corr':<8} {'MaxDiff':<10}")
    print("-"*100)

    for result in all_results:
        model = result["model"][:24]
        for prompt, comparisons in result.get("comparisons", {}).items():
            for backend, comp in comparisons.items():
                if comp.get("status") == "OK":
                    gen = "Y" if comp.get("gen_match") else "N"
                    t1 = "Y" if comp.get("top1_match") else "N"
                    t5 = "Y" if comp.get("top5_match") else "N"
                    t100 = f"{comp.get('top100_overlap', 0)*100:.0f}%" if comp.get("top100_overlap") else "-"
                    t1k = f"{comp.get('top1000_overlap', 0)*100:.0f}%" if comp.get("top1000_overlap") else "-"
                    corr = f"{comp.get('prob_correlation', 0):.4f}" if comp.get("prob_correlation") else "-"
                    maxd = f"{comp.get('prob_max_diff', 0):.6f}" if comp.get("prob_max_diff") else "-"
                    print(f"{model:<25} {backend:<12} {gen:<5} {t1:<5} {t5:<5} {t100:<6} {t1k:<6} {corr:<8} {maxd:<10}")
                else:
                    print(f"{model:<25} {backend:<12} {'ERR':<5}")
            break  # Only show first prompt in summary

    # Generation text comparison
    print("\n--- Generated Text Samples (first prompt) ---")
    first_prompt = TEST_PROMPTS[0]
    for result in all_results:
        model = result["model"]
        print(f"\n{model}:")
        for backend, data in result["backends"].items():
            if data["status"] == "OK":
                gen = data.get("generated_texts", {}).get(first_prompt, "N/A")[:60]
                print(f"  {backend:<15}: {gen}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Parse args
    models_to_test = MODELS
    if len(sys.argv) > 1:
        if sys.argv[1] == "--small":
            models_to_test = [m for m in MODELS if m[4] == "small"]
        elif sys.argv[1] == "--medium":
            models_to_test = [m for m in MODELS if m[4] in ["small", "medium"]]
        elif sys.argv[1] == "--quick":
            models_to_test = MODELS[:3]  # Just first 3 models

    all_results = []

    for hf_name, tl_name, mlx_name, display_name, category in models_to_test:
        try:
            result = test_model(hf_name, tl_name, mlx_name, display_name, category)
            all_results.append(result)

            # Save intermediate results
            output_file = OUT_DIR / "full_benchmark_results.json"
            output_file.write_text(json.dumps(all_results, indent=2, default=str))

        except Exception as e:
            print(f"FAILED to test {display_name}: {e}")
            traceback.print_exc()
            all_results.append({
                "model": display_name,
                "status": "FAILED",
                "error": str(e),
            })

    # Print summary
    print_summary(all_results)

    # Save final results
    output_file = OUT_DIR / "full_benchmark_results.json"
    output_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
