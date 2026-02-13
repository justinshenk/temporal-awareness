#!/usr/bin/env python3
"""Backend benchmark script - runs each backend multiple times and compares results."""

import json
import subprocess
import re
import statistics
import sys
from pathlib import Path

BACKENDS = ["TRANSFORMERLENS", "HUGGINGFACE", "PYVENE", "NNSIGHT", "MLX"]
RUNS_PER_BACKEND = 4
MODEL_RUNNER_PATH = Path("src/inference/model_runner.py")
OUT_DIR = Path("out/backends")

# Models to test
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def set_backend(backend: str):
    """Modify model_runner.py to use specified backend."""
    content = MODEL_RUNNER_PATH.read_text()
    new_content = re.sub(
        r'backend: ModelBackend = ModelBackend\.\w+,',
        f'backend: ModelBackend = ModelBackend.{backend},',
        content
    )
    MODEL_RUNNER_PATH.write_text(new_content)
    print(f"Set backend to {backend}")


def run_experiment(model_name: str):
    """Run experiment and return timing + memory stats."""
    result = subprocess.run(
        ["uv", "run", "python3", "scripts/intertemporal/run_full_experiment.py",
         "--test", "--model", model_name],
        capture_output=True,
        text=True,
        timeout=900  # 15 min for larger models
    )
    output = result.stdout + result.stderr

    # Extract total time
    total_match = re.search(r'^Total: ([\d.]+)ms$', output, re.MULTILINE)
    total_time = float(total_match.group(1)) if total_match else None

    # Extract max memory
    mem_matches = re.findall(r'mps_allocated_gb=([\d.]+)', output)
    max_memory = max(float(m) for m in mem_matches) if mem_matches else None

    # Extract key timings
    generate_match = re.search(r'generate: ([\d.]+)ms', output)
    generate_time = float(generate_match.group(1)) if generate_match else None

    batch_match = re.search(r'get_prob_trajectories_for_batch: ([\d.]+)ms', output)
    batch_time = float(batch_match.group(1)) if batch_match else None

    cache_match = re.search(r'run_with_cache: ([\d.]+)ms', output)
    cache_time = float(cache_match.group(1)) if cache_match else None

    # Check if experiment succeeded
    success = "PROFILER REPORT" in output and total_time is not None

    # Extract first sample's response for comparison
    response_match = re.search(r'"response_text": "([^"]+)"', output)
    response = response_match.group(1)[:100] if response_match else None

    # Extract choice probabilities
    choice_prob_match = re.search(r'"choice_prob": ([\d.]+)', output)
    choice_prob = float(choice_prob_match.group(1)) if choice_prob_match else None

    alt_prob_match = re.search(r'"alternative_prob": ([\d.]+)', output)
    alt_prob = float(alt_prob_match.group(1)) if alt_prob_match else None

    # Extract choice label
    choice_label_match = re.search(r'"choice_label": "([^"]+)"', output)
    choice_label = choice_label_match.group(1) if choice_label_match else None

    return {
        "success": success,
        "total_time": total_time,
        "max_memory": max_memory,
        "generate_time": generate_time,
        "batch_time": batch_time,
        "cache_time": cache_time,
        "response_preview": response,
        "choice_prob": choice_prob,
        "alt_prob": alt_prob,
        "choice_label": choice_label,
        "output": output if not success else None
    }


def benchmark_backend(backend: str, model_name: str, num_runs: int = RUNS_PER_BACKEND) -> dict:
    """Run multiple iterations for a backend."""
    set_backend(backend)
    results = []
    model_short = model_name.split("/")[-1]

    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)
        try:
            result = run_experiment(model_name)
            if result["success"]:
                mem_str = f"{result['max_memory']:.2f}GB" if result['max_memory'] else "N/A"
                print(f"OK - {result['total_time']:.0f}ms, {mem_str}")
                results.append(result)
            else:
                print("FAILED")
                error_file = OUT_DIR / f"{backend.lower()}_{model_short}_error_{i}.log"
                error_file.write_text(result["output"] or "No output")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        return {"backend": backend, "model": model_name, "status": "FAILED", "runs": 0}

    times = [r["total_time"] for r in results]
    memories = [r["max_memory"] for r in results if r["max_memory"]]

    return {
        "backend": backend,
        "model": model_name,
        "status": "OK",
        "runs": len(results),
        "time_avg": statistics.mean(times),
        "time_std": statistics.stdev(times) if len(times) > 1 else 0,
        "time_min": min(times),
        "time_max": max(times),
        "memory_avg": statistics.mean(memories) if memories else None,
        "memory_max": max(memories) if memories else None,
        "generate_avg": statistics.mean([r["generate_time"] for r in results if r["generate_time"]]),
        "batch_avg": statistics.mean([r["batch_time"] for r in results if r["batch_time"]]),
        "cache_avg": statistics.mean([r["cache_time"] for r in results if r["cache_time"]]),
        "response_preview": results[0]["response_preview"] if results else None,
        "choice_probs": [r["choice_prob"] for r in results if r["choice_prob"]],
        "alt_probs": [r["alt_prob"] for r in results if r["alt_prob"]],
        "choice_labels": [r["choice_label"] for r in results if r["choice_label"]],
    }


def test_activations_match(model_name: str) -> dict:
    """Test if activations match across backends that support caching."""
    print(f"\n{'='*60}")
    print(f"Testing activation consistency for {model_name}")
    print('='*60)

    # Backends that support run_with_cache properly
    cache_backends = ["TRANSFORMERLENS", "HUGGINGFACE", "PYVENE", "NNSIGHT"]

    results = {}
    for backend in cache_backends:
        set_backend(backend)
        print(f"  Testing {backend}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["uv", "run", "python3", "-c", f'''
import torch
import numpy as np
from src.inference import ModelRunner
from src.inference.backends import ModelBackend

runner = ModelRunner("{model_name}", backend=ModelBackend.{backend})
prompt = "What is 2+2?"
logits, cache = runner.run_with_cache(prompt)

# Get activation stats for comparison
stats = {{}}
for name, tensor in cache.items():
    if isinstance(tensor, torch.Tensor):
        stats[name] = {{
            "shape": list(tensor.shape),
            "mean": float(tensor.float().mean()),
            "std": float(tensor.float().std()),
            "min": float(tensor.float().min()),
            "max": float(tensor.float().max()),
        }}

# Print as JSON
import json
print("ACTIVATION_STATS:" + json.dumps(stats))
print("LOGITS_SHAPE:" + str(list(logits.shape)))
print("LOGITS_MEAN:" + str(float(logits.float().mean())))
'''],
                capture_output=True,
                text=True,
                timeout=300
            )
            output = result.stdout + result.stderr

            # Parse activation stats
            stats_match = re.search(r'ACTIVATION_STATS:(.+)', output)
            logits_shape_match = re.search(r'LOGITS_SHAPE:\[([^\]]+)\]', output)
            logits_mean_match = re.search(r'LOGITS_MEAN:([\d.e+-]+)', output)

            if stats_match:
                stats = json.loads(stats_match.group(1))
                results[backend] = {
                    "status": "OK",
                    "num_cached": len(stats),
                    "cache_keys": list(stats.keys())[:5],  # First 5 keys
                    "sample_stats": list(stats.values())[0] if stats else None,
                    "logits_shape": logits_shape_match.group(1) if logits_shape_match else None,
                    "logits_mean": float(logits_mean_match.group(1)) if logits_mean_match else None,
                }
                print(f"OK - {len(stats)} activations cached")
            else:
                results[backend] = {"status": "FAILED", "error": "No stats found"}
                print("FAILED - no stats")

        except Exception as e:
            results[backend] = {"status": "ERROR", "error": str(e)}
            print(f"ERROR: {e}")

    return results


def print_summary(all_results: dict):
    """Print benchmark summary tables."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY BY MODEL SIZE")
    print("="*100)

    for model in MODELS:
        model_short = model.split("/")[-1]
        model_results = {k: v for k, v in all_results.items() if v.get("model") == model}

        if not model_results:
            continue

        print(f"\n{model_short}")
        print("-"*100)
        print(f"{'Backend':<15} {'Status':<8} {'Runs':<5} {'Avg Time':<12} {'Std':<10} {'Avg Mem':<10} {'Cache':<10}")
        print("-"*100)

        for key, result in model_results.items():
            backend = result["backend"]
            if result["status"] == "OK":
                cache_str = f"{result.get('cache_avg', 0):.0f}ms" if result.get('cache_avg') else "-"
                mem_str = f"{result['memory_avg']:.2f}GB" if result.get('memory_avg') else "-"
                print(f"{backend:<15} {result['status']:<8} {result['runs']:<5} "
                      f"{result['time_avg']:>8.0f}ms   {result['time_std']:>6.1f}ms   "
                      f"{mem_str:<10} {cache_str:<10}")
            else:
                print(f"{backend:<15} {result['status']:<8} {result['runs']:<5} -")

    # Find fastest per model
    print("\n" + "="*100)
    print("FASTEST BACKEND PER MODEL")
    print("="*100)
    print(f"{'Model':<30} {'Fastest':<15} {'Time':<12} {'Memory':<10}")
    print("-"*100)

    for model in MODELS:
        model_short = model.split("/")[-1]
        model_results = {k: v for k, v in all_results.items()
                        if v.get("model") == model and v.get("status") == "OK"}
        if model_results:
            fastest = min(model_results.values(), key=lambda x: x["time_avg"])
            mem_str = f"{fastest['memory_avg']:.2f}GB" if fastest.get('memory_avg') else "-"
            print(f"{model_short:<30} {fastest['backend']:<15} {fastest['time_avg']:>8.0f}ms   {mem_str:<10}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Parse command line args
    models_to_test = MODELS
    if len(sys.argv) > 1:
        if sys.argv[1] == "--small":
            models_to_test = ["Qwen/Qwen2.5-0.5B-Instruct"]
        elif sys.argv[1] == "--medium":
            models_to_test = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
        elif sys.argv[1] == "--model":
            models_to_test = [sys.argv[2]]

    all_results = {}
    activation_results = {}

    for model in models_to_test:
        model_short = model.split("/")[-1]
        print(f"\n{'#'*80}")
        print(f"# TESTING MODEL: {model}")
        print('#'*80)

        for backend in BACKENDS:
            print(f"\n{'='*60}")
            print(f"Benchmarking {backend} with {model_short}")
            print('='*60)

            result = benchmark_backend(backend, model, num_runs=4)
            key = f"{backend}_{model_short}"
            all_results[key] = result

            # Save intermediate results
            (OUT_DIR / "benchmark_results.json").write_text(
                json.dumps(all_results, indent=2)
            )

        # Test activation consistency for this model
        activation_results[model] = test_activations_match(model)

    # Reset to default
    set_backend("MLX")

    # Print summary
    print_summary(all_results)

    # Print activation comparison
    print("\n" + "="*100)
    print("ACTIVATION CACHE COMPARISON")
    print("="*100)
    for model, results in activation_results.items():
        model_short = model.split("/")[-1]
        print(f"\n{model_short}:")
        for backend, data in results.items():
            if data["status"] == "OK":
                print(f"  {backend}: {data['num_cached']} activations, logits_mean={data.get('logits_mean', 'N/A'):.4f}")
            else:
                print(f"  {backend}: {data['status']} - {data.get('error', '')}")

    # Save all results
    (OUT_DIR / "benchmark_full_results.json").write_text(
        json.dumps({"benchmarks": all_results, "activations": activation_results}, indent=2)
    )

    print(f"\nResults saved to {OUT_DIR}/benchmark_full_results.json")


if __name__ == "__main__":
    main()
