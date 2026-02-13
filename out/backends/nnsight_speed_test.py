#!/usr/bin/env python3
"""Benchmark NNsight speed vs HuggingFace."""

import sys
sys.path.insert(0, "/Users/unrulyabstractions/work/temporal-awareness")

import time
import torch

def benchmark_forward(runner, prompt, n_runs=10):
    """Benchmark forward pass."""
    input_ids = runner.tokenize(prompt)

    # Warmup
    for _ in range(3):
        runner._backend.forward(input_ids)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        runner._backend.forward(input_ids)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs * 1000  # ms per run

def benchmark_run_with_cache(runner, prompt, n_runs=10):
    """Benchmark run_with_cache."""
    # Warmup
    for _ in range(3):
        runner.run_with_cache(prompt)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        runner.run_with_cache(prompt)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs * 1000  # ms per run

def benchmark_generate(runner, prompt, n_runs=5):
    """Benchmark generation."""
    # Warmup
    for _ in range(2):
        runner.generate(prompt, max_new_tokens=10, temperature=0.0)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        runner.generate(prompt, max_new_tokens=10, temperature=0.0)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs * 1000  # ms per run

def main():
    from src.inference.model_runner import ModelRunner
    from src.inference.backends import ModelBackend

    model_name = "gpt2"
    device = "cpu"
    dtype = torch.float32
    prompt = "The capital of France is"

    print("=" * 60)
    print("NNsight vs HuggingFace Speed Comparison")
    print("=" * 60)
    print(f"Model: {model_name}, Device: {device}")
    print(f"Prompt: '{prompt}' ({len(prompt.split())} words)")
    print()

    # Load models
    print("Loading HuggingFace...")
    hf_runner = ModelRunner(model_name, device=device, dtype=dtype, backend=ModelBackend.HUGGINGFACE)

    print("Loading NNsight...")
    nn_runner = ModelRunner(model_name, device=device, dtype=dtype, backend=ModelBackend.NNSIGHT)

    print("Loading Pyvene...")
    pv_runner = ModelRunner(model_name, device=device, dtype=dtype, backend=ModelBackend.PYVENE)

    print("\n" + "=" * 60)
    print("Benchmarks (ms per call, lower is better)")
    print("=" * 60)

    # Forward pass
    print("\n--- Forward Pass ---")
    hf_fwd = benchmark_forward(hf_runner, prompt)
    nn_fwd = benchmark_forward(nn_runner, prompt)
    pv_fwd = benchmark_forward(pv_runner, prompt)
    print(f"  HuggingFace: {hf_fwd:.2f} ms")
    print(f"  NNsight:     {nn_fwd:.2f} ms ({nn_fwd/hf_fwd:.1f}x slower)")
    print(f"  Pyvene:      {pv_fwd:.2f} ms ({pv_fwd/hf_fwd:.1f}x slower)")

    # Run with cache
    print("\n--- Run With Cache ---")
    hf_cache = benchmark_run_with_cache(hf_runner, prompt)
    nn_cache = benchmark_run_with_cache(nn_runner, prompt)
    pv_cache = benchmark_run_with_cache(pv_runner, prompt)
    print(f"  HuggingFace: {hf_cache:.2f} ms")
    print(f"  NNsight:     {nn_cache:.2f} ms ({nn_cache/hf_cache:.1f}x slower)")
    print(f"  Pyvene:      {pv_cache:.2f} ms ({pv_cache/hf_cache:.1f}x slower)")

    # Generation
    print("\n--- Generation (10 tokens) ---")
    hf_gen = benchmark_generate(hf_runner, prompt)
    nn_gen = benchmark_generate(nn_runner, prompt)
    pv_gen = benchmark_generate(pv_runner, prompt)
    print(f"  HuggingFace: {hf_gen:.2f} ms")
    print(f"  NNsight:     {nn_gen:.2f} ms ({nn_gen/hf_gen:.1f}x slower)")
    print(f"  Pyvene:      {pv_gen:.2f} ms ({pv_gen/hf_gen:.1f}x slower)")

    # Investigate NNsight overhead
    print("\n" + "=" * 60)
    print("NNsight Overhead Analysis")
    print("=" * 60)

    # Raw HF model forward (no wrapper)
    hf_model = hf_runner._model
    input_ids = hf_runner.tokenize(prompt)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            hf_model(input_ids)

    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            hf_model(input_ids)
    raw_hf = (time.perf_counter() - start) / 20 * 1000
    print(f"\nRaw HF model forward: {raw_hf:.2f} ms")

    # NNsight trace overhead
    nn_model = nn_runner._model
    input_ids_nn = nn_runner.tokenize(prompt)

    # Warmup
    for _ in range(3):
        with nn_model.trace(input_ids_nn):
            logits = nn_model.lm_head.output.save()

    start = time.perf_counter()
    for _ in range(20):
        with nn_model.trace(input_ids_nn):
            logits = nn_model.lm_head.output.save()
    nn_trace = (time.perf_counter() - start) / 20 * 1000
    print(f"NNsight trace (minimal): {nn_trace:.2f} ms ({nn_trace/raw_hf:.1f}x overhead)")

    # NNsight trace with activation capture
    start = time.perf_counter()
    for _ in range(20):
        with nn_model.trace(input_ids_nn):
            acts = {}
            for i in range(12):
                layer = nn_model.transformer.h[i]
                acts[f"attn_{i}"] = layer.attn.output[0].save()
                acts[f"mlp_{i}"] = layer.mlp.output.save()
                acts[f"out_{i}"] = layer.output[0].save()
            logits = nn_model.lm_head.output.save()
    nn_trace_full = (time.perf_counter() - start) / 20 * 1000
    print(f"NNsight trace (full cache): {nn_trace_full:.2f} ms ({nn_trace_full/raw_hf:.1f}x overhead)")

    # Check if it's the .save() calls
    print("\n--- Breakdown: .save() overhead ---")
    start = time.perf_counter()
    for _ in range(20):
        with nn_model.trace(input_ids_nn):
            # Just access output without saving
            _ = nn_model.lm_head.output
    nn_no_save = (time.perf_counter() - start) / 20 * 1000
    print(f"NNsight trace (no save): {nn_no_save:.2f} ms")

    start = time.perf_counter()
    for _ in range(20):
        with nn_model.trace(input_ids_nn):
            logits = nn_model.lm_head.output.save()
    nn_one_save = (time.perf_counter() - start) / 20 * 1000
    print(f"NNsight trace (1 save): {nn_one_save:.2f} ms")
    print(f"Cost per .save(): ~{(nn_one_save - nn_no_save):.2f} ms")


if __name__ == "__main__":
    main()
