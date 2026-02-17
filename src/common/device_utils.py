"""Device and memory utilities for GPU/MPS/CPU operations."""

from __future__ import annotations

import gc
import os
import sys
import torch

# Track memory across iterations for leak detection
_memory_history: list[dict] = []


def get_device() -> str:
    """Return the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_usage() -> dict:
    """Return current memory usage statistics for available accelerators and system RAM."""
    stats = {}

    # GPU memory
    if torch.cuda.is_available():
        stats["cuda_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    if hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["mps_alloc_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass

    # System RAM (cross-platform)
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        stats["ram_gb"] = proc.memory_info().rss / 1e9
        stats["ram_percent"] = proc.memory_percent()
    except ImportError:
        pass

    return stats


def log_memory(stage: str, iteration: int = -1, verbose: bool = False) -> None:
    """Print memory usage at a given stage and track history."""
    mem = get_memory_usage()
    if mem:
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
        if verbose:
            print(f"  [Memory @ {stage}] {mem_str}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Track for leak detection
        if iteration >= 0:
            _memory_history.append({"iteration": iteration, "stage": stage, **mem})


def check_memory_trend() -> None:
    """Print memory trend analysis to detect leaks."""
    if len(_memory_history) < 2:
        return

    # Compare first and last entries
    first = _memory_history[0]
    last = _memory_history[-1]

    print("\n  [Memory Trend Analysis]")
    for key in ["ram_gb", "mps_alloc_gb", "cuda_alloc_gb"]:
        if key in first and key in last:
            delta = last[key] - first[key]
            if abs(delta) > 0.1:  # Only report if > 100MB change
                print(
                    f"    {key}: {first[key]:.2f} -> {last[key]:.2f} (delta: {delta:+.2f} GB)"
                )
    print()


def clear_gpu_memory() -> None:
    """Clear GPU memory caches for CUDA and MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
