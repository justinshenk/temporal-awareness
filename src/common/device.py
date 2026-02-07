"""Device and memory utilities for GPU/MPS/CPU operations."""

from __future__ import annotations

import gc

import torch


def get_device() -> str:
    """Return the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_usage() -> dict:
    """Return current memory usage statistics for available accelerators."""
    stats = {}
    if torch.cuda.is_available():
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    if hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
    return stats


def log_memory(stage: str) -> None:
    """Print memory usage at a given stage."""
    mem = get_memory_usage()
    if mem:
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
        print(f"  [Memory @ {stage}] {mem_str}")


def clear_gpu_memory() -> None:
    """Clear GPU memory caches for CUDA and MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
