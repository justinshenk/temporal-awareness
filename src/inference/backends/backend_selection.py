"""Recommended backend selection based on use case and hardware.

Usage:
    from src.inference.backends import get_recommended_backend

    # Pure inference (fastest generation)
    backend = get_recommended_backend_inference()

    # Capturing internal activations
    backend = get_recommended_backend_internals()

    # Running interventions (steering, patching)
    backend = get_recommended_backend_interventions()
"""

from __future__ import annotations

import torch

from .model_backend import ModelBackend


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return torch.backends.mps.is_available()


def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def _mlx_available() -> bool:
    """Check if MLX is installed."""
    try:
        import mlx.core

        return True
    except ImportError:
        return False


def get_recommended_backend_inference() -> ModelBackend:
    """Get the recommended backend for pure inference (generation/logprobs).

    Prioritizes speed. MLX is fastest on Apple Silicon, HuggingFace
    is most compatible and second-fastest on all platforms.

    Returns:
        ModelBackend: Recommended backend for inference
    """
    if _is_apple_silicon() and _mlx_available():
        return ModelBackend.MLX
    return ModelBackend.HUGGINGFACE


def get_recommended_backend_internals() -> ModelBackend:
    """Get the recommended backend for capturing internal activations.

    Prioritizes activation caching support and hook infrastructure.
    TransformerLens has the best hook infrastructure with consistent
    naming conventions. HuggingFace is a good alternative with
    standard PyTorch hooks.

    Note: MLX does not support activation caching (returns empty dict).

    Returns:
        ModelBackend: Recommended backend for internals/caching
    """
    return ModelBackend.PYVENE


def get_recommended_backend_interventions() -> ModelBackend:
    """Get the recommended backend for running interventions.

    Interventions include activation patching, steering, and other
    modifications to internal model states during forward passes.

    Note: MLX does not support interventions.

    Returns:
        ModelBackend: Recommended backend for interventions

    """
    return ModelBackend.PYVENE
