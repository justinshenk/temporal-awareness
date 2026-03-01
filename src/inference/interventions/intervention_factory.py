"""Factory functions for creating interventions.

IMPORTANT DESIGN PRINCIPLES:
1. Use these utility functions to create Interventions - don't construct manually
2. NEVER access backend APIs directly - always use ModelRunner methods
3. All interventions work identically across backends (TL, NNsight, Pyvene)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from .intervention import Intervention
from .intervention_target import InterventionTarget

if TYPE_CHECKING:
    from ..model_runner import ModelRunner


def steering(
    layer: int,
    direction: Union[np.ndarray, list],
    strength: float = 1.0,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
    normalize: bool = True,
) -> Intervention:
    """Add direction to activations (mode=add)."""
    direction = np.array(direction, dtype=np.float32).flatten()
    if normalize and len(direction) > 0:
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

    return Intervention(
        layer=layer,
        mode="add",
        values=direction,
        target=_target(positions),
        component=component,
        strength=strength,
    )


def ablation(
    layer: int,
    values: Optional[Union[np.ndarray, list, float]] = None,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Set activations to fixed values (mode=set). Default: zero."""
    if values is None:
        values = np.array([0.0], dtype=np.float32)
    elif isinstance(values, (int, float)):
        values = np.array([float(values)], dtype=np.float32)
    else:
        values = np.array(values, dtype=np.float32)

    return Intervention(
        layer=layer,
        mode="set",
        values=values,
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def patch(
    layer: int,
    values: Union[np.ndarray, list],
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Replace activations with cached values (mode=set)."""
    return Intervention(
        layer=layer,
        mode="set",
        values=np.array(values, dtype=np.float32),
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def scale(
    layer: int,
    factor: float,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Multiply activations by factor (mode=mul)."""
    return Intervention(
        layer=layer,
        mode="mul",
        values=np.array([factor], dtype=np.float32),
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def interpolate(
    layer: int,
    source_values: Union[np.ndarray, list],
    target_values: Union[np.ndarray, list],
    alpha: float = 0.5,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Interpolate between source and target activations (mode=interpolate).

    Result: source + alpha * (target - source)
    - alpha=0: use source values
    - alpha=1: use target values

    Args:
        layer: Layer to intervene on
        source_values: Source activations (e.g., corrupted)
        target_values: Target activations (e.g., clean)
        alpha: Interpolation factor [0, 1]
        positions: Optional positions to target
        component: Component to intervene on

    Returns:
        Intervention that computes interpolated activations
    """
    return Intervention(
        layer=layer,
        mode="interpolate",
        values=np.array(source_values, dtype=np.float32),
        target_values=np.array(target_values, dtype=np.float32),
        alpha=alpha,
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def patch_embeddings(
    values: Union[np.ndarray, torch.Tensor],
    positions: Optional[Union[int, list[int]]] = None,
) -> Intervention:
    """Replace input embeddings (mode=set, component=embed).

    Use this for embedding-level interventions like EAP-IG.

    Args:
        values: Embedding values [seq_len, d_model] or [d_model]
        positions: Optional positions to target (None = all)

    Returns:
        Intervention that patches embeddings
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return Intervention(
        layer=0,  # Ignored for embedding interventions
        mode="set",
        values=np.array(values, dtype=np.float32),
        target=_target(positions),
        component="embed",
        strength=1.0,
    )


def interpolate_embeddings(
    source_values: Union[np.ndarray, torch.Tensor],
    target_values: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
    positions: Optional[Union[int, list[int]]] = None,
) -> Intervention:
    """Interpolate between source and target embeddings (mode=interpolate).

    Result: source + alpha * (target - source)

    Use this for embedding-level EAP-IG.

    Args:
        source_values: Source embeddings (e.g., corrupted)
        target_values: Target embeddings (e.g., clean)
        alpha: Interpolation factor [0, 1]
        positions: Optional positions to target

    Returns:
        Intervention that interpolates embeddings
    """
    if isinstance(source_values, torch.Tensor):
        source_values = source_values.detach().cpu().numpy()
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.detach().cpu().numpy()
    return Intervention(
        layer=0,  # Ignored for embedding interventions
        mode="interpolate",
        values=np.array(source_values, dtype=np.float32),
        target_values=np.array(target_values, dtype=np.float32),
        alpha=alpha,
        target=_target(positions),
        component="embed",
        strength=1.0,
    )


def _target(positions=None) -> InterventionTarget:
    if positions is not None:
        return InterventionTarget.at_positions(positions)
    return InterventionTarget.all()


def compute_mean_activations(
    runner: "ModelRunner",
    layer: int,
    prompts: Union[str, list[str]],
    component: str = "resid_post",
) -> np.ndarray:
    """Compute mean activations across prompts."""
    if isinstance(prompts, str):
        prompts = [prompts]

    hook_name = f"blocks.{layer}.hook_{component}"
    means = []

    for prompt in prompts:
        _, cache = runner.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]
        if isinstance(acts, torch.Tensor):
            acts = acts.detach().cpu().numpy()
        means.append(acts.mean(axis=(0, 1)))

    return np.mean(means, axis=0).astype(np.float32)


def get_activations(
    runner: "ModelRunner",
    layer: int,
    prompt: str,
    component: str = "resid_post",
) -> np.ndarray:
    """Get activations [seq_len, d_model] for a prompt."""
    hook_name = f"blocks.{layer}.hook_{component}"
    _, cache = runner.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
    acts = cache[hook_name]
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    return acts[0].astype(np.float32)


def random_direction(d_model: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random unit direction vector."""
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(d_model).astype(np.float32)
    return vec / np.linalg.norm(vec)
