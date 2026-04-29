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

from .intervention_base import Intervention
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
    target_values: Union[np.ndarray, list],
    alpha: float = 0.5,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Interpolate from current activation towards target values (mode=interpolate).

    Result: act + alpha * (target_values - act)
    - alpha=0: keep current activation unchanged
    - alpha=1: fully replace with target_values

    The actual current activation is used as the interpolation source.

    Args:
        layer: Layer to intervene on
        target_values: Target activations to interpolate towards
        alpha: Interpolation factor [0=keep current, 1=use target]
        positions: Optional positions to target
        component: Component to intervene on

    Returns:
        Intervention that interpolates current activation towards target
    """
    target_arr = np.array(target_values, dtype=np.float32)
    return Intervention(
        layer=layer,
        mode="interpolate",
        values=np.zeros(1, dtype=np.float32),  # Placeholder, not used in interpolate mode
        target_values=target_arr,
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
    target_values: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
    positions: Optional[Union[int, list[int]]] = None,
) -> Intervention:
    """Interpolate from current embeddings towards target embeddings (mode=interpolate).

    Result: act + alpha * (target_values - act)
    - alpha=0: keep current embeddings unchanged
    - alpha=1: fully replace with target_values

    The actual current embedding is used as the interpolation source.
    Use this for embedding-level EAP-IG.

    Args:
        target_values: Target embeddings to interpolate towards
        alpha: Interpolation factor [0=keep current, 1=use target]
        positions: Optional positions to target

    Returns:
        Intervention that interpolates current embeddings towards target
    """
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.detach().cpu().numpy()
    return Intervention(
        layer=0,  # Ignored for embedding interventions
        mode="interpolate",
        values=np.zeros(1, dtype=np.float32),  # Placeholder, not used in interpolate mode
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


def zero_ablation_intervention(
    layer: int,
    d_model: int,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Create zero ablation intervention (set activations to 0).

    Used for testing circuit necessity: if ablating a component significantly
    affects behavior, that component is necessary for the circuit.

    Args:
        layer: Layer to intervene on
        d_model: Model hidden dimension
        positions: Optional positions to target (None = all)
        component: Component to intervene on

    Returns:
        Intervention that sets activations to zero
    """
    return Intervention(
        layer=layer,
        mode="set",
        values=np.zeros(d_model, dtype=np.float32),
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def mean_ablation_intervention(
    layer: int,
    mean_activations: np.ndarray,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
) -> Intervention:
    """Create mean ablation intervention (set to pre-computed mean).

    Used for testing circuit necessity while preserving approximate activation
    magnitude. This is often preferred over zero ablation as it's less disruptive
    to downstream computations.

    Args:
        layer: Layer to intervene on
        mean_activations: Pre-computed mean activations [d_model]
        positions: Optional positions to target (None = all)
        component: Component to intervene on

    Returns:
        Intervention that sets activations to mean values
    """
    return Intervention(
        layer=layer,
        mode="set",
        values=np.array(mean_activations, dtype=np.float32),
        target=_target(positions),
        component=component,
        strength=1.0,
    )


def gaussian_noise_intervention(
    layer: int,
    d_model: int,
    sigma: float = 1.0,
    positions: Optional[Union[int, list[int]]] = None,
    component: str = "resid_post",
    seed: Optional[int] = None,
) -> Intervention:
    """Create Gaussian noise injection intervention (add N(0, sigma) noise).

    Used for robustness testing: if adding noise to a component doesn't
    significantly affect behavior, that component may have redundant pathways.

    Args:
        layer: Layer to intervene on
        d_model: Model hidden dimension
        sigma: Standard deviation of noise
        positions: Optional positions to target (None = all)
        component: Component to intervene on
        seed: Optional random seed for reproducibility

    Returns:
        Intervention that adds Gaussian noise to activations
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=d_model).astype(np.float32)
    return Intervention(
        layer=layer,
        mode="add",
        values=noise,
        target=_target(positions),
        component=component,
        strength=1.0,
    )
