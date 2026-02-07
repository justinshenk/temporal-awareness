"""Steering vector computation and application."""

from __future__ import annotations

import numpy as np
import torch

from ..data import PreferenceDataset
from ..models import ModelRunner
from ..models.intervention_utils import steering
from ..probes import prepare_samples


def compute_steering_vector(
    runner: ModelRunner,
    pref_data: PreferenceDataset,
    layer: int,
    position: int,
    max_samples: int = 500,
) -> tuple[np.ndarray, dict]:
    """Compute steering vector at a specific (layer, position).

    Args:
        runner: Model runner
        pref_data: Preference data
        layer: Target layer
        position: Target position
        max_samples: Max samples per class

    Returns:
        Tuple of (direction vector, stats dict)
    """
    # Prepare binary-labeled samples (class 0 vs class 1 based on choice)
    samples, labels = prepare_samples(pref_data, "choice", "choice", random_seed=42)
    # labels: ndarray of shape [n_samples] with values 0 or 1

    # Subsample to balance classes and cap total count
    if len(samples) > max_samples * 2:
        np.random.seed(42)
        idx_0 = np.where(labels == 0)[0]  # indices of class 0
        idx_1 = np.where(labels == 1)[0]  # indices of class 1
        if len(idx_0) > max_samples:
            idx_0 = np.random.choice(idx_0, max_samples, replace=False)
        if len(idx_1) > max_samples:
            idx_1 = np.random.choice(idx_1, max_samples, replace=False)
        selected = np.concatenate([idx_0, idx_1])  # [n_selected]
        samples = [samples[i] for i in selected]
        labels = labels[selected]  # [n_selected]

    hook_name = f"blocks.{layer}.hook_resid_post"

    def names_filter(name: str) -> bool:
        return hook_name in name

    # Extract the residual stream activation at the target (layer, position) for each sample
    acts_list = []
    for sample in samples:
        text = sample.prompt_text + (sample.response or "")
        with torch.no_grad():
            # cache[hook_name] shape: [batch, seq_len, d_model]
            _, cache = runner.run_with_cache(text, names_filter=names_filter)
        acts = cache[hook_name]  # [batch, seq_len, d_model]
        if isinstance(acts, torch.Tensor):
            acts = acts[0].cpu().numpy()  # [seq_len, d_model]
        # Clamp position to valid range (different prompts may have different lengths)
        pos_idx = min(position, acts.shape[0] - 1)
        acts_list.append(acts[pos_idx])  # [d_model]
        del cache

    activations = np.array(acts_list)  # [n_samples, d_model]
    acts_0 = activations[labels == 0]   # [n_class0, d_model]
    acts_1 = activations[labels == 1]   # [n_class1, d_model]

    # Steering direction = difference of class means (mean-diff method)
    mean_0, mean_1 = np.mean(acts_0, axis=0), np.mean(acts_1, axis=0)  # each [d_model]
    direction = mean_1 - mean_0  # [d_model]
    norm = np.linalg.norm(direction)  # scalar

    stats = {
        "layer": layer,
        "position": position,
        "direction_norm": float(norm),
        "n_class0": len(acts_0),
        "n_class1": len(acts_1),
    }

    return direction, stats


def apply_steering(
    runner: ModelRunner,
    prompt: str,
    direction: np.ndarray,
    layer: int,
    strength: float = 1.0,
    max_new_tokens: int = 100,
) -> str:
    """Apply steering vector and generate text.

    Args:
        runner: Model runner
        prompt: Input prompt
        direction: Steering direction vector
        layer: Target layer
        strength: Steering strength (can be negative to reverse)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Create a steering intervention that adds strength * direction to residual stream
    # direction: [d_model], applied at all positions during generation
    intervention = steering(
        layer=layer,
        direction=direction,  # [d_model]
        strength=strength,
        normalize=False,  # Caller is responsible for normalization
    )
    return runner.generate(prompt, max_new_tokens=max_new_tokens, intervention=intervention)
