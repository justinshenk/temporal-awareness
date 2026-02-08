"""Load intervention configs from JSON files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .core import Intervention, Target
from .utils import compute_mean_activations, get_activations, random_direction

if TYPE_CHECKING:
    from ..model_runner import ModelRunner


def load_intervention_from_dict(
    data: dict,
    runner: "ModelRunner",
    calibration_prompt: str = "The quick brown fox jumps over the lazy dog.",
    cache_prompt: Optional[str] = None,
) -> Intervention:
    """Load intervention from dict."""
    cache_prompt = cache_prompt or calibration_prompt
    layer = min(data["layer"], runner.n_layers - 1)
    component = data.get("component", "resid_post")

    return Intervention(
        layer=layer,
        mode=data["mode"],
        values=_resolve_values(
            data.get("values", 0), runner, layer, component, calibration_prompt, cache_prompt
        ),
        target=_parse_target(data.get("target", "all")),
        component=component,
        strength=data.get("strength", 1.0),
    )


def _parse_target(data) -> Target:
    if data == "all" or data is None:
        return Target.all()
    if isinstance(data, dict):
        axis = data.get("axis", "all")
        if axis == "all":
            return Target.all()
        if axis == "position":
            return Target.at_positions(data["positions"])
        if axis == "neuron":
            return Target.at_neurons(data["neurons"])
        if axis == "pattern":
            return Target.on_pattern(data["pattern"])
    raise ValueError(f"Invalid target: {data}")


def _resolve_values(spec, runner, layer, component, calibration_prompt, cache_prompt) -> np.ndarray:
    if isinstance(spec, (int, float)):
        return np.array([float(spec)], dtype=np.float32)
    if isinstance(spec, list):
        return np.array(spec, dtype=np.float32)
    if isinstance(spec, str):
        if spec == "random":
            return random_direction(runner.d_model)
        if spec == "mean":
            return compute_mean_activations(runner, layer, calibration_prompt, component)
        if spec == "cached":
            return get_activations(runner, layer, cache_prompt, component)
        if spec.endswith(".npy"):
            return np.load(spec).astype(np.float32)
    raise ValueError(f"Invalid values: {spec}")
