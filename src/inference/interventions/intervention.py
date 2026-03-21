"""Activation interventions for modifying model behavior during inference.

IMPORTANT: Use ModelRunner API, never access backends directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch

from ...common.base_schema import BaseSchema
from .intervention_target import InterventionTarget

Mode = Literal["add", "set", "mul", "interpolate"]

# Type alias for single or multiple interventions
Interventions = Union["Intervention", list["Intervention"]]

DEBUG_HOOKS = False


@dataclass
class Intervention(BaseSchema):
    """Intervention config: layer, mode, values, target.

    For interpolate mode:
        - Uses actual current activation as source (not pre-computed values)
        - target_values: target values to interpolate towards
        - alpha: interpolation factor (0=keep current, 1=use target)
        - Note: values field is ignored for interpolate mode

    For embedding interventions:
        - Set component="embed" to intervene on input embeddings
        - layer is ignored for embedding interventions
    """

    layer: int
    mode: Mode
    values: np.ndarray
    target: InterventionTarget = field(default_factory=InterventionTarget.all)
    component: str = "resid_post"
    strength: float = 1.0
    target_values: Optional[np.ndarray] = None
    alpha: float = 0.5

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)
        if self.target_values is not None and not isinstance(
            self.target_values, np.ndarray
        ):
            self.target_values = np.array(self.target_values, dtype=np.float32)
        if self.mode == "interpolate" and self.target_values is None:
            raise ValueError("interpolate mode requires target_values")

    @property
    def is_embedding(self) -> bool:
        """True if this intervention targets embeddings."""
        return self.component == "embed"

    @property
    def hook_name(self) -> str:
        if self.is_embedding:
            return "hook_embed"
        return f"blocks.{self.layer}.hook_{self.component}"

    @property
    def scaled_values(self) -> np.ndarray:
        return self.values * self.strength


def create_intervention_hook(
    config: Intervention,
    dtype: torch.dtype,
    device: str,
) -> tuple[Callable, None]:
    """Create a forward hook for the intervention. Returns (hook, None)."""
    values = torch.tensor(config.scaled_values, dtype=dtype, device=device)
    target = config.target
    mode = config.mode
    alpha = config.alpha

    target_values = None
    if mode == "interpolate" and config.target_values is not None:
        target_values = torch.tensor(config.target_values, dtype=dtype, device=device)

    if DEBUG_HOOKS:
        print(f"[hook] Creating hook: layer={config.layer}, mode={mode}, alpha={alpha}")
        print(f"[hook]   values.shape={values.shape}, target={target}")
        if target_values is not None:
            print(f"[hook]   target_values.shape={target_values.shape}")

    # All positions
    if target.is_all_positions:
        def full_hook(act, hook=None):
            if DEBUG_HOOKS:
                print(f"[hook] Applying FULL: act.shape={act.shape}, values.shape={values.shape}")
            return _apply_full(act, values, mode, target_values, alpha)
        return full_hook, None

    # Specific positions
    positions = list(target.positions)

    def hook(act, hook=None):
        if DEBUG_HOOKS:
            print(f"[hook] Applying to positions {positions[:5]}..., act.shape={act.shape}")
        for i, pos in enumerate(positions):
            if pos < act.shape[1]:
                v = values[i] if values.dim() > 1 and i < values.shape[0] else values
                tv = None
                if target_values is not None:
                    tv = (
                        target_values[i]
                        if target_values.dim() > 1 and i < target_values.shape[0]
                        else target_values
                    )
                act[:, pos] = _apply_position(act[:, pos], v, mode, tv, alpha)
        return act

    return hook, None


def _apply_full(
    act: torch.Tensor,
    values: torch.Tensor,
    mode: Mode,
    target_values: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """Apply intervention to full activation tensor."""
    if mode == "add":
        return act + values
    if mode == "mul":
        return act * values
    if mode == "interpolate":
        # Use actual current activation as source, interpolate towards target_values
        # Result: act + alpha * (target_values - act) = (1-alpha)*act + alpha*target_values
        if target_values.dim() == 2 and act.dim() == 3:
            seq = min(act.shape[1], target_values.shape[0])
            result = act.clone()
            tv = target_values[:seq].unsqueeze(0).expand(act.shape[0], -1, -1)
            result[:, :seq] = act[:, :seq] + alpha * (tv - act[:, :seq])
            return result
        return act + alpha * (target_values - act)
    # set mode
    if values.dim() <= 1:
        return values.expand_as(act)
    seq = min(act.shape[1], values.shape[0])
    result = act.clone()
    result[:, :seq] = values[:seq].unsqueeze(0).expand(act.shape[0], -1, -1)
    return result


def _apply_position(
    act: torch.Tensor,
    values: torch.Tensor,
    mode: Mode,
    target_values: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """Apply intervention to single position."""
    v = values[-1] if values.dim() > 1 else values
    if mode == "add":
        return act + v
    if mode == "mul":
        return act * v
    if mode == "interpolate":
        # Use actual current activation as source, interpolate towards target_values
        tv = target_values[-1] if target_values.dim() > 1 else target_values
        return act + alpha * (tv - act)
    return v.expand_as(act)


def load_intervention_from_dict(data: dict, n_layers: int) -> Intervention:
    """Load intervention from dict config."""
    layer = min(data["layer"], n_layers - 1)
    component = data.get("component", "resid_post")

    # Parse values
    values = data.get("values", 0)
    if isinstance(values, (int, float)):
        values = np.array([float(values)], dtype=np.float32)
    elif isinstance(values, list):
        values = np.array(values, dtype=np.float32)
    elif isinstance(values, str) and values.endswith(".npy"):
        values = np.load(values).astype(np.float32)
    else:
        values = np.array(values, dtype=np.float32)

    # Parse target
    target_data = data.get("target", "all")
    if target_data == "all" or target_data is None:
        target = InterventionTarget.all()
    elif isinstance(target_data, dict):
        positions = target_data.get("positions")
        layers = target_data.get("layers")
        target = InterventionTarget.at(
            positions=positions, layers=layers, component=component
        )
    else:
        target = InterventionTarget.all()

    return Intervention(
        layer=layer,
        mode=data["mode"],
        values=values,
        target=target,
        component=component,
        strength=data.get("strength", 1.0),
    )
