"""Captured internals from model forward passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from ..common import BaseSchema


@dataclass
class ActivationSpec(BaseSchema):
    """Specification for which activations to capture."""

    component: str  # e.g., "resid_pre", "resid_post", "attn_out", "mlp_out"
    layers: list[int] = field(default_factory=list)


@dataclass
class InternalsConfig(BaseSchema):
    """Configuration for capturing model internals.

    If None is passed to PreferenceQueryConfig.internals, ALL activations are captured.
    Use InternalsConfig.empty() to capture no activations.
    """

    activations: list[ActivationSpec] = field(default_factory=list)
    save_all: bool = False

    def get_names(self) -> list[str]:
        names = []
        for spec in self.activations:
            for layer in spec.layers:
                names.append(f"blocks.{layer}.hook_{spec.component}")
        return names


@dataclass
class CapturedInternals(BaseSchema):
    """Captured activations from a forward pass."""

    activations: dict  # name -> tensor
    activation_names: list[str]

    @classmethod
    def from_activation_names(cls, activation_names: Sequence[str], internals: dict):
        activations = {}
        for name in activation_names:
            if name in internals:
                activations[name] = internals[name][0].cpu()
        return CapturedInternals(
            activations=activations,
            activation_names=list(activations.keys()),
        )

    @classmethod
    def from_activation_names_in_trajectories(
        cls,
        activation_names: Sequence[str],
        trajectories: Sequence,
    ) -> list["CapturedInternals"]:
        """Extract CapturedInternals from trajectories that have internals."""
        results = []
        for traj in trajectories:
            if traj.has_internals():
                internals = getattr(traj, "internals", {})
                captured = cls.from_activation_names(activation_names, internals)
                results.append(captured)
        return results
