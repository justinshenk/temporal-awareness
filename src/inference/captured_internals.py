"""Captured internals from model forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CapturedInternals:
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
