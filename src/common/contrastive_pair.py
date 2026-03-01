"""ContrastivePair: two contrasting trajectories for patching and steering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_schema import BaseSchema
from .token_trajectory import TokenTrajectory
from .token_positions import PositionMapping
from .hook_utils import hook_name as make_hook_name, parse_hook_name
from ..inference.interventions import Intervention, InterventionTarget


@dataclass
class ContrastivePair(BaseSchema):
    """A pair of contrasting trajectories for activation patching.

    Attributes:
        short_traj: Trajectory for short-term choice
        long_traj: Trajectory for long-term choice
        position_mapping: Maps short positions to long positions
        labels: (short_label, long_label)
        choice_prefix: e.g. "I choose: "
    """

    short_traj: TokenTrajectory
    long_traj: TokenTrajectory
    position_mapping: PositionMapping = field(default_factory=PositionMapping)
    full_texts: tuple[str, str] = ("", "")
    labels: tuple[str, str] | None = None
    choice_prefix: str = ""
    sample_id: int = 0
    prompt_token_counts: tuple[int, int] | None = None
    choice_divergent_positions: tuple[int, int] | None = None

    # =========================================================================
    # Text and Label Properties
    # =========================================================================

    @property
    def short_text(self) -> str:
        return self.full_texts[0]

    @property
    def long_text(self) -> str:
        return self.full_texts[1]

    @property
    def short_label(self) -> str | None:
        return self.labels[0] if self.labels else None

    @property
    def long_label(self) -> str | None:
        return self.labels[1] if self.labels else None

    @property
    def short_divergent_position(self) -> int | None:
        """Position where A/B tokens diverge in short trajectory."""
        if self.choice_divergent_positions is None:
            return None
        return self.choice_divergent_positions[0]

    @property
    def long_divergent_position(self) -> int | None:
        """Position where A/B tokens diverge in long trajectory."""
        if self.choice_divergent_positions is None:
            return None
        return self.choice_divergent_positions[1]

    # =========================================================================
    # Length Properties
    # =========================================================================

    @property
    def short_length(self) -> int:
        return self.short_traj.n_sequence

    @property
    def long_length(self) -> int:
        return self.long_traj.n_sequence

    @property
    def max_length(self) -> int:
        return max(self.short_traj.n_sequence, self.long_traj.n_sequence)

    @property
    def short_prompt_length(self) -> int:
        if self.prompt_token_counts and len(self.prompt_token_counts) > 0:
            return self.prompt_token_counts[0]
        return 0

    @property
    def long_prompt_length(self) -> int:
        if self.prompt_token_counts and len(self.prompt_token_counts) > 1:
            return self.prompt_token_counts[1]
        return 0

    # =========================================================================
    # Trajectory Aliases
    # =========================================================================

    @property
    def short(self) -> TokenTrajectory:
        return self.short_traj

    @property
    def long(self) -> TokenTrajectory:
        return self.long_traj

    @property
    def reference(self) -> TokenTrajectory:
        """Short as reference (baseline behavior)."""
        return self.short_traj

    @property
    def counterfactual(self) -> TokenTrajectory:
        """Long as counterfactual (target behavior)."""
        return self.long_traj

    @property
    def positive(self) -> TokenTrajectory:
        """Long as positive (desired behavior)."""
        return self.long_traj

    @property
    def negative(self) -> TokenTrajectory:
        """Short as negative (baseline behavior)."""
        return self.short_traj

    @property
    def baseline(self) -> TokenTrajectory:
        """Short as baseline for patching."""
        return self.short_traj

    @property
    def target(self) -> TokenTrajectory:
        """Long as target for patching."""
        return self.long_traj

    # =========================================================================
    # Cache Access
    # =========================================================================

    @property
    def short_cache(self) -> dict:
        return self.short_traj.internals if self.short_traj.has_internals() else {}

    @property
    def long_cache(self) -> dict:
        return self.long_traj.internals if self.long_traj.has_internals() else {}

    @property
    def available_layers(self) -> list[int]:
        """Layers with cached activations."""
        layers = set()
        for name in self.short_cache.keys():
            parsed = parse_hook_name(name)
            if parsed:
                layers.add(parsed[0])
        return sorted(layers)

    @property
    def available_components(self) -> list[str]:
        """Components with cached activations."""
        components = set()
        for name in self.short_cache.keys():
            parsed = parse_hook_name(name)
            if parsed:
                components.add(parsed[1])
        return sorted(components)

    def _get_acts(self, cache: dict, layer: int, component: str) -> np.ndarray | None:
        """Get activations from cache."""
        hook = make_hook_name(layer, component)
        if hook not in cache:
            return None
        act = cache[hook]
        try:
            import torch

            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
        except ImportError:
            pass
        return act[0] if act.ndim == 3 and act.shape[0] == 1 else act

    # =========================================================================
    # Interventions
    # =========================================================================

    def get_interventions(
        self,
        target: InterventionTarget,
        layers: list[int],
        component: str,
        mode: str,
        alpha: float = 1.0,
    ) -> list[Intervention]:
        """Get interventions for all specified layers."""
        return [
            self._make_intervention(target, layer, component, mode, alpha)
            for layer in layers
            if self._get_acts(self.short_cache, layer, component) is not None
        ]

    def _make_intervention(
        self,
        target: InterventionTarget,
        layer: int,
        component: str,
        mode: str,
        alpha: float,
    ) -> Intervention:
        """Create intervention for a single layer."""
        short = self._get_acts(self.short_cache, layer, component)
        long = self._get_acts(self.long_cache, layer, component)

        if short is None or long is None:
            raise ValueError(f"Missing activations for layer {layer}")

        positions = target.positions

        if mode == "denoising":
            # Inject long into short context
            source, dest = short, long
            if positions:
                mapped = [self.position_mapping.get(p, p) for p in positions]
                mapped = [max(0, min(p, len(long) - 1)) for p in mapped]
                positions = [max(0, min(p, len(short) - 1)) for p in positions]
                src_vals = source[list(positions)]
                dst_vals = dest[mapped]
            else:
                min_len = min(len(short), len(long))
                src_vals, dst_vals = source[:min_len], dest[:min_len]
        else:
            # Inject short into long context
            source, dest = long, short
            if positions:
                mapped = [self.position_mapping.dst_to_src(p) or p for p in positions]
                mapped = [max(0, min(p, len(short) - 1)) for p in mapped]
                positions = [max(0, min(p, len(long) - 1)) for p in positions]
                src_vals = source[list(positions)]
                dst_vals = dest[mapped]
            else:
                min_len = min(len(short), len(long))
                src_vals, dst_vals = source[:min_len], dest[:min_len]

        patch_target = (
            InterventionTarget.at_positions(positions)
            if positions
            else InterventionTarget.all()
        )

        if alpha < 1.0:
            return Intervention(
                layer=layer,
                mode="interpolate",
                values=src_vals,
                target_values=dst_vals,
                alpha=alpha,
                target=patch_target,
                component=component,
            )

        return Intervention(
            layer=layer,
            mode="set",
            values=dst_vals,
            target=patch_target,
            component=component,
        )

    def get_steering_vector(
        self, layer: int, component: str = "resid_post"
    ) -> np.ndarray:
        """Get mean (long - short) difference."""
        short = self._get_acts(self.short_cache, layer, component)
        long = self._get_acts(self.long_cache, layer, component)
        if short is None or long is None:
            raise ValueError(f"Missing activations for layer {layer}")
        min_len = min(len(short), len(long))
        return (long[:min_len] - short[:min_len]).mean(axis=0)

    def print_summary(self) -> None:
        layers = self.available_layers
        layers_str = f"{layers[:5]}..." if len(layers) > 5 else str(layers)
        print(f"Short: {self.short_length}, Long: {self.long_length}, Layers: {layers_str}")
