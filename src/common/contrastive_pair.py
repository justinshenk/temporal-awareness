"""ContrastivePair: two contrasting trajectories for analysis and intervention.

A ContrastivePair stores two trajectories (short and long) along with their
cached activations, supporting various use cases:
- Activation patching (denoising/noising)
- Steering vector extraction
- Contrastive Activation Addition (CAA)
- Choice analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base_schema import BaseSchema
from .token_trajectory import TokenTrajectory
from .token_positions import PositionMapping
from ..inference.interventions import Intervention, InterventionTarget


@dataclass
class ContrastivePair(BaseSchema):
    """A pair of contrasting trajectories.

    Primary fields use short/long terminology (intertemporal domain).
    Property aliases provide alternative access patterns for different use cases.

    Attributes:
        short_traj: Trajectory for short-term choice
        long_traj: Trajectory for long-term choice
        position_mapping: Maps short positions to long positions for alignment

    Aliases:
        short/long: short_traj/long_traj
        reference/counterfactual: short=reference, long=counterfactual
        positive/negative: long=positive, short=negative
        baseline/target: short=baseline, long=target
    """

    short_traj: TokenTrajectory
    long_traj: TokenTrajectory
    position_mapping: PositionMapping = field(default_factory=PositionMapping)

    # Metadata
    full_texts: tuple[str, str] = ("", "")  # (short_text, long_text)
    labels: tuple[str, str] | None = None  # (short_term_label, long_term_label)
    choice_prefix: str = ""  # e.g. "I select: "

    @property
    def short_text(self) -> str:
        """Full text for short-term trajectory."""
        return self.full_texts[0]

    @property
    def long_text(self) -> str:
        """Full text for long-term trajectory."""
        return self.full_texts[1]

    @property
    def short_label(self) -> str | None:
        """Label for the short-term option."""
        return self.labels[0] if self.labels else None

    @property
    def long_label(self) -> str | None:
        """Label for the long-term option."""
        return self.labels[1] if self.labels else None

    # =========================================================================
    # Aliases: short / long
    # =========================================================================

    @property
    def short(self) -> TokenTrajectory:
        """Alias for short_traj."""
        return self.short_traj

    @property
    def long(self) -> TokenTrajectory:
        """Alias for long_traj."""
        return self.long_traj

    # =========================================================================
    # Aliases: reference / counterfactual
    # =========================================================================

    @property
    def reference(self) -> TokenTrajectory:
        """Alias: short as reference."""
        return self.short_traj

    @property
    def counterfactual(self) -> TokenTrajectory:
        """Alias: long as counterfactual."""
        return self.long_traj

    # =========================================================================
    # Aliases: positive / negative (contrastive learning convention)
    # =========================================================================

    @property
    def positive(self) -> TokenTrajectory:
        """Alias: long as positive (desired behavior)."""
        return self.long_traj

    @property
    def negative(self) -> TokenTrajectory:
        """Alias: short as negative (baseline behavior)."""
        return self.short_traj

    # =========================================================================
    # Aliases: baseline / target (patching convention)
    # =========================================================================

    @property
    def baseline(self) -> TokenTrajectory:
        """Alias: short as baseline (starting behavior)."""
        return self.short_traj

    @property
    def target(self) -> TokenTrajectory:
        """Alias: long as target (goal behavior)."""
        return self.long_traj

    # =========================================================================
    # Cache access
    # =========================================================================

    @property
    def short_cache(self) -> dict:
        """Get short activations from trajectory internals."""
        if self.short_traj.has_internals():
            return self.short_traj.internals
        return {}

    @property
    def long_cache(self) -> dict:
        """Get long activations from trajectory internals."""
        if self.long_traj.has_internals():
            return self.long_traj.internals
        return {}

    # =========================================================================
    # Interventions
    # =========================================================================

    def get_noising_intervention(
        self,
        target: InterventionTarget,
        layer: int,
        component: str = "resid_post",
    ) -> Intervention:
        """Create intervention to replace long activations with short ones.

        Injects "short-term" activations into a long-term forward pass.
        (In patching terms: corrupts the target behavior.)

        Args:
            target: Which positions/neurons to patch
            layer: Which layer to patch
            component: Which component (resid_pre, resid_post, etc.)

        Returns:
            Intervention that patches short values into long positions
        """
        hook_name = f"blocks.{layer}.hook_{component}"
        long_acts = self._get_activation(self.long_cache, hook_name)
        short_acts = self._get_activation(self.short_cache, hook_name)

        if short_acts is None:
            raise ValueError(f"No short activations found for {hook_name}")
        if long_acts is None:
            raise ValueError(f"No long activations found for {hook_name}")

        if target.axis == "position" and target.positions:
            # For noising: target.positions are LONG positions (base text)
            # Reverse map LONG -> SHORT to get corresponding SHORT positions
            short_positions = [
                self.position_mapping.dst_to_src(p) or p for p in target.positions
            ]
            # Clamp to valid range (handle both negative and out-of-bounds)
            short_positions = [max(0, min(p, len(short_acts) - 1)) for p in short_positions]
            long_positions = [max(0, min(p, len(long_acts) - 1)) for p in target.positions]

            short_at_pos = short_acts[short_positions]
            long_at_pos = long_acts[long_positions]
            values = short_at_pos - long_at_pos
            # Patch at LONG positions (the base text positions)
            patch_target = InterventionTarget.at_positions(long_positions)
        elif target.axis == "all":
            min_len = min(len(long_acts), len(short_acts))
            diff = short_acts[:min_len] - long_acts[:min_len]
            values = diff.mean(axis=0)
            patch_target = InterventionTarget.all()
        else:
            values = short_acts - long_acts
            patch_target = target

        return Intervention(
            layer=layer,
            mode="add",
            values=values,
            target=patch_target,
            component=component,
        )

    def get_denoising_intervention(
        self,
        target: InterventionTarget,
        layer: int,
        component: str = "resid_post",
    ) -> Intervention:
        """Create intervention to replace short activations with long ones.

        Injects "long-term" activations into a short-term forward pass.
        (In patching terms: recovers target behavior from baseline.)

        Args:
            target: Which positions/neurons to patch
            layer: Which layer to patch
            component: Which component (resid_pre, resid_post, etc.)

        Returns:
            Intervention that patches long values into short positions
        """
        hook_name = f"blocks.{layer}.hook_{component}"
        long_acts = self._get_activation(self.long_cache, hook_name)
        short_acts = self._get_activation(self.short_cache, hook_name)

        if long_acts is None:
            raise ValueError(f"No long activations found for {hook_name}")
        if short_acts is None:
            raise ValueError(f"No short activations found for {hook_name}")

        if target.axis == "position" and target.positions:
            # For denoising: target.positions are SHORT positions (base text)
            # Map SHORT -> LONG to get corresponding LONG positions
            long_positions = [
                self.position_mapping.get(p, p) for p in target.positions
            ]
            # Clamp to valid range (handle both negative and out-of-bounds)
            long_positions = [max(0, min(p, len(long_acts) - 1)) for p in long_positions]
            short_positions = [max(0, min(p, len(short_acts) - 1)) for p in target.positions]

            long_at_pos = long_acts[long_positions]
            short_at_pos = short_acts[short_positions]
            values = long_at_pos - short_at_pos
            # Patch at SHORT positions (the base text positions)
            patch_target = InterventionTarget.at_positions(short_positions)
        elif target.axis == "all":
            min_len = min(len(long_acts), len(short_acts))
            diff = long_acts[:min_len] - short_acts[:min_len]
            values = diff.mean(axis=0)
            patch_target = InterventionTarget.all()
        else:
            values = long_acts - short_acts
            patch_target = target

        return Intervention(
            layer=layer,
            mode="add",
            values=values,
            target=patch_target,
            component=component,
        )

    def get_intervention(
        self,
        target: InterventionTarget,
        layer: int,
        component: str = "resid_post",
        mode: str = "denoising",
    ) -> Intervention:
        """Get intervention for the specified mode.

        Args:
            target: Which positions/neurons to patch
            layer: Which layer to patch
            component: Which component (resid_pre, resid_post, etc.)
            mode: "denoising" or "noising"

        Returns:
            Intervention configured for the specified mode
        """
        if mode == "denoising":
            return self.get_denoising_intervention(target, layer, component)
        return self.get_noising_intervention(target, layer, component)

    def get_interventions(
        self,
        target: InterventionTarget,
        layers: list[int],
        component: str = "resid_post",
        mode: str = "denoising",
    ) -> list[Intervention]:
        """Get interventions across multiple layers for a single target.

        Skips layers where intervention creation fails.
        """
        interventions = []
        for layer in layers:
            try:
                interventions.append(
                    self.get_intervention(target, layer, component, mode)
                )
            except ValueError:
                continue
        return interventions

    def get_steering_vector(
        self,
        layer: int,
        component: str = "resid_post",
    ) -> np.ndarray:
        """Get steering vector (long - short) for this layer.

        Returns:
            Mean activation difference [hidden_size]
        """
        hook_name = f"blocks.{layer}.hook_{component}"
        long_acts = self._get_activation(self.long_cache, hook_name)
        short_acts = self._get_activation(self.short_cache, hook_name)

        if long_acts is None or short_acts is None:
            raise ValueError(f"Missing activations for {hook_name}")

        min_len = min(len(long_acts), len(short_acts))
        diff = long_acts[:min_len] - short_acts[:min_len]
        return diff.mean(axis=0)

    def _get_activation(self, cache: dict, hook_name: str) -> Optional[np.ndarray]:
        """Extract activation from cache, converting to numpy if needed."""
        if hook_name not in cache:
            return None

        act = cache[hook_name]

        try:
            import torch

            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
        except ImportError:
            pass

        if act.ndim == 3 and act.shape[0] == 1:
            act = act[0]

        return act

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def short_length(self) -> int:
        """Length of the short trajectory."""
        return self.short_traj.n_sequence

    @property
    def long_length(self) -> int:
        """Length of the long trajectory."""
        return self.long_traj.n_sequence

    @property
    def available_layers(self) -> list[int]:
        """List of layers that have cached activations."""
        layers = set()
        for hook_name in self.short_cache.keys():
            if hook_name.startswith("blocks.") and ".hook_" in hook_name:
                try:
                    layer = int(hook_name.split(".")[1])
                    layers.add(layer)
                except (IndexError, ValueError):
                    continue
        return sorted(layers)

    @property
    def available_components(self) -> list[str]:
        """List of components that have cached activations."""
        components = set()
        for hook_name in self.short_cache.keys():
            if ".hook_" in hook_name:
                try:
                    comp = hook_name.split(".hook_")[1]
                    components.add(comp)
                except IndexError:
                    continue
        return sorted(components)

    def print_summary(self) -> None:
        layers = self.available_layers
        layers_str = f"{layers[:5]}..." if len(layers) > 5 else str(layers)
        print(f"Short: {self.short_length} tokens, Long: {self.long_length} tokens")
        print(f"Layers: {layers_str}")
