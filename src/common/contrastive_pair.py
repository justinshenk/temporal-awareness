"""ContrastivePair: two contrasting trajectories for patching and steering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..inference.interventions import Intervention, InterventionTarget
from .base_schema import BaseSchema
from .hook_utils import hook_name as make_hook_name, parse_hook_name
from .token_positions import PositionMapping
from .token_trajectory import TokenTrajectory


@dataclass
class ContrastivePair(BaseSchema):
    """A pair of contrasting trajectories for activation patching.

    Attributes:
        clean_traj: Clean trajectory (baseline/reference behavior)
        corrupted_traj: Corrupted trajectory (target/counterfactual behavior)
        position_mapping: Maps clean positions to corrupted positions
        full_texts: (clean_text, corrupted_text) full prompt+response strings
        labels: (clean_label, corrupted_label) choice labels
        choice_prefix: e.g. "I choose: "
        sample_id: Unique identifier for this sample
        prompt_token_counts: (clean_prompt_len, corrupted_prompt_len)
        choice_divergent_positions: (clean_pos, corrupted_pos) where A/B diverge
    """

    clean_traj: TokenTrajectory
    corrupted_traj: TokenTrajectory
    position_mapping: PositionMapping = field(default_factory=PositionMapping)
    full_texts: tuple[str, str] = ("", "")
    prompt_texts: tuple[str, str] = ("", "")
    labels: tuple[str, str] | None = None
    choice_prefix: str = ""
    sample_id: int = 0
    prompt_token_counts: tuple[int, int] | None = None
    choice_divergent_positions: tuple[int, int] | None = None

    # =========================================================================
    # Text and Label Properties
    # =========================================================================

    @property
    def clean_text(self) -> str:
        return self.full_texts[0]

    @property
    def corrupted_text(self) -> str:
        return self.full_texts[1]

    @property
    def clean_prompt(self) -> str:
        return self.prompt_texts[0]

    @property
    def corrupted_prompt(self) -> str:
        return self.prompt_texts[1]

    @property
    def clean_label(self) -> str | None:
        return self.labels[0] if self.labels else None

    @property
    def corrupted_label(self) -> str | None:
        return self.labels[1] if self.labels else None

    @property
    def clean_divergent_position(self) -> int | None:
        """Position where A/B tokens diverge in clean trajectory."""
        if self.choice_divergent_positions is None:
            return None
        return self.choice_divergent_positions[0]

    @property
    def corrupted_divergent_position(self) -> int | None:
        """Position where A/B tokens diverge in corrupted trajectory."""
        if self.choice_divergent_positions is None:
            return None
        return self.choice_divergent_positions[1]

    # =========================================================================
    # Length Properties
    # =========================================================================

    @property
    def clean_length(self) -> int:
        return self.clean_traj.n_sequence

    @property
    def corrupted_length(self) -> int:
        return self.corrupted_traj.n_sequence

    @property
    def max_length(self) -> int:
        return max(self.clean_traj.n_sequence, self.corrupted_traj.n_sequence)

    @property
    def clean_prompt_length(self) -> int:
        if self.prompt_token_counts:
            return self.prompt_token_counts[0]
        return 0

    @property
    def corrupted_prompt_length(self) -> int:
        if self.prompt_token_counts:
            return self.prompt_token_counts[1]
        return 0

    # =========================================================================
    # Trajectory Aliases
    # =========================================================================

    @property
    def clean(self) -> TokenTrajectory:
        return self.clean_traj

    @property
    def corrupted(self) -> TokenTrajectory:
        return self.corrupted_traj

    @property
    def reference(self) -> TokenTrajectory:
        """Clean as reference (baseline behavior)."""
        return self.clean_traj

    @property
    def counterfactual(self) -> TokenTrajectory:
        """Corrupted as counterfactual (target behavior)."""
        return self.corrupted_traj

    @property
    def baseline(self) -> TokenTrajectory:
        """Clean as baseline for patching."""
        return self.clean_traj

    @property
    def target(self) -> TokenTrajectory:
        """Corrupted as target for patching."""
        return self.corrupted_traj

    # =========================================================================
    # Cache Access
    # =========================================================================

    @property
    def clean_cache(self) -> dict:
        return self.clean_traj.internals if self.clean_traj.has_internals() else {}

    @property
    def corrupted_cache(self) -> dict:
        return self.corrupted_traj.internals if self.corrupted_traj.has_internals() else {}

    @property
    def available_layers(self) -> list[int]:
        """Layers with cached activations."""
        layers = set()
        for name in self.clean_cache.keys():
            parsed = parse_hook_name(name)
            if parsed:
                layers.add(parsed[0])
        return sorted(layers)

    @property
    def available_components(self) -> list[str]:
        """Components with cached activations."""
        components = set()
        for name in self.clean_cache.keys():
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
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
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
            if self._get_acts(self.clean_cache, layer, component) is not None
        ]

    def _make_intervention(
        self,
        target: InterventionTarget,
        layer: int,
        component: str,
        mode: str,
        alpha: float,
    ) -> Intervention:
        """Create intervention for a single layer.

        Patching semantics:
        - Denoising: Run model on clean context, inject corrupted activations
        - Noising: Run model on corrupted context, inject clean activations

        The intervention.values contains the activations we inject INTO the running context.
        """
        clean_acts = self._get_acts(self.clean_cache, layer, component)
        corrupted_acts = self._get_acts(self.corrupted_cache, layer, component)

        if clean_acts is None or corrupted_acts is None:
            raise ValueError(f"Missing activations for layer {layer}")

        positions = target.positions

        if mode == "denoising":
            # Denoising: Run clean context, inject corrupted activations
            running_acts = clean_acts  # What we're running the model on
            patch_acts = corrupted_acts  # What we're injecting
            if positions:
                # Map positions from clean (running) to corrupted (patch) space
                patch_positions = [self.position_mapping.get(p, p) for p in positions]
                patch_positions = [
                    max(0, min(p, len(corrupted_acts) - 1)) for p in patch_positions
                ]
                running_positions = [
                    max(0, min(p, len(clean_acts) - 1)) for p in positions
                ]
                running_vals = running_acts[running_positions]
                patch_vals = patch_acts[patch_positions]
            else:
                # All positions: use position mapping for each position
                running_len = len(clean_acts)
                patch_positions = [
                    self.position_mapping.get(p, p) for p in range(running_len)
                ]
                patch_positions = [
                    max(0, min(p, len(corrupted_acts) - 1)) for p in patch_positions
                ]
                running_vals = running_acts
                patch_vals = patch_acts[patch_positions]
        else:
            # Noising: Run corrupted context, inject clean activations
            running_acts = corrupted_acts  # What we're running the model on
            patch_acts = clean_acts  # What we're injecting
            if positions:
                # Map positions from corrupted (running) to clean (patch) space
                patch_positions = [
                    self.position_mapping.dst_to_src_interpolated(p)
                    for p in positions
                ]
                patch_positions = [
                    max(0, min(p, len(clean_acts) - 1)) for p in patch_positions
                ]
                running_positions = [
                    max(0, min(p, len(corrupted_acts) - 1)) for p in positions
                ]
                running_vals = running_acts[running_positions]
                patch_vals = patch_acts[patch_positions]
            else:
                # All positions: use position mapping for each position
                running_len = len(corrupted_acts)
                patch_positions = [
                    self.position_mapping.dst_to_src_interpolated(p)
                    for p in range(running_len)
                ]
                patch_positions = [
                    max(0, min(p, len(clean_acts) - 1)) for p in patch_positions
                ]
                running_vals = running_acts
                patch_vals = patch_acts[patch_positions]

        patch_target = (
            InterventionTarget.at_positions(positions)
            if positions
            else InterventionTarget.all()
        )

        if alpha < 1.0:
            # Interpolate between running activations and patch activations
            return Intervention(
                layer=layer,
                mode="interpolate",
                values=running_vals,
                target_values=patch_vals,
                alpha=alpha,
                target=patch_target,
                component=component,
            )

        # Full replacement: inject patch activations into running context
        return Intervention(
            layer=layer,
            mode="set",
            values=patch_vals,
            target=patch_target,
            component=component,
        )

    def get_steering_vector(
        self, layer: int, component: str = "resid_post"
    ) -> np.ndarray:
        """Get mean (corrupted - clean) difference."""
        clean = self._get_acts(self.clean_cache, layer, component)
        corrupted = self._get_acts(self.corrupted_cache, layer, component)
        if clean is None or corrupted is None:
            raise ValueError(f"Missing activations for layer {layer}")
        min_len = min(len(clean), len(corrupted))
        return (corrupted[:min_len] - clean[:min_len]).mean(axis=0)

    def print_summary(self) -> None:
        layers = self.available_layers
        layers_str = f"{layers[:5]}..." if len(layers) > 5 else str(layers)
        print(
            f"Clean: {self.clean_length}, Corrupted: {self.corrupted_length}, Layers: {layers_str}"
        )

    def switch(self) -> ContrastivePair:
        """Return a new ContrastivePair with clean and corrupted swapped.

        Also inverts the position_mapping so it maps corrupted→clean instead of clean→corrupted.

        Note: Cached activations (accessible via clean_cache/corrupted_cache) are also swapped
        with the trajectories, since they are stored within each TokenTrajectory's internals.
        """
        return ContrastivePair(
            clean_traj=self.corrupted_traj,
            corrupted_traj=self.clean_traj,
            position_mapping=self.position_mapping.switch(),
            full_texts=(self.full_texts[1], self.full_texts[0]),
            prompt_texts=(self.prompt_texts[1], self.prompt_texts[0]),
            labels=(self.labels[1], self.labels[0]) if self.labels else None,
            choice_prefix=self.choice_prefix,
            sample_id=self.sample_id,
            prompt_token_counts=(
                (self.prompt_token_counts[1], self.prompt_token_counts[0])
                if self.prompt_token_counts
                else None
            ),
            choice_divergent_positions=(
                (self.choice_divergent_positions[1], self.choice_divergent_positions[0])
                if self.choice_divergent_positions
                else None
            ),
        )
