"""ContrastivePair: two contrasting trajectories for patching and steering."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inference.interventions import Intervention, InterventionTarget
from .base_schema import BaseSchema
from .choice import LabeledSimpleBinaryChoice
from .hook_utils import hook_name
from .patching_types import PatchingMode
from .time_value import TimeValue
from .token_positions import PositionMapping
from .token_trajectory import TokenTrajectory

DEBUG_INTERVENTIONS = False


@dataclass
class ContrastivePair(BaseSchema):
    """A pair of contrasting trajectories for activation patching.

    Attributes:
        clean_traj: Clean trajectory (baseline/reference behavior)
        corrupted_traj: Corrupted trajectory (target/counterfactual behavior)
        position_mapping: Maps clean positions to corrupted positions
        full_texts: (clean_text, corrupted_text) full prompt+response strings
        clean_labels: (option_a, option_b) labels for clean trajectory
        corrupted_labels: (option_a, option_b) labels for corrupted trajectory
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
    clean_labels: tuple[str, str] | None = None
    corrupted_labels: tuple[str, str] | None = None
    choice_prefix: str = ""
    sample_id: int = 0
    prompt_token_counts: tuple[int, int] | None = None
    choice_divergent_positions: tuple[int, int] | None = None
    time_horizons: tuple[TimeValue, TimeValue] | None = None

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
    # Characteristics
    # =========================================================================

    @property
    def same_labels(self) -> bool:
        if self.clean_labels is None or self.corrupted_labels is None:
            return False
        return self.clean_labels == self.corrupted_labels

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

    # =========================================================================
    # Interventions
    # =========================================================================

    def create_patching_intervention(
        self,
        target: InterventionTarget,
        mode: PatchingMode,
        clean_choice: LabeledSimpleBinaryChoice,
        corrupted_choice: LabeledSimpleBinaryChoice,
        alpha: float = 1.0,
    ) -> list[Intervention]:
        """Create interventions for activation patching.

        Gets activations from the choice objects (computed via forward pass),
        not from pre-cached trajectory activations.

        Args:
            target: InterventionTarget specifying layers/positions to patch
            mode: "denoising" (inject clean into corrupted) or "noising" (inject corrupted into clean)
            clean_choice: Result of runner.choose on clean prompt (has activations)
            corrupted_choice: Result of runner.choose on corrupted prompt (has activations)
            alpha: Interpolation strength (1.0 = full replacement)

        Returns:
            List of Intervention objects for each layer
        """
        # Get activations from the choice trees
        # Only the source choice needs internals (clean for denoising, corrupted for noising)
        clean_internals = self._get_choice_internals(clean_choice)
        corrupted_internals = self._get_choice_internals(corrupted_choice)

        source_internals = (
            clean_internals if mode == "denoising" else corrupted_internals
        )
        if not source_internals:
            raise ValueError(f"Missing internals in source choice for {mode} mode")

        # Resolve layers from source internals
        available = self._get_available_layers(source_internals)
        layers = target.resolve_layers(available)
        component = target.component or "resid_post"

        if DEBUG_INTERVENTIONS:
            print(
                f"[intervention] target.layers={target.layers}, available={len(available)}, resolved={layers}"
            )

        interventions = []
        for layer in layers:
            intervention = self._make_layer_intervention(
                layer,
                component,
                target,
                mode,
                clean_internals,
                corrupted_internals,
                alpha,
            )
            if intervention:
                interventions.append(intervention)

        return interventions

    def _get_choice_internals(self, choice: LabeledSimpleBinaryChoice) -> dict:
        """Extract internals from a choice object."""
        if not choice.tree or not choice.tree.trajs:
            return {}
        # Use first trajectory's internals (shared prefix has same activations)
        traj = choice.tree.trajs[0]
        return traj.internals if traj.has_internals() else {}

    def _get_available_layers(self, internals: dict) -> list[int]:
        """Get available layers from internals dict."""
        from .hook_utils import parse_hook_name

        layers = set()
        for name in internals.keys():
            parsed = parse_hook_name(name)
            if parsed:
                layers.add(parsed[0])
        return sorted(layers)

    def _make_layer_intervention(
        self,
        layer: int,
        component: str,
        target: InterventionTarget,
        mode: PatchingMode,
        clean_internals: dict,
        corrupted_internals: dict,
        alpha: float,
    ) -> Intervention | None:
        """Create intervention for a single layer."""
        hook = hook_name(layer, component)

        # Get source activations (only source needs internals)
        if mode == "denoising":
            patch_acts = clean_internals.get(hook)
            running_len = self.corrupted_length  # Destination sequence length
        else:
            patch_acts = corrupted_internals.get(hook)
            running_len = self.clean_length  # Destination sequence length

        if patch_acts is None:
            return None

        # Squeeze batch dimension if present: [1, seq, hidden] -> [seq, hidden]
        if patch_acts.ndim == 3 and patch_acts.shape[0] == 1:
            patch_acts = patch_acts.squeeze(0)

        positions = target.positions

        if mode == "denoising":
            # Running corrupted context, inject clean activations
            if positions:
                patch_positions = [
                    self.position_mapping.dst_to_src_interpolated(p) for p in positions
                ]
            else:
                patch_positions = [
                    self.position_mapping.dst_to_src_interpolated(p)
                    for p in range(running_len)
                ]
        else:
            # Running clean context, inject corrupted activations
            if positions:
                patch_positions = [self.position_mapping.get(p, p) for p in positions]
            else:
                patch_positions = [
                    self.position_mapping.get(p, p) for p in range(running_len)
                ]

        # Clamp positions to valid range
        patch_positions = [
            max(0, min(int(p), len(patch_acts) - 1)) for p in patch_positions
        ]
        patch_vals = patch_acts[patch_positions]

        # Properly convert tensor to numpy array
        if hasattr(patch_vals, "detach"):
            patch_vals = patch_vals.detach()
        if hasattr(patch_vals, "cpu"):
            patch_vals = patch_vals.cpu()
        if hasattr(patch_vals, "numpy"):
            patch_vals = patch_vals.numpy()

        patch_target = (
            InterventionTarget.at_positions(positions)
            if positions
            else InterventionTarget.all()
        )

        if DEBUG_INTERVENTIONS and layer == 0:
            print(f"[intervention] L{layer} mode={mode} alpha={alpha}")
            print(f"[intervention]   patch_acts.shape={patch_acts.shape}")
            print(
                f"[intervention]   running_len={running_len}, target.positions={positions}"
            )
            print(
                f"[intervention]   patch_positions[:5]={patch_positions[:5]}, len={len(patch_positions)}"
            )
            print(f"[intervention]   patch_vals.shape={patch_vals.shape}")
            print(f"[intervention]   patch_target={patch_target}")

        # Use set mode for full replacement, interpolate for partial
        if alpha >= 1.0:
            return Intervention(
                layer=layer,
                mode="set",
                values=patch_vals,
                target=patch_target,
                component=component,
            )
        else:
            return Intervention(
                layer=layer,
                mode="interpolate",
                values=patch_vals,  # Required but unused for interpolate mode
                target_values=patch_vals,
                alpha=alpha,
                target=patch_target,
                component=component,
            )

    def print_summary(self) -> None:
        print(f"Clean: {self.clean_length}, Corrupted: {self.corrupted_length}")

    def print_position_mapping_debug(self, prefix: str = "[debug]") -> None:
        """Print debug info about position mapping."""
        pm = self.position_mapping
        print(f"{prefix} Position mapping: src_len={pm.src_len}, dst_len={pm.dst_len}")
        print(
            f"{prefix} Anchors ({len(pm.anchors)}): {list(zip(pm.anchors, pm.anchor_texts))}"
        )
        print(
            f"{prefix} First interesting: {pm.first_interesting_marker} -> pos={pm.first_interesting_pos}"
        )
        if pm.anchors:
            for (src_pos, dst_pos), text in zip(pm.anchors[:3], pm.anchor_texts[:3]):
                print(
                    f"{prefix}   Anchor '{text}': src={src_pos} -> dst={dst_pos}, mapping[{src_pos}]={pm.mapping.get(src_pos)}"
                )
