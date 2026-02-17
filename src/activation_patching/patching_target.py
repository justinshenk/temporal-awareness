"""High-level target specification for activation patching.

Provides ActivationPatchingTarget which is a user-friendly way to specify
what layers and positions to patch, and converts to InterventionTarget(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..common.base_schema import BaseSchema
from ..inference.interventions import InterventionTarget


PositionMode = Literal["all", "prompt", "response", "each", "range", "explicit"]


@dataclass
class ActivationPatchingTarget(BaseSchema):
    """High-level specification for what to patch.

    Default: patches all layers and all positions.

    Attributes:
        position_mode: How to determine positions:
            - "all": All positions (default)
            - "prompt": Prompt positions only (0 to prompt_token_count)
            - "response": Response positions only (prompt_token_count to end)
            - "each": One target per position (for position sweeps)
            - "range": Range from token_positions[0] to token_positions[1]
            - "explicit": Specific positions from token_positions
        token_positions: Positions for "range" or "explicit" modes
        layers: Layer specification:
            - "all": All available layers, patched together (default)
            - "each": Each layer separately (for layer sweeps)
            - list[int]: Specific layers, patched together
        component: Component to patch (default: "resid_post")
    """

    position_mode: PositionMode = "all"
    token_positions: list[int] | None = None
    layers: str | list[int] = "all"
    component: str = "resid_post"

    # ── Factory Methods ─────────────────────────────────────────────────────

    @classmethod
    def all(cls) -> ActivationPatchingTarget:
        """Patch all positions across all layers."""
        return cls()

    @classmethod
    def at_positions(cls, positions: int | list[int]) -> ActivationPatchingTarget:
        """Patch specific positions across all layers."""
        if isinstance(positions, int):
            positions = [positions]
        return cls(position_mode="explicit", token_positions=positions)

    @classmethod
    def in_range(cls, start: int, end: int) -> ActivationPatchingTarget:
        """Patch positions in a range across all layers."""
        return cls(position_mode="range", token_positions=[start, end])

    @classmethod
    def at_layers(cls, layers: int | list[int]) -> ActivationPatchingTarget:
        """Patch all positions at specific layers."""
        if isinstance(layers, int):
            layers = [layers]
        return cls(layers=layers)

    @classmethod
    def response_only(cls) -> ActivationPatchingTarget:
        """Patch only response positions across all layers."""
        return cls(position_mode="response")

    @classmethod
    def prompt_only(cls) -> ActivationPatchingTarget:
        """Patch only prompt positions across all layers."""
        return cls(position_mode="prompt")

    # ── Conversion ──────────────────────────────────────────────────────────

    def to_intervention_targets(
        self,
        seq_len: int,
        prompt_token_count: int | None = None,
        available_layers: list[int] | None = None,
    ) -> tuple[list[InterventionTarget], list[int], bool]:
        """Convert to concrete intervention targets.

        Args:
            seq_len: Sequence length for position resolution
            prompt_token_count: Boundary between prompt and response
            available_layers: Available layers in the model

        Returns:
            Tuple of:
            - List of InterventionTarget objects
            - List of layer indices to patch
            - patch_together: True if layers should be patched in single forward pass,
              False if each layer should be patched separately
        """
        targets = self._resolve_positions(seq_len, prompt_token_count)
        layers, patch_together = self._resolve_layers(available_layers)
        return targets, layers, patch_together

    def _resolve_positions(
        self,
        seq_len: int,
        prompt_token_count: int | None,
    ) -> list[InterventionTarget]:
        """Convert position spec to InterventionTarget list."""
        if self.position_mode == "all":
            return [InterventionTarget.all()]

        if self.position_mode == "each":
            return [InterventionTarget.at_positions(i) for i in range(seq_len)]

        if self.position_mode == "response":
            start = prompt_token_count if prompt_token_count is not None else 0
            return [InterventionTarget.at_positions(list(range(start, seq_len)))]

        if self.position_mode == "prompt":
            end = prompt_token_count if prompt_token_count is not None else seq_len
            return [InterventionTarget.at_positions(list(range(0, end)))]

        if self.position_mode == "range":
            if not self.token_positions or len(self.token_positions) != 2:
                raise ValueError(
                    "range mode requires token_positions with [start, end]"
                )
            start, end = self.token_positions[0], self.token_positions[1]
            return [
                InterventionTarget.at_positions(list(range(start, min(end, seq_len))))
            ]

        if self.position_mode == "explicit":
            if not self.token_positions:
                raise ValueError("explicit mode requires token_positions")
            return [InterventionTarget.at_positions(self.token_positions)]

        raise ValueError(f"Unknown position_mode: {self.position_mode}")

    def _resolve_layers(
        self,
        available_layers: list[int] | None,
    ) -> tuple[list[int], bool]:
        """Resolve layer spec to (layers, patch_together).

        Returns:
            Tuple of (layer_indices, patch_together)
            - "all" -> (all_layers, True) - patch all together
            - "each" -> (all_layers, False) - patch each separately
            - [0,1,2] -> ([0,1,2], True) - patch specified together
        """
        all_layers = available_layers or []

        if self.layers == "all":
            return all_layers, True

        if self.layers == "each":
            return all_layers, False

        if isinstance(self.layers, list):
            return self.layers, True

        raise ValueError(f"Unknown layers specification: {self.layers}")
