"""Target specification for attribution patching.

Specifies what layers, positions, and components to compute attributions for,
and which methods to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..common.base_schema import BaseSchema


Method = Literal["standard", "eap", "eap_ig"]
Component = Literal["resid_post", "attn_out", "mlp_out"]
PositionMode = Literal["all", "prompt", "response"]


@dataclass
class AttributionTarget(BaseSchema):
    """Target specification for attribution computation.

    Unlike ActivationPatchingTarget which specifies what to patch,
    this specifies what to compute gradients for.

    Attributes:
        position_mode: Which positions to compute attributions for:
            - "all": All positions (default)
            - "prompt": Prompt positions only
            - "response": Response positions only
        layers: Layer specification:
            - "all": All layers (default)
            - list[int]: Specific layers only
        components: Components to compute attributions for
        methods: Attribution methods to use:
            - "standard": Basic attribution (clean - corrupted) * grad
            - "eap": Edge Attribution Patching
            - "eap_ig": EAP with Integrated Gradients
        ig_steps: Number of integration steps for EAP-IG
    """

    position_mode: PositionMode = "all"
    layers: str | list[int] = "all"
    components: list[Component] = field(
        default_factory=lambda: ["resid_post"]
    )
    methods: list[Method] = field(
        default_factory=lambda: ["standard", "eap"]
    )
    ig_steps: int = 10

    # ── Factory Methods ─────────────────────────────────────────────────────

    @classmethod
    def all(cls) -> "AttributionTarget":
        """Compute attributions for all positions and layers."""
        return cls()

    @classmethod
    def standard_only(
        cls,
        layers: str | list[int] = "all",
    ) -> "AttributionTarget":
        """Use only standard attribution (fastest).

        Args:
            layers: Layer specification ("all" or list of layer indices)
        """
        return cls(methods=["standard"], layers=layers)

    @classmethod
    def with_ig(
        cls,
        steps: int = 10,
        layers: str | list[int] = "all",
    ) -> "AttributionTarget":
        """Include EAP-IG for more accurate attributions.

        Args:
            steps: Number of integration steps
            layers: Layer specification
        """
        return cls(methods=["standard", "eap", "eap_ig"], ig_steps=steps, layers=layers)

    @classmethod
    def at_layers(cls, layers: int | list[int]) -> "AttributionTarget":
        """Compute attributions at specific layers only."""
        if isinstance(layers, int):
            layers = [layers]
        return cls(layers=layers)

    @classmethod
    def response_only(cls) -> "AttributionTarget":
        """Compute attributions for response positions only."""
        return cls(position_mode="response")

    # ── Resolution ──────────────────────────────────────────────────────────

    def resolve_layers(
        self,
        available_layers: list[int] | None = None,
    ) -> list[int]:
        """Resolve layer spec to concrete layer indices.

        Args:
            available_layers: Available layers in the model

        Returns:
            List of layer indices to compute attributions for
        """
        if available_layers is None:
            available_layers = []

        if self.layers == "all":
            return available_layers

        if isinstance(self.layers, list):
            # Filter to available layers
            return [l for l in self.layers if l in available_layers]

        raise ValueError(f"Unknown layers specification: {self.layers}")

    def should_include_position(
        self,
        position: int,
        seq_len: int,
        prompt_token_count: int | None = None,
    ) -> bool:
        """Check if a position should be included based on mode.

        Args:
            position: Position index
            seq_len: Total sequence length
            prompt_token_count: Boundary between prompt and response

        Returns:
            True if position should be included
        """
        if self.position_mode == "all":
            return True

        if prompt_token_count is None:
            return True

        if self.position_mode == "prompt":
            return position < prompt_token_count

        if self.position_mode == "response":
            return position >= prompt_token_count

        return True
