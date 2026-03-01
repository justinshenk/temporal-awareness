"""Settings for attribution patching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..common.base_schema import BaseSchema


Method = Literal["standard", "eap", "eap_ig"]
Component = Literal["resid_post", "attn_out", "mlp_out"]
GradPoint = Literal["clean", "corrupted", "both"]


@dataclass
class AttributionSettings(BaseSchema):
    """Settings for attribution computation.

    Attributes:
        components: Components to compute attributions for
        methods: Attribution methods to use
        ig_steps: Integration steps for EAP-IG
        grad_at: Where to compute gradients ("clean", "corrupted", or "both")
    """

    components: list[Component] = field(default_factory=lambda: ["resid_post"])
    methods: list[Method] = field(default_factory=lambda: ["standard", "eap"])
    ig_steps: int = 10
    grad_at: GradPoint = "both"

    @classmethod
    def all(cls) -> "AttributionSettings":
        """Default settings."""
        return cls()

    @classmethod
    def standard_only(cls) -> "AttributionSettings":
        """Use only standard attribution (fastest)."""
        return cls(methods=["standard"])

    @classmethod
    def with_ig(cls, steps: int = 10) -> "AttributionSettings":
        """Include EAP-IG for more accurate attributions."""
        return cls(methods=["standard", "eap", "eap_ig"], ig_steps=steps)
