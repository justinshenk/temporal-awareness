"""Settings for attribution patching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..common.base_schema import BaseSchema
from ..common.patching_types import PatchingComponent
from .attribution_quadrature import QuadratureMethod


Method = Literal["standard", "eap", "eap_ig"]


@dataclass
class AttributionSettings(BaseSchema):
    """Settings for attribution computation.

    Attributes:
        components: Components to compute attributions for
        methods: Attribution methods to use
        ig_steps: Integration steps for EAP-IG
        quadrature: Quadrature methods for EAP-IG integration

    Note:
        Gradient computation point is determined by mode:
        - noising: grad@clean (gradients at clean/source state)
        - denoising: grad@corrupted (gradients at corrupted/source state)
    """

    components: list[PatchingComponent] = field(default_factory=lambda: ["resid_post"])
    methods: list[Method] = field(default_factory=lambda: ["eap_ig"])
    ig_steps: int = 10
    quadrature: list[QuadratureMethod] = field(
        default_factory=lambda: [QuadratureMethod.CHEBYSHEV]
    )

    @classmethod
    def all(cls) -> "AttributionSettings":
        """Default settings."""
        return cls()
