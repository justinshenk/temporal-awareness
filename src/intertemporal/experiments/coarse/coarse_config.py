"""Configuration for coarse activation patching."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema


@dataclass
class CoarsePatchingConfig(BaseSchema):
    """Configuration for coarse activation patching analysis.

    Attributes:
        components: Model components to patch (e.g., resid_post, attn_out)
        layer_steps: Step sizes for layer sweep
        pos_steps: Step sizes for position sweep
    """

    components: list[str] = field(default_factory=lambda: ["resid_post"])
    layer_steps: list[int] | None = None
    pos_steps: list[int] | None = None
