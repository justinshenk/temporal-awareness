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
        min_layer_depth: First swept layer as a fraction of total layers.
            Defaults to full coverage; narrowing it bounds any localization
            claim, so it must be set deliberately.
        max_layer_depth: Last swept layer as a fraction of total layers.
    """

    components: list[str] = field(default_factory=lambda: ["resid_post"])
    layer_steps: list[int] | None = None
    pos_steps: list[int] | None = None
    min_layer_depth: float = 0.0
    max_layer_depth: float = 1.0
