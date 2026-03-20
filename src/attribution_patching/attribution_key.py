"""Typed attribution keys - no string parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from .attribution_quadrature import QuadratureMethod


Method = Literal["standard", "eap", "eap_ig"]
Component = Literal["resid_post", "attn_out", "mlp_out"]


class AttributionKey(NamedTuple):
    """Immutable, hashable key for attribution results.

    Use as dict key: results[AttributionKey("eap_ig", "attn_out", "midpoint")]
    """
    method: Method
    component: Component
    quadrature: str | None = None  # None for standard, required for eap/eap_ig

    def __str__(self) -> str:
        """Human-readable format: method/component[/quadrature]."""
        if self.quadrature:
            return f"{self.method}/{self.component}/{self.quadrature}"
        return f"{self.method}/{self.component}"

    @classmethod
    def from_str(cls, s: str) -> "AttributionKey":
        """Parse from string format."""
        parts = s.split("/")
        if len(parts) == 2:
            return cls(parts[0], parts[1], None)  # type: ignore
        elif len(parts) == 3:
            return cls(parts[0], parts[1], parts[2])  # type: ignore
        else:
            raise ValueError(f"Invalid key format: {s}")

    @classmethod
    def standard(cls, component: Component) -> "AttributionKey":
        return cls("standard", component, None)

    @classmethod
    def eap(cls, component: Component, quadrature: str) -> "AttributionKey":
        return cls("eap", component, quadrature)

    @classmethod
    def eap_ig(cls, component: Component, quadrature: str) -> "AttributionKey":
        return cls("eap_ig", component, quadrature)


# Components available per method (all methods support all components)
STANDARD_COMPONENTS: list[Component] = ["resid_post", "attn_out", "mlp_out"]
EAP_COMPONENTS: list[Component] = ["resid_post", "attn_out", "mlp_out"]


@dataclass
class AttributionConfig:
    """Configuration for which attributions to compute."""

    methods: list[Method]
    quadratures: list[QuadratureMethod]
    ig_steps: int = 10

    def get_keys(self) -> list[AttributionKey]:
        """Generate all attribution keys for this config."""
        keys = []

        if "standard" in self.methods:
            for comp in STANDARD_COMPONENTS:
                keys.append(AttributionKey.standard(comp))

        for quad in self.quadratures:
            quad_str = quad.value
            if "eap" in self.methods:
                for comp in EAP_COMPONENTS:
                    keys.append(AttributionKey.eap(comp, quad_str))
            if "eap_ig" in self.methods:
                for comp in EAP_COMPONENTS:
                    keys.append(AttributionKey.eap_ig(comp, quad_str))

        return keys
