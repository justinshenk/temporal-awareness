"""Target specification for interventions.

Specifies which positions and layers to intervene on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...common.base_schema import BaseSchema


COMPONENTS = ("resid_post", "attn_out", "mlp_out")

PositionMode = Literal["all", "explicit"]


@dataclass
class InterventionTarget(BaseSchema):
    """Specifies where to apply an intervention.

    Attributes:
        positions: Token positions to intervene on (None = all positions)
        layers: Layers to intervene on (None = all available layers)
        component: Component to intervene on (resid_post, attn_out, mlp_out)
    """

    positions: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    component: str = "resid_post"

    # ── Factory Methods ─────────────────────────────────────────────────────

    @classmethod
    def all(cls, component: str = "resid_post") -> InterventionTarget:
        """Intervene on all positions and all layers."""
        return cls(component=component)

    @classmethod
    def at_positions(
        cls,
        positions: int | list[int] | tuple[int, ...],
        component: str = "resid_post",
    ) -> InterventionTarget:
        """Intervene on specific positions across all layers."""
        if isinstance(positions, int):
            positions = (positions,)
        elif isinstance(positions, list):
            positions = tuple(positions)
        return cls(positions=positions, component=component)

    @classmethod
    def at_layers(
        cls,
        layers: int | list[int] | tuple[int, ...],
        component: str = "resid_post",
    ) -> InterventionTarget:
        """Intervene on all positions at specific layers."""
        if isinstance(layers, int):
            layers = (layers,)
        elif isinstance(layers, list):
            layers = tuple(layers)
        return cls(layers=layers, component=component)

    @classmethod
    def at(
        cls,
        positions: int | list[int] | tuple[int, ...] | None = None,
        layers: int | list[int] | tuple[int, ...] | None = None,
        component: str = "resid_post",
    ) -> InterventionTarget:
        """Intervene on specific positions and layers."""
        if positions is not None:
            if isinstance(positions, int):
                positions = (positions,)
            elif isinstance(positions, list):
                positions = tuple(positions)

        if layers is not None:
            if isinstance(layers, int):
                layers = (layers,)
            elif isinstance(layers, list):
                layers = tuple(layers)

        return cls(positions=positions, layers=layers, component=component)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_all_positions(self) -> bool:
        """True if intervening on all positions."""
        return self.positions is None

    @property
    def is_all_layers(self) -> bool:
        """True if intervening on all layers."""
        return self.layers is None

    @property
    def n_positions(self) -> int | None:
        """Number of positions, or None if all."""
        return len(self.positions) if self.positions else None

    @property
    def n_layers(self) -> int | None:
        """Number of layers, or None if all."""
        return len(self.layers) if self.layers else None

    # ── Resolution ──────────────────────────────────────────────────────────

    def resolve_layers(self, available_layers: list[int]) -> list[int]:
        """Resolve to concrete layer indices."""
        if self.layers is None:
            return available_layers
        return [l for l in self.layers if l in available_layers]

    def resolve_positions(self, seq_len: int) -> list[int]:
        """Resolve to concrete position indices."""
        if self.positions is None:
            return list(range(seq_len))
        return [p for p in self.positions if 0 <= p < seq_len]

    def with_layers(
        self, layers: int | list[int] | tuple[int, ...]
    ) -> InterventionTarget:
        """Return new target with specified layers."""
        if isinstance(layers, int):
            layers = (layers,)
        elif isinstance(layers, list):
            layers = tuple(layers)
        return InterventionTarget(
            positions=self.positions,
            layers=layers,
            component=self.component,
        )

    def with_positions(
        self, positions: int | list[int] | tuple[int, ...]
    ) -> InterventionTarget:
        """Return new target with specified positions."""
        if isinstance(positions, int):
            positions = (positions,)
        elif isinstance(positions, list):
            positions = tuple(positions)
        return InterventionTarget(
            positions=positions,
            layers=self.layers,
            component=self.component,
        )

    # ── String Representation ───────────────────────────────────────────────

    def __str__(self) -> str:
        parts = []
        if self.positions is not None:
            if len(self.positions) <= 3:
                parts.append(f"pos={list(self.positions)}")
            else:
                parts.append(f"pos=[{len(self.positions)} positions]")
        else:
            parts.append("pos=all")

        if self.layers is not None:
            if len(self.layers) <= 3:
                parts.append(f"L{list(self.layers)}")
            else:
                parts.append(f"L[{len(self.layers)} layers]")
        else:
            parts.append("L=all")

        parts.append(self.component)
        return f"Target({', '.join(parts)})"

    def __hash__(self) -> int:
        return hash((self.positions, self.layers, self.component))

    # ── Merge and Decompose ─────────────────────────────────────────────────

    @classmethod
    def merge(cls, targets: list[InterventionTarget]) -> InterventionTarget:
        """Merge multiple targets into one (union of positions and layers)."""
        if not targets:
            return cls.all()

        positions = set()
        layers = set()
        component = targets[0].component

        for t in targets:
            if t.positions is None:
                positions = None
                break
            positions.update(t.positions)

        for t in targets:
            if t.layers is None:
                layers = None
                break
            layers.update(t.layers)

        return cls(
            positions=tuple(sorted(positions)) if positions else None,
            layers=tuple(sorted(layers)) if layers else None,
            component=component,
        )

    def decompose(self) -> list[InterventionTarget]:
        """Decompose into per-layer targets."""
        if self.layers is None:
            return [self]
        return [
            InterventionTarget(
                positions=self.positions,
                layers=(layer,),
                component=self.component,
            )
            for layer in self.layers
        ]
