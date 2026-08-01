"""Resolve the scope of one geometry run from command line options.

The module constants in geometry_utils describe a single run: every layer of a
36-layer Qwen, every component, all 17 semantic positions. Two things break when
another model or another budget shows up. Layer 35 does not exist on a 32-layer
Llama, and the full position set costs about 112 GB of activations per model.

This module turns the options a caller passes into a RunScope, projecting the
default layers onto the model's real depth and validating everything against it.
Nothing is dropped silently: an out-of-range layer or an unknown position is an
error.
"""

from __future__ import annotations

from ..common.model_layers import (
    evenly_spaced_layers,
    scale_layers,
    validate_layers,
)
from ..common.semantic_positions import TURN_POSITIONS
from .geometry_config import DEFAULT_STORAGE_DTYPE, RunScope
from .geometry_utils import COMPONENTS, LAYERS, POSITIONS


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers, e.g. "0,12,31"."""
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError(f"no values in {raw!r}")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got {raw!r}") from exc


def parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated list of names, e.g. "resid_post,attn_out"."""
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError(f"no values in {raw!r}")
    return items


def resolve_layers(
    n_model_layers: int,
    layers: list[int] | None = None,
    n_layers: int | None = None,
) -> list[int]:
    """Choose the layers to extract for a model of depth `n_model_layers`."""
    if layers is not None and n_layers is not None:
        raise ValueError("give either explicit layers or a layer count, not both")

    if layers is not None:
        chosen = sorted(set(layers))
    elif n_layers is not None:
        chosen = evenly_spaced_layers(n_layers, n_model_layers)
    else:
        chosen = scale_layers(LAYERS, n_model_layers)

    return validate_layers(chosen, n_model_layers)


def resolve_positions(
    positions: list[str] | None = None,
    turn_only: bool = False,
) -> list[str]:
    """Choose the semantic positions to extract."""
    if positions is not None and turn_only:
        raise ValueError("give either explicit positions or --turn-only, not both")

    if turn_only:
        return list(TURN_POSITIONS)
    if positions is None:
        return list(POSITIONS)

    unknown = [p for p in positions if p not in POSITIONS]
    if unknown:
        raise ValueError(
            f"unknown positions {unknown} (valid: {POSITIONS}). "
            "Position names come from src/intertemporal/common/semantic_positions.py."
        )
    return positions


def resolve_scope(
    n_model_layers: int,
    layers: list[int] | None = None,
    n_layers: int | None = None,
    components: list[str] | None = None,
    positions: list[str] | None = None,
    turn_only: bool = False,
    dtype: str = DEFAULT_STORAGE_DTYPE,
) -> RunScope:
    """Resolve one run's layers, components, positions and storage dtype.

    With no options this reproduces the module constants, projected onto the
    model's depth. RunScope and TargetSpec reject an invalid dtype or component.
    """
    return RunScope(
        layers=resolve_layers(n_model_layers, layers=layers, n_layers=n_layers),
        components=list(COMPONENTS) if components is None else components,
        positions=resolve_positions(positions=positions, turn_only=turn_only),
        dtype=dtype,
    )
