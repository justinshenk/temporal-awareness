"""Hook name utilities for TransformerLens-style models.

Hook names follow pattern: blocks.{layer}.hook_{component}
"""

from __future__ import annotations

from typing import Callable

COMPONENTS = ("resid_pre", "resid_post", "attn_out", "mlp_out")
ATTRIBUTION_COMPONENTS = ("resid_post", "attn_out", "mlp_out")


def hook_name(layer: int, component: str) -> str:
    """Generate hook name: blocks.{layer}.hook_{component}"""
    return f"blocks.{layer}.hook_{component}"


def hook_names_for_layers(layers: list[int] | range, component: str) -> list[str]:
    """Generate hook names for multiple layers."""
    return [hook_name(layer, component) for layer in layers]


def hook_names_all(n_layers: int, components: list[str] | None = None) -> list[str]:
    """Generate all hook names for a model."""
    if components is None:
        components = list(COMPONENTS)
    return [hook_name(layer, comp) for layer in range(n_layers) for comp in components]


def hook_filter_for_component(component: str) -> Callable[[str], bool]:
    """Filter for a specific component."""
    target = f"hook_{component}"
    return lambda name: target in name


def hook_filter_exact(hook: str) -> Callable[[str], bool]:
    """Filter that matches exactly one hook name."""
    return lambda name: name == hook


def hook_filter_for_hooks(hooks: list[str]) -> Callable[[str], bool]:
    """Filter that matches any of the specified hooks."""
    hook_set = set(hooks)
    return lambda name: name in hook_set


def attribution_filter(name: str) -> bool:
    """Filter for hooks used in attribution (resid_post, attn_out, mlp_out)."""
    return "hook_resid_post" in name or "hook_attn_out" in name or "hook_mlp_out" in name


def parse_hook_name(name: str) -> tuple[int, str] | None:
    """Parse hook name to (layer, component) or None."""
    if not name.startswith("blocks.") or ".hook_" not in name:
        return None
    try:
        parts = name.split(".")
        layer = int(parts[1])
        component = name.split(".hook_")[1]
        return (layer, component)
    except (IndexError, ValueError):
        return None


def get_layer_from_hook(name: str) -> int | None:
    """Extract layer index from hook name."""
    parsed = parse_hook_name(name)
    return parsed[0] if parsed else None


def get_component_from_hook(name: str) -> str | None:
    """Extract component from hook name."""
    parsed = parse_hook_name(name)
    return parsed[1] if parsed else None
