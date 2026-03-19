"""Visualization for attribution patching results.

Uses "default + vary one axis" approach to limit combinatorial explosion.
For each axis (method, component, grad_at, quadrature), generates plots
while keeping other axes at their default values.

The default for each axis is the FIRST item in that axis's list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...attribution_patching import AttributionSummary, AttributionPatchingResult
from ...common import profile
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_multi_metric_heatmap,
    plot_patching_heatmap,
)
from .viz_config import (
    ATT_DEFAULT_COMPONENT,
    ATT_DEFAULT_GRAD_AT,
    ATT_DEFAULT_METHOD,
    ATT_DEFAULT_QUADRATURE,
    ATT_VIZ_AXES,
    ATT_VIZ_COMPARISON,
)


@dataclass
class ParsedResultKey:
    """Parsed attribution result key."""

    method: Literal["standard", "eap", "eap_ig"]
    component: str  # resid_post, attn_out, mlp_out
    grad_at: str | None  # clean, corrupted, or None if not specified
    quadrature: str | None  # midpoint, etc. or None
    raw_key: str


def parse_result_key(key: str) -> ParsedResultKey:
    """Parse a result key into its components.

    Examples:
        'resid' -> method=standard, component=resid_post, grad_at=None
        'resid_clean' -> method=standard, component=resid_post, grad_at=clean
        'eap_attn_corrupted' -> method=eap, component=attn_out, grad_at=corrupted
        'eap_ig_mlp_clean_midpoint' -> method=eap_ig, component=mlp_out, grad_at=clean, quadrature=midpoint
    """
    parts = key.split("_")
    grad_at = None
    quadrature = None

    # Check for grad_at suffix
    if parts[-1] in ["clean", "corrupted"]:
        grad_at = parts[-1]
        parts = parts[:-1]
    elif len(parts) >= 2 and parts[-2] in ["clean", "corrupted"]:
        # e.g., 'eap_ig_mlp_clean_midpoint'
        quadrature = parts[-1]
        grad_at = parts[-2]
        parts = parts[:-2]

    # Parse method and component
    if parts[0] == "eap" and len(parts) >= 2 and parts[1] == "ig":
        method: Literal["standard", "eap", "eap_ig"] = "eap_ig"
        comp_key = parts[2] if len(parts) > 2 else "attn"
    elif parts[0] == "eap":
        method = "eap"
        comp_key = parts[1] if len(parts) > 1 else "attn"
    else:
        method = "standard"
        comp_key = parts[0]

    # Map short component names to full names
    component_map = {
        "resid": "resid_post",
        "attn": "attn_out",
        "mlp": "mlp_out",
    }
    component = component_map.get(comp_key, comp_key)

    return ParsedResultKey(
        method=method,
        component=component,
        grad_at=grad_at,
        quadrature=quadrature,
        raw_key=key,
    )


def group_results_by_hierarchy(
    results: dict[str, AttributionPatchingResult],
) -> dict[str, dict[str, dict[str, AttributionPatchingResult]]]:
    """Group results by component -> method -> grad_at.

    Returns:
        Nested dict: component -> method -> grad_at -> result
    """
    grouped: dict[str, dict[str, dict[str, AttributionPatchingResult]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for key, result in results.items():
        parsed = parse_result_key(key)
        grad_key = parsed.grad_at or "default"
        if parsed.quadrature:
            grad_key = f"{grad_key}_{parsed.quadrature}"
        grouped[parsed.component][parsed.method][grad_key] = result

    return grouped


def _build_result_key(
    method: str, component: str, grad_at: str, quadrature: str | None
) -> str:
    """Build result key from components."""
    # Map component to short name
    comp_short = {"resid_post": "resid", "attn_out": "attn", "mlp_out": "mlp"}.get(
        component, component
    )

    if method == "standard":
        key = comp_short
    elif method == "eap":
        key = f"eap_{comp_short}"
    else:  # eap_ig
        key = f"eap_ig_{comp_short}"

    key = f"{key}_{grad_at}"
    if quadrature:
        key = f"{key}_{quadrature}"
    return key


def _discover_axes(
    results: dict[str, AttributionPatchingResult],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Discover unique values for each axis, preserving order of first appearance.

    Returns:
        (methods, components, grad_ats, quadratures) - each as ordered list
    """
    methods: list[str] = []
    components: list[str] = []
    grad_ats: list[str] = []
    quadratures: list[str] = []

    for key in results.keys():
        parsed = parse_result_key(key)
        if parsed.method not in methods:
            methods.append(parsed.method)
        if parsed.component not in components:
            components.append(parsed.component)
        if parsed.grad_at and parsed.grad_at not in grad_ats:
            grad_ats.append(parsed.grad_at)
        if parsed.quadrature and parsed.quadrature not in quadratures:
            quadratures.append(parsed.quadrature)

    return methods, components, grad_ats, quadratures


@profile
def visualize_att_patching(
    result: AttributionSummary | None,
    output_dir: Path,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
    title_prefix: str = "",
) -> None:
    """Visualize attribution patching results using 'vary one axis' approach.

    Instead of generating all combinations, uses the first value in each axis
    as the default, then generates plots varying ONE axis at a time.

    Creates:
        output_dir/
          by_method/
            standard.png, eap.png, eap_ig.png  (varying method, others default)
          by_component/
            resid_post.png, attn_out.png, mlp_out.png  (varying component)
          by_grad_at/
            clean.png, corrupted.png  (varying grad_at)
          by_quadrature/
            midpoint.png, gauss-legendre.png, ...  (varying quadrature)
          comparison.png

    Args:
        result: AttributionSummary to visualize
        output_dir: Directory to save plots
        position_labels: Optional labels for positions
        section_markers: Optional section markers for prompt structure
        title_prefix: Optional prefix for plot titles
    """
    if result is None or not result.results:
        print("[viz] No attribution patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common dimensions
    first_result = next(iter(result.results.values()))
    layers = first_result.layers
    n_positions = first_result.n_positions

    if position_labels is None:
        pos_labels = [f"p{i}" for i in range(n_positions)]
    else:
        pos_labels = position_labels[:n_positions]

    # Discover axes
    methods, components, grad_ats, quadratures = _discover_axes(result.results)

    # Use configured defaults, fallback to first available if not present
    default_method = ATT_DEFAULT_METHOD if ATT_DEFAULT_METHOD in methods else (methods[0] if methods else "standard")
    default_component = ATT_DEFAULT_COMPONENT if ATT_DEFAULT_COMPONENT in components else (components[0] if components else "resid_post")
    default_grad_at = ATT_DEFAULT_GRAD_AT if ATT_DEFAULT_GRAD_AT in grad_ats else (grad_ats[0] if grad_ats else "clean")
    default_quadrature = ATT_DEFAULT_QUADRATURE if ATT_DEFAULT_QUADRATURE in quadratures else (quadratures[0] if quadratures else None)

    def get_result(method: str, component: str, grad_at: str, quadrature: str | None):
        key = _build_result_key(method, component, grad_at, quadrature)
        return result.results.get(key)

    def plot_single(
        attr_result: AttributionPatchingResult,
        save_path: Path,
        title: str,
    ):
        if attr_result.scores.size == 0:
            return
        config = PatchingHeatmapConfig(
            title=title,
            subtitle=f"{attr_result.n_layers} layers, {n_positions} positions",
            cbar_label="Attribution Score",
            cmap="RdBu_r",
        )
        plot_patching_heatmap(
            attr_result.scores,
            layers,
            pos_labels,
            config=config,
            section_markers=section_markers,
            save_path=save_path,
        )

    # 1. By method (varying method, others default)
    if "method" in ATT_VIZ_AXES and len(methods) > 1:
        by_method_dir = output_dir / "by_method"
        by_method_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            attr_result = get_result(method, default_component, default_grad_at, default_quadrature)
            if attr_result:
                title = f"{title_prefix}{method} | {default_component}"
                plot_single(attr_result, by_method_dir / f"{method}.png", title)

    # 2. By component (varying component, others default)
    if "component" in ATT_VIZ_AXES and len(components) > 1:
        by_comp_dir = output_dir / "by_component"
        by_comp_dir.mkdir(parents=True, exist_ok=True)
        for component in components:
            attr_result = get_result(default_method, component, default_grad_at, default_quadrature)
            if attr_result:
                title = f"{title_prefix}{default_method} | {component}"
                plot_single(attr_result, by_comp_dir / f"{component}.png", title)

    # 3. By grad_at (varying grad_at, others default)
    if "grad_at" in ATT_VIZ_AXES and len(grad_ats) > 1:
        by_grad_dir = output_dir / "by_grad_at"
        by_grad_dir.mkdir(parents=True, exist_ok=True)
        for grad_at in grad_ats:
            attr_result = get_result(default_method, default_component, grad_at, default_quadrature)
            if attr_result:
                title = f"{title_prefix}{default_method} | grad@{grad_at}"
                plot_single(attr_result, by_grad_dir / f"{grad_at}.png", title)

    # 4. By quadrature (varying quadrature, others default)
    if "quadrature" in ATT_VIZ_AXES and len(quadratures) > 1:
        by_quad_dir = output_dir / "by_quadrature"
        by_quad_dir.mkdir(parents=True, exist_ok=True)
        for quadrature in quadratures:
            attr_result = get_result(default_method, default_component, default_grad_at, quadrature)
            if attr_result:
                title = f"{title_prefix}{default_method} | {quadrature}"
                plot_single(attr_result, by_quad_dir / f"{quadrature}.png", title)

    # 5. Multi-method comparison (limited to default component/grad_at/quadrature)
    if ATT_VIZ_COMPARISON and len(methods) > 1:
        matrices = {}
        for method in methods:
            attr_result = get_result(method, default_component, default_grad_at, default_quadrature)
            if attr_result and attr_result.scores.size > 0:
                matrices[method] = attr_result.scores

        if matrices:
            plot_multi_metric_heatmap(
                matrices,
                layers,
                pos_labels,
                section_markers=section_markers,
                save_path=output_dir / "comparison.png",
            )

    print(f"[viz] Attribution patching plots saved to {output_dir}")
