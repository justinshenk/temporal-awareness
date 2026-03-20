"""Visualization for attribution patching results."""

from __future__ import annotations

from pathlib import Path

from ...attribution_patching import AttributionSummary, AttributionPatchingResult
from ...attribution_patching.attribution_key import AttributionKey
from ...common import profile
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_patching_heatmap,
)


# Method priority for selecting best result per component
METHOD_PRIORITY = {"eap_ig": 3, "eap": 2, "standard": 1}


@profile
def visualize_att_patching(
    result: AttributionSummary | None,
    output_dir: Path,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
    title_prefix: str = "",
) -> None:
    """Visualize attribution patching results.

    Creates one plot per component: {component}_{mode}.png
    Uses the best available method (eap_ig > eap > standard).

    Args:
        result: AttributionSummary to visualize
        output_dir: Directory to save plots
        position_labels: Optional labels for positions
        section_markers: Optional section markers
        title_prefix: Optional prefix for titles
    """
    if result is None or not result.results:
        print("[viz] No attribution patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dimensions from first result
    first_result = next(iter(result.results.values()))
    layers = first_result.layers
    n_positions = first_result.n_positions
    mode = result.mode

    pos_labels = position_labels[:n_positions] if position_labels else [f"p{i}" for i in range(n_positions)]

    # Find best method for each component
    best_per_component: dict[str, tuple[str, AttributionPatchingResult]] = {}

    for key_str, attr_result in result.results.items():
        if attr_result.scores.size == 0:
            continue

        # Parse key: "method/component[/quadrature]"
        key = AttributionKey.from_str(key_str)
        component = key.component
        method = key.method

        if component not in best_per_component:
            best_per_component[component] = (method, attr_result)
        elif METHOD_PRIORITY.get(method, 0) > METHOD_PRIORITY.get(best_per_component[component][0], 0):
            best_per_component[component] = (method, attr_result)

    # Generate plot per component
    for component, (method, attr_result) in best_per_component.items():
        filename = f"{component}_{mode}.png"
        title = f"{title_prefix}{method} | {component}"

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
            save_path=output_dir / filename,
        )

    print(f"[viz] Attribution patching plots saved to {output_dir}")
