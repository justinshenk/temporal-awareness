"""Component analysis: decomposition and importance ranking.

This module contains the algorithmic logic for analyzing component
contributions, cumulative recovery, and position hub detection.
"""

from __future__ import annotations

import numpy as np

from ....activation_patching.coarse import SweepStepResults
from .processing_results import (
    ComponentImportance,
    CumulativeRecovery,
    HubRegion,
    PositionAnalysis,
    RankedComponent,
)


COMPONENTS = ["resid_pre", "resid_post", "attn_out", "mlp_out"]


def compute_cumulative_recovery(
    layer_data: dict[str, SweepStepResults | None],
) -> CumulativeRecovery | None:
    """Compute cumulative recovery across layers.

    Calculates the cumulative sum of attention and MLP recovery values
    across layers, identifying dips and key contributing layers.

    Args:
        layer_data: Layer sweep results keyed by component

    Returns:
        CumulativeRecovery with cumulative sums and analysis, or None if no data
    """
    attn_data = layer_data.get("attn_out")
    mlp_data = layer_data.get("mlp_out")

    if not attn_data or not mlp_data:
        return None

    layers = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not layers:
        return None

    attn_recovery = [attn_data[layer].recovery or 0 for layer in layers]
    mlp_recovery = [mlp_data[layer].recovery or 0 for layer in layers]

    attn_cumsum = list(np.cumsum(attn_recovery))
    mlp_cumsum = list(np.cumsum(mlp_recovery))
    total_cumsum = [a + m for a, m in zip(attn_cumsum, mlp_cumsum)]

    # Detect dips in attention recovery
    dip_layers = []
    if len(attn_recovery) > 3:
        window = 3
        for i in range(window, len(attn_recovery)):
            layer = layers[i]
            attn_val = attn_recovery[i]
            recent_avg = np.mean(attn_recovery[max(0, i - window) : i])

            # Type 1: Explicitly negative (counterproductive)
            if attn_val < -0.02:
                dip_layers.append(layer)
            # Type 2: Relative dip (current << recent average)
            elif recent_avg > 0.05 and attn_val < recent_avg * 0.3:
                dip_layers.append(layer)

    # Find top contributor layers
    total_per_layer = [a + m for a, m in zip(attn_recovery, mlp_recovery)]
    sorted_layers = sorted(
        zip(layers, total_per_layer), key=lambda x: x[1], reverse=True
    )
    key_layers = [layer for layer, _ in sorted_layers[:5]]

    return CumulativeRecovery(
        layers=layers,
        attn_cumsum=attn_cumsum,
        mlp_cumsum=mlp_cumsum,
        total_cumsum=total_cumsum,
        dip_layers=dip_layers,
        key_layers=key_layers,
        attn_recovery=attn_recovery,
        mlp_recovery=mlp_recovery,
    )


def rank_component_importance(
    layer_data: dict[str, SweepStepResults | None],
    top_n: int = 15,
) -> ComponentImportance | None:
    """Rank components by importance (recovery score).

    Creates a ranked list of layer-component pairs sorted by their
    denoising recovery score.

    Args:
        layer_data: Layer sweep results keyed by component
        top_n: Number of top components to include

    Returns:
        ComponentImportance with ranked components, or None if no data
    """
    all_components = []

    for comp in ["attn_out", "mlp_out"]:
        data = layer_data.get(comp)
        if not data:
            continue
        for layer, result in data.items():
            if result.recovery is not None and result.disruption is not None:
                all_components.append(
                    {
                        "label": f"L{layer}_{comp.replace('_out', '')}",
                        "layer": layer,
                        "recovery": result.recovery,
                        "disruption": result.disruption,
                        "comp": comp,
                    }
                )

    if not all_components:
        return None

    # Sort by recovery
    all_components.sort(key=lambda x: x["recovery"], reverse=True)
    top_components = all_components[:top_n]

    # Find layers that appear multiple times
    layers_in_top = [c["layer"] for c in top_components]
    layer_counts: dict[int, int] = {}
    for layer in layers_in_top:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    multi_layer_groups: dict[int, list[str]] = {}
    for c in top_components:
        if layer_counts.get(c["layer"], 0) > 1:
            if c["layer"] not in multi_layer_groups:
                multi_layer_groups[c["layer"]] = []
            multi_layer_groups[c["layer"]].append(c["label"])

    # Convert to RankedComponent objects
    ranked = [
        RankedComponent(
            label=c["label"],
            layer=c["layer"],
            recovery=c["recovery"],
            disruption=c["disruption"],
            component=c["comp"],
        )
        for c in top_components
    ]

    total_recovery = sum(c["recovery"] for c in all_components)

    return ComponentImportance(
        ranked_components=ranked,
        multi_layer_groups=multi_layer_groups,
        total_recovery=total_recovery,
    )


def detect_hub_regions(
    pos_data: dict[str, SweepStepResults | None],
    percentile: float = 85,
) -> PositionAnalysis | None:
    """Detect contiguous regions of high-activity positions.

    Identifies "hub" regions where the average effect across components
    exceeds a threshold based on the given percentile.

    Args:
        pos_data: Position sweep results keyed by component
        percentile: Percentile threshold for hub detection

    Returns:
        PositionAnalysis with detected hub regions, or None if no data
    """
    all_positions: set[int] = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return None

    positions = sorted(all_positions)

    # Collect all values for threshold computation
    all_values = []
    for mode_attr in ["recovery", "disruption"]:
        for comp in COMPONENTS:
            data = pos_data.get(comp)
            if not data:
                continue
            for pos in positions:
                if data.get(pos) is not None:
                    val = getattr(data[pos], mode_attr, None)
                    if val is not None:
                        all_values.append(val)

    if not all_values:
        return None

    hub_threshold = float(np.percentile(all_values, percentile))

    # Compute average effect per position
    avg_effects = []
    for pos in positions:
        effects = []
        for comp in COMPONENTS:
            data = pos_data.get(comp)
            if data and data.get(pos) is not None:
                rec = data[pos].recovery
                dis = data[pos].disruption
                if rec is not None:
                    effects.append(rec)
                if dis is not None:
                    effects.append(dis)
        avg_effects.append(np.mean(effects) if effects else 0)

    # Detect contiguous regions
    hub_regions = []
    in_region = False
    start = None

    for i, (pos, val) in enumerate(zip(positions, avg_effects)):
        if val >= hub_threshold and not in_region:
            in_region = True
            start = pos
        elif val < hub_threshold and in_region:
            in_region = False
            hub_regions.append(HubRegion(start=start, end=positions[i - 1]))

    if in_region and start is not None:
        hub_regions.append(HubRegion(start=start, end=positions[-1]))

    return PositionAnalysis(
        hub_regions=hub_regions,
        hub_threshold=hub_threshold,
    )
