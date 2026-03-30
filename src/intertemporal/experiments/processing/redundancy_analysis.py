"""Redundancy analysis: compute redundancy gaps across layers.

This module contains the algorithmic logic for analyzing redundancy
(the gap between noising disruption and denoising recovery).
"""

from __future__ import annotations

from ....activation_patching.coarse import SweepStepResults
from .processing_results import RedundancyAnalysis, RedundancyGap


def compute_redundancy_gaps(
    layer_data: dict[str, SweepStepResults | None],
) -> RedundancyAnalysis | None:
    """Compute redundancy gaps for all layer-component pairs.

    The redundancy gap is computed as disruption - recovery:
    - Positive gap: Component is necessary (noising hurts more than denoising helps)
    - Negative gap: Component is sufficient (denoising helps more than noising hurts)

    Args:
        layer_data: Layer sweep results keyed by component

    Returns:
        RedundancyAnalysis with all gaps and sorted layer indices, or None if no data
    """
    all_layers: set[int] = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return None

    layers = sorted(all_layers)
    plot_components = ["attn_out", "mlp_out", "resid_post"]

    gaps: list[RedundancyGap] = []
    layer_total_abs_gap: dict[int, float] = {layer: 0.0 for layer in layers}

    for comp in plot_components:
        data = layer_data.get(comp)
        for layer in layers:
            if data and data.get(layer) is not None:
                rec = data[layer].recovery
                dis = data[layer].disruption
                if rec is not None and dis is not None:
                    gap_value = dis - rec
                    gaps.append(
                        RedundancyGap(layer=layer, component=comp, gap=gap_value)
                    )
                    layer_total_abs_gap[layer] += abs(gap_value)

    # Sort layers by total absolute gap
    sorted_layers = sorted(
        layers, key=lambda l: layer_total_abs_gap[l], reverse=True
    )

    return RedundancyAnalysis(
        gaps=gaps,
        layers_sorted_by_magnitude=sorted_layers,
    )
