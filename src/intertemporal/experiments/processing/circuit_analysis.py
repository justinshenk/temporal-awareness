"""Circuit analysis: extract circuit hypothesis from activation patching results.

This module contains the algorithmic logic for extracting circuit hypotheses
from coarse activation patching results. The visualization code uses the
pre-computed results from this module.
"""

from __future__ import annotations

from ....activation_patching.coarse import SweepStepResults
from .processing_results import (
    CircuitHypothesis,
    LayerInfo,
    LayerPositionBinding,
    PositionInfo,
)


def extract_circuit_hypothesis(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
) -> CircuitHypothesis:
    """Extract circuit hypothesis from activation patching data.

    Analyzes layer and position patching results to identify:
    - Source positions: Early positions where information is read
    - Destination positions: Late positions (bottlenecks) where results are written
    - Key attention layers: Layers that route information
    - Key MLP layers: Layers that transform information
    - Multi-function layers: Layers appearing in both attention and MLP top lists
    - Layer-position bindings: Inferred relationships between layers and positions

    Args:
        layer_data: Layer sweep results keyed by component
        pos_data: Position sweep results keyed by component

    Returns:
        CircuitHypothesis containing all extracted information
    """
    source_positions = []
    destination_positions = []
    attn_layers = []
    mlp_layers = []
    multi_function_layers = []
    bindings = []

    # Extract position data
    resid_post_pos = pos_data.get("resid_post", {})
    if resid_post_pos:
        positions = sorted(resid_post_pos.keys())
        mid_pos = positions[len(positions) // 2] if positions else 0

        pos_data_list = []
        for p in positions:
            result = resid_post_pos[p]
            denoise = result.recovery or 0
            noise = result.disruption or 0
            pos_data_list.append((p, denoise, noise))

        # Sort by noising score (more reliable for finding bottlenecks)
        pos_data_list.sort(key=lambda x: x[2], reverse=True)

        # Split into source (early) and destination (late) positions
        # FILTER: Skip denoising-only positions (where denoise >> noise)
        for p, denoise, noise in pos_data_list[:20]:
            # Skip denoising-only positions that would be misleading
            if denoise > 0.1 and noise < 0.05:
                continue
            if denoise > noise * 3 and noise < 0.1:
                continue

            redundancy_gap = noise - denoise
            if p < mid_pos:
                # Source position: redundant if gap < 0 (denoise > noise)
                is_redundant = redundancy_gap < -0.1
                source_positions.append(
                    PositionInfo(
                        position=p,
                        recovery=denoise,
                        disruption=noise,
                        is_redundant=is_redundant,
                    )
                )
            else:
                # Destination position: bottleneck if noise > 0.5
                is_bottleneck = noise > 0.5
                destination_positions.append(
                    PositionInfo(
                        position=p,
                        recovery=denoise,
                        disruption=noise,
                        is_bottleneck=is_bottleneck,
                    )
                )

        # Limit to top 3-4 each
        source_positions = source_positions[:4]
        destination_positions = destination_positions[:4]

    # Extract attention layer data
    attn_data = layer_data.get("attn_out", {})
    if attn_data:
        attn_list = []
        for lyr, result in attn_data.items():
            denoise = result.recovery or 0
            noise = result.disruption or 0
            attn_list.append((lyr, denoise, noise))
        # Sort by noising (more reliable)
        attn_list.sort(key=lambda x: x[2], reverse=True)
        attn_layers = [
            LayerInfo(layer=lyr, recovery=denoise, disruption=noise, component="attn_out")
            for lyr, denoise, noise in attn_list[:5]
        ]

    # Extract MLP layer data
    mlp_data = layer_data.get("mlp_out", {})
    if mlp_data:
        mlp_list = []
        for lyr, result in mlp_data.items():
            denoise = result.recovery or 0
            noise = result.disruption or 0
            mlp_list.append((lyr, denoise, noise))
        mlp_list.sort(key=lambda x: x[2], reverse=True)
        mlp_layers = [
            LayerInfo(layer=lyr, recovery=denoise, disruption=noise, component="mlp_out")
            for lyr, denoise, noise in mlp_list[:5]
        ]

    # Find multi-function layers (appear in both attn and mlp top lists)
    attn_top_layers = {info.layer for info in attn_layers}
    mlp_top_layers = {info.layer for info in mlp_layers}
    multi_function_layers = sorted(attn_top_layers & mlp_top_layers)

    # Infer layer-position bindings
    if attn_layers and source_positions:
        attn_layers_sorted = sorted([info.layer for info in attn_layers])
        input_pos_sorted = sorted([info.position for info in source_positions])
        if attn_layers_sorted and input_pos_sorted:
            bindings.append(
                LayerPositionBinding(
                    layers=f"L{min(attn_layers_sorted)}-L{max(attn_layers_sorted)}",
                    positions=f"P{min(input_pos_sorted)}-P{max(input_pos_sorted)}",
                    binding_type="attn_reads_from",
                )
            )

    if mlp_layers and destination_positions:
        mlp_layers_sorted = sorted([info.layer for info in mlp_layers])
        output_pos_sorted = sorted([info.position for info in destination_positions])
        if mlp_layers_sorted and output_pos_sorted:
            bindings.append(
                LayerPositionBinding(
                    layers=f"L{min(mlp_layers_sorted)}-L{max(mlp_layers_sorted)}",
                    positions=f"P{min(output_pos_sorted)}-P{max(output_pos_sorted)}",
                    binding_type="mlp_writes_to",
                )
            )

    # Compute summary statistics
    top3_attn_noising = sum(info.disruption for info in attn_layers[:3])
    top3_mlp_noising = sum(info.disruption for info in mlp_layers[:3])

    return CircuitHypothesis(
        source_positions=source_positions,
        destination_positions=destination_positions,
        attn_layers=attn_layers,
        mlp_layers=mlp_layers,
        multi_function_layers=multi_function_layers,
        layer_position_bindings=bindings,
        top3_attn_noising=top3_attn_noising,
        top3_mlp_noising=top3_mlp_noising,
    )
