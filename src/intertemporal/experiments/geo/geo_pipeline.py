"""Geometry pipeline analysis runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ....common.logging import log
from ...geometry import (
    GeometryConfig,
    TargetSpec,
    analyze_geometry_data,
    generate_geo_samples,
)
from .geo_pipeline_config import GeometryPipelineConfig

if TYPE_CHECKING:
    from ..experiment_context import ExperimentContext


def run_geometry_pipeline_analysis(
    ctx: ExperimentContext,
    config: GeometryPipelineConfig,
    use_cache: bool = False,
) -> dict:
    """Run full geometry pipeline analysis.

    Performs:
    - Linear probes for time horizon classification
    - Cross-position similarity analysis
    - Continuous time probes
    - PCA visualization

    Args:
        ctx: Experiment context with model runner and data
        config: Geometry pipeline configuration
        use_cache: Whether to use cached results

    Returns:
        Results dict with analysis outputs
    """
    log("[geo] Running geometry pipeline analysis...")

    # Build target specs from config
    targets = [
        TargetSpec(layer=layer, component=component, position=position)
        for layer in config.layers
        for component in config.components
        for position in config.positions
    ]

    # Build geometry config
    geometry_output_dir = ctx.output_dir / "geometry"
    geo_config = GeometryConfig(
        targets=targets,
        output_dir=geometry_output_dir,
        model=ctx.pref_data.model if ctx.pref_data else "meta-llama/Llama-3.1-8B-Instruct",
        seed=config.seed,
        n_pca_components=config.n_pca_components,
    )

    log(f"[geo] Geometry pipeline: {len(targets)} targets, output: {geometry_output_dir}")

    # Run Phase 1: Generate samples
    data = generate_geo_samples(geo_config, use_cache=use_cache)

    # Run Phase 2: Analysis
    results = analyze_geometry_data(
        data,
        geo_config,
        skip_viz=config.skip_viz,
        skip_per_target_plots=config.skip_per_target_plots,
        run_cross_position_similarity=config.run_cross_position_similarity,
        run_continuous_time_probe=config.run_continuous_time_probe,
    )

    log(f"[geo] Geometry pipeline complete: {results['summary']['n_samples']} samples analyzed")

    return results
