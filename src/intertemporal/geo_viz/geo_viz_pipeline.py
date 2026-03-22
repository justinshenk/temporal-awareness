"""Main pipeline for geometric visualization.

Memory-optimized implementation:
- Streaming extraction (per-target files on disk)
- Streaming analysis (one target at a time)
- Memory-efficient plotting with explicit cleanup
"""

import gc
import json
import logging

from .geo_viz_analysis import run_streaming_analysis
from .geo_viz_config import GeoVizConfig
from .geo_viz_data import (
    collect_samples,
    extract_activations,
    load_cached_data,
    save_data,
)
from .geo_viz_plotting import generate_all_plots

logger = logging.getLogger(__name__)


def run_geo_viz_pipeline(
    config: GeoVizConfig,
    use_cache: bool = True,
    skip_extraction: bool = False,
) -> dict:
    """Run the full geometric visualization pipeline.

    Memory-optimized:
    - Activations stored on disk per-target (never all in RAM)
    - Analysis streams through targets one at a time
    - Results cached to disk for incremental re-runs

    Args:
        config: Pipeline configuration
        use_cache: Whether to use cached data if available
        skip_extraction: Skip sample generation and extraction (requires cache)

    Returns:
        Dictionary with summary (not full results to save memory)
    """
    logger.info("=" * 60)
    logger.info("Geometric Visualization Pipeline (Memory-Optimized)")
    logger.info("=" * 60)

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Step 1: Load or extract data
    data = None
    if use_cache:
        data = load_cached_data(config)
        if data is not None:
            logger.info(f"Loaded cached data: {len(data.samples)} samples, {len(data.get_target_keys())} targets")

    if data is None:
        if skip_extraction:
            raise ValueError("No cached data available and skip_extraction=True")

        # Generate samples
        logger.info("\n" + "=" * 60)
        logger.info("Generating Samples")
        logger.info("=" * 60)
        dataset = collect_samples()

        # Extract activations (streaming to disk)
        logger.info("\n" + "=" * 60)
        logger.info("Extracting Activations (Streaming)")
        logger.info("=" * 60)
        data = extract_activations(dataset, config.targets, config)

        # Clear dataset to free memory
        del dataset
        gc.collect()

    # Verify we have targets
    available_targets = set(data.get_target_keys())
    requested_targets = set(t.key for t in config.targets)
    missing = requested_targets - available_targets
    if missing:
        logger.warning(f"Missing {len(missing)} targets in data")
        logger.warning("Run without --cache to regenerate")

    # Step 2: Run streaming analysis (one target at a time)
    logger.info("\n" + "=" * 60)
    logger.info("Running Streaming Analysis")
    logger.info("=" * 60)
    linear_probe_results, pca_results, embedding_results = run_streaming_analysis(data, config)

    # Force GC before plotting
    gc.collect()

    # Step 3: Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("Generating Plots")
    logger.info("=" * 60)
    generate_all_plots(
        data, linear_probe_results, pca_results, embedding_results, config
    )

    # Build summary (minimal memory)
    summary = {
        "n_samples": len(data.samples),
        "n_targets": len(available_targets),
        "targets": list(available_targets),
        "linear_probe": {
            k: {"r2": v.r2_mean, "r2_std": v.r2_std, "corr": v.correlation}
            for k, v in linear_probe_results.items()
        },
        "pca": {
            k: {
                "top_pc": v.pc_correlations[0][0],
                "top_corr": v.pc_correlations[0][1],
            }
            for k, v in pca_results.items()
        },
    }

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(data.samples)}")
    logger.info(f"Targets: {len(available_targets)}")

    # Top R² scores
    logger.info("\nTop 10 Linear Probe R² Scores:")
    sorted_results = sorted(
        linear_probe_results.items(), key=lambda x: x[1].r2_mean, reverse=True
    )[:10]
    for k, v in sorted_results:
        pos_type = "DEST" if "Presponse" in k or "P14" in k else "SRC"
        logger.info(f"  [{pos_type}] {k}: R²={v.r2_mean:.3f}")

    logger.info(f"\nResults saved to {config.output_dir}")

    # Clear results from memory (they're on disk)
    data.clear_cache()
    del linear_probe_results, pca_results, embedding_results
    gc.collect()

    return {"summary": summary}
