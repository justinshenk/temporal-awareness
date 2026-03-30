"""Main pipeline for geometric visualization."""

import json
import logging

from .geo_viz_analysis import (
    compute_embeddings,
    linear_probe_analysis,
    pca_correlation_analysis,
)
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

    Args:
        config: Pipeline configuration
        use_cache: Whether to use cached data if available
        skip_extraction: Skip sample generation and extraction (requires cache)

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 60)
    logger.info("Geometric Visualization Pipeline")
    logger.info("=" * 60)

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Step 1: Load or collect data
    data = None
    if use_cache:
        data = load_cached_data(config)
        if data is not None:
            logger.info(f"Loaded cached data: {len(data.samples)} samples")

    if data is None:
        if skip_extraction:
            raise ValueError("No cached data available and skip_extraction=True")

        # Generate samples
        dataset = collect_samples()

        # Extract activations
        data = extract_activations(dataset, config.targets, config)

        # Cache data
        save_data(data, config)

    # Verify we have all requested targets
    missing = set(t.key for t in config.targets) - set(data.activations.keys())
    if missing:
        logger.warning(f"Missing targets in data: {missing}")
        logger.warning("Run without --cache to regenerate")

    # Step 2: Linear probe analysis
    logger.info("\n" + "=" * 60)
    logger.info("Linear Probe Analysis")
    logger.info("=" * 60)
    linear_probe_results = linear_probe_analysis(data, config)

    # Step 3: PCA correlation analysis
    logger.info("\n" + "=" * 60)
    logger.info("PCA Correlation Analysis")
    logger.info("=" * 60)
    pca_results = pca_correlation_analysis(data, config)

    # Step 4: Compute embeddings
    logger.info("\n" + "=" * 60)
    logger.info("Computing Embeddings")
    logger.info("=" * 60)
    embedding_results = compute_embeddings(data, config, pca_results)

    # Step 5: Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("Generating Plots")
    logger.info("=" * 60)
    generate_all_plots(
        data, linear_probe_results, pca_results, embedding_results, config
    )

    # Save summary
    summary = {
        "n_samples": len(data.samples),
        "targets": list(data.activations.keys()),
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
    logger.info(f"Targets: {len(data.activations)}")
    logger.info("\nLinear Probe R² Scores:")
    for k, v in sorted(
        linear_probe_results.items(), key=lambda x: x[1].r2_mean, reverse=True
    ):
        pos_type = "DEST" if "Pdest" in k else "SRC"
        logger.info(f"  [{pos_type}] {k}: {v.r2_mean:.3f}")

    logger.info(f"\nResults saved to {config.output_dir}")

    return {
        "data": data,
        "linear_probe": linear_probe_results,
        "pca": pca_results,
        "embeddings": embedding_results,
        "summary": summary,
    }
