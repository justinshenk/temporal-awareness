"""Main pipeline for geometric visualization.

Memory-optimized implementation:
- Streaming extraction (per-target files on disk)
- Streaming analysis (one target at a time)
- Memory-efficient plotting with explicit cleanup
"""

import gc
import json
import logging

from .geometry_analysis import (
    compute_continuous_time_probe,
    compute_cross_position_similarity,
    run_streaming_analysis,
)
from .geometry_config import GeometryConfig
from .geometry_data import (
    collect_samples,
    extract_activations,
    load_cached_data,
)
from .geometry_logit_lens import LogitLensResult, run_logit_lens_from_cache
from .geometry_plotting import generate_all_plots, plot_logit_lens

logger = logging.getLogger(__name__)


def run_geometry_pipeline(
    config: GeometryConfig,
    use_cache: bool = True,
    skip_extraction: bool = False,
    skip_viz: bool = False,
    skip_per_target_plots: bool = False,
    run_cross_position_similarity: bool = True,
    run_continuous_time_probe: bool = True,
    run_logit_lens: bool = False,
    logit_lens_runner=None,
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
        skip_viz: Skip all visualization (extraction and analysis only)
        skip_per_target_plots: Skip per-target plots (09_targets folder)
        run_cross_position_similarity: Run cross-position cosine similarity analysis
            (compares PC0 directions at source vs destination positions)
        run_continuous_time_probe: Run continuous time horizon regression
            (predicts time_horizon_months from source position activations)
        run_logit_lens: Run logit lens analysis with LayerNorm correction.
            Requires logit_lens_runner to be provided.
        logit_lens_runner: ModelRunner with TransformerLens backend for logit lens.
            Required if run_logit_lens is True.

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
        dataset = collect_samples(config.output_dir)

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

    # Targets actually analyzed = intersection of requested and available
    analyzed_targets = requested_targets & available_targets

    # Step 2: Run streaming analysis (one target at a time)
    logger.info("\n" + "=" * 60)
    logger.info("Running Streaming Analysis")
    logger.info("=" * 60)
    linear_probe_results, pca_results, embedding_results = run_streaming_analysis(data, config)

    # Force GC before additional analyses
    gc.collect()

    # Step 2b: Run continuous time probe on source positions (optional)
    continuous_time_results = None
    if run_continuous_time_probe:
        logger.info("\n" + "=" * 60)
        logger.info("Running Continuous Time Probe (Source Positions)")
        logger.info("=" * 60)
        continuous_time_results = compute_continuous_time_probe(data, config)

    # Step 2c: Compute cross-position similarity (optional)
    cross_position_results = None
    if run_cross_position_similarity:
        logger.info("\n" + "=" * 60)
        logger.info("Computing Cross-Position Similarity")
        logger.info("=" * 60)
        cross_position_results = compute_cross_position_similarity(pca_results, config)

    # Step 2d: Run logit lens analysis (optional)
    logit_lens_result = None
    if run_logit_lens:
        if logit_lens_runner is None:
            logger.warning("run_logit_lens=True but no runner provided, skipping logit lens")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Running Logit Lens Analysis (with LayerNorm)")
            logger.info("=" * 60)
            logit_lens_result = run_logit_lens_from_cache(
                logit_lens_runner, data, config,
                token_a_str="a", token_b_str="b"
            )
            if logit_lens_result is not None:
                # Save logit lens results
                logit_lens_dir = config.output_dir / "results" / "logit_lens"
                logit_lens_result.save(logit_lens_dir)
                logger.info(f"Saved logit lens results to {logit_lens_dir}")

    # Force GC before plotting
    gc.collect()

    # Step 3: Generate plots (unless skipped)
    if not skip_viz:
        logger.info("\n" + "=" * 60)
        logger.info("Generating Plots")
        logger.info("=" * 60)
        generate_all_plots(
            data, linear_probe_results, pca_results, embedding_results, config,
            cross_position_results=cross_position_results,
            continuous_time_results=continuous_time_results,
            skip_per_target_plots=skip_per_target_plots,
        )

        # Step 3b: Generate logit lens plots (if available)
        if logit_lens_result is not None:
            logger.info("Generating logit lens plots...")
            logit_lens_plot_dir = config.output_dir / "plots" / "12_logit_lens"
            plot_logit_lens(logit_lens_result, logit_lens_plot_dir)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Skipping Visualization (--no-viz)")
        logger.info("=" * 60)

    # Build summary (minimal memory)
    summary = {
        "n_samples": len(data.samples),
        "n_targets": len(analyzed_targets),
        "targets": sorted(analyzed_targets),
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

    if continuous_time_results:
        summary["continuous_time_probe"] = {
            k: {"r2": v.r2_mean, "r2_std": v.r2_std, "corr": v.correlation}
            for k, v in continuous_time_results.items()
        }

    if cross_position_results:
        summary["cross_position_similarity"] = {
            k: {"best_sim": v.best_similarity, "mean_sim": v.mean_similarity}
            for k, v in cross_position_results.items()
        }

    if logit_lens_result is not None:
        summary["logit_lens"] = logit_lens_result.to_dict()

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(data.samples)}")
    logger.info(f"Targets analyzed: {len(analyzed_targets)}")

    # Top R² scores (destination positions - binary choice decoding)
    logger.info("\nTop 10 Linear Probe R² Scores (Destination):")
    sorted_results = sorted(
        linear_probe_results.items(), key=lambda x: x[1].r2_mean, reverse=True
    )[:10]
    for k, v in sorted_results:
        pos_type = "DEST" if "_response" in k or "_dest" in k else "SRC"
        logger.info(f"  [{pos_type}] {k}: R²={v.r2_mean:.3f}")

    # Top R² scores for continuous time probe (source positions)
    if continuous_time_results:
        logger.info("\nTop 10 Continuous Time Probe R² Scores (Source):")
        sorted_continuous = sorted(
            continuous_time_results.items(), key=lambda x: x[1].r2_mean, reverse=True
        )[:10]
        for k, v in sorted_continuous:
            logger.info(f"  [SRC] {k}: R²={v.r2_mean:.3f}")

    # Logit lens summary
    if logit_lens_result is not None:
        logger.info("\nLogit Lens Summary:")
        final_layer = logit_lens_result.layers[-1]
        final_logit_diff_mean = logit_lens_result.logit_diffs[-1].mean()
        final_cosine_sim_mean = logit_lens_result.cosine_sims[-1].mean()
        logger.info(f"  Final layer L{final_layer}:")
        logger.info(f"    Mean logit diff: {final_logit_diff_mean:.3f}")
        logger.info(f"    Mean cosine sim: {final_cosine_sim_mean:.3f}")

    logger.info(f"\nResults saved to {config.output_dir}")

    # Clear results from memory (they're on disk)
    data.clear_cache()
    del linear_probe_results, pca_results, embedding_results
    if continuous_time_results is not None:
        del continuous_time_results
    if cross_position_results is not None:
        del cross_position_results
    if logit_lens_result is not None:
        del logit_lens_result
    gc.collect()

    return {"summary": summary}
