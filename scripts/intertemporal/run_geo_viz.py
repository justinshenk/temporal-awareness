#!/usr/bin/env python3
"""Geometric visualization of temporal awareness in model activations.

This script analyzes how time horizon information is encoded in transformer
activations using linear probes, PCA correlation analysis, and dimensionality
reduction visualizations.

Key findings from initial analysis:
- Time horizon is linearly decodable at dest positions (R² ~ 0.98)
- Source positions show no time signal (R² ~ -0.01)
- The signal emerges strongly in layers 19-24

Usage:
    uv run python scripts/intertemporal/run_geo_viz.py
    uv run python scripts/intertemporal/run_geo_viz.py --cache
    uv run python scripts/intertemporal/run_geo_viz.py --only-viz  # regenerate plots from cached data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.geo_viz import GeoVizConfig, run_geo_viz_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Default configuration with recommended targets based on our analysis
DEFAULT_GEO_VIZ_CFG = {
    "targets": [
        # Best performers for time decoding (dest positions)
        # These showed R² > 0.97 in linear probe analysis
        {"layer": 24, "component": "resid_pre", "position": "dest"},
        {"layer": 21, "component": "resid_post", "position": "dest"},
        {"layer": 21, "component": "attn_out", "position": "dest"},
        {"layer": 19, "component": "mlp_out", "position": "dest"},
        {"layer": 31, "component": "mlp_out", "position": "dest"},
        # Early layer for comparison (should show weaker signal)
        {"layer": 12, "component": "resid_post", "position": "dest"},
        # Source positions for comparison (should show no time signal)
        {"layer": 21, "component": "attn_out", "position": "source"},
        {"layer": 21, "component": "resid_post", "position": "source"},
        {"layer": 19, "component": "mlp_out", "position": "source"},
    ],
    "output_dir": "out/geo_viz",
    "model": DEFAULT_MODEL,
    "seed": 42,
    "n_pca_components": 50,
    "umap_n_neighbors": 30,
    "umap_min_dist": 0.1,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometric visualization of temporal awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--only-viz",
        action="store_true",
        help="Only run visualization (load cached data, skip extraction)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip sample generation and extraction (requires cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GEO_VIZ_CFG["output_dir"],
        help=f"Output directory (default: {DEFAULT_GEO_VIZ_CFG['output_dir']})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEO_VIZ_CFG["model"],
        help=f"Model identifier (default: {DEFAULT_GEO_VIZ_CFG['model']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GEO_VIZ_CFG["seed"],
        help=f"Random seed (default: {DEFAULT_GEO_VIZ_CFG['seed']})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (default: all)",
    )

    return parser.parse_args()


def main():
    """Run the geometric visualization pipeline."""
    args = parse_args()

    # Build config
    cfg_dict = DEFAULT_GEO_VIZ_CFG.copy()
    cfg_dict["output_dir"] = args.output_dir
    cfg_dict["model"] = args.model
    cfg_dict["seed"] = args.seed
    cfg_dict["max_samples"] = args.max_samples

    config = GeoVizConfig.from_dict(cfg_dict)

    logger.info("Configuration:")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Targets: {len(config.targets)}")
    for t in config.targets:
        logger.info(f"    - {t}")

    # Handle --only-viz (implies --cache and --skip-extraction)
    use_cache = args.cache or args.only_viz
    skip_extraction = args.skip_extraction or args.only_viz

    # Run pipeline
    results = run_geo_viz_pipeline(
        config,
        use_cache=use_cache,
        skip_extraction=skip_extraction,
    )

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    logger.info("\nLinear Probe R² (ability to decode time horizon):")
    for key, result in sorted(
        results["linear_probe"].items(),
        key=lambda x: x[1].r2_mean,
        reverse=True,
    ):
        pos = "DEST" if "Pdest" in key else "SRC"
        logger.info(
            f"  [{pos}] {key}: R²={result.r2_mean:.3f} (corr={result.correlation:.3f})"
        )

    logger.info(f"\nResults saved to: {config.output_dir}")
    logger.info("Key outputs:")
    logger.info(f"  - {config.output_dir}/plots/linear_probe_summary.png")
    logger.info(f"  - {config.output_dir}/plots/pca_*.png")
    logger.info(f"  - {config.output_dir}/plots/embeddings_*.png")
    logger.info(f"  - {config.output_dir}/plots/3d_*.html")
    logger.info(f"  - {config.output_dir}/summary.json")


if __name__ == "__main__":
    main()
