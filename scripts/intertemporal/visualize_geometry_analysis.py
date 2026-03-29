#!/usr/bin/env python3
"""Generate static plots from geometry analysis results.

This is Script 3 of the geometry pipeline. It handles:
- Summary dashboard heatmaps
- Linear probe visualizations
- Decision boundary plots
- Trajectory plots
- Component decomposition plots
- Per-target plots (optional, use --quick to skip)

Output structure:
    out/geometry/
        viz/
            01_dashboard/
            02_linear_probe/
            03_decision_boundary/
            04_trajectories/
            05_direction_alignment/
            06_scree/
            07_component_decomp/
            08_component_decomp_3d/
            09_targets/ (skipped with --quick)
            10_cross_position/
            11_continuous_time/

Usage:
    # Generate all plots
    uv run python scripts/intertemporal/visualize_geometry_analysis.py

    # Quick mode (skip per-target plots)
    uv run python scripts/intertemporal/visualize_geometry_analysis.py --quick

    # Custom data directory
    uv run python scripts/intertemporal/visualize_geometry_analysis.py --data-dir out/geo_test
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.common.semantic_positions import (
    PROMPT_POSITIONS,
    RESPONSE_POSITIONS,
)
from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.geometry import GeometryConfig, TargetSpec
from src.intertemporal.geometry.geometry_analysis import (
    ContinuousTimeProbeResult,
    CrossPositionSimilarityResult,
    EmbeddingResult,
    LinearProbeResult,
    PCAResult,
)
from src.common.device_utils import clear_gpu_memory
from src.intertemporal.geometry.geometry_data import load_visualization_data
from src.intertemporal.geometry.geometry_plotting import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (same as other scripts)
# =============================================================================

LAYERS = [
    0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35,
]

COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
ALL_POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS


def build_targets(
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> list[TargetSpec]:
    """Build target specifications for all layer/component/position combinations."""
    return [
        TargetSpec(layer=layer, component=component, position=position)
        for layer in layers
        for component in components
        for position in positions
    ]


DEFAULT_CONFIG = {
    "targets": build_targets(LAYERS, COMPONENTS, ALL_POSITIONS),
    "output_dir": "out/geometry",
    "model": DEFAULT_MODEL,
    "seed": 42,
    "n_pca_components": 10,
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate static plots from geometry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help=f"Data directory (default: {DEFAULT_CONFIG['output_dir']})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip per-target plots (09_targets folder)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="viz",
        help="Output subdirectory for plots (default: viz)",
    )

    return parser.parse_args()


# =============================================================================
# Result Loading
# =============================================================================


def _safe_key(key: str) -> str:
    """Convert target key to safe directory name."""
    return key.replace("/", "_").replace("\\", "_")


def load_analysis_results(
    data_dir: Path,
    config: GeometryConfig,
) -> tuple[
    dict[str, LinearProbeResult],
    dict[str, PCAResult],
    dict[str, EmbeddingResult],
    dict[str, CrossPositionSimilarityResult] | None,
    dict[str, ContinuousTimeProbeResult] | None,
]:
    """Load analysis results from disk.

    Uses analysis/ directory (same as geoapp).
    """
    results_dir = data_dir / "analysis"

    if not results_dir.exists():
        raise FileNotFoundError(f"No analysis results found in {results_dir}")

    logger.info(f"Loading analysis results from {results_dir}...")

    # Load linear probe results
    linear_probe_results = {}
    linear_probe_dir = results_dir / "linear_probe"
    if linear_probe_dir.exists():
        for target_dir in linear_probe_dir.iterdir():
            if target_dir.is_dir():
                try:
                    result = LinearProbeResult.load(target_dir)
                    linear_probe_results[result.target_key] = result
                except Exception as e:
                    logger.warning(f"Failed to load linear probe result from {target_dir}: {e}")

    logger.info(f"Loaded {len(linear_probe_results)} linear probe results")
    clear_gpu_memory(aggressive=True)

    # Load PCA results
    pca_results = {}
    pca_dir = results_dir / "pca"
    if pca_dir.exists():
        for target_dir in pca_dir.iterdir():
            if target_dir.is_dir():
                try:
                    result = PCAResult.load(target_dir)
                    pca_results[result.target_key] = result
                except Exception as e:
                    logger.warning(f"Failed to load PCA result from {target_dir}: {e}")

    logger.info(f"Loaded {len(pca_results)} PCA results")
    clear_gpu_memory(aggressive=True)

    # Load embedding results
    embedding_results = {}
    embedding_dir = results_dir / "embeddings"
    if embedding_dir.exists():
        for target_dir in embedding_dir.iterdir():
            if target_dir.is_dir():
                try:
                    result = EmbeddingResult.load(target_dir)
                    embedding_results[result.target_key] = result
                except Exception as e:
                    logger.warning(f"Failed to load embedding result from {target_dir}: {e}")

    logger.info(f"Loaded {len(embedding_results)} embedding results")
    clear_gpu_memory(aggressive=True)

    # Load cross-position similarity results (optional)
    cross_position_results = None
    cross_pos_dir = results_dir / "cross_position_similarity"
    if cross_pos_dir.exists():
        cross_position_results = {}
        for target_dir in cross_pos_dir.iterdir():
            if target_dir.is_dir():
                try:
                    result = CrossPositionSimilarityResult.load(target_dir)
                    key = f"L{result.layer}_{result.component}"
                    cross_position_results[key] = result
                except Exception:
                    pass
        logger.info(f"Loaded {len(cross_position_results)} cross-position similarity results")

    # Load continuous time probe results (optional)
    continuous_time_results = None
    cont_time_dir = results_dir / "continuous_time_probe"
    if cont_time_dir.exists():
        continuous_time_results = {}
        for target_dir in cont_time_dir.iterdir():
            if target_dir.is_dir():
                try:
                    result = ContinuousTimeProbeResult.load(target_dir)
                    continuous_time_results[result.target_key] = result
                except Exception:
                    pass
        logger.info(f"Loaded {len(continuous_time_results)} continuous time probe results")

    clear_gpu_memory(aggressive=True)
    return (
        linear_probe_results,
        pca_results,
        embedding_results,
        cross_position_results,
        continuous_time_results,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Generate static plots from geometry analysis."""
    args = parse_args()

    data_dir = Path(args.data_dir)

    # Check that data exists
    if not (data_dir / "data").exists():
        logger.error(f"Data directory not found: {data_dir / 'data'}")
        logger.error("Run generate_geometry_samples.py first.")
        return 1

    # Build config
    config = GeometryConfig(
        targets=DEFAULT_CONFIG["targets"],
        output_dir=data_dir,
        model=DEFAULT_CONFIG["model"],
        seed=DEFAULT_CONFIG["seed"],
        n_pca_components=DEFAULT_CONFIG["n_pca_components"],
    )

    # Load data (lightweight mode - only load what's needed for plotting)
    logger.info("=" * 60)
    logger.info("VISUALIZE GEOMETRY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")

    data = load_visualization_data(config)
    if data is None:
        logger.error("Failed to load data. Run generate_geometry_samples.py first.")
        return 1

    logger.info(f"Loaded {data.n_samples} samples (lightweight mode)")

    # Load analysis results
    try:
        (
            linear_probe_results,
            pca_results,
            embedding_results,
            cross_position_results,
            continuous_time_results,
        ) = load_analysis_results(data_dir, config)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run compute_geometry_analysis.py first.")
        return 1

    if not linear_probe_results:
        logger.error("No analysis results found. Run compute_geometry_analysis.py first.")
        return 1

    # Override output directory to use viz/ instead of plots/
    # Temporarily modify config
    original_output_dir = config.output_dir
    config.output_dir = data_dir

    # Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("Generating Plots")
    logger.info("=" * 60)

    # Rename output from plots/ to viz/
    plots_dir = data_dir / "plots"
    viz_dir = data_dir / args.output_subdir

    generate_all_plots(
        data=data,
        linear_probe_results=linear_probe_results,
        pca_results=pca_results,
        embedding_results=embedding_results,
        config=config,
        cross_position_results=cross_position_results,
        continuous_time_results=continuous_time_results,
        skip_per_target_plots=args.quick,
    )

    # If output went to plots/, rename to viz/
    if plots_dir.exists() and args.output_subdir != "plots":
        if viz_dir.exists():
            import shutil
            shutil.rmtree(viz_dir)
        plots_dir.rename(viz_dir)
        logger.info(f"Moved output from plots/ to {args.output_subdir}/")

    # Clear memory aggressively
    clear_gpu_memory(aggressive=True)

    logger.info("\n" + "=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {viz_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
