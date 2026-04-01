#!/usr/bin/env python3
"""Generate static plots from geometry analysis. STRICT MODE - crashes if data missing.

Usage:
    # Visualize ALL datasets in out/geo/
    uv run python scripts/intertemporal/visualize_geometry_analysis.py

    # Visualize specific dataset
    uv run python scripts/intertemporal/visualize_geometry_analysis.py investment

    # Quick mode (skip per-target plots)
    uv run python scripts/intertemporal/visualize_geometry_analysis.py --quick
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.device_utils import clear_gpu_memory
from src.intertemporal.common.semantic_positions import PROMPT_POSITIONS, RESPONSE_POSITIONS
from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.geometry import GeometryConfig, TargetSpec
from src.intertemporal.geometry.geometry_analysis import (
    ContinuousTimeProbeResult,
    CrossPositionSimilarityResult,
    LinearProbeResult,
    PCAResult,
)
from src.intertemporal.geometry.geometry_data import load_visualization_data
from src.intertemporal.geometry.geometry_plotting import generate_all_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LAYERS = [0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS


def build_targets() -> list[TargetSpec]:
    """Build all target specs."""
    return [
        TargetSpec(layer=l, component=c, position=p)
        for l in LAYERS for c in COMPONENTS for p in POSITIONS
    ]


def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Load linear probe and PCA results. STRICT - raises on failure."""
    import json
    linear_probe = {}
    pca = {}

    lp_dir = results_dir / "linear_probe"
    pca_dir = results_dir / "pca"

    if not lp_dir.exists():
        raise RuntimeError(f"Linear probe results missing: {lp_dir}")
    if not pca_dir.exists():
        raise RuntimeError(f"PCA results missing: {pca_dir}")

    # Load linear probe (simplified - no predictions needed for plotting)
    for d in lp_dir.iterdir():
        if d.is_dir():
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                # Create result with minimal data for plotting
                predictions = np.array([])  # Empty - not needed for most plots
                if (d / "predictions.npy").exists():
                    predictions = np.load(d / "predictions.npy")
                coefficients = np.array([])  # Empty - not needed for most plots
                if (d / "coefficients.npy").exists():
                    coefficients = np.load(d / "coefficients.npy")
                result = LinearProbeResult(
                    target_key=m.get("target_key", d.name),
                    r2_mean=m.get("r2_mean", 0.0),
                    r2_std=m.get("r2_std", 0.0),
                    correlation=m.get("correlation", 0.0) if m.get("correlation") is not None else 0.0,
                    predictions=predictions,
                    coefficients=coefficients,
                )
                linear_probe[result.target_key] = result

    # Load PCA (simplified)
    for d in pca_dir.iterdir():
        if d.is_dir():
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                components = np.array([[]])  # Minimal
                if (d / "components.npy").exists():
                    components = np.load(d / "components.npy")
                # pc_correlations format: [[pc_idx, corr], ...]
                pc_corrs = m.get("pc_correlations", [[0, 0.0]])
                result = PCAResult(
                    target_key=m.get("target_key", d.name),
                    explained_variance=np.array(m.get("explained_variance", [1.0])),
                    pc_correlations=pc_corrs,
                    components=components,
                    transformed=np.array([]),  # Not loaded - too large
                )
                pca[result.target_key] = result

    if not linear_probe:
        raise RuntimeError("No linear probe results loaded")
    if not pca:
        raise RuntimeError("No PCA results loaded")

    log.info(f"Loaded {len(linear_probe)} linear probes, {len(pca)} PCA results")
    return linear_probe, pca


def load_optional(results_dir: Path) -> tuple[dict | None, dict | None]:
    """Load optional results (cross-position, continuous-time)."""
    cross_pos = None
    cont_time = None

    cross_dir = results_dir / "cross_position_similarity"
    if cross_dir.exists():
        cross_pos = {}
        for d in cross_dir.iterdir():
            if d.is_dir():
                try:
                    r = CrossPositionSimilarityResult.load(d)
                    cross_pos[f"L{r.layer}_{r.component}"] = r
                except Exception:
                    pass
        log.info(f"Loaded {len(cross_pos)} cross-position results")

    cont_dir = results_dir / "continuous_time_probe"
    if cont_dir.exists():
        cont_time = {}
        for d in cont_dir.iterdir():
            if d.is_dir():
                try:
                    r = ContinuousTimeProbeResult.load(d)
                    cont_time[r.target_key] = r
                except Exception:
                    pass
        log.info(f"Loaded {len(cont_time)} continuous-time results")

    return cross_pos, cont_time


def cleanup_empty_dirs(viz_dir: Path) -> int:
    """Remove empty subdirectories. Returns count removed."""
    removed = 0
    for d in viz_dir.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
    return removed


def count_files(viz_dir: Path) -> dict[str, int]:
    """Count files in each viz subdirectory."""
    counts = {}
    for d in sorted(viz_dir.iterdir()):
        if d.is_dir():
            counts[d.name] = sum(1 for _ in d.rglob("*") if _.is_file())
    return counts


def discover_datasets(base_dir: Path) -> list[Path]:
    """Discover all valid dataset directories under base_dir."""
    datasets = []
    if not base_dir.exists():
        return datasets
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and (subdir / "data" / "samples").exists():
            datasets.append(subdir)
    return datasets


def process_dataset(data_dir: Path, quick: bool) -> int:
    """Process a single dataset directory."""
    results_dir = data_dir / "analysis"
    viz_dir = data_dir / "viz"

    # Validate
    if not (data_dir / "data").exists():
        log.error(f"Data missing: {data_dir / 'data'}")
        return 1
    if not results_dir.exists():
        log.error(f"Analysis missing: {results_dir}")
        return 1

    log.info("=" * 50)
    log.info(f"VISUALIZE: {data_dir.name}")
    log.info("=" * 50)

    # Load data
    config = GeometryConfig(
        targets=build_targets(),
        output_dir=data_dir,
        model=DEFAULT_MODEL,
        seed=42,
        n_pca_components=10,
    )

    data = load_visualization_data(config)
    if data is None:
        log.error("Failed to load visualization data")
        return 1

    log.info(f"Loaded {data.n_samples} samples")
    clear_gpu_memory(aggressive=True)

    # Load results
    linear_probe, pca = load_results(results_dir)
    cross_pos, cont_time = load_optional(results_dir)
    clear_gpu_memory(aggressive=True)

    # Generate plots
    log.info("\nGenerating plots...")
    generate_all_plots(
        data=data,
        linear_probe_results=linear_probe,
        pca_results=pca,
        embedding_results={},  # Not used for static plots
        config=config,
        cross_position_results=cross_pos,
        continuous_time_results=cont_time,
        skip_per_target_plots=quick,
    )

    # Move plots/ to viz/ if needed
    plots_dir = data_dir / "plots"
    if plots_dir.exists():
        import shutil
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        plots_dir.rename(viz_dir)

    # Cleanup empty directories
    if viz_dir.exists():
        removed = cleanup_empty_dirs(viz_dir)
        if removed:
            log.info(f"Removed {removed} empty directories")

    # Report
    clear_gpu_memory(aggressive=True)
    log.info("\n" + "=" * 50)
    log.info(f"COMPLETE: {data_dir.name}")
    log.info("=" * 50)

    if viz_dir.exists():
        counts = count_files(viz_dir)
        total = sum(counts.values())
        log.info(f"Generated {total} files in {len(counts)} directories:")
        for name, count in counts.items():
            log.info(f"  {name}: {count}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate static plots for datasets in out/geo/",
        epilog="Examples:\n"
               "  %(prog)s                  # Visualize ALL datasets in out/geo/\n"
               "  %(prog)s investment       # Visualize only out/geo/investment\n"
               "  %(prog)s --quick          # Skip per-target plots\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name to visualize (default: all datasets in out/geo/)",
    )
    p.add_argument(
        "--base-dir",
        default="out/geo",
        help="Base directory containing datasets (default: out/geo)",
    )
    p.add_argument("--quick", action="store_true", help="Skip per-target plots")
    args = p.parse_args()

    base_dir = Path(args.base_dir)

    # Determine which datasets to process
    if args.dataset:
        # Process specific dataset
        data_dir = base_dir / args.dataset
        if not data_dir.exists():
            log.error(f"Dataset not found: {data_dir}")
            return 1
        datasets = [data_dir]
    else:
        # Discover all datasets
        datasets = discover_datasets(base_dir)
        if not datasets:
            log.error(f"No valid datasets found in {base_dir}")
            return 1

    log.info("=" * 60)
    log.info("VISUALIZE GEOMETRY ANALYSIS")
    log.info(f"Processing {len(datasets)} dataset(s):")
    for d in datasets:
        log.info(f"  - {d.name}")
    log.info("=" * 60)

    # Process each dataset
    for data_dir in datasets:
        result = process_dataset(data_dir, args.quick)
        if result != 0:
            return result

    log.info("\n" + "=" * 60)
    log.info("ALL DATASETS COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
