#!/usr/bin/env python3
"""Compute linear probes for GeoApp. Separate from PCA/embeddings due to slow CV."""

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.device_utils import clear_gpu_memory
from src.intertemporal.geometry.geometry_utils import (
    COMPONENTS,
    LAYERS,
    POSITIONS,
    cache_position_mappings,
    load_horizons,
    load_target,
    parse_key,
    target_keys,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def run_linear_probes(data_dir: Path, keys: list[str], force: bool) -> None:
    """Run linear probes with 5-fold CV. SLOW but necessary for R2 scores."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    lp_dir = data_dir / "analysis" / "linear_probe"
    lp_dir.mkdir(parents=True, exist_ok=True)

    y = load_horizons(data_dir)
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)
    log.info(f"  Cached {len(mapping_cache)} position mappings")

    lp_all = {}
    skipped_count = 0

    for i, key in enumerate(keys):
        lp_cache = lp_dir / key / "metrics.json"

        if not force and lp_cache.exists():
            with open(lp_cache) as f:
                lp_all[key] = json.load(f)
            skipped_count += 1
            if skipped_count % 100 == 0:
                log.info(f"  SKIP {skipped_count} cached targets...")
            continue

        target_start = _time.time()
        log.info(f"  [{i}/{len(keys)}] {key} - LOADING...")

        t0 = _time.time()
        result = load_target(data_dir, key, sample_dirs, mapping_cache)
        load_time = _time.time() - t0
        if result is None:
            log.info(f"  [{i}/{len(keys)}] {key} - NO DATA (load={load_time:.1f}s)")
            continue
        X, valid_indices = result

        log.info(f"  [{i}/{len(keys)}] {key} - LOADED shape={X.shape} ({load_time:.1f}s)")

        y_sub = y[valid_indices]
        valid_mask = ~np.isnan(y_sub)
        X_valid = X[valid_mask]
        y_valid = y_sub[valid_mask]
        log.info(f"  [{i}/{len(keys)}] {key} - valid={len(y_valid)}/{len(X)} samples")

        if len(y_valid) < 10:
            lp = {
                "target_key": key,
                "r2_mean": 0.0,
                "r2_std": 0.0,
                "correlation": 0.0,
                "n_samples": len(y_valid),
                "skipped_reason": "too_few_samples_with_horizon",
            }
            lp_all[key] = lp
            (lp_dir / key).mkdir(exist_ok=True)
            with open(lp_cache, "w") as f:
                json.dump(lp, f)
            log.info(f"  [{i}/{len(keys)}] {key} - SKIP (too few samples)")
            continue

        # Ridge regression with 5-fold CV
        try:
            log.info(f"  [{i}/{len(keys)}] {key} - RIDGE CV starting...")
            t0 = _time.time()
            pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
            scores = cross_val_score(pipe, X_valid, y_valid, cv=5, scoring="r2")
            cv_time = _time.time() - t0
            log.info(f"  [{i}/{len(keys)}] {key} - RIDGE CV done ({cv_time:.1f}s) R2={np.mean(scores):.3f}")

            pipe.fit(X_valid, y_valid)
            corr = spearmanr(y_valid, pipe.predict(X_valid))[0]
            lp = {
                "target_key": key,
                "r2_mean": float(np.mean(scores)),
                "r2_std": float(np.std(scores)),
                "correlation": float(corr) if np.isfinite(corr) else 0.0,
                "n_samples": len(y_valid),
                "n_no_horizon": int((~valid_mask).sum()),
            }
            lp_all[key] = lp
            (lp_dir / key).mkdir(exist_ok=True)
            with open(lp_cache, "w") as f:
                json.dump(lp, f)
            del pipe, scores
        except Exception as e:
            raise RuntimeError(f"Linear probe failed for target {key}: {e}") from e

        total_time = _time.time() - target_start
        log.info(f"  [{i}/{len(keys)}] {key} - COMPLETE ({total_time:.1f}s total)")

        del X
        if i % 10 == 0:
            clear_gpu_memory(aggressive=True)

    # Update summary.json with linear probe results
    summary_file = data_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["linear_probe"] = lp_all
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"  Completed {len(lp_all)} linear probes")


def load_probe_results(data_dir: Path) -> dict:
    """Load all linear probe results from disk."""
    lp_dir = data_dir / "analysis" / "linear_probe"
    results = {}
    if not lp_dir.exists():
        return results
    for d in lp_dir.iterdir():
        if d.is_dir():
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    results[d.name] = json.load(f)
    return results


def visualize_probes(data_dir: Path) -> None:
    """Generate linear probe visualization plots."""
    import matplotlib.pyplot as plt

    results = load_probe_results(data_dir)
    if not results:
        log.warning("No linear probe results to visualize")
        return

    viz_dir = data_dir / "viz" / "linear_probes"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Extract R2 scores by layer and component
    layer_r2: dict[int, list[float]] = {l: [] for l in LAYERS}
    comp_r2: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    pos_r2: dict[str, list[float]] = {p: [] for p in POSITIONS}

    for key, m in results.items():
        parsed = parse_key(key)
        if not parsed:
            continue
        layer, comp, pos, _ = parsed
        r2 = m.get("r2_mean", 0.0)
        if layer in layer_r2:
            layer_r2[layer].append(r2)
        if comp in comp_r2:
            comp_r2[comp].append(r2)
        if pos in pos_r2:
            pos_r2[pos].append(r2)

    # Plot 1: R2 by layer
    fig, ax = plt.subplots(figsize=(10, 5))
    layers_sorted = sorted(layer_r2.keys())
    means = [np.mean(layer_r2[l]) if layer_r2[l] else 0 for l in layers_sorted]
    ax.bar(range(len(layers_sorted)), means)
    ax.set_xticks(range(len(layers_sorted)))
    ax.set_xticklabels([f"L{l}" for l in layers_sorted])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean R² Score")
    ax.set_title("Linear Probe R² by Layer")
    plt.tight_layout()
    plt.savefig(viz_dir / "r2_by_layer.png", dpi=150)
    plt.close()

    # Plot 2: R2 by component
    fig, ax = plt.subplots(figsize=(8, 5))
    comp_means = [np.mean(comp_r2[c]) if comp_r2[c] else 0 for c in COMPONENTS]
    ax.bar(COMPONENTS, comp_means)
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean R² Score")
    ax.set_title("Linear Probe R² by Component")
    plt.tight_layout()
    plt.savefig(viz_dir / "r2_by_component.png", dpi=150)
    plt.close()

    # Plot 3: R2 by position (top 10)
    pos_means = [(p, np.mean(pos_r2[p]) if pos_r2[p] else 0) for p in POSITIONS]
    pos_means.sort(key=lambda x: x[1], reverse=True)
    top_pos = pos_means[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([p[0] for p in top_pos], [p[1] for p in top_pos])
    ax.set_xlabel("Mean R² Score")
    ax.set_ylabel("Position")
    ax.set_title("Top 10 Positions by Linear Probe R²")
    plt.tight_layout()
    plt.savefig(viz_dir / "r2_by_position_top10.png", dpi=150)
    plt.close()

    # Plot 4: Heatmap - Layer x Component
    fig, ax = plt.subplots(figsize=(8, 10))
    heatmap = np.zeros((len(LAYERS), len(COMPONENTS)))
    for i, l in enumerate(LAYERS):
        for j, c in enumerate(COMPONENTS):
            vals = [m.get("r2_mean", 0.0) for k, m in results.items()
                    if parse_key(k) and parse_key(k)[0] == l and parse_key(k)[1] == c]
            heatmap[i, j] = np.mean(vals) if vals else 0
    im = ax.imshow(heatmap, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(COMPONENTS)))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha='right')
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f"L{l}" for l in LAYERS])
    ax.set_title("Linear Probe R² Heatmap")
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    plt.savefig(viz_dir / "r2_heatmap.png", dpi=150)
    plt.close()

    log.info(f"  Generated {len(list(viz_dir.glob('*.png')))} probe visualizations")


def process_dataset(data_dir: Path, force: bool, visualize: bool) -> int:
    """Process a single dataset directory."""
    if not (data_dir / "data").exists():
        log.error(f"No data: {data_dir}")
        return 1

    keys = target_keys()
    total = len(keys)

    lp_dir = data_dir / "analysis" / "linear_probe"
    existing = len(list(lp_dir.glob("*/metrics.json"))) if lp_dir.exists() else 0

    log.info("=" * 50)
    log.info(f"LINEAR PROBES: {data_dir.name}")
    log.info(f"Targets: {total}, Cached: {existing}")
    log.info("=" * 50)

    if force or existing < total:
        run_linear_probes(data_dir, keys, force)
    else:
        log.info("All linear probes already computed")

    if visualize:
        log.info("\nGenerating visualizations...")
        visualize_probes(data_dir)

    return 0


def discover_datasets(base_dir: Path) -> list[Path]:
    """Discover all valid dataset directories under base_dir."""
    datasets = []
    if not base_dir.exists():
        return datasets
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and (subdir / "data" / "samples").exists():
            datasets.append(subdir)
    return datasets


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute linear probes for datasets in out/geo/",
        epilog="Examples:\n"
               "  %(prog)s                  # Process ALL datasets in out/geo/\n"
               "  %(prog)s investment       # Process only out/geo/investment\n"
               "  %(prog)s --force          # Force recompute all\n"
               "  %(prog)s --viz            # Also generate visualizations\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name to process (default: all datasets in out/geo/)",
    )
    p.add_argument(
        "--base-dir",
        default="out/geo",
        help="Base directory containing datasets (default: out/geo)",
    )
    p.add_argument("--force", action="store_true", help="Force recompute all")
    p.add_argument("--viz", action="store_true", help="Generate visualizations after computing")
    args = p.parse_args()

    base_dir = Path(args.base_dir)

    if args.dataset:
        data_dir = base_dir / args.dataset
        if not data_dir.exists():
            log.error(f"Dataset not found: {data_dir}")
            return 1
        datasets = [data_dir]
    else:
        datasets = discover_datasets(base_dir)
        if not datasets:
            log.error(f"No valid datasets found in {base_dir}")
            return 1

    log.info("=" * 60)
    log.info(f"COMPUTE LINEAR PROBES")
    log.info(f"Processing {len(datasets)} dataset(s):")
    for d in datasets:
        log.info(f"  - {d.name}")
    log.info("=" * 60)

    for data_dir in datasets:
        result = process_dataset(data_dir, args.force, args.viz)
        if result != 0:
            return result

    log.info("\n" + "=" * 60)
    log.info("ALL LINEAR PROBES COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
