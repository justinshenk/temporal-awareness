#!/usr/bin/env python3
"""Compute geometry analysis including embeddings for GeoApp.

This is Script 2 of the geometry pipeline. It handles:
- Linear probe analysis (time horizon decoding)
- PCA analysis
- Cross-position similarity analysis
- Continuous time probe analysis
- 3D embeddings for GeoApp (PCA, UMAP, t-SNE - all methods)
- 1D trajectory caches for instant plot loading
- Color data pre-computation for each sample

After running this script, the GeoApp will load ALL visualizations instantly
with NO runtime computation.

Output structure:
    out/geometry/
        analysis/
            linear_probe/
                L{layer}_{component}_{position}/
                    metrics.json
                    predictions.npy
                    coefficients.npy
            pca/
                L{layer}_{component}_{position}/
                    metrics.json
                    explained_variance.npy
                    components.npy
                    transformed.npy
            embeddings/
                pca/L{layer}_{component}_{position}.npy
                umap/L{layer}_{component}_{position}.npy
                tsne/L{layer}_{component}_{position}.npy
            trajectories/
                layers_{component}_{position}.npz
                positions_L{layer}_{component}.npz
            cross_position_similarity/
            continuous_time_probe/
        data/
            samples/
                sample_*/
                    choice.json  (includes color data)
        summary.json

Usage:
    # Compute all analysis (no options needed)
    uv run python scripts/intertemporal/compute_geometry_analysis.py

    # Custom data directory
    uv run python scripts/intertemporal/compute_geometry_analysis.py --data-dir out/geo_test
"""

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.time_value import TimeValue
from src.intertemporal.common.semantic_positions import (
    PROMPT_POSITIONS,
    RESPONSE_POSITIONS,
)
from src.intertemporal.data.default_configs import DEFAULT_MODEL

# These are the ONLY positions that have activation data extracted
# (defined in semantic_positions.py)
VALID_POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS
from src.intertemporal.geometry import GeometryConfig, TargetSpec
from src.intertemporal.geometry.geometry_analysis import (
    compute_continuous_time_probe,
    compute_cross_position_similarity,
    run_streaming_analysis,
)
from src.intertemporal.geometry.geometry_data import ActivationData, load_cached_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_all_positions(data_dir: Path) -> list[str]:
    """Get ALL positions from the data, not just a hardcoded subset.

    Reads the first sample's position_mapping.json to discover all named positions.
    """
    samples_dir = data_dir / "data" / "samples"

    # Find first sample directory (filter out hidden files like .DS_Store)
    sample_dirs = sorted(
        d for d in samples_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ) if samples_dir.exists() else []
    if not sample_dirs:
        logger.error(f"No sample directories found in {samples_dir}")
        return []

    # Read position_mapping.json from first sample
    position_file = sample_dirs[0] / "position_mapping.json"
    if not position_file.exists():
        logger.error(f"No position_mapping.json found in {sample_dirs[0]}")
        return []

    with open(position_file) as f:
        mapping = json.load(f)

    positions = sorted(mapping.get("named_positions", {}).keys())
    logger.info(f"Found {len(positions)} positions in data: {positions}")
    return positions


# =============================================================================
# Configuration
# =============================================================================

LAYERS = [
    0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35,
]

COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

# NOTE: Positions are loaded dynamically from the data (see get_all_positions())
# This ensures we compute embeddings for ALL positions, not just a subset.

# Embedding methods
# PCA is fast and essential
# UMAP is slower but useful
# t-SNE is VERY slow (~5-10 min per target) - skip for now
EMBEDDING_METHODS = ["pca"]


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
    # NOTE: targets are built dynamically in main() after loading all positions from data
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
        description="Compute ALL geometry analysis including ALL GeoApp embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help=f"Data directory (default: {DEFAULT_CONFIG['output_dir']})",
    )

    return parser.parse_args()


# =============================================================================
# Color Data Computation
# =============================================================================


def compute_color_data_for_sample(
    prompt_sample_path: Path,
    preference_sample_path: Path,
) -> dict:
    """Compute color data fields for a single sample.

    Returns dict with:
        - log_time_horizon: log10 of time horizon in months (None if no horizon)
        - option_time_delta: long_term_time - short_term_time (in years)
        - option_reward_delta: long_term_reward - short_term_reward
        - option_confidence_delta: choice_prob - alternative_prob
        - matches_largest_reward: True if chosen option has largest reward
        - matches_rational: True if choice matches rational choice given horizon
        - matches_associated: True if choice matches option closest to horizon
    """
    with open(prompt_sample_path) as f:
        prompt_data = json.load(f)

    with open(preference_sample_path) as f:
        pref_data = json.load(f)

    # Extract time horizon
    time_horizon_data = prompt_data["prompt"].get("time_horizon")
    if time_horizon_data is None:
        log_time_horizon = None
        time_horizon_months = None
    else:
        tv = TimeValue.from_dict(time_horizon_data)
        time_horizon_months = tv.to_months()
        log_time_horizon = math.log10(max(time_horizon_months, 1e-10))

    # Extract option times (in years)
    short_term_time = pref_data.get("short_term_time")  # Already in years
    long_term_time = pref_data.get("long_term_time")  # Already in years

    # Option time delta
    if short_term_time is not None and long_term_time is not None:
        option_time_delta = long_term_time - short_term_time
    else:
        option_time_delta = None

    # Extract rewards
    short_term_reward = pref_data.get("short_term_reward")
    long_term_reward = pref_data.get("long_term_reward")

    # Option reward delta
    if short_term_reward is not None and long_term_reward is not None:
        option_reward_delta = long_term_reward - short_term_reward
    else:
        option_reward_delta = None

    # Confidence delta
    choice_prob = pref_data.get("choice_prob", 0.5)
    alternative_prob = pref_data.get("alternative_prob", 0.5)
    option_confidence_delta = choice_prob - alternative_prob

    # Matches largest reward
    chose_long_term = pref_data.get("chose_long_term", False)
    if short_term_reward is not None and long_term_reward is not None:
        if long_term_reward > short_term_reward:
            matches_largest_reward = chose_long_term
        elif short_term_reward > long_term_reward:
            matches_largest_reward = not chose_long_term
        else:
            # Equal rewards - any choice matches
            matches_largest_reward = True
    else:
        matches_largest_reward = None

    # Matches rational (based on time horizon vs long term time)
    if time_horizon_months is not None and long_term_time is not None:
        time_horizon_years = time_horizon_months / 12.0
        if time_horizon_years < long_term_time:
            # Short term is rational (won't be around for long term)
            matches_rational = not chose_long_term
        elif time_horizon_years > long_term_time:
            # Long term is rational
            matches_rational = chose_long_term
        else:
            # Ambiguous
            matches_rational = None
    else:
        matches_rational = None

    # Matches associated (option closest to time horizon)
    if time_horizon_months is not None and short_term_time is not None and long_term_time is not None:
        time_horizon_years = time_horizon_months / 12.0
        short_dist = abs(time_horizon_years - short_term_time)
        long_dist = abs(time_horizon_years - long_term_time)

        if short_dist < long_dist:
            matches_associated = not chose_long_term
        elif long_dist < short_dist:
            matches_associated = chose_long_term
        else:
            # Equidistant
            matches_associated = None
    else:
        matches_associated = None

    return {
        "log_time_horizon": log_time_horizon,
        "option_time_delta": option_time_delta,
        "option_reward_delta": option_reward_delta,
        "option_confidence_delta": option_confidence_delta,
        "matches_largest_reward": matches_largest_reward,
        "matches_rational": matches_rational,
        "matches_associated": matches_associated,
    }


def update_choice_json_with_color_data(data_dir: Path, n_samples: int) -> int:
    """Update all choice.json files with color data.

    Returns number of samples updated.
    """
    samples_dir = data_dir / "samples"
    updated = 0

    logger.info(f"Updating choice.json files with color data for {n_samples} samples...")

    for sample_idx in range(n_samples):
        sample_dir = samples_dir / f"sample_{sample_idx}"
        if not sample_dir.exists():
            continue

        choice_path = sample_dir / "choice.json"
        prompt_sample_path = sample_dir / "prompt_sample.json"
        preference_sample_path = sample_dir / "preference_sample.json"

        if not all(p.exists() for p in [choice_path, prompt_sample_path, preference_sample_path]):
            continue

        # Load existing choice.json
        with open(choice_path) as f:
            choice_data = json.load(f)

        # Skip if already has color data
        if "log_time_horizon" in choice_data:
            continue

        # Compute color data
        color_data = compute_color_data_for_sample(prompt_sample_path, preference_sample_path)

        # Merge into choice.json
        choice_data.update(color_data)

        # Write back
        with open(choice_path, "w") as f:
            json.dump(choice_data, f, indent=4)

        updated += 1

        if updated % 500 == 0:
            logger.info(f"  Updated {updated} samples...")

    logger.info(f"Color data update complete: {updated} samples updated")
    return updated


# =============================================================================
# Embedding Computation
# =============================================================================


def compute_embeddings(
    data: ActivationData,
    output_dir: Path,
    methods: list[str],
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> None:
    """Compute and save 3D embeddings for GeoApp.

    Saves to analysis/embeddings/{method}/L{layer}_{component}_{position}.npy
    """
    import traceback

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    embeddings_dir = output_dir / "analysis" / "embeddings"

    # Create directories for each method
    for method in methods:
        (embeddings_dir / method).mkdir(parents=True, exist_ok=True)

    # Count work - total targets (not methods * targets)
    total_targets = len(layers) * len(components) * len(positions)
    current_target = 0
    computed = 0
    cached = 0

    logger.info(f"Computing embeddings for {total_targets} targets ({methods})...")
    print(f"[VERBOSE] Starting compute_embeddings: {total_targets} targets, methods={methods}", flush=True)

    for layer in layers:
        for component in components:
            for position in positions:
                key = f"L{layer}_{component}_{position}"
                current_target += 1

                print(f"Processing {key}...", flush=True)

                # Check what needs to be computed
                methods_to_compute = []
                for method in methods:
                    cache_path = embeddings_dir / method / f"{key}.npy"
                    if cache_path.exists():
                        cached += 1
                    else:
                        methods_to_compute.append(method)

                if not methods_to_compute:
                    print(f"  [CACHED] All methods already computed for {key}", flush=True)
                    # Print progress every 10 targets
                    if current_target % 10 == 0:
                        print(f"Progress: {current_target}/{total_targets}", flush=True)
                    continue

                print(f"  Methods to compute: {methods_to_compute}", flush=True)

                # Load activations
                try:
                    print(f"  Loading activations for {key}...", flush=True)
                    X = data.load_target(key)
                    print(f"  Loaded activations: shape={X.shape}", flush=True)
                except (ValueError, FileNotFoundError) as e:
                    print(f"  [ERROR] Failed to load activations for {key}: {e}", flush=True)
                    # Print progress every 10 targets
                    if current_target % 10 == 0:
                        print(f"Progress: {current_target}/{total_targets}", flush=True)
                    continue
                except Exception as e:
                    print(f"  [ERROR] Unexpected error loading activations for {key}:", flush=True)
                    traceback.print_exc()
                    # Print progress every 10 targets
                    if current_target % 10 == 0:
                        print(f"Progress: {current_target}/{total_targets}", flush=True)
                    continue

                if X is None or X.shape[0] < 4:
                    print(f"  [SKIP] Insufficient samples for {key} (shape={X.shape if X is not None else None})", flush=True)
                    data.unload_target(key)
                    # Print progress every 10 targets
                    if current_target % 10 == 0:
                        print(f"Progress: {current_target}/{total_targets}", flush=True)
                    continue

                n_samples = X.shape[0]

                for method in methods_to_compute:
                    cache_path = embeddings_dir / method / f"{key}.npy"
                    print(f"  Computing {method}...", flush=True)

                    try:
                        if method == "pca":
                            n_comp = min(3, n_samples - 1, X.shape[1])
                            if n_comp < 1:
                                print(f"    [SKIP] n_comp < 1 for PCA", flush=True)
                                continue
                            model = PCA(n_components=n_comp, random_state=42)
                            embedding = model.fit_transform(X).astype(np.float32)
                        elif method == "umap":
                            n_neighbors = min(15, max(2, n_samples - 1))
                            model = UMAP(
                                n_components=3,
                                n_neighbors=n_neighbors,
                                min_dist=0.1,
                                random_state=42,
                            )
                            embedding = model.fit_transform(X).astype(np.float32)
                        elif method == "tsne":
                            perplexity = min(30.0, max(1.0, (n_samples - 1) / 3))
                            model = TSNE(
                                n_components=3,
                                perplexity=perplexity,
                                random_state=42,
                                max_iter=1000,
                                init="pca",
                            )
                            embedding = model.fit_transform(X).astype(np.float32)
                        else:
                            print(f"    [SKIP] Unknown method: {method}", flush=True)
                            continue

                        # Pad to 3 components if needed
                        if embedding.shape[1] < 3:
                            padded = np.zeros((n_samples, 3), dtype=np.float32)
                            padded[:, :embedding.shape[1]] = embedding
                            embedding = padded

                        np.save(cache_path, embedding)
                        computed += 1
                        print(f"  Saved {method} to {cache_path}", flush=True)

                    except Exception as e:
                        print(f"  [ERROR] Failed to compute {method} for {key}: {e}", flush=True)
                        print(f"  [ERROR] Full traceback:", flush=True)
                        traceback.print_exc()
                        logger.warning(f"Failed to compute {method} for {key}: {e}")

                data.unload_target(key)

                # Print progress every 10 targets
                if current_target % 10 == 0:
                    print(f"Progress: {current_target}/{total_targets}", flush=True)
                    gc.collect()

    print(f"[VERBOSE] Embeddings complete: {computed} computed, {cached} cached", flush=True)
    logger.info(f"Embeddings complete: {computed} computed, {cached} cached")


def compute_trajectories(
    data: ActivationData,
    output_dir: Path,
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> None:
    """Compute and cache trajectory data for 1D plots.

    Trajectories aggregate PC1 values across layers (1DxLayer) or positions (1DxPos).
    """
    trajectories_dir = output_dir / "analysis" / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = output_dir / "analysis" / "embeddings" / "pca"

    # 1DxLayer: PC1 across all layers for each component/position
    logger.info("Computing layer trajectories (1D x Layer)...")
    layer_traj_count = 0
    layer_traj_cached = 0

    for component in components:
        for position in positions:
            cache_file = trajectories_dir / f"layers_{component}_{position}.npz"
            if cache_file.exists():
                layer_traj_cached += 1
                continue

            # Collect PC1 for all layers
            layer_data = {}
            sample_indices = None

            for layer in layers:
                key = f"L{layer}_{component}_{position}"
                pca_file = embeddings_dir / f"{key}.npy"

                if pca_file.exists():
                    try:
                        embedding = np.load(pca_file)
                        if embedding.shape[1] >= 1:
                            layer_data[layer] = embedding[:, 0]
                            if sample_indices is None:
                                # Get valid sample indices from data loader
                                try:
                                    X = data.load_target(key)
                                    sample_indices = list(range(X.shape[0]))
                                    data.unload_target(key)
                                except Exception:
                                    sample_indices = list(range(len(embedding)))
                    except Exception:
                        pass

            if layer_data:
                np.savez_compressed(
                    cache_file,
                    layers=np.array(list(layer_data.keys())),
                    pc1_values=np.array([layer_data[l] for l in sorted(layer_data.keys())]),
                    sample_indices=np.array(sample_indices) if sample_indices else np.array([]),
                )
                layer_traj_count += 1

    logger.info(f"Layer trajectories: {layer_traj_count} computed, {layer_traj_cached} cached")

    # 1DxPos: PC1 across all positions for each layer/component
    logger.info("Computing position trajectories (1D x Position)...")
    pos_traj_count = 0
    pos_traj_cached = 0

    for layer in layers:
        for component in components:
            cache_file = trajectories_dir / f"positions_L{layer}_{component}.npz"
            if cache_file.exists():
                pos_traj_cached += 1
                continue

            # Collect PC1 for all positions
            pos_data = {}

            for position in positions:
                key = f"L{layer}_{component}_{position}"
                pca_file = embeddings_dir / f"{key}.npy"

                if pca_file.exists():
                    try:
                        embedding = np.load(pca_file)
                        if embedding.shape[1] >= 1:
                            pc1 = embedding[:, 0]
                            # Get sample indices
                            try:
                                X = data.load_target(key)
                                indices = list(range(X.shape[0]))
                                data.unload_target(key)
                            except Exception:
                                indices = list(range(len(pc1)))
                            pos_data[position] = {"pc1": pc1, "indices": indices}
                    except Exception:
                        pass

            if pos_data:
                pos_order = [p for p in positions if p in pos_data]
                np.savez_compressed(
                    cache_file,
                    positions=np.array(pos_order),
                    pc1_values=np.array([pos_data[p]["pc1"] for p in pos_order], dtype=object),
                    sample_indices_list=np.array([pos_data[p]["indices"] for p in pos_order], dtype=object),
                )
                pos_traj_count += 1

    logger.info(f"Position trajectories: {pos_traj_count} computed, {pos_traj_cached} cached")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run geometry analysis and embedding computation."""
    args = parse_args()

    data_dir = Path(args.data_dir)

    # Check that data exists
    if not (data_dir / "data").exists():
        logger.error(f"Data directory not found: {data_dir / 'data'}")
        logger.error("Run generate_geometry_samples.py first.")
        return 1

    # Load data
    logger.info("=" * 60)
    logger.info("COMPUTE GEOMETRY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")

    # Use ONLY positions that have activation data extracted
    # These are defined in semantic_positions.py - DO NOT try to use other positions!
    all_positions = VALID_POSITIONS
    logger.info(f"Will compute embeddings for {len(all_positions)} valid positions: {all_positions}")

    # Build config with ALL positions
    all_targets = build_targets(LAYERS, COMPONENTS, all_positions)
    config = GeometryConfig(
        targets=all_targets,
        output_dir=data_dir,
        model=DEFAULT_CONFIG["model"],
        seed=DEFAULT_CONFIG["seed"],
        n_pca_components=DEFAULT_CONFIG["n_pca_components"],
    )

    data = load_cached_data(config)
    if data is None:
        logger.error("Failed to load data. Run generate_geometry_samples.py first.")
        return 1

    logger.info(f"Loaded {len(data.samples)} samples, {len(data.get_target_keys())} targets")

    # Create analysis directory structure
    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Move results to analysis if they exist in old location
    old_results_dir = data_dir / "results"
    if old_results_dir.exists() and not (analysis_dir / "linear_probe").exists():
        logger.info("Migrating results/ to analysis/...")
        for subdir in old_results_dir.iterdir():
            if subdir.is_dir():
                target = analysis_dir / subdir.name
                if not target.exists():
                    import shutil
                    shutil.copytree(subdir, target)

    # Phase 0: Update choice.json with color data
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0: Pre-computing Color Data")
    logger.info("=" * 60)

    update_choice_json_with_color_data(data_dir / "data", data.n_samples)

    # Phase 1: Run streaming analysis (linear probe + PCA)
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Streaming Analysis (Linear Probe + PCA)")
    logger.info("=" * 60)

    # Redirect results to analysis/
    config.output_dir = data_dir  # results will go to data_dir/results, we'll move later

    linear_results, pca_results, embedding_results = run_streaming_analysis(data, config)

    # Also run cross-position similarity
    logger.info("\nComputing cross-position similarity...")
    cross_position_results = compute_cross_position_similarity(pca_results, config)

    # And continuous time probe
    logger.info("\nComputing continuous time probe...")
    continuous_time_results = compute_continuous_time_probe(data, config)

    # Build and save summary
    summary = {
        "n_samples": len(data.samples),
        "n_targets": len(linear_results),
        "targets": sorted(linear_results.keys()),
        "linear_probe": {
            k: {"r2": v.r2_mean, "r2_std": v.r2_std, "corr": v.correlation}
            for k, v in linear_results.items()
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

    with open(data_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Move results to analysis if needed
    if old_results_dir.exists():
        for subdir in old_results_dir.iterdir():
            if subdir.is_dir():
                target = analysis_dir / subdir.name
                if not target.exists():
                    import shutil
                    shutil.move(str(subdir), str(target))
        # Remove empty results dir
        try:
            old_results_dir.rmdir()
        except OSError:
            pass

    gc.collect()

    # Phase 2: Compute embeddings for GeoApp (ALL methods, ALL components)
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Computing 3D Embeddings for GeoApp")
    logger.info("=" * 60)
    logger.info(f"Methods: {EMBEDDING_METHODS}")
    logger.info(f"Components: {COMPONENTS}")

    compute_embeddings(
        data=data,
        output_dir=data_dir,
        methods=EMBEDDING_METHODS,
        layers=LAYERS,
        components=COMPONENTS,
        positions=all_positions,
    )

    gc.collect()

    # Phase 3: Compute trajectories (all components, all positions)
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Computing 1D Trajectory Caches")
    logger.info("=" * 60)

    compute_trajectories(
        data=data,
        output_dir=data_dir,
        layers=LAYERS,
        components=COMPONENTS,
        positions=all_positions,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    # Calculate cache sizes
    embeddings_dir = data_dir / "analysis" / "embeddings"
    if embeddings_dir.exists():
        npy_size = sum(f.stat().st_size for f in embeddings_dir.rglob("*.npy"))
        logger.info(f"Embeddings cache: {npy_size / 1024 / 1024:.1f} MB")

    trajectories_dir = data_dir / "analysis" / "trajectories"
    if trajectories_dir.exists():
        npz_size = sum(f.stat().st_size for f in trajectories_dir.rglob("*.npz"))
        logger.info(f"Trajectories cache: {npz_size / 1024 / 1024:.1f} MB")

    logger.info(f"\nOutput directory: {data_dir / 'analysis'}")
    logger.info("GeoApp will now load all visualizations instantly!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
