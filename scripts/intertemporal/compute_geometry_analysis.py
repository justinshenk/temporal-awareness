#!/usr/bin/env python3
"""Compute geometry analysis for GeoApp. MINIMAL MEMORY - streams everything."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# NaN VALIDATION UTILITIES
# =============================================================================


class NaNDetectedError(Exception):
    """Raised when NaN values are detected in computations."""
    pass


def _print_nan_warning(context: str, details: str) -> None:
    """Print a loud, impossible-to-miss warning about NaN values."""
    warning_msg = f"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! CRITICAL: NaN VALUES DETECTED !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! Context: {context}
!!! {details}
!!!
!!! This indicates INVALID DATA - zero variance, corrupt inputs, or bugs
!!! The pipeline will NOT silently save NaN values
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    print(warning_msg, file=sys.stderr)
    logging.getLogger(__name__).critical(warning_msg)


def validate_array_no_nan(
    arr: np.ndarray,
    context: str,
    target_key: str | None = None,
    crash: bool = True,
) -> bool:
    """Check array for NaN/Inf values. Optionally crash with clear error.

    Args:
        arr: NumPy array to validate
        context: Description of what this array represents (e.g., "PCA explained_variance")
        target_key: Optional target key for more specific error messages
        crash: If True, raise NaNDetectedError on invalid values. If False, just warn.

    Returns:
        True if array is valid (no NaN/Inf), False otherwise

    Raises:
        NaNDetectedError: If crash=True and invalid values are detected
    """
    if arr is None:
        return True

    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))

    if has_nan or has_inf:
        nan_count = np.sum(np.isnan(arr)) if has_nan else 0
        inf_count = np.sum(np.isinf(arr)) if has_inf else 0

        target_info = f"Target: {target_key}" if target_key else "Target: unknown"
        details = f"{target_info} | NaN count: {nan_count}, Inf count: {inf_count}, Shape: {arr.shape}"

        _print_nan_warning(context, details)

        if crash:
            raise NaNDetectedError(
                f"NaN/Inf detected in {context}. {details}. "
                f"Refusing to continue with invalid data."
            )
        return False

    return True


def validate_input_data(X: np.ndarray, target_key: str) -> None:
    """Validate input activation data before any computation.

    Args:
        X: Input activation matrix
        target_key: Target key for error messages

    Raises:
        NaNDetectedError: If input contains NaN/Inf values
    """
    validate_array_no_nan(X, "INPUT ACTIVATIONS", target_key, crash=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.device_utils import clear_gpu_memory
from src.intertemporal.geometry.geometry_utils import (
    COMPONENTS,
    LAYERS,
    METHODS,
    POSITIONS,
    cache_position_mappings,
    load_horizons,
    load_target,
    target_keys,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def get_max_relpos_counts(data_dir: Path) -> dict[str, int]:
    """Scan all samples to find max rel_pos count for each position."""
    samples_dir = data_dir / "data" / "samples"
    if not samples_dir.exists():
        return {}

    counts: dict[str, int] = {}
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")]

    for d in sample_dirs:
        mapping_file = d / "position_mapping.json"
        if not mapping_file.exists():
            continue
        with open(mapping_file) as f:
            mapping = json.load(f)
        if "named_positions" not in mapping:
            raise KeyError(f"named_positions missing from {mapping_file}")
        named_positions = mapping["named_positions"]
        for pos, abs_positions in named_positions.items():
            if isinstance(abs_positions, list):
                counts[pos] = max(counts.get(pos, 0), len(abs_positions))
            elif abs_positions is not None:
                counts[pos] = max(counts.get(pos, 0), 1)

    return counts


def target_keys_with_relpos(relpos_counts: dict[str, int]) -> list[str]:
    """Generate target keys including per-rel_pos variants.

    For each position with N tokens, generates:
    - L{layer}_{component}_{position} (combined/first token)
    - L{layer}_{component}_{position}_r0, _r1, ..., _r{N-1} (per-token)
    """
    keys = []
    for l in LAYERS:
        for c in COMPONENTS:
            for p in POSITIONS:
                # Combined key (always)
                keys.append(f"L{l}_{c}_{p}")
                # Per-rel_pos keys (always generate, even for single-token positions)
                # This ensures format_tail:0 exists alongside format_tail
                if p not in relpos_counts:
                    raise KeyError(f"Position '{p}' not found in relpos_counts. Available: {list(relpos_counts.keys())}")
                max_relpos = relpos_counts[p]
                if max_relpos >= 1:
                    for r in range(max_relpos):
                        keys.append(f"L{l}_{c}_{p}_r{r}")
    return keys


def count_files(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern))) if d.exists() else 0


N_PCA_COMPONENTS = 10
MIN_SAMPLES = 4
RANDOM_SEED = 42


def _compute_pca_for_target(
    X: np.ndarray,
    y_valid: np.ndarray,
    valid_mask: np.ndarray,
    target_key: str = "unknown",
) -> dict:
    """Compute PCA and correlations for a single target."""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    n_components = min(N_PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        return {}

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(X)

    # ==========================================================================
    # CRITICAL: Validate PCA outputs - NO SILENT NaN FAILURES
    # ==========================================================================
    validate_array_no_nan(
        pca.explained_variance_ratio_,
        "PCA explained_variance_ratio_",
        target_key,
        crash=True,
    )
    validate_array_no_nan(
        pca.components_,
        "PCA components_",
        target_key,
        crash=True,
    )
    validate_array_no_nan(
        Xp,
        "PCA transformed data (fit_transform output)",
        target_key,
        crash=True,
    )
    # ==========================================================================

    Xp_valid = Xp[valid_mask]

    pc_correlations = []
    for j in range(n_components):
        corr = spearmanr(y_valid, Xp_valid[:, j])[0]
        pc_correlations.append([j, float(corr) if np.isfinite(corr) else 0.0])

    # Validate correlations (warn but don't crash - NaN correlations are replaced with 0)
    for j, (_, corr_val) in enumerate(pc_correlations):
        if not np.isfinite(corr_val) or corr_val == 0.0:
            # Check if original was NaN
            raw_corr = spearmanr(y_valid, Xp_valid[:, j])[0]
            if np.isnan(raw_corr):
                _print_nan_warning(
                    f"PC{j} correlation",
                    f"Target: {target_key} | Spearman correlation returned NaN (replaced with 0.0)"
                )

    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pc_correlations": pc_correlations,
        "components": pca.components_.astype(np.float32),
    }


def run_pca_analysis(data_dir: Path, keys: list[str], force: bool) -> None:
    """PCA analysis only. Linear probes computed separately via compute_linear_probes.py."""
    pca_dir = data_dir / "analysis" / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    y = load_horizons(data_dir)
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)
    log.info(f"Cached {len(mapping_cache)} position mappings")

    pca_all = {}
    processed = 0

    for i, key in enumerate(keys):
        pca_cache = pca_dir / key / "metrics.json"

        # Skip cached - but VALIDATE loaded data first
        if not force and pca_cache.exists():
            with open(pca_cache) as f:
                cached_data = json.load(f)

            # =======================================================================
            # CRITICAL: Validate cached data - refuse to use NaN-contaminated cache
            # =======================================================================
            if "explained_variance" in cached_data:
                ev_array = np.array(cached_data["explained_variance"])
                if not validate_array_no_nan(ev_array, "CACHED PCA explained_variance", key, crash=False):
                    log.error(
                        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        f"!!! CORRUPTED CACHE: {pca_cache}\n"
                        f"!!! Contains NaN values in explained_variance\n"
                        f"!!! Delete this file and re-run, or use --force\n"
                        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    )
                    raise NaNDetectedError(
                        f"Cached PCA metrics contain NaN for {key}. "
                        f"Delete {pca_cache} and re-run."
                    )

            pca_all[key] = cached_data
            continue

        # Load activations
        result = load_target(data_dir, key, sample_dirs, mapping_cache)
        if result is None:
            continue
        X, valid_indices = result

        # Get valid horizon values
        y_sub = y[valid_indices]
        valid_mask = ~np.isnan(y_sub)
        y_valid = y_sub[valid_mask]

        if len(y_valid) < MIN_SAMPLES:
            continue

        # =======================================================================
        # CRITICAL: Validate input data BEFORE any computation
        # =======================================================================
        validate_input_data(X, key)

        # Compute PCA - skip targets that fail (e.g., zero variance L0 positions)
        try:
            pca_result = _compute_pca_for_target(X, y_valid, valid_mask, target_key=key)
        except NaNDetectedError as e:
            log.warning(f"Skipping {key}: {e}")
            continue
        if not pca_result:
            continue

        # Save results
        metrics = {
            "target_key": key,
            "explained_variance": pca_result["explained_variance"],
            "pc_correlations": pca_result["pc_correlations"],
        }
        pca_all[key] = metrics
        (pca_dir / key).mkdir(exist_ok=True)
        with open(pca_cache, "w") as f:
            json.dump(metrics, f)
        np.save(pca_dir / key / "components.npy", pca_result["components"])

        processed += 1
        if processed % 50 == 0:
            log.info(f"Processed {processed} targets ({i}/{len(keys)})")
            clear_gpu_memory(aggressive=True)

        del X

    # Save summary
    summary = {
        "n_samples": len(y),
        "layers": LAYERS,
        "components": COMPONENTS,
        "positions": POSITIONS,
        "pca": {
            k: {"top_pc": v["pc_correlations"][0][0], "top_corr": v["pc_correlations"][0][1]}
            for k, v in pca_all.items()
            if v.get("pc_correlations") and len(v["pc_correlations"]) > 0
        },
    }
    with open(data_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Completed {processed} PCA analyses")
    clear_gpu_memory(aggressive=True)


def _compute_single_embedding(
    key: str,
    X: np.ndarray,
    methods: list[str],
    emb_dir: Path,
    pca_dir: Path | None = None,
) -> dict[str, str | None]:
    """Compute embeddings for a single target. Returns {method: error_or_None}.

    If pca_dir is provided, also saves PCA metrics (variance explained, components)
    for scree/alignment analysis.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    n = X.shape[0]
    results = {}

    for m in methods:
        try:
            if m == "pca":
                pca_model = PCA(n_components=min(3, n-1, X.shape[1]), random_state=42)
                e = pca_model.fit_transform(X)

                # =============================================================
                # CRITICAL: Validate PCA embedding output
                # =============================================================
                if np.any(np.isnan(pca_model.explained_variance_ratio_)):
                    _print_nan_warning(
                        "PCA EMBEDDING explained_variance",
                        f"Target: {key} | PCA returned NaN explained variance - indicates zero variance input"
                    )
                    raise NaNDetectedError(f"PCA embedding has NaN explained_variance for {key}")

                # Save PCA metrics for scree/alignment analysis
                if pca_dir is not None:
                    pca_metrics_dir = pca_dir / key
                    pca_metrics_dir.mkdir(parents=True, exist_ok=True)
                    metrics = {
                        "explained_variance": pca_model.explained_variance_ratio_.tolist(),
                        "n_samples": n,
                    }
                    with open(pca_metrics_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                    np.save(pca_metrics_dir / "components.npy", pca_model.components_.astype(np.float32))

            elif m == "umap":
                e = UMAP(n_components=3, n_neighbors=min(15, max(2, n-1)), min_dist=0.1, random_state=42, n_jobs=1).fit_transform(X)

                # =============================================================
                # CRITICAL: Validate UMAP output
                # =============================================================
                if np.any(np.isnan(e)):
                    _print_nan_warning(
                        "UMAP EMBEDDING",
                        f"Target: {key} | UMAP returned NaN values - check input data variance"
                    )
                    raise NaNDetectedError(f"UMAP embedding has NaN values for {key}")

            elif m == "tsne":
                # Subsample for large datasets (t-SNE is O(n^2))
                max_tsne_samples = 2000
                if n > max_tsne_samples:
                    np.random.seed(42)
                    idx = np.random.choice(n, max_tsne_samples, replace=False)
                    X_tsne = X[idx]
                    n_tsne = max_tsne_samples
                else:
                    X_tsne = X
                    n_tsne = n
                    idx = None
                # Use random init for robustness
                e_sub = TSNE(n_components=3, perplexity=min(30.0, max(5.0, (n_tsne-1)/3)), random_state=42, max_iter=300, init="random").fit_transform(X_tsne)

                # =============================================================
                # CRITICAL: Validate t-SNE output
                # =============================================================
                if np.any(np.isnan(e_sub)):
                    _print_nan_warning(
                        "t-SNE EMBEDDING",
                        f"Target: {key} | t-SNE returned NaN values - check input data"
                    )
                    raise NaNDetectedError(f"t-SNE embedding has NaN values for {key}")

                # Pad back to full size if subsampled
                if idx is not None:
                    e = np.zeros((n, 3), dtype=np.float32)
                    e[idx] = e_sub
                else:
                    e = e_sub
            else:
                continue

            # =================================================================
            # FINAL VALIDATION: Check for any remaining NaN/Inf values
            # =================================================================
            if not np.all(np.isfinite(e)):
                nan_count = np.sum(np.isnan(e))
                inf_count = np.sum(np.isinf(e))
                _print_nan_warning(
                    f"{m.upper()} FINAL OUTPUT",
                    f"Target: {key} | NaN: {nan_count}, Inf: {inf_count} | REFUSING TO SAVE"
                )
                raise NaNDetectedError(
                    f"{m.upper()} embedding contains invalid values for {key}. "
                    f"NaN: {nan_count}, Inf: {inf_count}"
                )

            # Pad if needed
            if e.shape[1] < 3:
                pad = np.zeros((n, 3), dtype=np.float32)
                pad[:, :e.shape[1]] = e
                e = pad

            np.save(emb_dir / m / f"{key}.npy", e.astype(np.float32))
            results[m] = None  # Success
        except NaNDetectedError:
            # Re-raise NaN errors - these should crash the pipeline
            raise
        except Exception as ex:
            results[m] = f"{key}: {ex}"

    return results


def run_embeddings(data_dir: Path, keys: list[str], methods: list[str], force: bool) -> dict[str, int]:
    """Compute embeddings. Loads each target once, computes all needed methods, deletes immediately.

    Also saves PCA metrics (variance explained, components) for scree/alignment analysis.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    emb_dir = data_dir / "analysis" / "embeddings"
    pca_dir = data_dir / "analysis" / "pca"  # For saving PCA metrics
    for m in methods:
        (emb_dir / m).mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    counts = {m: 0 for m in methods}
    failed = {m: [] for m in methods}

    # Cache position mappings once for all targets (avoids millions of JSON reads)
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)
    log.info(f"  Cached {len(mapping_cache)} position mappings")

    # Determine which keys need computation
    # Also validate cached embeddings - refuse to use NaN-contaminated files
    keys_to_compute = []
    for key in keys:
        need = [m for m in methods if force or not (emb_dir / m / f"{key}.npy").exists()]
        if need:
            keys_to_compute.append((key, need))
        else:
            # =======================================================================
            # CRITICAL: Validate cached embeddings - refuse NaN-contaminated cache
            # =======================================================================
            for m in methods:
                cache_file = emb_dir / m / f"{key}.npy"
                if cache_file.exists():
                    try:
                        cached_emb = np.load(cache_file)
                        if np.any(~np.isfinite(cached_emb)):
                            nan_count = np.sum(np.isnan(cached_emb))
                            inf_count = np.sum(np.isinf(cached_emb))
                            _print_nan_warning(
                                f"CACHED {m.upper()} EMBEDDING",
                                f"Target: {key} | File: {cache_file} | NaN: {nan_count}, Inf: {inf_count}"
                            )
                            raise NaNDetectedError(
                                f"Cached {m.upper()} embedding for {key} contains invalid values. "
                                f"File: {cache_file}. Delete and re-run."
                            )
                    except NaNDetectedError:
                        raise
                    except Exception as ex:
                        log.warning(f"Failed to validate cached {m} embedding for {key}: {ex}")
                counts[m] += 1  # Valid cache

    log.info(f"  {len(keys_to_compute)} targets need computation, {len(keys) - len(keys_to_compute)} cached")

    # Use parallel processing with limited workers to avoid memory issues
    # Reduce to 2 workers for better memory stability
    max_workers = min(2, os.cpu_count() or 1)
    log.info(f"  Using {max_workers} parallel workers")

    # Process in batches for progress logging
    batch_size = 20
    processed = 0

    for batch_start in range(0, len(keys_to_compute), batch_size):
        batch = keys_to_compute[batch_start:batch_start + batch_size]
        log.info(f"  Embeddings {processed}/{len(keys_to_compute)}")

        # Load data for this batch
        batch_data = []
        for key, need in batch:
            result = load_target(data_dir, key, sample_dirs, mapping_cache)
            if result is None or result[0].shape[0] < 4:
                for m in need:
                    failed[m].append(key)
                continue
            X, _valid_indices = result  # Embeddings don't need indices (no y correlation)

            # =======================================================================
            # CRITICAL: Validate input data - CRASH on NaN, don't silently fix
            # =======================================================================
            if np.any(~np.isfinite(X)):
                nan_count = np.sum(np.isnan(X))
                inf_count = np.sum(np.isinf(X))
                _print_nan_warning(
                    "EMBEDDING INPUT DATA",
                    f"Target: {key} | NaN: {nan_count}, Inf: {inf_count} | Input activations are INVALID"
                )
                raise NaNDetectedError(
                    f"Input activations for {key} contain invalid values. "
                    f"NaN: {nan_count}, Inf: {inf_count}. "
                    f"Check data extraction pipeline."
                )

            batch_data.append((key, X, need))

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_embedding, key, X, need, emb_dir, pca_dir): key
                for key, X, need in batch_data
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results = future.result()
                    for m, error in results.items():
                        if error is None:
                            counts[m] += 1
                        else:
                            failed[m].append(error)
                except Exception as ex:
                    for m in methods:
                        failed[m].append(f"{key}: {ex}")

        processed += len(batch)
        clear_gpu_memory(aggressive=True)

    log.info(f"  Embeddings {len(keys_to_compute)}/{len(keys_to_compute)} DONE")

    # Log failures - always show first 5 errors
    for m in methods:
        if failed[m]:
            log.warning(f"  {m.upper()} failed for {len(failed[m])} targets")
            for f in failed[m][:5]:
                log.warning(f"    {f}")

    return counts


def align_pc_signs_continuity(pc_values: list[np.ndarray]) -> list[np.ndarray]:
    """Align PC signs using continuity (unbiased method).

    Works backwards from last layer (most structured) to ensure smooth trajectories.
    Flips signs where correlation with next layer is negative.

    Args:
        pc_values: List of PC projections, one per layer/position. Shape (n_samples,) each.

    Returns:
        Sign-aligned PC values (same structure, signs may be flipped).
    """
    if len(pc_values) <= 1:
        return pc_values

    aligned = [arr.copy() for arr in pc_values]

    # Work backwards from last layer (most structured)
    for i in range(len(aligned) - 2, -1, -1):
        # Correlation with next layer
        corr = np.corrcoef(aligned[i], aligned[i + 1])[0, 1]
        if np.isfinite(corr) and corr < 0:
            aligned[i] *= -1

    return aligned


def align_pc_signs_position_trajectory(pc_values: list[np.ndarray]) -> list[np.ndarray]:
    """Align PC signs for position trajectories with potentially variable lengths.

    Uses a hybrid strategy:
    1. For adjacent positions with same length: correlation-based alignment
    2. For different lengths: use overlapping indices (assumes first N samples are shared)

    Works backwards from the last position (most semantically meaningful for responses).

    Args:
        pc_values: List of PC projections, one per position. Each may have different length.

    Returns:
        Sign-aligned PC values (same structure, signs may be flipped).
    """
    if len(pc_values) <= 1:
        return pc_values

    aligned = [arr.copy() for arr in pc_values]

    # Work backwards from last position
    for i in range(len(aligned) - 2, -1, -1):
        curr = aligned[i]
        next_pos = aligned[i + 1]

        # Find overlap - use minimum length (assumes samples are ordered consistently)
        min_len = min(len(curr), len(next_pos))
        if min_len < 2:
            continue  # Not enough samples to correlate

        # Compute correlation on overlapping samples
        curr_overlap = curr[:min_len]
        next_overlap = next_pos[:min_len]

        # Correlation between this position and next
        corr = np.corrcoef(curr_overlap, next_overlap)[0, 1]

        if np.isfinite(corr) and corr < 0:
            aligned[i] = -aligned[i]

    return aligned


def run_trajectories(data_dir: Path, force: bool) -> None:
    """Compute trajectories using BOTH approaches:
    1. Aligned: Per-target PCA with sign alignment (original method)
    2. Shared: Single PCA on all samples, then project per-layer/position

    This allows UI to switch between the two visualization modes.
    """
    log.info("  Computing ALIGNED trajectories (per-target PCA + sign alignment)...")
    run_trajectories_aligned(data_dir, force)

    log.info("  Computing SHARED trajectories (single PCA subspace)...")
    run_trajectories_shared(data_dir, force)

    clear_gpu_memory(aggressive=True)


def run_trajectories_aligned(data_dir: Path, force: bool) -> None:
    """Compute trajectories using per-target PCA with sign alignment (ORIGINAL method).

    Each layer/position has its own PCA embedding. Signs are aligned using
    continuity (correlation with adjacent layers/positions).

    Output files: layers_{c}_{p}.npz, positions_L{l}_{c}.npz (no suffix)
    """
    traj = data_dir / "analysis" / "trajectories"
    pca_dir = data_dir / "analysis" / "embeddings" / "pca"
    traj.mkdir(parents=True, exist_ok=True)

    # Load relpos_counts
    relpos_file = data_dir / "analysis" / "relpos_counts.json"
    if relpos_file.exists():
        with open(relpos_file) as f:
            relpos_counts = json.load(f)
    else:
        relpos_counts = {}

    # Build list of all position variants
    all_positions = []
    for p in POSITIONS:
        all_positions.append(p)
        if p in relpos_counts:
            for r in range(relpos_counts[p]):
                all_positions.append(f"{p}_r{r}")

    # Layer trajectories (aligned)
    layer_traj_count = 0
    for c in COMPONENTS:
        for p in all_positions:
            f = traj / f"layers_{c}_{p}.npz"
            if not force and f.exists():
                layer_traj_count += 1
                continue
            layers, pc1_values, pc2_values = [], [], []
            n_samples = 0
            for l in LAYERS:
                src = pca_dir / f"L{l}_{c}_{p}.npy"
                if src.exists():
                    try:
                        arr = np.load(src, mmap_mode='r')
                        target_key = f"L{l}_{c}_{p}"
                        pc1 = np.array(arr[:, 0])
                        pc2 = np.array(arr[:, 1]) if arr.shape[1] > 1 else np.zeros(arr.shape[0])

                        if np.any(~np.isfinite(pc1)) or np.any(~np.isfinite(pc2)):
                            _print_nan_warning(
                                "TRAJECTORY PCA EMBEDDING",
                                f"Target: {target_key} | File: {src} | PC values contain NaN/Inf"
                            )
                            raise NaNDetectedError(f"PCA embedding {src} contains invalid values.")

                        layers.append(l)
                        pc1_values.append(pc1)
                        pc2_values.append(pc2)
                        n_samples = arr.shape[0]
                    except NaNDetectedError:
                        raise
                    except Exception as e:
                        log.error(f"  Layer trajectory error L{l}_{c}_{p}: {e}")
                        raise
            if layers:
                # Sign alignment for smooth trajectories
                pc1_values = align_pc_signs_continuity(pc1_values)
                pc2_values = align_pc_signs_continuity(pc2_values)
                sample_indices = np.arange(n_samples)
                np.savez_compressed(
                    f,
                    layers=np.array(layers),
                    pc1_values=np.array(pc1_values),
                    pc2_values=np.array(pc2_values),
                    sample_indices=sample_indices,
                )
                layer_traj_count += 1
            del layers, pc1_values, pc2_values

    log.info(f"    Aligned layer trajectories: {layer_traj_count} files")

    # Position trajectories (aligned)
    pos_traj_count = 0
    for l in LAYERS:
        for c in COMPONENTS:
            f = traj / f"positions_L{l}_{c}.npz"
            if not force and f.exists():
                pos_traj_count += 1
                continue
            pos_list, pc1_values, pc2_values, sample_indices_list = [], [], [], []
            for p in POSITIONS:
                src = pca_dir / f"L{l}_{c}_{p}.npy"
                if src.exists():
                    try:
                        arr = np.load(src, mmap_mode='r')
                        target_key = f"L{l}_{c}_{p}"
                        pc1 = np.array(arr[:, 0])
                        pc2 = np.array(arr[:, 1]) if arr.shape[1] > 1 else np.zeros(arr.shape[0])

                        if np.any(~np.isfinite(pc1)) or np.any(~np.isfinite(pc2)):
                            _print_nan_warning(
                                "TRAJECTORY PCA EMBEDDING",
                                f"Target: {target_key} | File: {src} | PC values contain NaN/Inf"
                            )
                            raise NaNDetectedError(f"PCA embedding {src} contains invalid values.")

                        pos_list.append(p)
                        pc1_values.append(pc1)
                        pc2_values.append(pc2)
                        sample_indices_list.append(np.arange(arr.shape[0]))
                    except NaNDetectedError:
                        raise
                    except Exception as e:
                        log.error(f"  Position trajectory error L{l}_{c}_{p}: {e}")
                        raise
            if pos_list:
                pc1_values = align_pc_signs_position_trajectory(pc1_values)
                pc2_values = align_pc_signs_position_trajectory(pc2_values)
                np.savez_compressed(
                    f,
                    positions=np.array(pos_list),
                    pc1_values=np.array(pc1_values, dtype=object),
                    pc2_values=np.array(pc2_values, dtype=object),
                    sample_indices_list=np.array(sample_indices_list, dtype=object),
                )
                pos_traj_count += 1
            del pos_list, pc1_values, pc2_values, sample_indices_list

    log.info(f"    Aligned position trajectories: {pos_traj_count} files")


def run_trajectories_shared(data_dir: Path, force: bool) -> None:
    """Compute trajectories using SHARED PCA subspace (NEW method).

    For layer trajectories: Fit ONE PCA on ALL samples from ALL layers
    For position trajectories: Fit ONE PCA on ALL samples from ALL positions

    Output files: layers_{c}_{p}_shared.npz, positions_L{l}_{c}_shared.npz
    """
    from sklearn.decomposition import PCA

    traj = data_dir / "analysis" / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)

    # Load relpos_counts
    relpos_file = data_dir / "analysis" / "relpos_counts.json"
    if relpos_file.exists():
        with open(relpos_file) as f:
            relpos_counts = json.load(f)
    else:
        relpos_counts = {}

    # Build list of all position variants
    all_positions = []
    for p in POSITIONS:
        all_positions.append(p)
        if p in relpos_counts:
            for r in range(relpos_counts[p]):
                all_positions.append(f"{p}_r{r}")

    # Cache position mappings for activation loading
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)

    # =========================================================================
    # LAYER TRAJECTORIES (SHARED): Fit ONE PCA on ALL layers
    # =========================================================================
    layer_traj_count = 0
    for c in COMPONENTS:
        for p in all_positions:
            f = traj / f"layers_{c}_{p}_shared.npz"
            if not force and f.exists():
                layer_traj_count += 1
                continue

            all_activations = []
            layer_boundaries = []
            layers_with_data = []
            sample_indices_per_layer = []

            current_idx = 0
            for l in LAYERS:
                key = f"L{l}_{c}_{p}"
                result = load_target(data_dir, key, sample_dirs, mapping_cache)
                if result is None:
                    continue
                X, valid_indices = result
                validate_input_data(X, key)

                all_activations.append(X)
                n_samples = X.shape[0]
                layer_boundaries.append((l, current_idx, current_idx + n_samples))
                layers_with_data.append(l)
                sample_indices_per_layer.append(valid_indices)
                current_idx += n_samples

            if not all_activations:
                continue

            X_all = np.vstack(all_activations)
            n_components = min(10, X_all.shape[0] - 1, X_all.shape[1])
            if n_components < 2:
                continue

            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_projected = pca.fit_transform(X_all)
            validate_array_no_nan(X_projected, f"Shared layer trajectory PCA {c}/{p}", crash=True)

            layers, pc1_values, pc2_values = [], [], []
            for l, start_idx, end_idx in layer_boundaries:
                pc1 = X_projected[start_idx:end_idx, 0]
                pc2 = X_projected[start_idx:end_idx, 1] if n_components > 1 else np.zeros(end_idx - start_idx)
                layers.append(l)
                pc1_values.append(pc1)
                pc2_values.append(pc2)

            sample_indices = sample_indices_per_layer[0] if sample_indices_per_layer else np.arange(pc1_values[0].shape[0])

            np.savez_compressed(
                f,
                layers=np.array(layers),
                pc1_values=np.array(pc1_values),
                pc2_values=np.array(pc2_values),
                sample_indices=np.array(sample_indices),
            )
            layer_traj_count += 1
            del all_activations, X_all, X_projected, pc1_values, pc2_values

    log.info(f"    Shared layer trajectories: {layer_traj_count} files")

    # =========================================================================
    # POSITION TRAJECTORIES (SHARED): Fit ONE PCA on ALL positions
    # =========================================================================
    pos_traj_count = 0
    for l in LAYERS:
        for c in COMPONENTS:
            # Combined positions
            f = traj / f"positions_L{l}_{c}_shared.npz"
            if force or not f.exists():
                _compute_position_trajectory_shared(
                    data_dir, l, c, POSITIONS, f,
                    sample_dirs, mapping_cache
                )
                pos_traj_count += 1

            # Per-rel_pos trajectories
            for p in POSITIONS:
                if p not in relpos_counts or relpos_counts[p] <= 1:
                    continue
                relpos_positions = [f"{p}_r{r}" for r in range(relpos_counts[p])]
                f_relpos = traj / f"positions_L{l}_{c}_{p}_relpos_shared.npz"
                if force or not f_relpos.exists():
                    _compute_position_trajectory_shared(
                        data_dir, l, c, relpos_positions, f_relpos,
                        sample_dirs, mapping_cache
                    )
                    pos_traj_count += 1

    log.info(f"    Shared position trajectories: {pos_traj_count} files")


def _compute_position_trajectory_shared(
    data_dir: Path,
    layer: int,
    component: str,
    positions: list[str],
    output_file: Path,
    sample_dirs: list,
    mapping_cache: dict,
) -> None:
    """Compute position trajectory with shared PCA subspace.

    Fits ONE PCA on all samples from all positions, then projects each position's samples.
    """
    from sklearn.decomposition import PCA

    # Collect activations from ALL positions
    all_activations = []
    position_boundaries = []  # (position, start_idx, end_idx)
    positions_with_data = []
    sample_indices_per_position = []

    current_idx = 0
    for p in positions:
        key = f"L{layer}_{component}_{p}"
        result = load_target(data_dir, key, sample_dirs, mapping_cache)
        if result is None:
            continue
        X, valid_indices = result

        # Validate input
        validate_input_data(X, key)

        all_activations.append(X)
        n_samples = X.shape[0]
        position_boundaries.append((p, current_idx, current_idx + n_samples))
        positions_with_data.append(p)
        sample_indices_per_position.append(list(valid_indices))
        current_idx += n_samples

    if not all_activations:
        log.warning(f"  No data for position trajectory L{layer}/{component}")
        return

    # Stack all activations and fit ONE PCA
    X_all = np.vstack(all_activations)
    log.info(f"  Position trajectory L{layer}/{component}: fitting PCA on {X_all.shape[0]} samples from {len(positions_with_data)} positions")

    n_components = min(10, X_all.shape[0] - 1, X_all.shape[1])
    if n_components < 2:
        log.warning(f"  Skipping L{layer}/{component}: not enough samples for PCA")
        return

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_projected = pca.fit_transform(X_all)

    # Validate PCA output
    validate_array_no_nan(X_projected, f"Position trajectory PCA L{layer}/{component}", crash=True)

    # Extract PC1 and PC2 for each position
    pos_list, pc1_values, pc2_values, sample_indices_list = [], [], [], []
    for i, (p, start_idx, end_idx) in enumerate(position_boundaries):
        pc1 = X_projected[start_idx:end_idx, 0]
        pc2 = X_projected[start_idx:end_idx, 1] if n_components > 1 else np.zeros(end_idx - start_idx)
        pos_list.append(p)
        pc1_values.append(pc1)
        pc2_values.append(pc2)
        sample_indices_list.append(sample_indices_per_position[i])

    np.savez_compressed(
        output_file,
        positions=np.array(pos_list),
        pc1_values=np.array(pc1_values, dtype=object),
        pc2_values=np.array(pc2_values, dtype=object),
        sample_indices_list=np.array(sample_indices_list, dtype=object),
    )

    # Clean up
    del all_activations, X_all, X_projected, pc1_values, pc2_values


def process_dataset(data_dir: Path, force: bool, fast: bool = False, lenient: bool = False) -> int:
    """Process a single dataset directory."""
    if not (data_dir / "data").exists():
        log.error(f"No data: {data_dir}")
        return 1

    # Get rel_pos counts for this dataset
    relpos_counts = get_max_relpos_counts(data_dir)

    # All keys including per-rel_pos (for PCA, embeddings, everything)
    all_keys = target_keys_with_relpos(relpos_counts)
    total_all = len(all_keys)

    # Combined keys (legacy - used only for trajectory count estimation)
    combined_keys = target_keys()
    total_combined = len(combined_keys)

    analysis = data_dir / "analysis"
    expected_traj = len(COMPONENTS) * len(POSITIONS) + len(LAYERS) * len(COMPONENTS)

    # Count samples
    samples_dir = data_dir / "data" / "samples"
    n_samples = sum(1 for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_"))

    # Save rel_pos counts for UI
    relpos_file = analysis / "relpos_counts.json"
    relpos_file.parent.mkdir(parents=True, exist_ok=True)
    with open(relpos_file, "w") as f:
        json.dump(relpos_counts, f, indent=2)

    # Determine which embedding methods to run
    methods = ["pca"] if fast else METHODS

    # Check cache
    pca_count = count_files(analysis / "pca", "*/metrics.json")
    emb_counts = {m: count_files(analysis / "embeddings" / m, "*.npy") for m in methods}
    traj_count = count_files(analysis / "trajectories", "*.npz")

    log.info("=" * 50)
    log.info(f"DATASET: {data_dir.name} | {n_samples} samples")
    log.info(f"Targets: {total_combined} combined, {total_all} total (with per-rel_pos)")
    log.info(f"Cache: PCA={pca_count}/{total_all} Embeddings={list(emb_counts.values())} Traj={traj_count}/{expected_traj}")
    log.info("=" * 50)

    # Phase 1: PCA Analysis - ALL keys including per-rel_pos (for scree/alignment)
    if force or pca_count < total_all:
        log.info(f"\n[1] PCA Analysis ({total_all} targets including per-rel_pos)")
        run_pca_analysis(data_dir, all_keys, force)
    else:
        log.info("\n[1] PCA Analysis COMPLETE")

    # Phase 2: Embeddings - ALL keys including per-rel_pos
    # Check against total_all (includes per-rel_pos)
    need = [m for m in methods if force or emb_counts[m] < total_all]
    if need:
        log.info(f"\n[2] Embeddings (combined + per-rel_pos): {need}")
        log.info(f"  Computing {total_all} targets...")
        counts = run_embeddings(data_dir, all_keys, need, force)
        log.info(f"  {counts}")
    else:
        log.info("\n[2] Embeddings COMPLETE")

    # Phase 3: Trajectories - combined keys only
    if force or traj_count < expected_traj:
        log.info("\n[3] Trajectories")
        run_trajectories(data_dir, force)
    else:
        log.info("\n[3] Trajectories COMPLETE")

    # ==========================================================================
    # STRICT VERIFICATION - CRASHES ON MISSING DATA
    # ==========================================================================
    log.info("\n" + "=" * 60)
    log.info("STRICT VERIFICATION - WILL CRASH IF DATA IS MISSING")
    log.info("=" * 60)

    errors = []

    # 1. Verify PCA embeddings for all combined keys
    for m in methods:
        missing_combined = []
        for key in combined_keys:
            emb_file = analysis / "embeddings" / m / f"{key}.npy"
            if not emb_file.exists():
                missing_combined.append(key)

        if missing_combined:
            errors.append(f"{m.upper()} embeddings missing for {len(missing_combined)} combined keys: {missing_combined[:5]}...")
        else:
            log.info(f"  {m.upper()} embeddings: {len(combined_keys)} combined keys OK")

    # 2. Verify trajectory files for all layer/component combinations
    expected_layer_traj = []
    for c in COMPONENTS:
        for p in POSITIONS:
            expected_layer_traj.append(f"layers_{c}_{p}.npz")

    missing_layer_traj = []
    for traj_file in expected_layer_traj:
        if not (analysis / "trajectories" / traj_file).exists():
            missing_layer_traj.append(traj_file)

    if missing_layer_traj:
        errors.append(f"Layer trajectories missing: {len(missing_layer_traj)} files: {missing_layer_traj[:5]}...")
    else:
        log.info(f"  Layer trajectories: {len(expected_layer_traj)} files OK")

    # 3. Verify position trajectory files for all layers
    expected_pos_traj = []
    for l in LAYERS:
        for c in COMPONENTS:
            expected_pos_traj.append(f"positions_L{l}_{c}.npz")

    missing_pos_traj = []
    for traj_file in expected_pos_traj:
        if not (analysis / "trajectories" / traj_file).exists():
            missing_pos_traj.append(traj_file)

    if missing_pos_traj:
        errors.append(f"Position trajectories missing: {len(missing_pos_traj)} files: {missing_pos_traj[:5]}...")
    else:
        log.info(f"  Position trajectories: {len(expected_pos_traj)} files OK")

    log.info("=" * 60)

    # Handle verification errors
    if errors:
        log.warning("\n" + "!" * 60)
        log.warning("VERIFICATION ISSUES - DATA IS INCOMPLETE")
        log.warning("!" * 60)
        for err in errors:
            log.warning(f"  - {err}")
        log.warning("!" * 60)

        if lenient:
            log.warning("LENIENT MODE: Continuing despite missing data")
            log.warning("Some visualizations may not work correctly")
            log.warning("!" * 60)
        else:
            log.error("Re-run with --force to regenerate all data")
            log.error("Or use --lenient to continue anyway")
            log.error("!" * 60)
            raise RuntimeError(f"VERIFICATION FAILED: {len(errors)} errors. Data is incomplete.")

    log.info("VERIFICATION COMPLETE" + (" (with warnings)" if errors else " - ALL DATA COMPLETE"))
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
        description="Compute geometry analysis for datasets in out/geo/",
        epilog="Examples:\n"
               "  %(prog)s                  # Process ALL datasets in out/geo/\n"
               "  %(prog)s investment       # Process only out/geo/investment\n"
               "  %(prog)s --force          # Force recompute all\n",
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
    p.add_argument("--fast", action="store_true", help="Fast mode: PCA only, skip UMAP and t-SNE")
    p.add_argument("--lenient", action="store_true", help="Lenient mode: warn on missing data instead of failing")
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
    log.info("COMPUTE GEOMETRY ANALYSIS")
    if args.fast:
        log.info("MODE: FAST (PCA only, skipping UMAP and t-SNE)")
    log.info(f"Processing {len(datasets)} dataset(s):")
    for d in datasets:
        log.info(f"  - {d.name}")
    log.info("=" * 60)

    # Process each dataset
    for data_dir in datasets:
        result = process_dataset(data_dir, args.force, fast=args.fast, lenient=args.lenient)
        if result != 0:
            return result

    log.info("\n" + "=" * 60)
    log.info("ALL DATASETS COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
