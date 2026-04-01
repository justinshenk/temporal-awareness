#!/usr/bin/env python3
"""Compute geometry analysis for GeoApp. MINIMAL MEMORY - streams everything."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.device_utils import clear_gpu_memory
from src.intertemporal.common.semantic_positions import PROMPT_POSITIONS, RESPONSE_POSITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LAYERS = [0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS
METHODS = ["pca", "umap", "tsne"]


def target_keys() -> list[str]:
    """Generate combined (aggregated) target keys only."""
    return [f"L{l}_{c}_{p}" for l in LAYERS for c in COMPONENTS for p in POSITIONS]


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
        named_positions = mapping.get("named_positions", {})
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
                # Per-rel_pos keys (if position has multiple tokens)
                max_relpos = relpos_counts.get(p, 1)
                if max_relpos > 1:
                    for r in range(max_relpos):
                        keys.append(f"L{l}_{c}_{p}_r{r}")
    return keys


def count_files(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern))) if d.exists() else 0


import re
_KEY_PATTERN = re.compile(r"L(\d+)_(.+)")
_RELPOS_PATTERN = re.compile(r"(.+)_r(\d+)$")


def parse_key(key: str) -> tuple[int, str, str, int | None] | None:
    """Parse target key into (layer, component, position, rel_pos).

    Key format: L{layer}_{component}_{position} or L{layer}_{component}_{position}_r{rel_pos}
    Component is one of: resid_pre, attn_out, mlp_out, resid_post
    rel_pos is None for combined keys, integer for per-token keys.
    """
    m = _KEY_PATTERN.match(key)
    if not m:
        return None
    layer = int(m.group(1))
    rest = m.group(2)
    for comp in COMPONENTS:
        if rest.startswith(comp + "_"):
            pos_part = rest[len(comp) + 1:]
            # Check for _r{N} suffix
            rel_match = _RELPOS_PATTERN.match(pos_part)
            if rel_match:
                pos = rel_match.group(1)
                rel_pos = int(rel_match.group(2))
                return (layer, comp, pos, rel_pos)
            else:
                return (layer, comp, pos_part, None)
    return None


def get_abs_pos(mapping: dict, pos: str) -> int | list[int] | None:
    """Get absolute position(s) from mapping.

    Returns single int, list of ints, or None. Caller should try each position.
    """
    abs_pos = mapping.get("named_positions", {}).get(pos)
    if abs_pos is None:
        return None
    return abs_pos


def find_activation_file(sample_dir: Path, layer: int, comp: str, abs_pos: int | list[int]) -> Path | None:
    """Find activation file, trying multiple positions if abs_pos is a list."""
    if isinstance(abs_pos, list):
        # Try each position in order until we find one that exists
        for p in abs_pos:
            f = sample_dir / f"L{layer}_{comp}_{p}.npy"
            if f.exists():
                return f
        return None
    else:
        f = sample_dir / f"L{layer}_{comp}_{abs_pos}.npy"
        return f if f.exists() else None


def cache_position_mappings(data_dir: Path) -> tuple[list[Path], dict[int, dict]]:
    """Cache all position mappings once. Returns (sample_dirs, mapping_cache)."""
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    mapping_cache = {}
    for i, d in enumerate(sample_dirs):
        mapping_file = d / "position_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapping_cache[i] = json.load(f)
    return sample_dirs, mapping_cache


def load_target(
    data_dir: Path,
    key: str,
    sample_dirs: list[Path] | None = None,
    mapping_cache: dict[int, dict] | None = None,
) -> np.ndarray | None:
    """Load activations for ONE target.

    For combined keys (no _r{N} suffix): loads first available token per sample.
    For per-rel_pos keys (_r{N} suffix): loads only the specific rel_pos token.

    Args:
        data_dir: Dataset directory
        key: Target key (e.g., "L0_resid_pre_time_horizon" or "L0_resid_pre_time_horizon_r1")
        sample_dirs: Pre-cached list of sample directories (optional, loaded if None)
        mapping_cache: Pre-cached position mappings {sample_idx: mapping_dict} (optional)
    """
    parsed = parse_key(key)
    if not parsed:
        return None
    layer, comp, pos, rel_pos = parsed

    # Use cached mappings if provided, otherwise load fresh
    if sample_dirs is None or mapping_cache is None:
        sample_dirs, mapping_cache = cache_position_mappings(data_dir)

    # Single pass: load all valid activation files
    valid_files: list[Path] = []
    dim = None

    for i, d in enumerate(sample_dirs):
        if i not in mapping_cache:
            continue
        abs_pos = get_abs_pos(mapping_cache[i], pos)
        if abs_pos is None:
            continue

        # For per-rel_pos: select specific token index
        if rel_pos is not None:
            if isinstance(abs_pos, list):
                if rel_pos >= len(abs_pos):
                    continue  # This sample doesn't have this rel_pos
                abs_pos = abs_pos[rel_pos]
            elif rel_pos > 0:
                continue  # Single token, but asking for rel_pos > 0

        act_file = find_activation_file(d, layer, comp, abs_pos)
        if act_file:
            if dim is None:
                dim = np.load(act_file, mmap_mode='r').shape[0]
            valid_files.append(act_file)

    if len(valid_files) < 4 or dim is None:
        return None

    # Load all valid activations into pre-allocated array
    X = np.empty((len(valid_files), dim), dtype=np.float32)
    for idx, act_file in enumerate(valid_files):
        X[idx] = np.load(act_file)

    return X


def load_horizons(data_dir: Path) -> np.ndarray:
    """Load log time horizons. Called once."""
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    h = np.empty(len(sample_dirs), dtype=np.float32)
    for i, d in enumerate(sample_dirs):
        f = d / "choice.json"
        if f.exists():
            with open(f) as fp:
                h[i] = json.load(fp).get("time_horizon_months", 60.0)
        else:
            h[i] = 60.0
    return np.log10(h + 0.1)


def run_analysis(data_dir: Path, keys: list[str], force: bool) -> None:
    """Linear probe + PCA. Streams one target at a time."""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    analysis = data_dir / "analysis"
    lp_dir, pca_dir = analysis / "linear_probe", analysis / "pca"
    lp_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    y = load_horizons(data_dir)
    lp_all, pca_all = {}, {}

    # Cache position mappings once for all targets
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)
    log.info(f"  Cached {len(mapping_cache)} position mappings")

    for i, key in enumerate(keys):
        if i % 50 == 0:
            log.info(f"  Analysis {i}/{len(keys)}")
            clear_gpu_memory(aggressive=True)

        lp_cache = lp_dir / key / "metrics.json"
        pca_cache = pca_dir / key / "metrics.json"

        if not force and lp_cache.exists() and pca_cache.exists():
            with open(lp_cache) as f:
                lp_all[key] = json.load(f)
            with open(pca_cache) as f:
                pca_all[key] = json.load(f)
            continue

        X = load_target(data_dir, key, sample_dirs, mapping_cache)
        if X is None:
            continue

        y_sub = y[:len(X)]

        # Linear probe
        try:
            pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
            scores = cross_val_score(pipe, X, y_sub, cv=5, scoring="r2")
            pipe.fit(X, y_sub)
            corr = spearmanr(y_sub, pipe.predict(X))[0]
            lp = {
                "target_key": key,
                "r2_mean": float(np.mean(scores)),
                "r2_std": float(np.std(scores)),
                "correlation": float(corr) if np.isfinite(corr) else 0.0,
                "n_samples": len(y_sub),
            }
            lp_all[key] = lp
            (lp_dir / key).mkdir(exist_ok=True)
            with open(lp_cache, "w") as f:
                json.dump(lp, f)
            del pipe, scores
        except Exception:
            pass

        # PCA
        try:
            n = min(10, X.shape[0] - 1, X.shape[1])
            if n >= 1:
                pca = PCA(n_components=n, random_state=42)
                Xp = pca.fit_transform(X)
                corrs = []
                for j in range(n):
                    corr_val = spearmanr(y_sub, Xp[:, j])[0]
                    corrs.append([j, float(corr_val) if np.isfinite(corr_val) else 0.0])
                pm = {
                    "target_key": key,
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                    "pc_correlations": corrs,
                }
                pca_all[key] = pm
                (pca_dir / key).mkdir(exist_ok=True)
                with open(pca_cache, "w") as f:
                    json.dump(pm, f)
                np.save(pca_dir / key / "components.npy", pca.components_.astype(np.float32))
                del pca, Xp
        except Exception:
            pass

        del X
        clear_gpu_memory(aggressive=True)

    # Save summary
    with open(data_dir / "summary.json", "w") as f:
        json.dump({
            "n_samples": len(y),
            "layers": LAYERS,
            "components": COMPONENTS,
            "positions": POSITIONS,
            "linear_probe": lp_all,
            "pca": {k: {"top_pc": v["pc_correlations"][0][0], "top_corr": v["pc_correlations"][0][1]}
                   for k, v in pca_all.items() if v.get("pc_correlations")}
        }, f, indent=2)

    del y, lp_all, pca_all
    clear_gpu_memory(aggressive=True)


def _compute_single_embedding(
    key: str,
    X: np.ndarray,
    methods: list[str],
    emb_dir: Path,
) -> dict[str, str | None]:
    """Compute embeddings for a single target. Returns {method: error_or_None}."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    n = X.shape[0]
    results = {}

    for m in methods:
        try:
            if m == "pca":
                e = PCA(n_components=min(3, n-1, X.shape[1]), random_state=42).fit_transform(X)
            elif m == "umap":
                e = UMAP(n_components=3, n_neighbors=min(15, max(2, n-1)), min_dist=0.1, random_state=42, n_jobs=1).fit_transform(X)
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
                # Pad back to full size if subsampled
                if idx is not None:
                    e = np.zeros((n, 3), dtype=np.float32)
                    e[idx] = e_sub
                else:
                    e = e_sub
            else:
                continue

            # Validate output
            if not np.all(np.isfinite(e)):
                e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

            # Pad if needed
            if e.shape[1] < 3:
                pad = np.zeros((n, 3), dtype=np.float32)
                pad[:, :e.shape[1]] = e
                e = pad

            np.save(emb_dir / m / f"{key}.npy", e.astype(np.float32))
            results[m] = None  # Success
        except Exception as ex:
            results[m] = f"{key}: {ex}"

    return results


def run_embeddings(data_dir: Path, keys: list[str], methods: list[str], force: bool) -> dict[str, int]:
    """Compute embeddings. Loads each target once, computes all needed methods, deletes immediately."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    emb_dir = data_dir / "analysis" / "embeddings"
    for m in methods:
        (emb_dir / m).mkdir(parents=True, exist_ok=True)

    counts = {m: 0 for m in methods}
    failed = {m: [] for m in methods}

    # Cache position mappings once for all targets (avoids millions of JSON reads)
    sample_dirs, mapping_cache = cache_position_mappings(data_dir)
    log.info(f"  Cached {len(mapping_cache)} position mappings")

    # Determine which keys need computation
    keys_to_compute = []
    for key in keys:
        need = [m for m in methods if force or not (emb_dir / m / f"{key}.npy").exists()]
        if need:
            keys_to_compute.append((key, need))
        else:
            for m in methods:
                counts[m] += 1  # Already exists

    log.info(f"  {len(keys_to_compute)} targets need computation, {len(keys) - len(keys_to_compute)} cached")

    # Use parallel processing with limited workers to avoid memory issues
    max_workers = min(4, os.cpu_count() or 1)
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
            X = load_target(data_dir, key, sample_dirs, mapping_cache)
            if X is None or X.shape[0] < 4:
                for m in need:
                    failed[m].append(key)
                continue

            # Validate data
            if np.any(~np.isfinite(X)):
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            batch_data.append((key, X, need))

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_embedding, key, X, need, emb_dir): key
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


def run_trajectories(data_dir: Path, force: bool) -> None:
    """Compute trajectories from PCA embeddings. Memory-mapped reads."""
    traj = data_dir / "analysis" / "trajectories"
    pca = data_dir / "analysis" / "embeddings" / "pca"
    traj.mkdir(parents=True, exist_ok=True)

    # Layer trajectories
    for c in COMPONENTS:
        for p in POSITIONS:
            f = traj / f"layers_{c}_{p}.npz"
            if not force and f.exists():
                continue
            layers, values = [], []
            for l in LAYERS:
                src = pca / f"L{l}_{c}_{p}.npy"
                if src.exists():
                    try:
                        # Memory-mapped read - only loads first column
                        arr = np.load(src, mmap_mode='r')
                        layers.append(l)
                        values.append(np.array(arr[:, 0]))
                    except Exception:
                        pass
            if layers:
                np.savez_compressed(f, layers=np.array(layers), pc1_values=np.array(values))
            del layers, values

    # Position trajectories
    for l in LAYERS:
        for c in COMPONENTS:
            f = traj / f"positions_L{l}_{c}.npz"
            if not force and f.exists():
                continue
            pos_list, values = [], []
            for p in POSITIONS:
                src = pca / f"L{l}_{c}_{p}.npy"
                if src.exists():
                    try:
                        arr = np.load(src, mmap_mode='r')
                        pos_list.append(p)
                        values.append(np.array(arr[:, 0]))
                    except Exception:
                        pass
            if pos_list:
                np.savez_compressed(f, positions=np.array(pos_list), pc1_values=np.array(values, dtype=object))
            del pos_list, values

    clear_gpu_memory(aggressive=True)


def process_dataset(data_dir: Path, force: bool) -> int:
    """Process a single dataset directory."""
    if not (data_dir / "data").exists():
        log.error(f"No data: {data_dir}")
        return 1

    # Get rel_pos counts for this dataset
    relpos_counts = get_max_relpos_counts(data_dir)

    # Combined keys for analysis (linear probe, PCA)
    combined_keys = target_keys()
    total_combined = len(combined_keys)

    # All keys including per-rel_pos for embeddings
    all_keys = target_keys_with_relpos(relpos_counts)
    total_all = len(all_keys)

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

    # Check cache
    lp_count = count_files(analysis / "linear_probe", "*/metrics.json")
    pca_count = count_files(analysis / "pca", "*/metrics.json")
    emb_counts = {m: count_files(analysis / "embeddings" / m, "*.npy") for m in METHODS}
    traj_count = count_files(analysis / "trajectories", "*.npz")

    log.info("=" * 50)
    log.info(f"DATASET: {data_dir.name} | {n_samples} samples")
    log.info(f"Targets: {total_combined} combined, {total_all} total (with per-rel_pos)")
    log.info(f"Cache: LP={lp_count}/{total_combined} PCA={pca_count}/{total_combined}")
    log.info(f"       Embeddings={list(emb_counts.values())} Traj={traj_count}/{expected_traj}")
    log.info("=" * 50)

    # Phase 1: Analysis - combined keys only
    if force or lp_count < total_combined or pca_count < total_combined:
        log.info("\n[1] Analysis (combined keys only)")
        run_analysis(data_dir, combined_keys, force)
    else:
        log.info("\n[1] Analysis COMPLETE")

    # Phase 2: Embeddings - ALL keys including per-rel_pos
    # Check against total_all (includes per-rel_pos)
    need = [m for m in METHODS if force or emb_counts[m] < total_all]
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

    # Verify - warn if incomplete but don't fail
    # (some per-rel_pos targets may have insufficient samples)
    log.info("\n" + "=" * 50)
    log.info("VERIFICATION")
    for m in METHODS:
        c = count_files(analysis / "embeddings" / m, "*.npy")
        # At minimum, combined keys should exist
        if c >= total_combined:
            log.info(f"  {m.upper()}: {c}/{total_all} OK (>= {total_combined} combined)")
        else:
            log.warning(f"  {m.upper()}: {c}/{total_all} (< {total_combined} combined - some missing)")
    log.info("=" * 50)

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
    log.info(f"COMPUTE GEOMETRY ANALYSIS")
    log.info(f"Processing {len(datasets)} dataset(s):")
    for d in datasets:
        log.info(f"  - {d.name}")
    log.info("=" * 60)

    # Process each dataset
    for data_dir in datasets:
        result = process_dataset(data_dir, args.force)
        if result != 0:
            return result

    log.info("\n" + "=" * 60)
    log.info("ALL DATASETS COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
