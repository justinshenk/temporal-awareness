"""Analysis functions for geometric visualization.

Memory-optimized implementation:
- Process one target at a time
- Save results to disk immediately
- Clear memory after each target
- Store only essential PCA components
"""

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from .geo_viz_config import (
    GeoVizConfig,
    ACTIVATION_DTYPE,
    ANALYSIS_GC_INTERVAL,
    MAX_STORED_PCA_COMPONENTS,
)
from .geo_viz_data import ActivationData, get_time_horizon_months

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes (with disk serialization)
# =============================================================================


@dataclass(slots=True)
class LinearProbeResult:
    """Results from linear probe analysis. Uses __slots__."""

    target_key: str
    r2_mean: float
    r2_std: float
    correlation: float
    predictions: np.ndarray
    coefficients: np.ndarray

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "target_key": self.target_key,
            "r2_mean": float(self.r2_mean),
            "r2_std": float(self.r2_std),
            "correlation": float(self.correlation),
            "n_samples": len(self.predictions),
        }

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "predictions.npy", self.predictions.astype(ACTIVATION_DTYPE))
        np.save(path / "coefficients.npy", self.coefficients.astype(ACTIVATION_DTYPE))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "LinearProbeResult":
        """Load from disk."""
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        return cls(
            target_key=metrics["target_key"],
            r2_mean=metrics["r2_mean"],
            r2_std=metrics["r2_std"],
            correlation=metrics["correlation"],
            predictions=np.load(path / "predictions.npy"),
            coefficients=np.load(path / "coefficients.npy"),
        )


@dataclass(slots=True)
class PCAResult:
    """Results from PCA analysis. Uses __slots__."""

    target_key: str
    explained_variance: np.ndarray
    components: np.ndarray
    transformed: np.ndarray
    pc_correlations: list[tuple[int, float, float]]

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "target_key": self.target_key,
            "n_components": len(self.explained_variance),
            "n_samples": self.transformed.shape[0],
            "top_correlations": [
                {"pc": int(pc), "corr": float(corr), "pval": float(pval)}
                for pc, corr, pval in self.pc_correlations[:10]
            ],
        }

    def save(self, path: Path):
        """Save to disk (only stores essential components)."""
        path.mkdir(parents=True, exist_ok=True)

        # Store only top N components to save space
        n_store = MAX_STORED_PCA_COMPONENTS or len(self.explained_variance)
        n_store = min(n_store, len(self.explained_variance))

        np.save(path / "explained_variance.npy", self.explained_variance[:n_store].astype(ACTIVATION_DTYPE))
        np.save(path / "components.npy", self.components[:n_store].astype(ACTIVATION_DTYPE))
        np.save(path / "transformed.npy", self.transformed[:, :n_store].astype(ACTIVATION_DTYPE))

        with open(path / "metrics.json", "w") as f:
            json.dump({
                **self.to_dict(),
                "stored_components": n_store,
                "pc_correlations": [(int(pc), float(corr), float(pval)) for pc, corr, pval in self.pc_correlations],
            }, f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "PCAResult":
        """Load from disk."""
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        return cls(
            target_key=metrics["target_key"],
            explained_variance=np.load(path / "explained_variance.npy"),
            components=np.load(path / "components.npy"),
            transformed=np.load(path / "transformed.npy"),
            pc_correlations=[(pc, corr, pval) for pc, corr, pval in metrics["pc_correlations"]],
        )


@dataclass(slots=True)
class EmbeddingResult:
    """Results from dimensionality reduction. Uses __slots__."""

    target_key: str
    umap_embedding: np.ndarray | None
    tsne_embedding: np.ndarray | None
    pca_embedding: np.ndarray

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "pca_embedding.npy", self.pca_embedding.astype(ACTIVATION_DTYPE))
        if self.umap_embedding is not None:
            np.save(path / "umap_embedding.npy", self.umap_embedding.astype(ACTIVATION_DTYPE))
        if self.tsne_embedding is not None:
            np.save(path / "tsne_embedding.npy", self.tsne_embedding.astype(ACTIVATION_DTYPE))
        with open(path / "metadata.json", "w") as f:
            json.dump({"target_key": self.target_key}, f)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingResult":
        """Load from disk."""
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        umap_path = path / "umap_embedding.npy"
        tsne_path = path / "tsne_embedding.npy"

        return cls(
            target_key=metadata["target_key"],
            pca_embedding=np.load(path / "pca_embedding.npy"),
            umap_embedding=np.load(umap_path) if umap_path.exists() else None,
            tsne_embedding=np.load(tsne_path) if tsne_path.exists() else None,
        )


# =============================================================================
# Single-Target Analysis Functions
# =============================================================================


def _get_log_horizons(data: ActivationData) -> np.ndarray:
    """Get log-transformed time horizons for all samples."""
    horizons = np.array([get_time_horizon_months(s) for s in data.samples], dtype=ACTIVATION_DTYPE)
    return np.log10(horizons + 1)


def analyze_single_target(
    target_key: str,
    X: np.ndarray,
    log_horizons: np.ndarray,
    config: GeoVizConfig,
) -> tuple[LinearProbeResult, PCAResult, EmbeddingResult]:
    """Run all analyses on a single target. Memory efficient."""
    n_samples = X.shape[0]

    # Handle sample count mismatch
    if n_samples < len(log_horizons):
        target_log_horizons = log_horizons[:n_samples]
    else:
        target_log_horizons = log_horizons

    # Linear probe
    linear_result = _linear_probe_single(target_key, X, target_log_horizons)

    # PCA
    pca_result = _pca_single(target_key, X, target_log_horizons, config)

    # Embeddings (uses PCA result)
    embedding_result = _embeddings_single(target_key, pca_result, config)

    return linear_result, pca_result, embedding_result


def _linear_probe_single(
    target_key: str,
    X: np.ndarray,
    log_horizons: np.ndarray,
) -> LinearProbeResult:
    """Linear probe analysis for a single target."""
    n_samples = X.shape[0]

    if n_samples < 10:
        return LinearProbeResult(
            target_key=target_key,
            r2_mean=0.0,
            r2_std=0.0,
            correlation=0.0,
            predictions=np.zeros(n_samples, dtype=ACTIVATION_DTYPE),
            coefficients=np.zeros(X.shape[1], dtype=ACTIVATION_DTYPE),
        )

    ridge = Ridge(alpha=1.0)
    try:
        cv_folds = min(5, n_samples // 2)
        scores = cross_val_score(ridge, X, log_horizons, cv=cv_folds, scoring="r2")
        r2_mean = float(scores.mean())
        r2_std = float(scores.std())
    except Exception:
        r2_mean = 0.0
        r2_std = 0.0

    try:
        ridge.fit(X, log_horizons)
        predictions = ridge.predict(X).astype(ACTIVATION_DTYPE)
        correlation = float(np.corrcoef(predictions, log_horizons)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
        coefficients = ridge.coef_.astype(ACTIVATION_DTYPE)
    except Exception:
        predictions = np.zeros(n_samples, dtype=ACTIVATION_DTYPE)
        correlation = 0.0
        coefficients = np.zeros(X.shape[1], dtype=ACTIVATION_DTYPE)

    return LinearProbeResult(
        target_key=target_key,
        r2_mean=r2_mean,
        r2_std=r2_std,
        correlation=correlation,
        predictions=predictions,
        coefficients=coefficients,
    )


def _pca_single(
    target_key: str,
    X: np.ndarray,
    log_horizons: np.ndarray,
    config: GeoVizConfig,
) -> PCAResult:
    """PCA analysis for a single target."""
    n_samples = X.shape[0]
    n_components = min(config.n_pca_components, n_samples - 1, X.shape[1])

    if n_components < 1:
        return PCAResult(
            target_key=target_key,
            explained_variance=np.array([0.0], dtype=ACTIVATION_DTYPE),
            components=np.zeros((1, X.shape[1]), dtype=ACTIVATION_DTYPE),
            transformed=np.zeros((n_samples, 1), dtype=ACTIVATION_DTYPE),
            pc_correlations=[(0, 0.0, 1.0)],
        )

    pca = PCA(n_components=n_components, random_state=config.seed)
    X_pca = pca.fit_transform(X).astype(ACTIVATION_DTYPE)

    # Correlate each PC with log horizon
    pc_correlations = []
    for i in range(n_components):
        try:
            corr, pval = spearmanr(X_pca[:, i], log_horizons)
            if np.isnan(corr):
                corr, pval = 0.0, 1.0
        except Exception:
            corr, pval = 0.0, 1.0
        pc_correlations.append((i, float(corr), float(pval)))

    # Sort by absolute correlation
    pc_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    return PCAResult(
        target_key=target_key,
        explained_variance=pca.explained_variance_ratio_.astype(ACTIVATION_DTYPE),
        components=pca.components_.astype(ACTIVATION_DTYPE),
        transformed=X_pca,
        pc_correlations=pc_correlations,
    )


def _embeddings_single(
    target_key: str,
    pca_result: PCAResult,
    config: GeoVizConfig,
) -> EmbeddingResult:
    """Compute embeddings for a single target."""
    X_pca = pca_result.transformed
    n_samples = X_pca.shape[0]
    n_pcs = min(20, X_pca.shape[1])

    # Check for degenerate activations
    variance = np.var(X_pca[:, :n_pcs])
    if variance < 1e-10:
        return EmbeddingResult(
            target_key=target_key,
            pca_embedding=X_pca[:, :2] if X_pca.shape[1] >= 2 else np.zeros((n_samples, 2), dtype=ACTIVATION_DTYPE),
            umap_embedding=None,
            tsne_embedding=None,
        )

    pca_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.zeros((n_samples, 2), dtype=ACTIVATION_DTYPE)

    # UMAP
    umap_2d = None
    try:
        import umap
        n_neighbors = min(config.umap_n_neighbors, n_samples - 1)
        if n_neighbors >= 2:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=config.umap_min_dist,
                random_state=config.seed,
            )
            umap_2d = reducer.fit_transform(X_pca[:, :n_pcs]).astype(ACTIVATION_DTYPE)
    except Exception:
        pass

    # t-SNE
    tsne_2d = None
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, n_samples // 4))
        if n_samples > perplexity * 3:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=config.seed)
            tsne_2d = tsne.fit_transform(X_pca[:, :n_pcs]).astype(ACTIVATION_DTYPE)
    except Exception:
        pass

    return EmbeddingResult(
        target_key=target_key,
        pca_embedding=pca_2d,
        umap_embedding=umap_2d,
        tsne_embedding=tsne_2d,
    )


# =============================================================================
# Streaming Pipeline
# =============================================================================


def _safe_key(key: str) -> str:
    """Convert target key to safe directory name."""
    return key.replace("/", "_").replace("\\", "_")


def run_streaming_analysis(
    data: ActivationData,
    config: GeoVizConfig,
) -> tuple[dict[str, LinearProbeResult], dict[str, PCAResult], dict[str, EmbeddingResult]]:
    """Run analysis with streaming - process one target at a time.

    Memory efficient: loads each target, analyzes, saves to disk, unloads.
    """
    results_dir = config.output_dir / "results"
    linear_dir = results_dir / "linear_probe"
    pca_dir = results_dir / "pca"
    embedding_dir = results_dir / "embeddings"

    for d in [linear_dir, pca_dir, embedding_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Pre-compute log horizons once
    log_horizons = _get_log_horizons(data)

    target_keys = data.get_target_keys()
    n_targets = len(target_keys)

    logger.info(f"Running streaming analysis on {n_targets} targets...")

    linear_results = {}
    pca_results = {}
    embedding_results = {}

    for i, target_key in enumerate(target_keys):
        if i % 20 == 0:
            logger.info(f"  Analyzing target {i}/{n_targets}: {target_key}")

        safe_key = _safe_key(target_key)
        linear_path = linear_dir / safe_key
        pca_path = pca_dir / safe_key
        embedding_path = embedding_dir / safe_key

        # Check if already analyzed (skip if cached)
        if linear_path.exists() and pca_path.exists() and embedding_path.exists():
            try:
                linear_results[target_key] = LinearProbeResult.load(linear_path)
                pca_results[target_key] = PCAResult.load(pca_path)
                embedding_results[target_key] = EmbeddingResult.load(embedding_path)
                continue
            except Exception as e:
                logger.warning(f"  Cache load failed for {target_key}: {e}")

        # Load activations
        try:
            X = data.load_target(target_key)
        except Exception as e:
            logger.warning(f"  Failed to load {target_key}: {e}")
            continue

        # Analyze
        linear_result, pca_result, embedding_result = analyze_single_target(
            target_key, X, log_horizons, config
        )

        # Save to disk
        linear_result.save(linear_path)
        pca_result.save(pca_path)
        embedding_result.save(embedding_path)

        # Store in results
        linear_results[target_key] = linear_result
        pca_results[target_key] = pca_result
        embedding_results[target_key] = embedding_result

        # Unload to free memory
        data.unload_target(target_key)

        # Periodic GC
        if i % ANALYSIS_GC_INTERVAL == 0:
            gc.collect()

    logger.info(f"Analysis complete: {len(linear_results)} targets")

    return linear_results, pca_results, embedding_results


# =============================================================================
# Legacy Functions (backwards compatibility)
# =============================================================================


def linear_probe_analysis(
    data: ActivationData, config: GeoVizConfig
) -> dict[str, LinearProbeResult]:
    """Run linear probe analysis. Uses streaming internally."""
    linear_results, _, _ = run_streaming_analysis(data, config)
    return linear_results


def pca_correlation_analysis(
    data: ActivationData, config: GeoVizConfig
) -> dict[str, PCAResult]:
    """Run PCA analysis. Returns cached results."""
    results_dir = config.output_dir / "results" / "pca"
    pca_results = {}

    for target_key in data.get_target_keys():
        safe_key = _safe_key(target_key)
        pca_path = results_dir / safe_key
        if pca_path.exists():
            try:
                pca_results[target_key] = PCAResult.load(pca_path)
            except Exception:
                pass

    return pca_results


def compute_embeddings(
    data: ActivationData, config: GeoVizConfig, pca_results: dict[str, PCAResult]
) -> dict[str, EmbeddingResult]:
    """Compute embeddings. Returns cached results."""
    results_dir = config.output_dir / "results" / "embeddings"
    embedding_results = {}

    for target_key in data.get_target_keys():
        safe_key = _safe_key(target_key)
        embedding_path = results_dir / safe_key
        if embedding_path.exists():
            try:
                embedding_results[target_key] = EmbeddingResult.load(embedding_path)
            except Exception:
                pass

    return embedding_results
