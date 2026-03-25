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
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .geometry_config import (
    GeometryConfig,
    ACTIVATION_DTYPE,
    ANALYSIS_GC_INTERVAL,
    MAX_STORED_PCA_COMPONENTS,
)
from .geometry_data import ActivationData, get_time_horizon_months

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
    """Results from PCA dimensionality reduction. Uses __slots__."""

    target_key: str
    pca_embedding: np.ndarray

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "pca_embedding.npy", self.pca_embedding.astype(ACTIVATION_DTYPE))
        with open(path / "metadata.json", "w") as f:
            json.dump({"target_key": self.target_key}, f)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingResult":
        """Load from disk."""
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        return cls(
            target_key=metadata["target_key"],
            pca_embedding=np.load(path / "pca_embedding.npy"),
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
    config: GeometryConfig,
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
    """Linear probe analysis for a single target.

    Uses StandardScaler + Ridge in a pipeline to avoid data leakage during CV.
    Without standardization, R² can be extremely negative (e.g., -25) because
    Ridge regression with unnormalized high-dimensional data can produce
    predictions far worse than predicting the mean.

    Note on R² values:
    - R² can be negative when model predictions are worse than predicting the mean
    - Values below -1 indicate severe numerical issues or poor fit
    - This is especially common at source positions where time horizon info
      may not be linearly decodable from activations
    """
    from sklearn.model_selection import KFold

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

    # Check target variance - if near-zero, linear probe won't work
    target_var = np.var(log_horizons)
    if target_var < 1e-10:
        logger.warning(f"  {target_key}: near-zero target variance ({target_var:.2e})")
        return LinearProbeResult(
            target_key=target_key,
            r2_mean=0.0,
            r2_std=0.0,
            correlation=0.0,
            predictions=np.full(n_samples, np.mean(log_horizons), dtype=ACTIVATION_DTYPE),
            coefficients=np.zeros(X.shape[1], dtype=ACTIVATION_DTYPE),
        )

    # Use pipeline to ensure StandardScaler is fit only on training folds during CV
    # Higher alpha (10.0) provides better regularization for high-dimensional data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])

    try:
        cv_folds = min(5, n_samples // 2)
        # Use KFold with shuffle to ensure balanced folds
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, log_horizons, cv=kfold, scoring="r2")
        r2_mean = float(scores.mean())
        r2_std = float(scores.std())

        # Log warning for extremely negative R² (indicates numerical issues)
        if r2_mean < -1.0:
            logger.warning(
                f"  {target_key}: R²={r2_mean:.2f} (extremely negative, "
                f"indicating poor linear fit or numerical issues)"
            )
    except Exception as e:
        logger.warning(f"  {target_key}: CV failed with {e}")
        r2_mean = 0.0
        r2_std = 0.0

    try:
        pipe.fit(X, log_horizons)
        predictions = pipe.predict(X).astype(ACTIVATION_DTYPE)
        correlation = float(np.corrcoef(predictions, log_horizons)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
        # Get coefficients from the ridge step (in scaled space)
        coefficients = pipe.named_steps["ridge"].coef_.astype(ACTIVATION_DTYPE)
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
    config: GeometryConfig,
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
    config: GeometryConfig,
) -> EmbeddingResult:
    """Compute PCA 2D embedding for a single target."""
    X_pca = pca_result.transformed
    n_samples = X_pca.shape[0]

    # Check for degenerate activations
    if X_pca.shape[1] < 2 or np.var(X_pca[:, :2]) < 1e-10:
        pca_2d = np.zeros((n_samples, 2), dtype=ACTIVATION_DTYPE)
    else:
        pca_2d = X_pca[:, :2].astype(ACTIVATION_DTYPE)

    return EmbeddingResult(
        target_key=target_key,
        pca_embedding=pca_2d,
    )


# =============================================================================
# Streaming Pipeline
# =============================================================================


def _safe_key(key: str) -> str:
    """Convert target key to safe directory name."""
    return key.replace("/", "_").replace("\\", "_")


def run_streaming_analysis(
    data: ActivationData,
    config: GeometryConfig,
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

    # Get target keys from data, filtered to only requested positions
    all_target_keys = data.get_target_keys()
    requested_positions = {t.position for t in config.targets}
    target_keys = [
        k for k in all_target_keys
        if any(k.endswith(f"_{pos}") for pos in requested_positions)
    ]
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
    data: ActivationData, config: GeometryConfig
) -> dict[str, LinearProbeResult]:
    """Run linear probe analysis. Uses streaming internally."""
    linear_results, _, _ = run_streaming_analysis(data, config)
    return linear_results


def pca_correlation_analysis(
    data: ActivationData, config: GeometryConfig
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
    data: ActivationData, config: GeometryConfig, pca_results: dict[str, PCAResult]
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


# =============================================================================
# Cross-Position Cosine Similarity Analysis
# =============================================================================


@dataclass(slots=True)
class CrossPositionSimilarityResult:
    """Results from cross-position cosine similarity analysis.

    Computes cosine similarity between PC0 direction vectors at source vs
    destination positions across layers. This reveals whether the temporal
    information direction at source correlates with the decision direction
    at destination.
    """

    layer: int
    component: str
    source_positions: list[str]
    dest_positions: list[str]
    # Matrix of similarities: [n_source, n_dest]
    similarity_matrix: np.ndarray
    # Best similarities
    best_source: str
    best_dest: str
    best_similarity: float
    # Mean similarity across all pairs
    mean_similarity: float

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "layer": self.layer,
            "component": self.component,
            "source_positions": self.source_positions,
            "dest_positions": self.dest_positions,
            "best_source": self.best_source,
            "best_dest": self.best_dest,
            "best_similarity": float(self.best_similarity),
            "mean_similarity": float(self.mean_similarity),
        }

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "similarity_matrix.npy", self.similarity_matrix.astype(ACTIVATION_DTYPE))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "CrossPositionSimilarityResult":
        """Load from disk."""
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        return cls(
            layer=metrics["layer"],
            component=metrics["component"],
            source_positions=metrics["source_positions"],
            dest_positions=metrics["dest_positions"],
            similarity_matrix=np.load(path / "similarity_matrix.npy"),
            best_source=metrics["best_source"],
            best_dest=metrics["best_dest"],
            best_similarity=metrics["best_similarity"],
            mean_similarity=metrics["mean_similarity"],
        )


def _parse_target_key(key: str) -> tuple[int, str, str] | None:
    """Parse target key to (layer, component, position)."""
    parts = key.split("_P")
    if len(parts) != 2:
        return None
    base = parts[0]
    position = parts[1]

    layer_match = re.match(r"L(\d+)_(.+)", base)
    if not layer_match:
        return None
    layer = int(layer_match.group(1))
    component = layer_match.group(2)

    return layer, component, position


def compute_cross_position_similarity(
    pca_results: dict[str, PCAResult],
    config: GeometryConfig,
) -> dict[str, CrossPositionSimilarityResult]:
    """Compute cosine similarity between PC0 directions at source vs destination.

    For each (layer, component) pair, computes the similarity between the
    PC0 direction at each source position and each destination position.

    Source positions: time_horizon, short_term_time, short_term_reward,
                     long_term_time, long_term_reward, source
    Dest positions: response, dest

    Returns:
        Dict mapping "L{layer}_{component}" to CrossPositionSimilarityResult
    """
    source_positions = {"time_horizon", "short_term_time", "short_term_reward",
                       "long_term_time", "long_term_reward", "source"}
    dest_positions = {"response", "dest"}

    # Group targets by (layer, component)
    layer_component_targets: dict[str, dict] = {}
    for key, pca_result in pca_results.items():
        parsed = _parse_target_key(key)
        if parsed is None:
            continue
        layer, component, position = parsed
        lc_key = f"L{layer}_{component}"

        if lc_key not in layer_component_targets:
            layer_component_targets[lc_key] = {"source": {}, "dest": {}, "layer": layer, "component": component}

        if position in source_positions:
            layer_component_targets[lc_key]["source"][position] = pca_result
        elif position in dest_positions:
            layer_component_targets[lc_key]["dest"][position] = pca_result

    results_dir = config.output_dir / "results" / "cross_position_similarity"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for lc_key, targets in layer_component_targets.items():
        source_targets = targets["source"]
        dest_targets = targets["dest"]

        if not source_targets or not dest_targets:
            continue

        safe_key = _safe_key(lc_key)
        result_path = results_dir / safe_key

        # Check cache
        if result_path.exists():
            try:
                results[lc_key] = CrossPositionSimilarityResult.load(result_path)
                continue
            except Exception:
                pass

        source_keys = sorted(source_targets.keys())
        dest_keys = sorted(dest_targets.keys())

        # Extract PC0 directions (normalized)
        source_directions = {}
        for pos, pca_result in source_targets.items():
            if pca_result.components.shape[0] > 0:
                direction = pca_result.components[0].copy()
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                source_directions[pos] = direction

        dest_directions = {}
        for pos, pca_result in dest_targets.items():
            if pca_result.components.shape[0] > 0:
                direction = pca_result.components[0].copy()
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                dest_directions[pos] = direction

        # Compute similarity matrix
        n_source = len(source_keys)
        n_dest = len(dest_keys)
        similarity_matrix = np.zeros((n_source, n_dest), dtype=ACTIVATION_DTYPE)

        for i, src_pos in enumerate(source_keys):
            for j, dst_pos in enumerate(dest_keys):
                if src_pos in source_directions and dst_pos in dest_directions:
                    src_dir = source_directions[src_pos]
                    dst_dir = dest_directions[dst_pos]
                    if len(src_dir) == len(dst_dir):
                        similarity_matrix[i, j] = abs(np.dot(src_dir, dst_dir))

        # Find best
        best_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        best_source = source_keys[best_idx[0]]
        best_dest = dest_keys[best_idx[1]]
        best_similarity = float(similarity_matrix[best_idx])
        mean_similarity = float(np.mean(similarity_matrix[similarity_matrix > 0])) if np.any(similarity_matrix > 0) else 0.0

        result = CrossPositionSimilarityResult(
            layer=targets["layer"],
            component=targets["component"],
            source_positions=source_keys,
            dest_positions=dest_keys,
            similarity_matrix=similarity_matrix,
            best_source=best_source,
            best_dest=best_dest,
            best_similarity=best_similarity,
            mean_similarity=mean_similarity,
        )

        result.save(result_path)
        results[lc_key] = result

    logger.info(f"Computed cross-position similarity for {len(results)} layer-component pairs")
    return results


# =============================================================================
# Continuous Time Horizon Probe at Source Positions
# =============================================================================


@dataclass(slots=True)
class ContinuousTimeProbeResult:
    """Results from continuous time horizon regression at source positions.

    Unlike the binary choice probe, this predicts the actual time_horizon_months
    value using a Ridge regression on activations at source positions.
    """

    target_key: str
    r2_mean: float
    r2_std: float
    correlation: float
    # Coefficients for interpretability
    coefficients: np.ndarray
    # Predictions vs actuals for plotting
    predictions: np.ndarray
    actuals: np.ndarray
    # Model parameters
    alpha: float

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "target_key": self.target_key,
            "r2_mean": float(self.r2_mean),
            "r2_std": float(self.r2_std),
            "correlation": float(self.correlation),
            "alpha": float(self.alpha),
            "n_samples": len(self.predictions),
        }

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "coefficients.npy", self.coefficients.astype(ACTIVATION_DTYPE))
        np.save(path / "predictions.npy", self.predictions.astype(ACTIVATION_DTYPE))
        np.save(path / "actuals.npy", self.actuals.astype(ACTIVATION_DTYPE))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "ContinuousTimeProbeResult":
        """Load from disk."""
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        return cls(
            target_key=metrics["target_key"],
            r2_mean=metrics["r2_mean"],
            r2_std=metrics["r2_std"],
            correlation=metrics["correlation"],
            coefficients=np.load(path / "coefficients.npy"),
            predictions=np.load(path / "predictions.npy"),
            actuals=np.load(path / "actuals.npy"),
            alpha=metrics.get("alpha", 1.0),
        )


def _get_time_horizons_months(data: ActivationData) -> np.ndarray:
    """Get time horizons in months for all samples."""
    return np.array([get_time_horizon_months(s) for s in data.samples], dtype=ACTIVATION_DTYPE)


def compute_continuous_time_probe(
    data: ActivationData,
    config: GeometryConfig,
    alpha: float = 1.0,
) -> dict[str, ContinuousTimeProbeResult]:
    """Run continuous time horizon regression on source positions.

    Unlike the existing log-transformed probe, this predicts the raw
    time_horizon_months value, which provides a cleaner interpretability
    for how the model encodes continuous time at source positions.

    Only runs on source positions (where the time horizon information
    actually appears in the prompt).

    Args:
        data: Activation data
        config: Pipeline config
        alpha: Ridge regression alpha parameter

    Returns:
        Dict mapping target_key to ContinuousTimeProbeResult
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    source_positions = {"time_horizon", "short_term_time", "short_term_reward",
                       "long_term_time", "long_term_reward", "source"}

    target_keys = data.get_target_keys()
    time_horizons = _get_time_horizons_months(data)

    results_dir = config.output_dir / "results" / "continuous_time_probe"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for target_key in target_keys:
        parsed = _parse_target_key(target_key)
        if parsed is None:
            continue

        _, _, position = parsed
        if position not in source_positions:
            continue

        safe_key = _safe_key(target_key)
        result_path = results_dir / safe_key

        # Check cache
        if result_path.exists():
            try:
                results[target_key] = ContinuousTimeProbeResult.load(result_path)
                continue
            except Exception:
                pass

        # Load activations
        try:
            X = data.load_target(target_key)
        except Exception as e:
            logger.warning(f"Failed to load {target_key}: {e}")
            continue

        n_samples = X.shape[0]
        target_horizons = time_horizons[:n_samples] if n_samples < len(time_horizons) else time_horizons

        if n_samples < 10:
            data.unload_target(target_key)
            continue

        # Use log-transformed target for better regression behavior
        log_horizons = np.log10(target_horizons + 1)

        # Use pipeline with StandardScaler to match the main linear probe
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        # Cross-validation
        try:
            cv_folds = min(5, n_samples // 2)
            scores = cross_val_score(pipe, X, log_horizons, cv=cv_folds, scoring="r2")
            r2_mean = float(scores.mean())
            r2_std = float(scores.std())
        except Exception:
            r2_mean = 0.0
            r2_std = 0.0

        # Full fit
        try:
            pipe.fit(X, log_horizons)
            predictions_log = pipe.predict(X).astype(ACTIVATION_DTYPE)
            # Convert back to months for storage
            predictions = (10 ** predictions_log - 1).astype(ACTIVATION_DTYPE)
            correlation = float(np.corrcoef(predictions_log, log_horizons)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
            coefficients = pipe.named_steps["ridge"].coef_.astype(ACTIVATION_DTYPE)
        except Exception:
            predictions = np.zeros(n_samples, dtype=ACTIVATION_DTYPE)
            correlation = 0.0
            coefficients = np.zeros(X.shape[1], dtype=ACTIVATION_DTYPE)

        result = ContinuousTimeProbeResult(
            target_key=target_key,
            r2_mean=r2_mean,
            r2_std=r2_std,
            correlation=correlation,
            coefficients=coefficients,
            predictions=predictions,
            actuals=target_horizons,
            alpha=alpha,
        )

        result.save(result_path)
        results[target_key] = result

        data.unload_target(target_key)

    logger.info(f"Computed continuous time probe for {len(results)} source targets")
    return results


# =============================================================================
# No-Horizon Projection Analysis
# =============================================================================


@dataclass(slots=True)
class NoHorizonProjectionResult:
    """Results from projecting no-horizon samples onto horizon-fitted PCA space."""

    target_key: str
    # Indices
    horizon_indices: np.ndarray  # Indices of samples WITH time horizon
    no_horizon_indices: np.ndarray  # Indices of samples WITHOUT time horizon
    # PCA projections (fitted on horizon samples only)
    horizon_projected: np.ndarray  # Shape: (n_horizon, n_components)
    no_horizon_projected: np.ndarray  # Shape: (n_no_horizon, n_components)
    # Centroids in PC space
    short_horizon_centroid: np.ndarray  # Centroid of short-horizon samples
    long_horizon_centroid: np.ndarray  # Centroid of long-horizon samples
    no_horizon_centroid: np.ndarray  # Centroid of no-horizon samples
    # Distance metrics
    dist_to_short: float  # Mean distance from no-horizon to short-horizon centroid
    dist_to_long: float  # Mean distance from no-horizon to long-horizon centroid
    bias_ratio: float  # dist_to_long / dist_to_short (>1 means closer to short)
    # PCA info
    explained_variance: np.ndarray
    components: np.ndarray
    # Per-sample distances for no-horizon samples
    no_horizon_dist_to_short: np.ndarray
    no_horizon_dist_to_long: np.ndarray
    # Horizon values for samples with horizons (for coloring)
    horizon_values_months: np.ndarray

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "target_key": self.target_key,
            "n_horizon_samples": len(self.horizon_indices),
            "n_no_horizon_samples": len(self.no_horizon_indices),
            "dist_to_short": float(self.dist_to_short),
            "dist_to_long": float(self.dist_to_long),
            "bias_ratio": float(self.bias_ratio),
            "short_horizon_centroid": self.short_horizon_centroid.tolist(),
            "long_horizon_centroid": self.long_horizon_centroid.tolist(),
            "no_horizon_centroid": self.no_horizon_centroid.tolist(),
        }

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "horizon_indices.npy", self.horizon_indices)
        np.save(path / "no_horizon_indices.npy", self.no_horizon_indices)
        np.save(path / "horizon_projected.npy", self.horizon_projected.astype(ACTIVATION_DTYPE))
        np.save(path / "no_horizon_projected.npy", self.no_horizon_projected.astype(ACTIVATION_DTYPE))
        np.save(path / "explained_variance.npy", self.explained_variance.astype(ACTIVATION_DTYPE))
        np.save(path / "components.npy", self.components.astype(ACTIVATION_DTYPE))
        np.save(path / "no_horizon_dist_to_short.npy", self.no_horizon_dist_to_short.astype(ACTIVATION_DTYPE))
        np.save(path / "no_horizon_dist_to_long.npy", self.no_horizon_dist_to_long.astype(ACTIVATION_DTYPE))
        np.save(path / "horizon_values_months.npy", self.horizon_values_months.astype(ACTIVATION_DTYPE))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "NoHorizonProjectionResult":
        """Load from disk."""
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        return cls(
            target_key=metrics["target_key"],
            horizon_indices=np.load(path / "horizon_indices.npy"),
            no_horizon_indices=np.load(path / "no_horizon_indices.npy"),
            horizon_projected=np.load(path / "horizon_projected.npy"),
            no_horizon_projected=np.load(path / "no_horizon_projected.npy"),
            short_horizon_centroid=np.array(metrics["short_horizon_centroid"]),
            long_horizon_centroid=np.array(metrics["long_horizon_centroid"]),
            no_horizon_centroid=np.array(metrics["no_horizon_centroid"]),
            dist_to_short=metrics["dist_to_short"],
            dist_to_long=metrics["dist_to_long"],
            bias_ratio=metrics["bias_ratio"],
            explained_variance=np.load(path / "explained_variance.npy"),
            components=np.load(path / "components.npy"),
            no_horizon_dist_to_short=np.load(path / "no_horizon_dist_to_short.npy"),
            no_horizon_dist_to_long=np.load(path / "no_horizon_dist_to_long.npy"),
            horizon_values_months=np.load(path / "horizon_values_months.npy"),
        )


def _split_horizon_indices(data: ActivationData) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into has_horizon and no_horizon groups."""
    has_horizon = []
    no_horizon = []
    for i, sample in enumerate(data.samples):
        if sample.prompt.time_horizon is None:
            no_horizon.append(i)
        else:
            has_horizon.append(i)
    return np.array(has_horizon, dtype=np.int32), np.array(no_horizon, dtype=np.int32)


def _get_horizon_months_array(data: ActivationData) -> np.ndarray:
    """Get time horizon in months for each sample. Returns actual value or NaN for no-horizon."""
    horizons = []
    for sample in data.samples:
        if sample.prompt.time_horizon is None:
            horizons.append(np.nan)
        else:
            horizons.append(sample.prompt.time_horizon.to_months())
    return np.array(horizons, dtype=ACTIVATION_DTYPE)


def analyze_no_horizon_projection(
    target_key: str,
    X: np.ndarray,
    data: ActivationData,
    config: GeometryConfig,
    short_threshold_months: float = 12.0,  # <1 year = short horizon
    long_threshold_months: float = 60.0,  # >5 years = long horizon
) -> NoHorizonProjectionResult | None:
    """Analyze where no-horizon samples project in PCA space fitted on horizon samples.

    Args:
        target_key: Target identifier
        X: Activation matrix (n_samples, n_features)
        data: ActivationData with sample information
        config: GeometryConfig
        short_threshold_months: Threshold for "short" horizon classification
        long_threshold_months: Threshold for "long" horizon classification

    Returns:
        NoHorizonProjectionResult or None if insufficient data
    """
    n_samples = X.shape[0]
    horizon_indices, no_horizon_indices = _split_horizon_indices(data)

    # Limit to available samples
    horizon_indices = horizon_indices[horizon_indices < n_samples]
    no_horizon_indices = no_horizon_indices[no_horizon_indices < n_samples]

    if len(horizon_indices) < 10 or len(no_horizon_indices) < 2:
        logger.debug(f"Insufficient data for no-horizon analysis: "
                    f"{len(horizon_indices)} horizon, {len(no_horizon_indices)} no-horizon")
        return None

    # Get horizon samples
    X_horizon = X[horizon_indices]
    X_no_horizon = X[no_horizon_indices]

    # Fit PCA on horizon samples only
    n_components = min(config.n_pca_components, len(horizon_indices) - 1, X.shape[1])
    if n_components < 2:
        return None

    pca = PCA(n_components=n_components, random_state=config.seed)
    horizon_projected = pca.fit_transform(X_horizon).astype(ACTIVATION_DTYPE)

    # Project no-horizon samples into this space
    no_horizon_projected = pca.transform(X_no_horizon).astype(ACTIVATION_DTYPE)

    # Get horizons for horizon samples
    horizons_months = _get_horizon_months_array(data)
    horizon_values = horizons_months[horizon_indices]

    # Classify horizon samples as short/long
    short_mask = horizon_values < short_threshold_months
    long_mask = horizon_values > long_threshold_months

    if short_mask.sum() < 3 or long_mask.sum() < 3:
        logger.debug(f"Insufficient short/long samples: {short_mask.sum()} short, {long_mask.sum()} long")
        return None

    # Compute centroids (using first 2 PCs for visualization, but full space for distances)
    n_pcs_for_centroid = min(10, n_components)  # Use top 10 PCs for distance calculation
    short_horizon_centroid = horizon_projected[short_mask, :n_pcs_for_centroid].mean(axis=0)
    long_horizon_centroid = horizon_projected[long_mask, :n_pcs_for_centroid].mean(axis=0)
    no_horizon_centroid = no_horizon_projected[:, :n_pcs_for_centroid].mean(axis=0)

    # Compute distances for each no-horizon sample
    no_horizon_dist_to_short = np.linalg.norm(
        no_horizon_projected[:, :n_pcs_for_centroid] - short_horizon_centroid,
        axis=1
    ).astype(ACTIVATION_DTYPE)
    no_horizon_dist_to_long = np.linalg.norm(
        no_horizon_projected[:, :n_pcs_for_centroid] - long_horizon_centroid,
        axis=1
    ).astype(ACTIVATION_DTYPE)

    # Mean distances
    dist_to_short = float(no_horizon_dist_to_short.mean())
    dist_to_long = float(no_horizon_dist_to_long.mean())
    bias_ratio = dist_to_long / (dist_to_short + 1e-8)

    return NoHorizonProjectionResult(
        target_key=target_key,
        horizon_indices=horizon_indices,
        no_horizon_indices=no_horizon_indices,
        horizon_projected=horizon_projected,
        no_horizon_projected=no_horizon_projected,
        short_horizon_centroid=short_horizon_centroid.astype(ACTIVATION_DTYPE),
        long_horizon_centroid=long_horizon_centroid.astype(ACTIVATION_DTYPE),
        no_horizon_centroid=no_horizon_centroid.astype(ACTIVATION_DTYPE),
        dist_to_short=dist_to_short,
        dist_to_long=dist_to_long,
        bias_ratio=bias_ratio,
        explained_variance=pca.explained_variance_ratio_.astype(ACTIVATION_DTYPE),
        components=pca.components_.astype(ACTIVATION_DTYPE),
        no_horizon_dist_to_short=no_horizon_dist_to_short,
        no_horizon_dist_to_long=no_horizon_dist_to_long,
        horizon_values_months=horizon_values.astype(ACTIVATION_DTYPE),
    )


def run_no_horizon_analysis(
    data: ActivationData,
    config: GeometryConfig,
) -> dict[str, NoHorizonProjectionResult]:
    """Run no-horizon projection analysis on all targets.

    Returns:
        Dictionary mapping target_key -> NoHorizonProjectionResult
    """
    results_dir = config.output_dir / "results" / "no_horizon"
    results_dir.mkdir(parents=True, exist_ok=True)

    target_keys = data.get_target_keys()
    n_targets = len(target_keys)

    logger.info(f"Running no-horizon projection analysis on {n_targets} targets...")

    results = {}

    for i, target_key in enumerate(target_keys):
        if i % 20 == 0:
            logger.info(f"  Analyzing target {i}/{n_targets}: {target_key}")

        safe_key = _safe_key(target_key)
        result_path = results_dir / safe_key

        # Check cache
        if result_path.exists():
            try:
                results[target_key] = NoHorizonProjectionResult.load(result_path)
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
        result = analyze_no_horizon_projection(target_key, X, data, config)
        if result is not None:
            result.save(result_path)
            results[target_key] = result

        # Unload
        data.unload_target(target_key)

        if i % ANALYSIS_GC_INTERVAL == 0:
            gc.collect()

    logger.info(f"No-horizon analysis complete: {len(results)} targets with results")

    return results
