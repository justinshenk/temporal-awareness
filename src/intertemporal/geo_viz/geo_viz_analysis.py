"""Analysis functions for geometric visualization."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from .geo_viz_config import GeoVizConfig
from .geo_viz_data import ActivationData, get_time_horizon_months

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeResult:
    """Results from linear probe analysis."""

    target_key: str
    r2_mean: float
    r2_std: float
    correlation: float
    predictions: np.ndarray
    coefficients: np.ndarray


@dataclass
class PCAResult:
    """Results from PCA analysis."""

    target_key: str
    explained_variance: np.ndarray
    components: np.ndarray
    transformed: np.ndarray
    pc_correlations: list[tuple[int, float, float]]  # (pc_idx, corr, pval)


@dataclass
class EmbeddingResult:
    """Results from dimensionality reduction."""

    target_key: str
    umap_embedding: np.ndarray | None
    tsne_embedding: np.ndarray | None
    pca_embedding: np.ndarray  # First 2 PCs


def linear_probe_analysis(
    data: ActivationData, config: GeoVizConfig
) -> dict[str, LinearProbeResult]:
    """Run linear probe analysis for time horizon decoding.

    Tests whether log(time_horizon) can be linearly decoded from activations.

    Args:
        data: Activation data
        config: Configuration

    Returns:
        Dictionary mapping target_key to LinearProbeResult
    """
    logger.info("Running linear probe analysis...")

    # Get target variable
    horizons = np.array([get_time_horizon_months(s) for s in data.samples])
    log_horizons = np.log10(horizons + 1)

    results = {}

    for target_key, X in data.activations.items():
        logger.info(f"  {target_key}: {X.shape}")

        # Cross-validated R² score
        ridge = Ridge(alpha=1.0)
        try:
            scores = cross_val_score(ridge, X, log_horizons, cv=5, scoring="r2")
            r2_mean = scores.mean()
            r2_std = scores.std()
        except Exception as e:
            logger.warning(f"    CV failed: {e}")
            r2_mean = 0.0
            r2_std = 0.0

        # Fit full model for predictions and coefficients
        ridge.fit(X, log_horizons)
        predictions = ridge.predict(X)
        correlation = np.corrcoef(predictions, log_horizons)[0, 1]

        results[target_key] = LinearProbeResult(
            target_key=target_key,
            r2_mean=r2_mean,
            r2_std=r2_std,
            correlation=correlation,
            predictions=predictions,
            coefficients=ridge.coef_,
        )

        logger.info(f"    R²={r2_mean:.3f}±{r2_std:.3f}, corr={correlation:.3f}")

    return results


def pca_correlation_analysis(
    data: ActivationData, config: GeoVizConfig
) -> dict[str, PCAResult]:
    """Run PCA and correlate components with time horizon.

    Args:
        data: Activation data
        config: Configuration

    Returns:
        Dictionary mapping target_key to PCAResult
    """
    logger.info("Running PCA correlation analysis...")

    horizons = np.array([get_time_horizon_months(s) for s in data.samples])
    log_horizons = np.log10(horizons + 1)

    results = {}

    for target_key, X in data.activations.items():
        logger.info(f"  {target_key}")

        # Fit PCA
        n_components = min(config.n_pca_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, random_state=config.seed)
        X_pca = pca.fit_transform(X)

        # Correlate each PC with log horizon
        pc_correlations = []
        for i in range(n_components):
            corr, pval = spearmanr(X_pca[:, i], log_horizons)
            pc_correlations.append((i, corr, pval))

        # Sort by absolute correlation
        pc_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Log top correlations
        top_pcs = pc_correlations[:3]
        for pc_idx, corr, pval in top_pcs:
            logger.info(f"    PC{pc_idx}: r={corr:.3f} (p={pval:.2e})")

        results[target_key] = PCAResult(
            target_key=target_key,
            explained_variance=pca.explained_variance_ratio_,
            components=pca.components_,
            transformed=X_pca,
            pc_correlations=pc_correlations,
        )

    return results


def compute_embeddings(
    data: ActivationData, config: GeoVizConfig, pca_results: dict[str, PCAResult]
) -> dict[str, EmbeddingResult]:
    """Compute 2D embeddings using UMAP, t-SNE, and PCA.

    Args:
        data: Activation data
        config: Configuration
        pca_results: PCA results (reuse transformed data)

    Returns:
        Dictionary mapping target_key to EmbeddingResult
    """
    logger.info("Computing embeddings...")

    results = {}

    for target_key, X in data.activations.items():
        logger.info(f"  {target_key}")

        pca_result = pca_results[target_key]
        X_pca = pca_result.transformed

        # Check for degenerate activations (constant or near-constant)
        variance = np.var(X_pca[:, :20])
        if variance < 1e-10:
            logger.warning(f"    Skipping embeddings: activations are constant (var={variance:.2e})")
            results[target_key] = EmbeddingResult(
                target_key=target_key,
                pca_embedding=X_pca[:, :2],
                umap_embedding=None,
                tsne_embedding=None,
            )
            continue

        # PCA 2D (first two components)
        pca_2d = X_pca[:, :2]

        # UMAP
        umap_2d = None
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                random_state=config.seed,
            )
            # Use PCA-reduced data for faster UMAP
            umap_2d = reducer.fit_transform(X_pca[:, :20])
            logger.info("    UMAP computed")
        except Exception as e:
            logger.warning(f"    UMAP failed: {e}")

        # t-SNE
        tsne_2d = None
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(X) // 4),
                random_state=config.seed,
            )
            tsne_2d = tsne.fit_transform(X_pca[:, :20])
            logger.info("    t-SNE computed")
        except Exception as e:
            logger.warning(f"    t-SNE failed: {e}")

        results[target_key] = EmbeddingResult(
            target_key=target_key,
            umap_embedding=umap_2d,
            tsne_embedding=tsne_2d,
            pca_embedding=pca_2d,
        )

    return results
