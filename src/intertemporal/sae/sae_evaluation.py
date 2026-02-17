"""Evaluation metrics and visualization for SAE clustering."""

import json
from pathlib import Path

import numpy as np
import torch
import pacmap
import pacmap.pacmap as _pm
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .sae_plots import plot_cluster_distribution, plot_embedding, plot_gradient_embedding


# =============================================================================
# Metrics
# =============================================================================


def compute_purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Fraction of samples in majority class per cluster."""
    n = len(cluster_labels)
    if n == 0:
        return 0.0
    correct = 0
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        if mask.sum() > 0:
            labels_in_cluster = true_labels[mask]
            correct += (labels_in_cluster == np.bincount(labels_in_cluster).argmax()).sum()
    return correct / n


def compute_cluster_balance(cluster_dist: list[int]) -> float:
    """Normalized entropy of cluster size distribution."""
    total = sum(cluster_dist)
    if total == 0:
        return 0.0
    non_empty = [c for c in cluster_dist if c > 0]
    if len(non_empty) <= 1:
        return 0.0
    probs = np.array(non_empty) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(non_empty))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_clustering_metrics(labels: np.ndarray, n_clusters: int, sentences: list[dict]) -> dict:
    """Compute NMI, ARI, purity for horizon and choice labels."""
    horizon = np.array([s["time_horizon_bucket"] for s in sentences])
    choice = np.array([s["llm_choice"] for s in sentences])

    valid_h = horizon >= 0
    valid_c = choice >= 0

    result = {
        "horizon_nmi": normalized_mutual_info_score(horizon[valid_h], labels[valid_h]) if valid_h.sum() > 0 else 0.0,
        "horizon_ari": adjusted_rand_score(horizon[valid_h], labels[valid_h]) if valid_h.sum() > 0 else 0.0,
        "horizon_purity": compute_purity(labels[valid_h], horizon[valid_h]) if valid_h.sum() > 0 else 0.0,
        "choice_nmi": normalized_mutual_info_score(choice[valid_c], labels[valid_c]) if valid_c.sum() > 0 else 0.0,
        "choice_ari": adjusted_rand_score(choice[valid_c], labels[valid_c]) if valid_c.sum() > 0 else 0.0,
        "choice_purity": compute_purity(labels[valid_c], choice[valid_c]) if valid_c.sum() > 0 else 0.0,
    }

    cluster_dist = np.bincount(labels, minlength=n_clusters).tolist()
    result["cluster_balance"] = compute_cluster_balance(cluster_dist)
    result["active_clusters"] = sum(1 for c in cluster_dist if c > 0)
    result["n_sentences"] = len(sentences)
    result["cluster_distribution"] = cluster_dist

    return result


# =============================================================================
# Embeddings
# =============================================================================


def _patch_pacmap_annoy():
    """Monkey-patch PaCMAP to use sklearn NearestNeighbors instead of annoy (broken on Apple Silicon)."""
    _original = _pm.generate_pair

    def _patched(X, n_neighbors, n_MN, n_FP, distance="euclidean", verbose=True):
        n = X.shape[0]
        n_neighbors_extra = min(n_neighbors + 50, n - 1)
        n_neighbors = min(n_neighbors, n - 1)
        n_FP = min(n_FP, n - 1)
        n_MN = min(n_MN, n - 1)

        metric = "minkowski" if distance == "euclidean" else distance
        nn = NearestNeighbors(n_neighbors=n_neighbors_extra + 1, metric=metric, algorithm="auto")
        nn.fit(X)
        knn_distances, indices = nn.kneighbors(X)
        nbrs = indices[:, 1:].astype(np.int32)
        knn_distances = knn_distances[:, 1:].astype(np.float32)

        sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
        scaled_dist = _pm.scale_dist(knn_distances, sig, nbrs)
        pair_neighbors = _pm.sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)

        option = _pm.distance_to_option(distance=distance)
        if _pm._RANDOM_STATE is None:
            pair_MN = _pm.sample_MN_pair(X, n_MN, option)
            pair_FP = _pm.sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
        else:
            pair_MN = _pm.sample_MN_pair_deterministic(X, n_MN, _pm._RANDOM_STATE, option)
            pair_FP = _pm.sample_FP_pair_deterministic(X, pair_neighbors, n_neighbors, n_FP, _pm._RANDOM_STATE)

        return pair_neighbors, pair_MN, pair_FP, None

    _pm.generate_pair = _patched


_patch_pacmap_annoy()


def compute_embeddings(features: np.ndarray) -> dict[str, np.ndarray]:
    """Compute 2D embeddings via UMAP, t-SNE, and PaCMAP."""
    n = len(features)
    embeddings = {}

    try:
        embeddings["umap"] = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), random_state=42).fit_transform(features)
    except Exception:
        pass

    try:
        embeddings["tsne"] = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1)).fit_transform(features)
    except Exception:
        pass

    try:
        embeddings["pacmap"] = pacmap.PaCMAP(n_components=2, n_neighbors=min(10, n - 1), random_state=42).fit_transform(features)
    except Exception:
        pass

    return embeddings


# =============================================================================
# Colorings
# =============================================================================

CHOICE_NAMES = {-1: "unknown", 0: "short-term", 1: "long-term"}


def format_horizon(months: float | None) -> str:
    if months is None:
        return "none"
    if months < 1:
        return f"{round(months * 30.44)}d"
    if months < 12:
        return f"{round(months)}mo"
    years = months / 12
    return f"{int(years)}y" if years == int(years) else f"{years:.1f}y"


def format_formatting_id(fid: int | None) -> str:
    if fid is None:
        return "none"
    return f"F{abs(fid) % 1000}"


def format_formatting_sign(fid: int | None) -> str:
    if fid is None:
        return "none"
    if fid > 0:
        return "positive"
    if fid < 0:
        return "negative"
    return "zero"


def format_bool(val: bool | None) -> str:
    if val is None:
        return "none"
    return "yes" if val else "no"


def build_colorings(sentences: list[dict], labels: np.ndarray) -> dict[str, list[str]]:
    """Build categorical label arrays for visualization."""
    return {
        "cluster": [f"C{c}" for c in labels],
        "choice": [CHOICE_NAMES.get(s["llm_choice"], "unknown") for s in sentences],
        "time_horizon": [format_horizon(s["time_horizon_months"]) for s in sentences],
        "source": [s["source"] for s in sentences],
        "section": [s["section"] for s in sentences],
        "formatting_id": [format_formatting_id(s.get("formatting_id")) for s in sentences],
        "formatting_id_sign": [format_formatting_sign(s.get("formatting_id")) for s in sentences],
        "matches_rational": [format_bool(s.get("matches_rational")) for s in sentences],
        "matches_associated": [format_bool(s.get("matches_associated")) for s in sentences],
    }


def build_gradient_colorings(sentences: list[dict]) -> dict[str, list[float]]:
    """Build continuous value arrays for gradient visualization."""
    return {
        "choice_time_months": [s["llm_choice_time_months"] for s in sentences],
        "time_horizon_months": [s["time_horizon_months"] for s in sentences],
    }


# =============================================================================
# Visualization
# =============================================================================


def generate_plots(
    analysis_dir: Path,
    features: np.ndarray,
    sentences: list[dict],
    labels: np.ndarray,
    title_prefix: str = "",
) -> None:
    """Generate all embedding plots with categorical and gradient colorings."""
    embeddings = compute_embeddings(features)

    # Categorical colorings
    for name, color_labels in build_colorings(sentences, labels).items():
        coloring_dir = analysis_dir / name
        coloring_dir.mkdir(parents=True, exist_ok=True)
        for method, coords in embeddings.items():
            title = f"{method.upper()}{' — ' + title_prefix if title_prefix else ''} — {name}"
            try:
                plot_embedding(coords, color_labels, title, coloring_dir / f"{method}.png")
            except Exception:
                pass

    # Gradient colorings
    for name, values in build_gradient_colorings(sentences).items():
        coloring_dir = analysis_dir / name
        coloring_dir.mkdir(parents=True, exist_ok=True)
        for method, coords in embeddings.items():
            title = f"{method.upper()}{' — ' + title_prefix if title_prefix else ''} — {name}"
            try:
                plot_gradient_embedding(coords, values, title, coloring_dir / f"{method}.png")
            except Exception:
                pass


# =============================================================================
# SAE Cluster Analysis
# =============================================================================


def cluster_analysis(sentences: list[dict], features: torch.Tensor, analysis_dir: str) -> dict:
    """Run cluster analysis on SAE features."""
    path = Path(analysis_dir)
    path.mkdir(parents=True, exist_ok=True)

    labels = features.argmax(dim=1).cpu().numpy()
    n_clusters = features.shape[1]

    result = compute_clustering_metrics(labels, n_clusters, sentences)

    with open(path / "cluster_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        plot_cluster_distribution(result["cluster_distribution"], f"Cluster Distribution ({len(sentences)} sentences)", path / "cluster_distribution.png")
    except Exception:
        pass

    try:
        generate_plots(path, features.cpu().numpy(), sentences, labels)
    except Exception:
        pass

    print(f"    Horizon NMI: {result['horizon_nmi']:.4f}, Choice NMI: {result['choice_nmi']:.4f}, Active: {result['active_clusters']}/{n_clusters}")
    return result


# =============================================================================
# Baseline Clustering
# =============================================================================


BASELINE_METHODS = {
    "spherical_kmeans": ("Spherical KMeans", lambda X, k: _spherical_kmeans(X, k)),
    "agglomerative": ("Agglomerative (cosine)", lambda X, k: _agglomerative(X, k)),
    "pca_kmeans": ("PCA + KMeans", lambda X, k: _pca_kmeans(X, k)),
}


def _spherical_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / (norms + 1e-8)
    return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X_unit)


def _agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)


def _pca_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    n_components = min(X.shape[0], X.shape[1], 100)
    X_reduced = PCA(n_components=n_components, random_state=42).fit_transform(X)
    return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X_reduced)


def baseline_cluster_analysis(X: np.ndarray, sentences: list[dict], n_clusters: int, analysis_dir: str) -> dict:
    """Run baseline clustering methods."""
    path = Path(analysis_dir)
    path.mkdir(parents=True, exist_ok=True)

    results = {}

    for method_key, (label, runner) in BASELINE_METHODS.items():
        print(f"    Baseline: {label} (k={n_clusters})...", end=" ", flush=True)

        try:
            labels = runner(X, n_clusters)
        except Exception as e:
            print(f"FAILED: {e}")
            results[method_key] = {"error": str(e)}
            continue

        metrics = compute_clustering_metrics(labels, n_clusters, sentences)
        metrics["method"] = method_key
        results[method_key] = metrics

        print(f"Horizon NMI: {metrics['horizon_nmi']:.4f}, Choice NMI: {metrics['choice_nmi']:.4f}, Active: {metrics['active_clusters']}/{n_clusters}")

        method_dir = path / method_key
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_cluster_distribution(metrics["cluster_distribution"], f"{label} ({len(sentences)} sentences)", method_dir / "cluster_distribution.png")
        except Exception:
            pass

        try:
            generate_plots(method_dir, X, sentences, labels, title_prefix=label)
        except Exception:
            pass

    with open(path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
