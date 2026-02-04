"""Evaluation metrics for SAE clustering quality."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .sae import SAE
from .data import Sentence
from .inference import normalize_activations_raw
from .utils import get_device


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
            correct += (
                labels_in_cluster == np.bincount(labels_in_cluster).argmax()
            ).sum()
    return correct / n


def compute_cluster_balance(cluster_dist: list[int]) -> float:
    """Normalised entropy of cluster size distribution."""
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


# ── Pipeline analysis helpers ────────────────────────────────────────────────


def get_sentences(
    samples: list[dict],
    activations: list[dict],
) -> list[dict]:
    """Flatten samples + activations into a list of sentence dicts.

    Each dict has: sentence (Sentence metadata), sample_id,
    time_horizon_bucket, llm_choice, activations ({layer_key: ndarray}).
    """
    result = []
    for sample_idx, sample in enumerate(samples):
        raw_sentences = sample.get("sentences", [])
        if sample_idx >= len(activations) or not activations[sample_idx]:
            continue
        sample_acts = activations[sample_idx]
        for sentence_idx, raw in enumerate(raw_sentences):
            if sentence_idx not in sample_acts:
                continue
            sentence = Sentence.from_dict(raw)
            result.append(
                {
                    "text": sentence.text,
                    "source": sentence.source,
                    "section": sentence.section,
                    "sample_id": sample.get("sample_id"),
                    "time_horizon_bucket": sample.get("time_horizon_bucket", -1),
                    "time_horizon_months": sample.get("time_horizon_months"),
                    "llm_choice": sample.get("llm_choice", -1),
                    "activations": sample_acts[sentence_idx],
                }
            )
    return result


def get_sae_features_for_sentences(
    sae_model: SAE,
    sentences: list[dict],
    filter_sentence=None,
) -> tuple[torch.Tensor, list[dict]]:
    """Extract SAE feature activations for a list of sentence dicts.

    Uses the SAE's stored activation_mean for normalization.
    Returns (features_tensor, filtered_sentences) — only sentences that
    matched the filter and had activations for the SAE's layer.
    """
    layer_key = f"layer_{sae_model.layer}"

    vectors = []
    kept = []
    for s in sentences:
        if filter_sentence is not None and not filter_sentence(Sentence.from_dict(s)):
            continue
        act = s["activations"].get(layer_key)
        if act is not None:
            vectors.append(act)
            kept.append(s)

    if not vectors:
        raise ValueError(f"No activations found for layer {sae_model.layer}")

    X_raw = np.stack(vectors, axis=0)
    activation_mean = sae_model.activation_mean.cpu().numpy()
    X_norm = normalize_activations_raw(X_raw, activation_mean)

    device = get_device()
    X_tensor = torch.from_numpy(X_norm).float().to(device)
    sae_model = sae_model.to(device)

    with torch.no_grad():
        features = sae_model.get_all_activations(X_tensor)

    return features, kept


CHOICE_NAMES = {-1: "unknown", 0: "short-term", 1: "long-term"}


def _format_horizon(months: float | None) -> str:
    """Convert time_horizon_months to a readable label."""
    if months is None:
        return "none"
    days = months * 30.44
    if days < 14:
        return f"{round(days)}d"
    weeks = months * 4.345
    if months < 1:
        return f"{round(weeks)}w"
    if months < 12:
        return f"{round(months)}mo"
    years = months / 12
    if years == int(years):
        return f"{int(years)}y"
    return f"{years:.1f}y"


def _patch_pacmap_annoy():
    """Monkey-patch PaCMAP to use sklearn NearestNeighbors instead of annoy.

    Annoy is broken on Apple Silicon (get_nns_by_item returns empty).
    """
    import pacmap.pacmap as _pm
    from sklearn.neighbors import NearestNeighbors

    _original_generate_pair = _pm.generate_pair

    def _patched_generate_pair(
        X, n_neighbors, n_MN, n_FP, distance="euclidean", verbose=True
    ):
        n, dim = X.shape
        n_neighbors_extra = min(n_neighbors + 50, n - 1)
        n_neighbors = min(n_neighbors, n - 1)
        n_FP = min(n_FP, n - 1)
        n_MN = min(n_MN, n - 1)

        metric = "minkowski" if distance == "euclidean" else distance
        nn = NearestNeighbors(
            n_neighbors=n_neighbors_extra + 1, metric=metric, algorithm="auto"
        )
        nn.fit(X)
        knn_distances, indices = nn.kneighbors(X)
        nbrs = indices[:, 1:].astype(np.int32)
        knn_distances = knn_distances[:, 1:].astype(np.float32)

        sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
        scaled_dist = _pm.scale_dist(knn_distances, sig, nbrs)
        pair_neighbors = _pm.sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)

        if _pm._RANDOM_STATE is None:
            option = _pm.distance_to_option(distance=distance)
            pair_MN = _pm.sample_MN_pair(X, n_MN, option)
            pair_FP = _pm.sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
        else:
            option = _pm.distance_to_option(distance=distance)
            pair_MN = _pm.sample_MN_pair_deterministic(
                X, n_MN, _pm._RANDOM_STATE, option
            )
            pair_FP = _pm.sample_FP_pair_deterministic(
                X, pair_neighbors, n_neighbors, n_FP, _pm._RANDOM_STATE
            )

        return pair_neighbors, pair_MN, pair_FP, None

    _pm.generate_pair = _patched_generate_pair


# Apply patch once at import time
_patch_pacmap_annoy()


def _compute_embeddings(features_np: np.ndarray) -> dict[str, np.ndarray]:
    """Compute 2D embeddings via UMAP, t-SNE, and PaCMAP."""
    from sklearn.manifold import TSNE
    import umap
    import pacmap

    n = len(features_np)
    embeddings = {}

    try:
        embeddings["umap"] = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            random_state=42,
        ).fit_transform(features_np)
    except Exception as e:
        print(f"    Warning: UMAP failed: {e}")

    try:
        embeddings["tsne"] = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, n - 1),
        ).fit_transform(features_np)
    except Exception as e:
        print(f"    Warning: t-SNE failed: {e}")

    try:
        embeddings["pacmap"] = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=min(10, n - 1),
            random_state=42,
        ).fit_transform(features_np)
    except Exception as e:
        print(f"    Warning: PaCMAP failed (n={n}): {e}")

    return embeddings


def _build_colorings(
    sentences: list[dict],
    cluster_labels: np.ndarray,
) -> dict[str, list[str]]:
    """Build label arrays for each coloring variant."""
    return {
        "cluster": [f"C{c}" for c in cluster_labels],
        "choice": [CHOICE_NAMES.get(s["llm_choice"], "unknown") for s in sentences],
        "time_horizon": [_format_horizon(s["time_horizon_months"]) for s in sentences],
        "source": [s["source"] for s in sentences],
        "section": [s["section"] for s in sentences],
    }


def cluster_analysis(
    sentences: list[dict],
    sentence_features: torch.Tensor,
    analysis_dir: str,
) -> dict:
    """Run cluster analysis on SAE features and save results.

    Computes NMI, ARI, purity for horizon and choice labels.
    Generates UMAP, t-SNE, and PaCMAP plots with multiple colorings.
    """
    from .plots import plot_cluster_distribution, plot_embedding

    analysis_path = Path(analysis_dir)
    analysis_path.mkdir(parents=True, exist_ok=True)

    cluster_labels = sentence_features.argmax(dim=1).cpu().numpy()
    n_clusters = sentence_features.shape[1]

    horizon_labels = np.array([s["time_horizon_bucket"] for s in sentences])
    choice_labels = np.array([s["llm_choice"] for s in sentences])

    # Horizon metrics
    valid_h = horizon_labels >= 0
    if valid_h.sum() > 0:
        horizon_nmi = normalized_mutual_info_score(
            horizon_labels[valid_h], cluster_labels[valid_h]
        )
        horizon_ari = adjusted_rand_score(
            horizon_labels[valid_h], cluster_labels[valid_h]
        )
        horizon_purity = compute_purity(
            cluster_labels[valid_h], horizon_labels[valid_h]
        )
    else:
        horizon_nmi = horizon_ari = horizon_purity = 0.0

    # Choice metrics
    valid_c = choice_labels >= 0
    if valid_c.sum() > 0:
        choice_nmi = normalized_mutual_info_score(
            choice_labels[valid_c], cluster_labels[valid_c]
        )
        choice_ari = adjusted_rand_score(
            choice_labels[valid_c], cluster_labels[valid_c]
        )
        choice_purity = compute_purity(cluster_labels[valid_c], choice_labels[valid_c])
    else:
        choice_nmi = choice_ari = choice_purity = 0.0

    cluster_dist = np.bincount(cluster_labels, minlength=n_clusters).tolist()
    active_clusters = sum(1 for c in cluster_dist if c > 0)

    result = {
        "horizon_nmi": horizon_nmi,
        "horizon_ari": horizon_ari,
        "horizon_purity": horizon_purity,
        "choice_nmi": choice_nmi,
        "choice_ari": choice_ari,
        "choice_purity": choice_purity,
        "cluster_balance": compute_cluster_balance(cluster_dist),
        "active_clusters": active_clusters,
        "n_sentences": len(sentences),
        "cluster_distribution": cluster_dist,
    }

    # Save metrics
    with open(analysis_path / "cluster_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Cluster distribution bar chart
    try:
        plot_cluster_distribution(
            cluster_dist,
            f"Cluster Distribution ({len(sentences)} sentences)",
            analysis_path / "cluster_distribution.png",
        )
    except Exception as e:
        print(f"    Warning: Could not generate cluster plot: {e}")

    # Embedding plots: {coloring}/{method}.png
    try:
        features_np = sentence_features.cpu().numpy()
        embeddings = _compute_embeddings(features_np)
        colorings = _build_colorings(sentences, cluster_labels)

        for coloring_name, labels in colorings.items():
            coloring_dir = analysis_path / coloring_name
            coloring_dir.mkdir(parents=True, exist_ok=True)
            for method, coords in embeddings.items():
                title = f"{method.upper()} — {coloring_name}"
                path = coloring_dir / f"{method}.png"
                try:
                    plot_embedding(coords, labels, title, path)
                except Exception as e:
                    print(f"    Warning: {coloring_name}/{method} plot failed: {e}")
    except Exception as e:
        print(f"    Warning: Embedding plots failed: {e}")

    print(
        f"    Horizon NMI: {horizon_nmi:.4f}, Choice NMI: {choice_nmi:.4f}, "
        f"Active: {active_clusters}/{n_clusters}"
    )

    return result


# ── Baseline clustering ──────────────────────────────────────────────────────


def get_normalized_vectors_for_sentences(
    layer: int,
    activation_mean: np.ndarray,
    sentences: list[dict],
    filter_sentence=None,
) -> tuple[np.ndarray, list[dict]]:
    """Extract normalized activation vectors for a layer from sentence dicts.

    Returns (X_norm, filtered_sentences).
    """
    layer_key = f"layer_{layer}"
    vectors = []
    kept = []
    for s in sentences:
        if filter_sentence is not None and not filter_sentence(Sentence.from_dict(s)):
            continue
        act = s["activations"].get(layer_key)
        if act is not None:
            vectors.append(act)
            kept.append(s)

    if not vectors:
        raise ValueError(f"No activations found for layer {layer}")

    X_raw = np.stack(vectors, axis=0)
    X_norm = normalize_activations_raw(X_raw, activation_mean)
    return X_norm, kept


BASELINE_METHODS = {
    "spherical_kmeans": "Spherical KMeans",
    "agglomerative": "Agglomerative (cosine)",
    "pca_kmeans": "PCA + KMeans",
}


def _run_spherical_kmeans(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize then KMeans."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / (norms + 1e-8)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X_unit)
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    return labels, centers


def _run_agglomerative(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative clustering with cosine metric and average linkage."""
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    labels = agg.fit_predict(X)
    centers = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        mask = labels == i
        if mask.any():
            centers[i] = X[mask].mean(axis=0)
    return labels, centers


def _run_pca_kmeans(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA to min(100, d) dims then KMeans."""
    n_components = min(X.shape[0], X.shape[1], 100)
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X_reduced)
    centers = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        mask = labels == i
        if mask.any():
            centers[i] = X[mask].mean(axis=0)
    return labels, centers


_BASELINE_RUNNERS = {
    "spherical_kmeans": _run_spherical_kmeans,
    "agglomerative": _run_agglomerative,
    "pca_kmeans": _run_pca_kmeans,
}


def _compute_clustering_metrics(
    cluster_labels: np.ndarray,
    n_clusters: int,
    sentences: list[dict],
) -> dict:
    """Compute the same metrics as cluster_analysis for a given set of labels."""
    horizon_labels = np.array([s["time_horizon_bucket"] for s in sentences])
    choice_labels = np.array([s["llm_choice"] for s in sentences])

    valid_h = horizon_labels >= 0
    if valid_h.sum() > 0:
        horizon_nmi = normalized_mutual_info_score(
            horizon_labels[valid_h], cluster_labels[valid_h]
        )
        horizon_ari = adjusted_rand_score(
            horizon_labels[valid_h], cluster_labels[valid_h]
        )
        horizon_purity = compute_purity(cluster_labels[valid_h], horizon_labels[valid_h])
    else:
        horizon_nmi = horizon_ari = horizon_purity = 0.0

    valid_c = choice_labels >= 0
    if valid_c.sum() > 0:
        choice_nmi = normalized_mutual_info_score(
            choice_labels[valid_c], cluster_labels[valid_c]
        )
        choice_ari = adjusted_rand_score(
            choice_labels[valid_c], cluster_labels[valid_c]
        )
        choice_purity = compute_purity(cluster_labels[valid_c], choice_labels[valid_c])
    else:
        choice_nmi = choice_ari = choice_purity = 0.0

    cluster_dist = np.bincount(cluster_labels, minlength=n_clusters).tolist()
    active_clusters = sum(1 for c in cluster_dist if c > 0)

    return {
        "horizon_nmi": horizon_nmi,
        "horizon_ari": horizon_ari,
        "horizon_purity": horizon_purity,
        "choice_nmi": choice_nmi,
        "choice_ari": choice_ari,
        "choice_purity": choice_purity,
        "cluster_balance": compute_cluster_balance(cluster_dist),
        "active_clusters": active_clusters,
        "n_sentences": len(sentences),
        "cluster_distribution": cluster_dist,
    }


def baseline_cluster_analysis(
    X_norm: np.ndarray,
    sentences: list[dict],
    n_clusters: int,
    analysis_dir: str,
) -> dict:
    """Run baseline clustering methods and save results to analysis_dir/cluster_baseline/.

    Returns a dict mapping method name to its metrics dict.
    """
    from .plots import plot_cluster_distribution, plot_embedding

    base_path = Path(analysis_dir) / "cluster_baseline"
    base_path.mkdir(parents=True, exist_ok=True)

    all_methods_results = {}

    for method_key, runner in _BASELINE_RUNNERS.items():
        method_label = BASELINE_METHODS[method_key]
        print(f"    Baseline: {method_label} (k={n_clusters})...", end=" ", flush=True)

        try:
            labels, centers = runner(X_norm, n_clusters)
        except Exception as e:
            print(f"FAILED: {e}")
            all_methods_results[method_key] = {"error": str(e)}
            continue

        metrics = _compute_clustering_metrics(labels, n_clusters, sentences)
        metrics["method"] = method_key
        all_methods_results[method_key] = metrics

        print(
            f"Horizon NMI: {metrics['horizon_nmi']:.4f}, "
            f"Choice NMI: {metrics['choice_nmi']:.4f}, "
            f"Active: {metrics['active_clusters']}/{n_clusters}"
        )

        # Per-method subdirectory for plots
        method_dir = base_path / method_key
        method_dir.mkdir(parents=True, exist_ok=True)

        # Cluster distribution plot
        try:
            plot_cluster_distribution(
                metrics["cluster_distribution"],
                f"{method_label} Cluster Dist ({len(sentences)} sentences)",
                method_dir / "cluster_distribution.png",
            )
        except Exception as e:
            print(f"      Warning: cluster dist plot failed: {e}")

        # Embedding plots
        try:
            embeddings = _compute_embeddings(X_norm)
            colorings = _build_colorings(sentences, labels)
            for coloring_name, color_labels in colorings.items():
                coloring_dir = method_dir / coloring_name
                coloring_dir.mkdir(parents=True, exist_ok=True)
                for emb_method, coords in embeddings.items():
                    title = f"{emb_method.upper()} — {method_label} — {coloring_name}"
                    try:
                        plot_embedding(
                            coords, color_labels, title,
                            coloring_dir / f"{emb_method}.png",
                        )
                    except Exception as e:
                        print(f"      Warning: {coloring_name}/{emb_method} failed: {e}")
        except Exception as e:
            print(f"      Warning: embedding plots failed: {e}")

    # Save combined results
    with open(base_path / "results.json", "w") as f:
        json.dump(all_methods_results, f, indent=2)

    return all_methods_results
