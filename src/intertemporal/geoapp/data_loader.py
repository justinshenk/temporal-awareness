"""Data loader for geo_viz output with caching for PCA, UMAP, and t-SNE."""

# Disable numba parallelism to avoid threading issues with Dash
# This is the safest approach on macOS ARM where TBB isn't available
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import json
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# Lock for UMAP/t-SNE computations to prevent numba threading conflicts
_compute_lock = threading.Lock()


@dataclass
class GeoVizDataLoader:
    """Load and cache geo_viz data for interactive visualization."""

    data_dir: Path
    _samples: list = field(default_factory=list, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)
    _target_keys: list = field(default_factory=list, repr=False)
    _pca_cache: dict = field(default_factory=dict, repr=False)
    _umap_cache: dict = field(default_factory=dict, repr=False)
    _tsne_cache: dict = field(default_factory=dict, repr=False)
    _activations_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self._load_metadata()

    def _load_metadata(self):
        """Load samples and metadata."""
        samples_path = self.data_dir / "data" / "samples.json"
        if samples_path.exists():
            with open(samples_path) as f:
                self._samples = json.load(f)

        metadata_path = self.data_dir / "data" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        # Discover target keys from activation files
        targets_dir = self.data_dir / "data" / "targets"
        if targets_dir.exists():
            self._target_keys = sorted([
                f.stem for f in targets_dir.glob("*.npy")
            ])

    @property
    def samples(self) -> list:
        return self._samples

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def get_sample_text(self, idx: int) -> str:
        """Get the full prompt text for a sample."""
        if 0 <= idx < len(self._samples):
            return self._samples[idx].get("text", "")
        return ""

    def get_sample_info(self, idx: int) -> dict:
        """Get full sample info including prompt details."""
        if 0 <= idx < len(self._samples):
            return self._samples[idx]
        return {}

    def get_layers(self) -> list[int]:
        """Get available layers (cached after first call)."""
        if not hasattr(self, "_layers_cache"):
            layers = set()
            for key in self._target_keys:
                match = re.match(r"L(\d+)_", key)
                if match:
                    layers.add(int(match.group(1)))
            self._layers_cache = sorted(layers)
        return self._layers_cache

    def get_components(self) -> list[str]:
        """Get available components."""
        return ["resid_pre", "attn_out", "mlp_out", "resid_post"]

    def get_positions(self) -> list[str]:
        """Get available positions (cached after first call)."""
        if not hasattr(self, "_positions_cache"):
            positions = set()
            for key in self._target_keys:
                # Match both P{position} patterns
                match = re.search(r"_P(.+)$", key)
                if match:
                    positions.add(match.group(1))
            self._positions_cache = sorted(positions, key=self._position_sort_key)
        return self._positions_cache

    def _position_sort_key(self, pos: str) -> tuple:
        """Sort positions: numeric first, then named."""
        # Try to extract numeric value
        if pos.isdigit():
            return (0, int(pos))
        # Named positions sorted alphabetically
        return (1, pos)

    def get_named_positions(self) -> list[str]:
        """Get just the named (non-numeric) positions."""
        return [p for p in self.get_positions() if not p.isdigit()]

    def get_numeric_positions(self) -> list[int]:
        """Get just the numeric positions sorted."""
        return sorted([int(p) for p in self.get_positions() if p.isdigit()])

    def _make_target_key(self, layer: int, component: str, position: str) -> str:
        """Construct target key."""
        return f"L{layer}_{component}_P{position}"

    def load_activations(self, layer: int, component: str, position: str) -> np.ndarray | None:
        """Load raw activations for a target."""
        key = self._make_target_key(layer, component, position)
        if key in self._activations_cache:
            return self._activations_cache[key]

        # Try direct npy file
        target_path = self.data_dir / "data" / "targets" / f"{key}.npy"
        if target_path.exists():
            activations = np.load(target_path)
            self._activations_cache[key] = activations
            return activations

        # Try folder structure
        target_path = self.data_dir / "data" / "targets" / key / "activations.npy"
        if target_path.exists():
            activations = np.load(target_path)
            self._activations_cache[key] = activations
            return activations

        return None

    def load_pca(
        self, layer: int, component: str, position: str, n_components: int = 3
    ) -> np.ndarray | None:
        """Load or compute PCA embedding."""
        key = self._make_target_key(layer, component, position)
        cache_key = f"{key}_pca_{n_components}"

        if cache_key in self._pca_cache:
            return self._pca_cache[cache_key]

        # Try to load pre-computed PCA
        pca_path = self.data_dir / "results" / "pca" / key / "transformed.npy"
        if pca_path.exists():
            transformed = np.load(pca_path)
            if transformed.shape[1] >= n_components:
                result = transformed[:, :n_components]
                self._pca_cache[cache_key] = result
                return result

        # Compute from activations
        activations = self.load_activations(layer, component, position)
        if activations is None:
            return None

        n_comp = min(n_components, activations.shape[0] - 1, activations.shape[1])
        if n_comp < 1:
            return None

        pca = PCA(n_components=n_comp)
        result = pca.fit_transform(activations).astype(np.float32)
        self._pca_cache[cache_key] = result
        return result

    def load_umap(
        self,
        layer: int,
        component: str,
        position: str,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> np.ndarray | None:
        """Compute UMAP embedding (3D)."""
        key = self._make_target_key(layer, component, position)
        cache_key = f"{key}_umap_{n_components}_{n_neighbors}_{min_dist}"

        if cache_key in self._umap_cache:
            return self._umap_cache[cache_key]

        activations = self.load_activations(layer, component, position)
        if activations is None:
            return None

        if activations.shape[0] < n_neighbors:
            n_neighbors = max(2, activations.shape[0] - 1)

        # Use lock to prevent numba threading conflicts
        with _compute_lock:
            umap = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
            )
            result = umap.fit_transform(activations).astype(np.float32)

        self._umap_cache[cache_key] = result
        return result

    def load_tsne(
        self,
        layer: int,
        component: str,
        position: str,
        n_components: int = 3,
        perplexity: float = 30.0,
    ) -> np.ndarray | None:
        """Compute t-SNE embedding (3D)."""
        key = self._make_target_key(layer, component, position)
        cache_key = f"{key}_tsne_{n_components}_{perplexity}"

        if cache_key in self._tsne_cache:
            return self._tsne_cache[cache_key]

        activations = self.load_activations(layer, component, position)
        if activations is None:
            return None

        # Adjust perplexity if too large for dataset
        n_samples = activations.shape[0]
        effective_perplexity = min(perplexity, (n_samples - 1) / 3)
        if effective_perplexity < 5:
            effective_perplexity = 5

        # Use lock to prevent numba threading conflicts
        with _compute_lock:
            tsne = TSNE(
                n_components=n_components,
                perplexity=effective_perplexity,
                random_state=42,
                max_iter=1000,
                init="pca",
            )
            result = tsne.fit_transform(activations).astype(np.float32)

        self._tsne_cache[cache_key] = result
        return result

    def _extract_nested(self, sample: dict, path: str, default=None):
        """Extract value from nested dict using dot notation path."""
        parts = path.split(".")
        value = sample
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_sample_metadata(self, color_by: str) -> np.ndarray:
        """Get sample metadata for coloring."""
        if not self._samples:
            return np.array([])

        # Legacy fields (check top-level first, then nested)
        if color_by == "time_horizon":
            vals = []
            for s in self._samples:
                th = s.get("time_horizon_months")
                if th is None:
                    # Try nested prompt.time_horizon.value
                    th = self._extract_nested(s, "prompt.time_horizon.value", 0)
                vals.append(th if th is not None else 0)
            return np.array(vals)
        elif color_by == "log_time_horizon":
            vals = []
            for s in self._samples:
                th = s.get("time_horizon_months")
                if th is None:
                    th = self._extract_nested(s, "prompt.time_horizon.value", None)
                # Use 0 for no-horizon samples -> log10(0+1) = 0, distinct from others
                vals.append(th if th is not None else 0)
            return np.log10(np.array(vals) + 1)
        elif color_by == "time_scale":
            return np.array([s.get("time_scale", "unknown") for s in self._samples])
        elif color_by == "choice_type":
            return np.array([s.get("choice_type", "unknown") for s in self._samples])
        elif color_by == "short_term_first":
            return np.array([s.get("short_term_first", False) for s in self._samples])
        elif color_by == "has_horizon":
            # True if sample has a time horizon
            vals = []
            for s in self._samples:
                has_th = s.get("time_horizon_months") is not None
                if not has_th:
                    has_th = self._extract_nested(s, "prompt.time_horizon.value") is not None
                vals.append(has_th)
            return np.array(vals)

        # New fields from nested structure
        elif color_by == "long_term_delay":
            # Long-term option delay in years
            vals = []
            for s in self._samples:
                delay = self._extract_nested(s, "prompt.preference_pair.long_term.time.value", 0)
                unit = self._extract_nested(s, "prompt.preference_pair.long_term.time.unit", "years")
                # Convert to years for consistency
                if unit == "months":
                    delay = delay / 12
                elif unit == "decades":
                    delay = delay * 10
                elif unit == "centuries":
                    delay = delay * 100
                vals.append(delay)
            return np.array(vals)
        elif color_by == "context_id":
            return np.array([s.get("context_id", 0) for s in self._samples])
        elif color_by == "formatting_id":
            return np.array([s.get("formatting_id", "unknown") for s in self._samples])
        elif color_by == "sample_idx":
            return np.arange(len(self._samples))
        else:
            return np.arange(len(self._samples))

    def get_no_horizon_mask(self) -> np.ndarray:
        """Get boolean mask for samples without time horizon."""
        return np.array([
            s.get("time_horizon_months", None) is None
            for s in self._samples
        ])

    def get_color_options(self) -> list[str]:
        """Get available color-by options (cached constant list)."""
        if not hasattr(self, "_color_options_cache"):
            self._color_options_cache = [
                "long_term_delay",
                "log_time_horizon",
                "time_horizon",
                "context_id",
                "formatting_id",
                "has_horizon",
                "short_term_first",
                "sample_idx",
            ]
        return self._color_options_cache

    def load_linear_probe_metrics(self, layer: int, component: str, position: str) -> dict | None:
        """Load linear probe metrics for a target."""
        key = self._make_target_key(layer, component, position)
        metrics_path = self.data_dir / "results" / "linear_probe" / key / "metrics.json"
        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            return json.load(f)

    def load_pca_metrics(self, layer: int, component: str, position: str) -> dict | None:
        """Load PCA metrics for a target."""
        key = self._make_target_key(layer, component, position)
        metrics_path = self.data_dir / "results" / "pca" / key / "metrics.json"
        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            return json.load(f)

    def load_all_layer_embeddings(
        self,
        component: str,
        position: str,
        method: str = "pca",
        n_components: int = 3,
    ) -> dict[int, np.ndarray]:
        """Load embeddings for all layers for trajectory visualization."""
        layers = self.get_layers()
        embeddings = {}

        for layer in layers:
            if method == "pca":
                emb = self.load_pca(layer, component, position, n_components)
            elif method == "umap":
                emb = self.load_umap(layer, component, position, n_components)
            elif method == "tsne":
                emb = self.load_tsne(layer, component, position, n_components)
            else:
                emb = None

            if emb is not None:
                embeddings[layer] = emb

        return embeddings

    def warmup(
        self,
        methods: list[str] | None = None,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        positions: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Pre-compute embeddings synchronously to avoid threading issues during requests.

        Call this method at startup before the server starts handling requests.
        This ensures all UMAP/t-SNE computations happen sequentially, avoiding
        numba threading conflicts.

        Args:
            methods: Embedding methods to precompute. Defaults to ["pca", "umap", "tsne"].
            layers: Layers to precompute. Defaults to all available layers.
            components: Components to precompute. Defaults to ["resid_pre"].
            positions: Positions to precompute. Defaults to all named positions.
            progress_callback: Optional callback(current, total, description) for progress updates.

        Returns:
            Number of embeddings cached.
        """
        if methods is None:
            methods = ["pca", "umap", "tsne"]
        if layers is None:
            layers = self.get_layers()
        if components is None:
            components = ["resid_pre"]
        if positions is None:
            positions = self.get_named_positions() or self.get_positions()[:1]

        total = len(methods) * len(layers) * len(components) * len(positions)
        current = 0
        cached = 0

        for method in methods:
            for layer in layers:
                for component in components:
                    for position in positions:
                        current += 1
                        desc = f"{method.upper()} L{layer} {component} @ {position}"
                        if progress_callback:
                            progress_callback(current, total, desc)

                        result = None
                        if method == "pca":
                            result = self.load_pca(layer, component, position)
                        elif method == "umap":
                            result = self.load_umap(layer, component, position)
                        elif method == "tsne":
                            result = self.load_tsne(layer, component, position)

                        if result is not None:
                            cached += 1

        return cached

    def preload_activations(
        self,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        positions: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Pre-load raw activations into cache for faster embedding computation.

        Args:
            layers: Layers to preload. Defaults to all available layers.
            components: Components to preload. Defaults to all components.
            positions: Positions to preload. Defaults to all positions.
            progress_callback: Optional callback(current, total, description) for progress updates.

        Returns:
            Number of activation arrays loaded.
        """
        if layers is None:
            layers = self.get_layers()
        if components is None:
            components = self.get_components()
        if positions is None:
            positions = self.get_positions()

        total = len(layers) * len(components) * len(positions)
        current = 0
        loaded = 0

        for layer in layers:
            for component in components:
                for position in positions:
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, f"L{layer} {component} @ {position}")

                    result = self.load_activations(layer, component, position)
                    if result is not None:
                        loaded += 1

        return loaded
