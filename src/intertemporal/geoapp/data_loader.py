"""Data loader for geometry output with position mapping format.

The format stores:
- Per-sample activation files: targets/sample_{idx}/L{layer}_{component}_{abs_pos}.npy
- Position mapping: sample_position_mapping.json maps abs_pos -> semantic format_pos
- samples.json with prompt/preference info

This loader aggregates activations by semantic position across samples.
"""

import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

_compute_lock = threading.Lock()


@dataclass
class GeometryDataLoader:
    """Load and cache geometry data for interactive visualization.

    Handles the per-sample activation format with semantic position mapping.
    """

    data_dir: Path
    _samples: list = field(default_factory=list, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)
    _position_mapping: dict = field(default_factory=dict, repr=False)
    _layers: list = field(default_factory=list, repr=False)
    _semantic_positions: set = field(default_factory=set, repr=False)
    _pca_cache: dict = field(default_factory=dict, repr=False)
    _umap_cache: dict = field(default_factory=dict, repr=False)
    _tsne_cache: dict = field(default_factory=dict, repr=False)
    _activations_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self._load_data()

    def _load_data(self):
        """Load samples, position mapping, and discover available targets."""
        # Load samples
        samples_path = self.data_dir / "data" / "samples.json"
        if samples_path.exists():
            with open(samples_path) as f:
                self._samples = json.load(f)

        # Load choices and merge into samples
        choices_path = self.data_dir / "data" / "choices.json"
        if choices_path.exists():
            with open(choices_path) as f:
                choices = json.load(f)
            for i, choice in enumerate(choices):
                if i < len(self._samples):
                    if "chosen_time_months" in choice:
                        self._samples[i]["time_horizon_months"] = choice["chosen_time_months"]
                    if "chose_long_term" in choice:
                        self._samples[i]["chosen_time"] = choice["chose_long_term"]
                    if "chosen_reward" in choice:
                        self._samples[i]["chosen_reward"] = choice["chosen_reward"]

                    # Compute derived color fields
                    self._compute_derived_fields(i, choice)

        # Load metadata
        metadata_path = self.data_dir / "data" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        # Load position mapping
        mapping_path = self.data_dir / "data" / "sample_position_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                self._position_mapping = json.load(f)

        # Discover layers and semantic positions from mapping and files
        self._discover_targets()

    def _compute_derived_fields(self, sample_idx: int, choice: dict) -> None:
        """Compute derived color fields for a sample based on choice and prompt.

        Computes:
        - matches_largest_reward: Did they choose the option with higher reward?
        - matches_rational: For same reward, shorter time is rational
        - matches_associated: Does choice align with time horizon framing?
        """
        sample = self._samples[sample_idx]
        prompt = sample.get("prompt", {})
        pp = prompt.get("preference_pair", {})

        short_term = pp.get("short_term", {})
        long_term = pp.get("long_term", {})

        short_reward = short_term.get("reward", {}).get("value", 0)
        long_reward = long_term.get("reward", {}).get("value", 0)

        chose_long = choice.get("chose_long_term", False)

        # matches_largest_reward: Did they choose the larger reward?
        if long_reward > short_reward:
            sample["matches_largest_reward"] = chose_long
        elif short_reward > long_reward:
            sample["matches_largest_reward"] = not chose_long
        else:
            # Equal rewards - neither matches "largest", mark as True (rational either way)
            sample["matches_largest_reward"] = True

        # matches_rational: For same reward, shorter time is rational
        # For different rewards, choosing larger reward is rational
        if short_reward == long_reward:
            # Same reward - shorter time is rational
            sample["matches_rational"] = not chose_long
        else:
            # Different rewards - larger reward is rational
            sample["matches_rational"] = sample["matches_largest_reward"]

        # matches_associated: Does choice align with time horizon framing?
        # If time horizon is "short" (e.g., days/weeks), expect short-term choice
        # If time horizon is "long" (e.g., years/decades), expect long-term choice
        time_horizon = prompt.get("time_horizon")
        if time_horizon is None:
            # No horizon framing - mark as N/A (True for now)
            sample["matches_associated"] = True
        else:
            th_value = time_horizon.get("value", 0)
            th_unit = time_horizon.get("unit", "months")

            # Convert to months for comparison
            unit_to_months = {
                "seconds": 1 / 2592000,
                "minutes": 1 / 43200,
                "hours": 1 / 720,
                "days": 1 / 30,
                "weeks": 1 / 4.3,
                "months": 1,
                "years": 12,
                "decades": 120,
                "centuries": 1200,
                "millennia": 12000,
            }
            th_months = th_value * unit_to_months.get(th_unit, 1)

            # Threshold: < 1 month is "short", >= 1 month is "long"
            horizon_suggests_long = th_months >= 1
            sample["matches_associated"] = chose_long == horizon_suggests_long

    def _discover_targets(self):
        """Discover available layers and semantic positions."""
        layers = set()
        semantic_positions = set()

        # Get semantic positions from position mapping
        if self._position_mapping and "mappings" in self._position_mapping:
            for sample_mapping in self._position_mapping["mappings"]:
                for pos_info in sample_mapping.get("positions", []):
                    format_pos = pos_info.get("format_pos")
                    if format_pos:
                        semantic_positions.add(format_pos)

        # Get layers from target files
        targets_dir = self.data_dir / "data" / "targets"
        if targets_dir.exists():
            # Check first sample directory for layer info
            sample_dirs = sorted(targets_dir.glob("sample_*"))
            if sample_dirs:
                for npy_file in sample_dirs[0].glob("L*_*.npy"):
                    match = re.match(r"L(\d+)_", npy_file.stem)
                    if match:
                        layers.add(int(match.group(1)))

        self._layers = sorted(layers)
        self._semantic_positions = semantic_positions

    def _get_abs_positions_for_semantic(
        self, sample_idx: int, semantic_pos: str
    ) -> list[int]:
        """Get absolute positions for a semantic position in a sample."""
        if not self._position_mapping or "mappings" not in self._position_mapping:
            return []

        for sample_mapping in self._position_mapping["mappings"]:
            if sample_mapping.get("sample_idx") == sample_idx:
                positions = []
                for pos_info in sample_mapping.get("positions", []):
                    if pos_info.get("format_pos") == semantic_pos:
                        positions.append(pos_info["abs_pos"])
                return positions
        return []

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
        """Get available layers."""
        return self._layers

    def get_components(self) -> list[str]:
        """Get available components."""
        return ["resid_pre", "attn_out", "mlp_out", "resid_post"]

    def get_positions(self) -> list[str]:
        """Get available semantic positions."""
        # Return in a sensible order
        order = [
            "time_horizon", "post_time_horizon",
            "response_choice_prefix", "response_choice",
            "response_reasoning_prefix", "response_reasoning",
        ]
        result = []
        for pos in order:
            if pos in self._semantic_positions:
                result.append(pos)
        # Add any remaining positions
        for pos in sorted(self._semantic_positions):
            if pos not in result:
                result.append(pos)
        return result

    def load_activations(
        self, layer: int, component: str, position: str
    ) -> np.ndarray | None:
        """Load aggregated activations for a semantic position.

        Aggregates activations across all samples by:
        1. Finding absolute positions for the semantic position in each sample
        2. Loading the corresponding .npy files
        3. Averaging across positions within each sample (if multiple tokens)
        4. Stacking into (n_samples, d_model) array
        """
        cache_key = f"L{layer}_{component}_{position}"
        if cache_key in self._activations_cache:
            return self._activations_cache[cache_key]

        targets_dir = self.data_dir / "data" / "targets"
        if not targets_dir.exists():
            return None

        activations_list = []

        for sample_idx in range(len(self._samples)):
            abs_positions = self._get_abs_positions_for_semantic(sample_idx, position)
            if not abs_positions:
                continue

            sample_dir = targets_dir / f"sample_{sample_idx}"
            if not sample_dir.exists():
                continue

            # Load activations for all absolute positions and average
            sample_activations = []
            for abs_pos in abs_positions:
                npy_path = sample_dir / f"L{layer}_{component}_{abs_pos}.npy"
                if npy_path.exists():
                    act = np.load(npy_path)
                    sample_activations.append(act)

            if sample_activations:
                # Average across token positions within sample
                avg_activation = np.mean(sample_activations, axis=0)
                activations_list.append(avg_activation)

        if not activations_list:
            return None

        result = np.stack(activations_list).astype(np.float32)
        self._activations_cache[cache_key] = result
        return result

    def load_pca(
        self, layer: int, component: str, position: str, n_components: int = 3
    ) -> np.ndarray | None:
        """Load or compute PCA embedding."""
        cache_key = f"L{layer}_{component}_{position}_pca_{n_components}"

        if cache_key in self._pca_cache:
            return self._pca_cache[cache_key]

        # Try pre-computed PCA from results
        pca_key = f"L{layer}_{component}_{position}"
        pca_path = self.data_dir / "results" / "pca" / pca_key / "transformed.npy"
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
        transformed = pca.fit_transform(activations).astype(np.float32)

        if transformed.shape[1] < n_components:
            padded = np.zeros((transformed.shape[0], n_components), dtype=np.float32)
            padded[:, : transformed.shape[1]] = transformed
            result = padded
        else:
            result = transformed

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
        """Compute UMAP embedding."""
        cache_key = f"L{layer}_{component}_{position}_umap_{n_components}_{n_neighbors}_{min_dist}"

        if cache_key in self._umap_cache:
            return self._umap_cache[cache_key]

        activations = self.load_activations(layer, component, position)
        if activations is None or activations.shape[0] < 2:
            return None

        if activations.shape[0] < n_neighbors:
            n_neighbors = max(2, activations.shape[0] - 1)

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
        """Compute t-SNE embedding."""
        cache_key = f"L{layer}_{component}_{position}_tsne_{n_components}_{perplexity}"

        if cache_key in self._tsne_cache:
            return self._tsne_cache[cache_key]

        activations = self.load_activations(layer, component, position)
        if activations is None or activations.shape[0] < 4:
            return None

        effective_perplexity = min(perplexity, max(1.0, (activations.shape[0] - 1) / 3))

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
        """Extract value from nested dict using dot notation."""
        parts = path.split(".")
        value = sample
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def _convert_to_months(self, value: float, unit: str) -> float:
        """Convert a time value to months."""
        unit_lower = unit.lower()
        conversions = {
            "months": 1.0, "month": 1.0,
            "years": 12.0, "year": 12.0,
            "weeks": 1.0 / 4.33, "week": 1.0 / 4.33,
            "days": 1.0 / 30.0, "day": 1.0 / 30.0,
            "hours": 1.0 / 720.0, "hour": 1.0 / 720.0,
            "minutes": 1.0 / 43200.0, "minute": 1.0 / 43200.0,
            "seconds": 1.0 / 2592000.0, "second": 1.0 / 2592000.0,
            "decades": 120.0, "decade": 120.0,
            "centuries": 1200.0, "century": 1200.0,
            "millennia": 12000.0, "millennium": 12000.0,
        }
        return value * conversions.get(unit_lower, 1.0)

    def get_sample_metadata(self, color_by: str) -> np.ndarray:
        """Get sample metadata for coloring."""
        if not self._samples:
            return np.array([])

        if color_by == "time_horizon":
            vals = []
            for s in self._samples:
                th_value = self._extract_nested(s, "prompt.time_horizon.value")
                th_unit = self._extract_nested(s, "prompt.time_horizon.unit", "months")
                if th_value is not None:
                    th = self._convert_to_months(th_value, th_unit)
                else:
                    th = -1
                vals.append(th)
            return np.array(vals)
        elif color_by == "log_time_horizon":
            vals = []
            for s in self._samples:
                th_value = self._extract_nested(s, "prompt.time_horizon.value")
                th_unit = self._extract_nested(s, "prompt.time_horizon.unit", "months")
                if th_value is not None:
                    th = self._convert_to_months(th_value, th_unit)
                    vals.append(np.log10(th + 1))
                else:
                    vals.append(-1)
            return np.array(vals)
        elif color_by == "chosen_time":
            return np.array([s.get("chosen_time", False) for s in self._samples])
        elif color_by == "chosen_reward":
            return np.array([s.get("chosen_reward", 0) for s in self._samples])
        elif color_by == "matches_largest_reward":
            return np.array([s.get("matches_largest_reward", False) for s in self._samples])
        elif color_by == "matches_rational":
            return np.array([s.get("matches_rational", False) for s in self._samples])
        elif color_by == "matches_associated":
            return np.array([s.get("matches_associated", False) for s in self._samples])
        elif color_by == "has_horizon":
            return np.array([
                self._extract_nested(s, "prompt.time_horizon.value") is not None
                for s in self._samples
            ])
        elif color_by == "short_term_first":
            return np.array([s.get("short_term_first", False) for s in self._samples])
        elif color_by == "context_id":
            return np.array([s.get("context_id", 0) for s in self._samples])
        elif color_by == "sample_idx":
            return np.arange(len(self._samples))
        else:
            return np.arange(len(self._samples))

    def get_no_horizon_mask(self) -> np.ndarray:
        """Get boolean mask for samples without time horizon."""
        return np.array([
            self._extract_nested(s, "prompt.time_horizon.value") is None
            for s in self._samples
        ])

    def get_color_options(self) -> list[str]:
        """Get available color-by options."""
        return [
            "time_horizon",
            "chosen_time",
            "chosen_reward",
            "matches_largest_reward",
            "matches_rational",
            "matches_associated",
            "has_horizon",
            "short_term_first",
            "context_id",
            "sample_idx",
        ]

    def get_position_labels(self) -> dict[str, str]:
        """Get human-readable labels for positions."""
        labels = {
            "time_horizon": "Time Horizon",
            "post_time_horizon": "Post Time Horizon",
            "response_choice_prefix": "Choice Prefix (I choose:)",
            "response_choice": "Choice (a/b)",
            "response_reasoning_prefix": "Reasoning Prefix",
            "response_reasoning": "Reasoning Content",
            "situation_marker": "SITUATION:",
            "task_marker": "TASK:",
            "consider_marker": "CONSIDER:",
            "action_marker": "ACTION:",
            "format_marker": "FORMAT:",
            "format_choice_prefix": "Format Choice Prefix",
            "format_reasoning_prefix": "Format Reasoning Prefix",
            "left_label": "Left Option Label",
            "right_label": "Right Option Label",
            "left_reward": "Left Reward",
            "right_reward": "Right Reward",
            "left_time": "Left Time",
            "right_time": "Right Time",
            "situation": "Situation Text",
            "role": "Role",
            "task_in_question": "Task Description",
            "reasoning_ask": "Reasoning Ask",
            "reward_units": "Reward Units",
            "chat_prefix": "Chat Prefix",
            "situation_content": "Situation Content",
            "task_content": "Task Content",
            "consider_content": "Consider Content",
            "action_content": "Action Content",
            "format_content": "Format Content",
            "prompt_other": "Other Prompt Tokens",
            "response_other": "Other Response Tokens",
        }
        # Add any missing positions
        for pos in self._semantic_positions:
            if pos not in labels:
                labels[pos] = pos.replace("_", " ").title()
        return labels

    def get_enriched_position_labels(self) -> dict[str, str]:
        """Get position labels - same as get_position_labels for compatibility."""
        return self.get_position_labels()

    def get_semantic_to_positions_mapping(self) -> dict[str, list[str]]:
        """Get mapping from semantic positions to themselves (for API compatibility)."""
        return {pos: [pos] for pos in self.get_positions()}

    def get_markers(self) -> dict[str, str]:
        """Get section markers from DefaultPromptFormat."""
        from ..formatting.configs.default_prompt_format import DefaultPromptFormat

        fmt = DefaultPromptFormat()
        markers = {}
        marker_keys = [
            "situation_marker", "task_marker", "consider_marker",
            "action_marker", "format_marker",
        ]
        for key in marker_keys:
            if key in fmt.prompt_const_keywords:
                markers[key] = fmt.prompt_const_keywords[key]
        return markers

    def get_model_name(self) -> str:
        """Get the model name from metadata."""
        if self._metadata and "model_name" in self._metadata:
            return self._metadata["model_name"]
        return "Qwen3-4B"

    def get_prompt_template_structure(self) -> list[dict]:
        """Get prompt template structure for UI display."""
        available = set(self.get_positions())
        labels = self.get_position_labels()

        template_order = [
            ("situation_marker", "marker"),
            ("situation", "variable"),
            ("task_marker", "marker"),
            ("left_label", "variable"),
            ("left_reward", "variable"),
            ("left_time", "variable"),
            ("right_label", "variable"),
            ("right_reward", "variable"),
            ("right_time", "variable"),
            ("consider_marker", "marker"),
            ("time_horizon", "variable"),
            ("post_time_horizon", "variable"),
            ("action_marker", "marker"),
            ("format_marker", "marker"),
            ("response_choice_prefix", "marker"),
            ("response_choice", "semantic"),
            ("response_reasoning_prefix", "marker"),
            ("response_reasoning", "semantic"),
        ]

        elements = []
        for name, elem_type in template_order:
            elements.append({
                "name": name,
                "label": labels.get(name, name.replace("_", " ").title()),
                "type": elem_type,
                "available": name in available,
            })
        return elements

    def load_linear_probe_metrics(
        self, layer: int, component: str, position: str
    ) -> dict | None:
        """Load linear probe metrics for a target."""
        key = f"L{layer}_{component}_{position}"
        metrics_path = self.data_dir / "results" / "linear_probe" / key / "metrics.json"
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)

    def load_pca_metrics(
        self, layer: int, component: str, position: str
    ) -> dict | None:
        """Load PCA metrics for a target."""
        key = f"L{layer}_{component}_{position}"
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
        """Load embeddings for all layers."""
        embeddings = {}
        for layer in self._layers:
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
        """Pre-compute embeddings."""
        if methods is None:
            methods = ["pca", "umap", "tsne"]
        if layers is None:
            layers = self._layers
        if components is None:
            components = ["resid_pre"]
        if positions is None:
            positions = self.get_positions() or ["response_choice"]

        total = len(methods) * len(layers) * len(components) * len(positions)
        current = 0
        cached = 0

        for method in methods:
            for layer in layers:
                for component in components:
                    for position in positions:
                        current += 1
                        if progress_callback:
                            progress_callback(
                                current, total,
                                f"{method.upper()} L{layer} {component} @ {position}",
                            )

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

    def get_position_mapping_for_sample(self, sample_idx: int) -> dict | None:
        """Get the full position mapping for a sample."""
        if not self._position_mapping or "mappings" not in self._position_mapping:
            return None
        for mapping in self._position_mapping["mappings"]:
            if mapping.get("sample_idx") == sample_idx:
                return mapping
        return None

    def get_tokens_for_sample(self, sample_idx: int) -> list[dict]:
        """Get token info for a sample including format_pos labels."""
        mapping = self.get_position_mapping_for_sample(sample_idx)
        if mapping:
            return mapping.get("positions", [])
        return []
