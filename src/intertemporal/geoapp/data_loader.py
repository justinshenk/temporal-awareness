"""Data loader for geometry output with position mapping format.

This is a LOAD-ONLY loader. All embeddings must be pre-computed using
compute_geometry_analysis.py. The loader will NOT compute embeddings
at runtime - it only loads from analysis/embeddings/.

The format stores:
- Per-sample activation files: samples/sample_{idx}/L{layer}_{component}_{abs_pos}.npy
- Per-sample position mapping: samples/sample_{idx}/position_mapping.json
- Per-sample prompt sample: samples/sample_{idx}/prompt_sample.json
- Per-sample preference sample: samples/sample_{idx}/preference_sample.json
- Per-sample choice info: samples/sample_{idx}/choice.json

Pre-computed embeddings are loaded from:
- analysis/embeddings/pca/L{layer}_{component}_{position}.npy
- analysis/embeddings/umap/L{layer}_{component}_{position}.npy
- analysis/embeddings/tsne/L{layer}_{component}_{position}.npy

This loader aggregates activations by semantic position across samples.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


def _log(component: str, message: str, **kwargs):
    """Log with timestamp and component info."""
    ts = time.strftime("%H:%M:%S")
    extras = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[{ts}] [LOADER] [{component}] {message}" + (f" | {extras}" if extras else ""))

# Module-level caches that survive server reloads (uvicorn --reload)
# This prevents losing warmup work when files change and server restarts
_GLOBAL_PCA_CACHE: dict = {}
_GLOBAL_UMAP_CACHE: dict = {}
_GLOBAL_TSNE_CACHE: dict = {}
_GLOBAL_ACTIVATIONS_CACHE: dict = {}


@dataclass
class GeometryDataLoader:
    """Load and cache geometry data for interactive visualization.

    Handles the per-sample activation format with semantic position mapping.
    Uses module-level caches that survive server reloads.
    """

    data_dir: Path
    _samples: list = field(default_factory=list, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)
    _position_mapping: dict = field(default_factory=dict, repr=False)
    _layers: list = field(default_factory=list, repr=False)
    _semantic_positions: set = field(default_factory=set, repr=False)
    # Instance caches point to global caches for reload survival
    _pca_cache: dict = field(default_factory=lambda: _GLOBAL_PCA_CACHE, repr=False)
    _umap_cache: dict = field(default_factory=lambda: _GLOBAL_UMAP_CACHE, repr=False)
    _tsne_cache: dict = field(default_factory=lambda: _GLOBAL_TSNE_CACHE, repr=False)
    _activations_cache: dict = field(default_factory=lambda: _GLOBAL_ACTIVATIONS_CACHE, repr=False)
    # Cache for preloaded metadata values (color options)
    _metadata_cache: dict = field(default_factory=dict, repr=False)
    # Summary from summary.json - contains precomputed positions/layers
    _summary: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        # Cache key prefix for this data directory (ensures different dirs don't share cache)
        self._cache_prefix = str(self.data_dir.resolve())
        self._load_data()

    def _load_data(self):
        """Load samples, position mapping, and discover available targets.

        Data is loaded from per-sample files in samples/sample_*/

        Raises:
            ValueError: If required data is missing (metadata, samples, position mappings).
        """
        start_time = time.time()
        _log("load_data", f"Starting data load from {self.data_dir}")

        # Load metadata - REQUIRED
        metadata_path = self.data_dir / "data" / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(
                f"Required metadata file not found: {metadata_path}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )
        with open(metadata_path) as f:
            self._metadata = json.load(f)

        # Load from per-sample files - REQUIRED
        samples_dir = self.data_dir / "data" / "samples"
        if not samples_dir.exists():
            raise ValueError(
                f"Required samples directory not found: {samples_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        mappings = []
        sample_dirs = sorted(
            samples_dir.glob("sample_*"),
            key=lambda x: int(x.name.split("_")[1])  # Numeric sort, not lexicographic
        )
        if not sample_dirs:
            raise ValueError(
                f"No sample directories found in: {samples_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        for sample_dir in sample_dirs:
            sample_idx = len(self._samples)

            # Load prompt sample - REQUIRED for each sample
            prompt_path = sample_dir / "prompt_sample.json"
            if not prompt_path.exists():
                raise ValueError(
                    f"Required prompt_sample.json not found: {prompt_path}\n"
                    "Data may be corrupted. Re-run compute_geometry_analysis.py."
                )
            with open(prompt_path) as f:
                sample = json.load(f)
            self._samples.append(sample)

            # Load and merge choice info - REQUIRED
            choice_path = sample_dir / "choice.json"
            if not choice_path.exists():
                raise ValueError(
                    f"Required choice.json not found: {choice_path}\n"
                    "Data may be corrupted. Re-run compute_geometry_analysis.py."
                )
            with open(choice_path) as f:
                choice = json.load(f)

            # REQUIRED fields - crash if missing
            # Note: time_scale is computed at runtime from time_horizon_months
            required_fields = ["time_horizon_months", "chose_long_term", "chosen_time_months", "chosen_reward", "choice_prob"]
            missing = [f for f in required_fields if f not in choice]
            if missing:
                raise KeyError(
                    f"Required fields missing from {choice_path}: {missing}\n"
                    "Re-run data extraction to regenerate choice.json files."
                )

            # Load all required fields
            # Note: time_scale is computed at runtime from time_horizon_months
            sample["time_horizon_months"] = choice["time_horizon_months"]
            sample["chosen_time_months"] = choice["chosen_time_months"]
            # Note: chosen_time in metadata returns chosen_time_months (numeric)
            sample["chosen_reward"] = choice["chosen_reward"]
            sample["choice_prob"] = choice["choice_prob"]
            # Compute derived choice fields
            sample["chose_long_term"] = choice["chose_long_term"]
            sample["chose_short_term"] = not choice["chose_long_term"]
            # Load alt (unchosen) option fields if present
            sample["alt_time_months"] = choice.get("alt_time_months")
            sample["alt_reward"] = choice.get("alt_reward")
            sample["alt_prob"] = choice.get("alt_prob", 1.0 - choice["choice_prob"])

            # Load preference_sample.json for response data (optional)
            preference_path = sample_dir / "preference_sample.json"
            if preference_path.exists():
                with open(preference_path) as f:
                    preference = json.load(f)
                # Store response-related fields
                if "choice_label" in preference:
                    sample["response_label"] = preference["choice_label"]
                if "choice_term" in preference:
                    sample["response_term"] = preference["choice_term"]
                # Get response text - prefer response_text if set, else derive from response_texts[choice_idx]
                if "response_text" in preference and preference["response_text"]:
                    sample["response_text"] = preference["response_text"]
                elif "response_texts" in preference and "choice_idx" in preference:
                    choice_idx = preference["choice_idx"]
                    response_texts = preference["response_texts"]
                    if isinstance(response_texts, list) and 0 <= choice_idx < len(response_texts):
                        sample["response_text"] = response_texts[choice_idx]

            # Load pre-computed color fields from choice.json
            precomputed_fields = [
                "log_time_horizon",
                "option_time_delta",
                "option_reward_delta",
                "option_confidence_delta",
            ]
            for field_name in precomputed_fields:
                if field_name in choice:
                    sample[field_name] = choice[field_name]

            # NOTE: Derived fields (matches_*, log_time_horizon, etc.) are computed
            # on-demand in _compute_metadata_values() using STRICT required fields

            # Load position mapping - REQUIRED
            mapping_file = sample_dir / "position_mapping.json"
            if not mapping_file.exists():
                raise ValueError(
                    f"Required position_mapping.json not found: {mapping_file}\n"
                    "Data may be corrupted. Re-run compute_geometry_analysis.py."
                )
            with open(mapping_file) as f:
                mappings.append(json.load(f))

        self._position_mapping = {"mappings": mappings}

        # Discover layers and semantic positions from mapping and files
        self._discover_targets()

        # Load summary.json to get precomputed positions/layers
        summary_path = self.data_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                self._summary = json.load(f)
        else:
            self._summary = {}

        elapsed = time.time() - start_time
        _log("load_data", "Data loaded", n_samples=len(self._samples), n_layers=len(self._layers), n_positions=len(self._semantic_positions), elapsed_ms=f"{elapsed*1000:.1f}")

    def _discover_targets(self):
        """Discover available layers and semantic positions."""
        layers = set()
        semantic_positions = set()

        # Get semantic positions from position mapping
        # _position_mapping is set in _load_data() and always contains "mappings"
        if self._position_mapping:
            for sample_mapping in self._position_mapping["mappings"]:
                for pos_info in sample_mapping["positions"]:
                    semantic_positions.add(pos_info["format_pos"])

        # Get layers from target files
        samples_dir = self.data_dir / "data" / "samples"
        if samples_dir.exists():
            # Check first sample directory for layer info
            sample_dirs = sorted(
                samples_dir.glob("sample_*"),
                key=lambda x: int(x.name.split("_")[1])  # Numeric sort, not lexicographic
            )
            if sample_dirs:
                # Layer subfolders format: sample_0/L0/, sample_0/L1/, etc.
                for layer_dir in sample_dirs[0].glob("L*"):
                    if layer_dir.is_dir():
                        match = re.match(r"L(\d+)$", layer_dir.name)
                        if match:
                            layers.add(int(match.group(1)))
                if not layers:
                    raise ValueError(
                        f"No layer subfolders found in {sample_dirs[0]}.\n"
                        "Expected format: sample_0/L0/, sample_0/L1/, etc.\n"
                        "Data may be corrupted or in wrong format. Re-run compute_geometry_analysis.py."
                    )

        self._layers = sorted(layers)
        self._semantic_positions = semantic_positions

    def _parse_position(self, position: str) -> tuple[str, int | None]:
        """Parse position string into (format_pos, rel_pos).

        Supports formats:
        - "time_horizon" -> ("time_horizon", None) - combined mode
        - "time_horizon:0" -> ("time_horizon", 0) - specific rel_pos
        """
        if ":" in position:
            parts = position.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return position, None
        return position, None

    def _get_abs_positions_for_semantic(
        self, sample_idx: int, semantic_pos: str, rel_pos: int | None = None
    ) -> list[int]:
        """Get absolute positions for a semantic position in a sample.

        Args:
            sample_idx: Sample index.
            semantic_pos: Semantic position name (e.g., "time_horizon").
            rel_pos: If specified, only return the position at this relative offset.
                     If None, return all positions for the semantic position.

        Raises:
            ValueError: If position_mapping not loaded.
            KeyError: If required keys are missing from position_mapping.
        """
        if not self._position_mapping:
            raise ValueError("position_mapping not loaded")

        for sample_mapping in self._position_mapping["mappings"]:
            if sample_mapping["sample_idx"] == sample_idx:
                positions = []
                for pos_info in sample_mapping["positions"]:
                    if pos_info["format_pos"] == semantic_pos:
                        if rel_pos is None or pos_info["rel_pos"] == rel_pos:
                            positions.append(pos_info["abs_pos"])
                return positions
        # Sample not found in mappings - this is valid (sample may have no positions for this semantic)
        return []

    def get_rel_pos_counts(self) -> dict[str, int]:
        """Get the maximum rel_pos count for each semantic position.

        Returns a dict mapping semantic position names to the max rel_pos + 1 (count).
        Useful for showing how many tokens each position spans.

        First checks for precomputed relpos_counts.json from compute_geometry_analysis.py.
        Falls back to computing from position mappings if not found.

        Raises:
            ValueError: If position_mapping not loaded (when fallback is needed).
        """
        # Check for precomputed file first
        relpos_file = self.data_dir / "analysis" / "relpos_counts.json"
        if relpos_file.exists():
            with open(relpos_file) as f:
                return json.load(f)

        # Fallback: compute from position mappings
        if not self._position_mapping:
            raise ValueError("position_mapping not loaded and no precomputed relpos_counts.json found")

        counts: dict[str, int] = {}
        for sample_mapping in self._position_mapping["mappings"]:
            for pos_info in sample_mapping["positions"]:
                format_pos = pos_info["format_pos"]
                rel_pos = pos_info["rel_pos"]
                current_max = counts.get(format_pos, 0)
                counts[format_pos] = max(current_max, rel_pos + 1)

        return counts

    @property
    def samples(self) -> list:
        return self._samples

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def get_sample_text(self, idx: int) -> str:
        """Get the full prompt text for a sample.

        Raises:
            IndexError: If idx is out of range.
            KeyError: If sample does not have 'text' field.
        """
        if not (0 <= idx < len(self._samples)):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self._samples)})")
        return self._samples[idx]["text"]

    def get_sample_info(self, idx: int) -> dict:
        """Get full sample info including prompt details.

        Raises:
            IndexError: If idx is out of range.
        """
        if not (0 <= idx < len(self._samples)):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self._samples)})")
        return self._samples[idx]

    def get_layers(self) -> list[int]:
        """Get available layers."""
        return self._layers

    def get_components(self) -> list[str]:
        """Get available components."""
        return ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]

    def get_positions(self) -> list[str]:
        """Get available semantic positions in canonical order."""
        # Canonical ordering: prompt positions first (in prompt order), then response positions
        canonical_order = [
            # Prompt constraint positions
            "time_horizon", "post_time_horizon",
            # Prompt info positions (left/right pairs)
            "left_label", "right_label",
            "left_time", "right_time",
            "left_reward", "right_reward",
            # Prompt section tails (in prompt order)
            "task_tail", "options_tail", "objective_tail",
            "action_tail", "format_tail",
            "chat_suffix", "chat_suffix_tail",
            # Response positions
            "response_choice_prefix", "response_choice",
            "response_reasoning_prefix", "response_reasoning",
        ]
        result = []
        for pos in canonical_order:
            if pos in self._semantic_positions:
                result.append(pos)
        # Add any remaining positions (shouldn't happen with canonical list)
        for pos in sorted(self._semantic_positions):
            if pos not in result:
                result.append(pos)
        return result

    def get_precomputed_positions(self) -> list[str]:
        """Get positions that have precomputed embeddings (from summary.json).

        Returns only positions that have PCA embeddings precomputed.
        Falls back to get_positions() if summary.json is not available.
        """
        if self._summary and "positions" in self._summary:
            return self._summary["positions"]
        # Fallback: return all semantic positions (may include ones without embeddings)
        return self.get_positions()

    def get_positions_in_prompt_order(self) -> list[str]:
        """Get positions sorted by their actual order in the prompt.

        Uses position_mapping from samples to determine the actual token order,
        returning positions sorted by their median abs_pos across samples.
        Falls back to get_precomputed_positions() if mapping is unavailable.
        """
        cache_key = f"{self._cache_prefix}|positions_prompt_order"
        if cache_key in self._activations_cache:
            return self._activations_cache[cache_key]

        if not self._position_mapping:
            raise ValueError("position_mapping not loaded")

        mappings = self._position_mapping["mappings"]
        if not mappings:
            return self.get_precomputed_positions()

        # Collect first abs_pos for each semantic position across samples
        position_abs_pos: dict[str, list[int]] = {}
        for mapping in mappings[:100]:  # Sample first 100 for efficiency
            seen_in_sample: dict[str, int] = {}
            for p in mapping["positions"]:
                fmt = p["format_pos"]
                if fmt not in seen_in_sample:
                    seen_in_sample[fmt] = p["abs_pos"]
            for pos, abs_pos in seen_in_sample.items():
                if pos not in position_abs_pos:
                    position_abs_pos[pos] = []
                position_abs_pos[pos].append(abs_pos)

        # Calculate median abs_pos for each position
        position_median: dict[str, int] = {}
        for pos, abs_positions in position_abs_pos.items():
            position_median[pos] = sorted(abs_positions)[len(abs_positions) // 2]

        # Get precomputed positions and sort by their median abs_pos
        precomputed = self.get_precomputed_positions()
        result = sorted(
            [p for p in precomputed if p in position_median],
            key=lambda p: position_median[p]
        )
        # Add any positions not found in mapping at the end
        for p in precomputed:
            if p not in result:
                result.append(p)

        self._activations_cache[cache_key] = result
        return result

    def get_precomputed_layers(self) -> list[int]:
        """Get layers that have precomputed embeddings (from summary.json).

        Returns only layers that have PCA embeddings precomputed.
        Falls back to get_layers() if summary.json is not available.
        """
        if self._summary and "layers" in self._summary:
            return self._summary["layers"]
        return self.get_layers()

    def get_available_methods(self) -> list[str]:
        """Get available dimensionality reduction methods.

        Returns ONLY methods that have complete data (same file count as PCA).
        Server validates all required methods are present.
        """
        emb_dir = self.data_dir / "analysis" / "embeddings"
        methods = []

        pca_dir = emb_dir / "pca"
        pca_count = len(list(pca_dir.glob("*.npy"))) if pca_dir.exists() else 0
        if pca_count > 0:
            methods.append("pca")

        umap_dir = emb_dir / "umap"
        if umap_dir.exists() and len(list(umap_dir.glob("*.npy"))) == pca_count:
            methods.append("umap")

        tsne_dir = emb_dir / "tsne"
        if tsne_dir.exists() and len(list(tsne_dir.glob("*.npy"))) == pca_count:
            methods.append("tsne")

        return methods

    def get_positions_with_data(self, layer: int = 0, component: str = "resid_post") -> set[str]:
        """Get positions that have actual activation data for the given layer/component.

        This checks which positions have at least one sample with activation files,
        which is more accurate than just checking the position mapping.
        """
        cache_key = f"{self._cache_prefix}|positions_with_data_{layer}_{component}"
        if cache_key in self._activations_cache:
            return self._activations_cache[cache_key]

        positions_with_data = set()
        for pos in self._semantic_positions:
            # Try to get valid sample indices - if any exist, the position has data
            indices = self.get_valid_sample_indices(layer, component, pos)
            if indices:
                positions_with_data.add(pos)

        self._activations_cache[cache_key] = positions_with_data
        return positions_with_data

    def get_valid_sample_indices(
        self, layer: int, component: str, position: str
    ) -> list[int]:
        """Get list of sample indices that have valid activations for this target.

        Returns the sample indices in the same order as the activations/embeddings.
        Supports position format "format_pos:rel_pos" for specific relative position.

        Raises:
            ValueError: If samples directory does not exist.
        """
        cache_key = f"{self._cache_prefix}|L{layer}_{component}_{position}_indices"
        if cache_key in self._activations_cache:
            return self._activations_cache[cache_key]

        samples_dir = self.data_dir / "data" / "samples"
        if not samples_dir.exists():
            raise ValueError(
                f"Samples directory not found: {samples_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        # Parse position to get format_pos and optional rel_pos
        format_pos, rel_pos = self._parse_position(position)
        valid_indices = []

        for sample_idx in range(len(self._samples)):
            abs_positions = self._get_abs_positions_for_semantic(
                sample_idx, format_pos, rel_pos
            )
            if not abs_positions:
                continue

            sample_dir = samples_dir / f"sample_{sample_idx}"
            if not sample_dir.exists():
                continue

            # Check if at least one activation file exists
            # Format: sample_dir/L{layer}/{component}_{abs_pos}.npy
            has_data = False
            for abs_pos in abs_positions:
                npy_path = sample_dir / f"L{layer}" / f"{component}_{abs_pos}.npy"
                if npy_path.exists():
                    has_data = True
                    break

            if has_data:
                valid_indices.append(sample_idx)

        self._activations_cache[cache_key] = valid_indices
        return valid_indices

    def load_activations(
        self, layer: int, component: str, position: str
    ) -> np.ndarray:
        """Load aggregated activations for a semantic position.

        Aggregates activations across all samples by:
        1. Finding absolute positions for the semantic position in each sample
        2. Loading the corresponding .npy files (parallelized for I/O)
        3. Averaging across positions within each sample (if multiple tokens)
        4. Stacking into (n_samples, d_model) array

        Supports position format "format_pos:rel_pos" for specific relative position.

        Raises:
            ValueError: If activation files do not exist or fail to load (STRICT mode).
        """
        cache_key = f"{self._cache_prefix}|L{layer}_{component}_{position}"
        if cache_key in self._activations_cache:
            _log("load_activations", "Cache HIT", key=f"L{layer}_{component}_{position}")
            return self._activations_cache[cache_key]

        _log("load_activations", "Cache MISS - loading from disk", key=f"L{layer}_{component}_{position}")
        start_time = time.time()

        samples_dir = self.data_dir / "data" / "samples"
        if not samples_dir.exists():
            raise ValueError(
                f"Samples directory not found: {samples_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        # Parse position to get format_pos and optional rel_pos
        format_pos, rel_pos = self._parse_position(position)

        # Collect all files to load with their sample index
        load_tasks: list[tuple[int, Path]] = []  # (sample_idx, npy_path)
        sample_to_paths: dict[int, list[Path]] = {}

        for sample_idx in range(len(self._samples)):
            abs_positions = self._get_abs_positions_for_semantic(
                sample_idx, format_pos, rel_pos
            )
            if not abs_positions:
                continue

            sample_dir = samples_dir / f"sample_{sample_idx}"
            if not sample_dir.exists():
                continue

            paths_for_sample = []
            for abs_pos in abs_positions:
                # Format: sample_dir/L{layer}/{component}_{abs_pos}.npy
                npy_path = sample_dir / f"L{layer}" / f"{component}_{abs_pos}.npy"
                if npy_path.exists():
                    paths_for_sample.append(npy_path)
                    load_tasks.append((sample_idx, npy_path))

            if paths_for_sample:
                sample_to_paths[sample_idx] = paths_for_sample

        if not load_tasks:
            raise ValueError(
                f"No activation files found for L{layer}_{component}_{position}.\n"
                "Run compute_geometry_analysis.py to generate activation data first."
            )

        # Parallel I/O for loading .npy files - STRICT: crash on any failure
        loaded_data: dict[Path, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_path = {executor.submit(np.load, path): path for _, path in load_tasks}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    loaded_data[path] = future.result()
                except Exception as e:
                    raise ValueError(
                        f"Failed to load activation file: {path}\n"
                        f"Error: {e}\n"
                        "Data may be corrupted. Re-run compute_geometry_analysis.py."
                    ) from e

        # Aggregate per sample
        activations_list = []
        for sample_idx in sorted(sample_to_paths.keys()):
            sample_activations = []
            for path in sample_to_paths[sample_idx]:
                if path in loaded_data:
                    sample_activations.append(loaded_data[path])

            if sample_activations:
                # Average across token positions within sample
                avg_activation = np.mean(sample_activations, axis=0)
                activations_list.append(avg_activation)

        if not activations_list:
            raise ValueError(
                f"No valid activations loaded for L{layer}_{component}_{position}.\n"
                "All activation files failed to load. Data may be corrupted."
            )

        result = np.stack(activations_list).astype(np.float32)
        self._activations_cache[cache_key] = result

        elapsed = time.time() - start_time
        _log("load_activations", "Loaded activations", shape=result.shape, n_files=len(load_tasks), elapsed_ms=f"{elapsed*1000:.1f}")
        return result

    def _get_embedding_path(self, method: str, layer: int, component: str, position: str) -> Path:
        """Get path to pre-computed embedding file.

        Embeddings are stored in analysis/embeddings/{method}/L{layer}_{component}_{position}.npy
        For per-rel_pos embeddings: L{layer}_{component}_{position}_r{rel_pos}.npy

        Position format: "format_pos" or "format_pos:rel_pos"
        """
        # Parse position to get base_pos and optional rel_pos
        if ":" in position:
            parts = position.rsplit(":", 1)
            base_pos = parts[0]
            try:
                rel_pos = int(parts[1])
                # Per-rel_pos embedding: add _r{rel_pos} suffix
                return self.data_dir / "analysis" / "embeddings" / method / f"L{layer}_{component}_{base_pos}_r{rel_pos}.npy"
            except ValueError:
                # Invalid rel_pos, treat as base position
                pass
        # Combined embedding (no rel_pos suffix)
        return self.data_dir / "analysis" / "embeddings" / method / f"L{layer}_{component}_{position}.npy"

    def _load_embedding(self, method: str, layer: int, component: str, position: str) -> np.ndarray | None:
        """Load pre-computed embedding from disk.

        Path: analysis/embeddings/{method}/L{layer}_{component}_{position}.npy

        For per-rel_pos positions (e.g., "chat_suffix:0"):
        Path: analysis/embeddings/{method}/L{layer}_{component}_{base_pos}_r{rel_pos}.npy

        Returns None if not found - does NOT compute. NO FALLBACKS.
        """
        path = self._get_embedding_path(method, layer, component, position)
        if path.exists():
            return np.load(path)
        return None

    def load_pca(
        self, layer: int, component: str, position: str, n_components: int = 3
    ) -> np.ndarray:
        """Load pre-computed PCA embedding.

        This is a LOAD-ONLY method. It does NOT compute PCA.
        Run compute_geometry_analysis.py to pre-compute embeddings.

        Raises:
            ValueError: If embedding file does not exist (STRICT mode).
        """
        cache_key = f"{self._cache_prefix}|L{layer}_{component}_{position}_pca_{n_components}"
        key_short = f"L{layer}_{component}_{position}"

        # Check memory cache first
        if cache_key in self._pca_cache:
            _log("load_pca", "Memory cache HIT", key=key_short)
            return self._pca_cache[cache_key]

        _log("load_pca", "Memory cache MISS", key=key_short)
        start_time = time.time()

        # Load from pre-computed embeddings
        disk_result = self._load_embedding("pca", layer, component, position)
        if disk_result is not None:
            if disk_result.shape[1] >= n_components:
                result = disk_result[:, :n_components]
            else:
                # Pad if needed
                result = np.zeros((disk_result.shape[0], n_components), dtype=np.float32)
                result[:, :disk_result.shape[1]] = disk_result
            self._pca_cache[cache_key] = result
            elapsed = time.time() - start_time
            _log("load_pca", "Loaded from disk", key=key_short, shape=result.shape, elapsed_ms=f"{elapsed*1000:.1f}")
            return result

        # Not found - embeddings MUST be pre-computed
        raise ValueError(
            f"PCA embedding not found for L{layer}_{component}_{position}.\n"
            f"Expected path: {self._get_embedding_path('pca', layer, component, position)}\n"
            "Run compute_geometry_analysis.py to pre-compute embeddings first."
        )

    def load_umap(
        self,
        layer: int,
        component: str,
        position: str,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> np.ndarray:
        """Load pre-computed UMAP embedding.

        This is a LOAD-ONLY method. It does NOT compute UMAP.
        Run compute_geometry_analysis.py --full to pre-compute UMAP embeddings.

        Raises:
            ValueError: If embedding file does not exist (STRICT mode).
        """
        cache_key = f"{self._cache_prefix}|L{layer}_{component}_{position}_umap_{n_components}_{n_neighbors}_{min_dist}"
        key_short = f"L{layer}_{component}_{position}"

        # Check memory cache first
        if cache_key in self._umap_cache:
            _log("load_umap", "Memory cache HIT", key=key_short)
            return self._umap_cache[cache_key]

        _log("load_umap", "Memory cache MISS", key=key_short)
        start_time = time.time()

        # Load from pre-computed embeddings
        disk_result = self._load_embedding("umap", layer, component, position)
        if disk_result is not None:
            if disk_result.shape[1] >= n_components:
                result = disk_result[:, :n_components]
            else:
                result = np.zeros((disk_result.shape[0], n_components), dtype=np.float32)
                result[:, :disk_result.shape[1]] = disk_result
            self._umap_cache[cache_key] = result
            elapsed = time.time() - start_time
            _log("load_umap", "Loaded from disk", key=key_short, shape=result.shape, elapsed_ms=f"{elapsed*1000:.1f}")
            return result

        # Not found - CRASH: embeddings MUST be pre-computed
        raise ValueError(
            f"UMAP embedding not found for L{layer}_{component}_{position}.\n"
            f"Checked path: {self._get_embedding_path('umap', layer, component, position)}\n"
            "Run compute_geometry_analysis.py --full to pre-compute UMAP embeddings."
        )

    def load_tsne(
        self,
        layer: int,
        component: str,
        position: str,
        n_components: int = 3,
        perplexity: float = 30.0,
    ) -> np.ndarray:
        """Load pre-computed t-SNE embedding.

        This is a LOAD-ONLY method. It does NOT compute t-SNE.
        Run compute_geometry_analysis.py --full to pre-compute t-SNE embeddings.

        Raises:
            ValueError: If embedding file does not exist (STRICT mode).
        """
        cache_key = f"{self._cache_prefix}|L{layer}_{component}_{position}_tsne_{n_components}_{perplexity}"
        key_short = f"L{layer}_{component}_{position}"

        # Check memory cache first
        if cache_key in self._tsne_cache:
            _log("load_tsne", "Memory cache HIT", key=key_short)
            return self._tsne_cache[cache_key]

        _log("load_tsne", "Memory cache MISS", key=key_short)
        start_time = time.time()

        # Load from pre-computed embeddings
        disk_result = self._load_embedding("tsne", layer, component, position)
        if disk_result is not None:
            if disk_result.shape[1] >= n_components:
                result = disk_result[:, :n_components]
            else:
                result = np.zeros((disk_result.shape[0], n_components), dtype=np.float32)
                result[:, :disk_result.shape[1]] = disk_result
            self._tsne_cache[cache_key] = result
            elapsed = time.time() - start_time
            _log("load_tsne", "Loaded from disk", key=key_short, shape=result.shape, elapsed_ms=f"{elapsed*1000:.1f}")
            return result

        # Not found - CRASH: embeddings MUST be pre-computed
        raise ValueError(
            f"t-SNE embedding not found for L{layer}_{component}_{position}.\n"
            f"Checked path: {self._get_embedding_path('tsne', layer, component, position)}\n"
            "Run compute_geometry_analysis.py --full to pre-compute t-SNE embeddings."
        )

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
        """Convert a time value to months. Crashes on unknown units."""
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
        if unit_lower not in conversions:
            raise ValueError(f"Unknown time unit: '{unit}'. Valid units: {list(conversions.keys())}")
        return value * conversions[unit_lower]

    def _get_time_scale(self, months: float) -> int:
        """Classify time horizon into discrete scale categories.

        Uses extended scale to cover full range from seconds to centuries.

        Returns:
            0 = Seconds (< 1 minute)
            1 = Minutes (< 1 hour)
            2 = Hours (< 1 day)
            3 = Days (< 1 week)
            4 = Weeks (< 1 month)
            5 = Months (< 1 year)
            6 = Years (< 10 years)
            7 = Decades (< 100 years)
            8 = Centuries (100+ years)
            -1 = No horizon (sentinel value)
        """
        if months < 0:  # Sentinel for no horizon
            return -1

        # Convert thresholds to months
        # 1 month = 30.44 days = 730.5 hours = 43830 minutes
        MINUTE_IN_MONTHS = 1 / 43830  # ~2.3e-5
        HOUR_IN_MONTHS = 1 / 730.5    # ~0.00137
        DAY_IN_MONTHS = 1 / 30.44     # ~0.0329
        WEEK_IN_MONTHS = 7 / 30.44    # ~0.23

        if months < MINUTE_IN_MONTHS:  # < 1 minute
            return 0  # Seconds
        elif months < HOUR_IN_MONTHS:  # < 1 hour
            return 1  # Minutes
        elif months < DAY_IN_MONTHS:  # < 1 day
            return 2  # Hours
        elif months < WEEK_IN_MONTHS:  # < 1 week
            return 3  # Days
        elif months < 1:  # < 1 month
            return 4  # Weeks
        elif months < 12:  # < 1 year
            return 5  # Months
        elif months < 120:  # < 10 years
            return 6  # Years
        elif months < 1200:  # < 100 years
            return 7  # Decades
        else:  # 100+ years
            return 8  # Centuries

    def get_time_scale_labels(self) -> list[str]:
        """Get human-readable labels for time scale categories."""
        return ["Seconds", "Minutes", "Hours", "Days", "Weeks", "Months", "Years", "Decades", "Centuries"]

    def get_sample_metadata(self, color_by: str) -> np.ndarray:
        """Get sample metadata for coloring.

        First checks metadata cache, then falls back to sample fields or computation.
        """
        if not self._samples:
            return np.array([])

        # Check metadata cache first
        cache_key = f"{self._cache_prefix}|metadata_{color_by}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        result = self._compute_metadata_values(color_by)
        self._metadata_cache[cache_key] = result
        return result

    def _compute_metadata_values(self, color_by: str) -> np.ndarray:
        """Compute metadata values for coloring.

        Uses pre-loaded values from choice.json.
        """
        # Time horizon fields
        # Use -1 as sentinel for None values (no-horizon samples)
        if color_by == "time_horizon":
            return np.array([
                s["time_horizon_months"] if s["time_horizon_months"] is not None else -1.0
                for s in self._samples
            ])
        elif color_by == "log_time_horizon":
            return np.array([
                np.log10(s["time_horizon_months"] + 1) if s["time_horizon_months"] is not None else np.nan
                for s in self._samples
            ])
        elif color_by == "time_horizon_days":
            # Compute from months (30.44 days per month on average)
            return np.array([
                s["time_horizon_months"] * 30.44 if s["time_horizon_months"] is not None else np.nan
                for s in self._samples
            ])
        elif color_by == "time_horizon_years":
            # Compute from months
            return np.array([
                s["time_horizon_months"] / 12.0 if s["time_horizon_months"] is not None else np.nan
                for s in self._samples
            ])
        elif color_by == "time_scale":
            # Compute time_scale from time_horizon_months and return labels
            labels = self.get_time_scale_labels()  # ["Weeks", "Months", "Years", "Decades"]
            values = []
            for s in self._samples:
                months = s["time_horizon_months"]
                if months is None:
                    values.append("No Horizon")
                else:
                    scale_idx = self._get_time_scale(months)
                    values.append(labels[scale_idx] if 0 <= scale_idx < len(labels) else "Unknown")
            return np.array(values)
        elif color_by == "has_horizon":
            return np.array([s["time_horizon_months"] is not None for s in self._samples])
        # Choice fields
        elif color_by == "term_chosen":
            # Return categorical labels for term chosen
            return np.array(["Long" if s["chose_long_term"] else "Short" for s in self._samples])
        elif color_by == "chosen_time":
            return np.array([
                s["chosen_time_months"] if s["chosen_time_months"] is not None else -1.0
                for s in self._samples
            ])
        elif color_by == "chosen_time_scale":
            # Compute time_scale from chosen_time_months and return labels
            labels = self.get_time_scale_labels()  # ["Weeks", "Months", "Years", "Decades"]
            values = []
            for s in self._samples:
                months = s["chosen_time_months"]
                if months is None:
                    values.append("Unknown")
                else:
                    scale_idx = self._get_time_scale(months)
                    values.append(labels[scale_idx] if 0 <= scale_idx < len(labels) else "Unknown")
            return np.array(values)
        elif color_by == "chosen_reward":
            return np.array([s["chosen_reward"] for s in self._samples])
        elif color_by == "choice_confidence":
            return np.array([s["choice_prob"] for s in self._samples])
        # Alternative fields
        elif color_by == "alt_time":
            return np.array([
                s["alt_time_months"] if s["alt_time_months"] is not None else -1.0
                for s in self._samples
            ])
        elif color_by == "alt_reward":
            return np.array([s["alt_reward"] for s in self._samples])
        elif color_by == "alt_prob":
            return np.array([s["alt_prob"] for s in self._samples])
        elif color_by == "matches_largest_reward":
            # Computed: True if model chose the option with higher reward
            vals = []
            for s in self._samples:
                chose_long = s["chose_long_term"]
                pp = s["prompt"]["preference_pair"]
                short_reward = pp["short_term"]["reward"]["value"]
                long_reward = pp["long_term"]["reward"]["value"]
                long_has_more = long_reward > short_reward
                vals.append(chose_long == long_has_more)
            return np.array(vals)
        elif color_by == "matches_rational":
            # Computed: True if model chose rationally given time horizon
            # Rational = choose long-term if time_horizon >= long_term_time
            # For no-horizon samples, returns NaN
            vals = []
            for s in self._samples:
                chose_long = s["chose_long_term"]
                th_months = s["time_horizon_months"]
                if th_months is None:
                    vals.append(np.nan)
                    continue
                pp = s["prompt"]["preference_pair"]
                long_time = pp["long_term"]["time"]
                long_months = self._convert_to_months(long_time["value"], long_time["unit"])
                rational_long = th_months >= long_months
                vals.append(chose_long == rational_long)
            return np.array(vals)
        elif color_by == "matches_associated":
            # Computed: True if model chose option with time closest to horizon
            # For no-horizon samples, returns NaN
            vals = []
            for s in self._samples:
                chose_long = s["chose_long_term"]
                th_months = s["time_horizon_months"]
                if th_months is None:
                    vals.append(np.nan)
                    continue
                pp = s["prompt"]["preference_pair"]
                short_time = pp["short_term"]["time"]
                long_time = pp["long_term"]["time"]
                short_months = self._convert_to_months(short_time["value"], short_time["unit"])
                long_months = self._convert_to_months(long_time["value"], long_time["unit"])
                long_closer = abs(th_months - long_months) < abs(th_months - short_months)
                vals.append(chose_long == long_closer)
            return np.array(vals)
        # Computed deltas
        elif color_by == "option_confidence_delta":
            # Compute: |choice_prob - 0.5| * 2: 0 = uncertain, 1 = very confident
            return np.array([abs(s["choice_prob"] - 0.5) * 2 for s in self._samples])
        elif color_by == "option_time_delta":
            # Difference in time between long_term and short_term options (months)
            # STRICT: prompt.preference_pair REQUIRED
            vals = []
            for s in self._samples:
                pp = s["prompt"]["preference_pair"]
                short_time = pp["short_term"]["time"]
                long_time = pp["long_term"]["time"]
                short_months = self._convert_to_months(short_time["value"], short_time["unit"])
                long_months = self._convert_to_months(long_time["value"], long_time["unit"])
                vals.append(long_months - short_months)
            return np.array(vals)
        elif color_by == "option_reward_delta":
            # Difference in reward between long_term and short_term options
            # STRICT: prompt.preference_pair REQUIRED
            vals = []
            for s in self._samples:
                pp = s["prompt"]["preference_pair"]
                short_reward = pp["short_term"]["reward"]["value"]
                long_reward = pp["long_term"]["reward"]["value"]
                vals.append(long_reward - short_reward)
            return np.array(vals)
        elif color_by == "reward_ratio":
            # Ratio of long_term_reward / short_term_reward
            # STRICT: prompt.preference_pair REQUIRED
            vals = []
            for s in self._samples:
                pp = s["prompt"]["preference_pair"]
                short_reward = pp["short_term"]["reward"]["value"]
                long_reward = pp["long_term"]["reward"]["value"]
                if short_reward <= 0:
                    raise ValueError(f"short_term reward must be positive, got {short_reward}")
                vals.append(long_reward / short_reward)
            return np.array(vals)
        elif color_by == "time_ratio":
            # Ratio of long_term_time / short_term_time
            # STRICT: prompt.preference_pair REQUIRED
            vals = []
            for s in self._samples:
                pp = s["prompt"]["preference_pair"]
                short_time = pp["short_term"]["time"]
                long_time = pp["long_term"]["time"]
                short_months = self._convert_to_months(short_time["value"], short_time["unit"])
                long_months = self._convert_to_months(long_time["value"], long_time["unit"])
                if short_months <= 0:
                    raise ValueError(f"short_term time must be positive, got {short_months}")
                vals.append(long_months / short_months)
            return np.array(vals)
        elif color_by == "short_term_reward":
            # STRICT: prompt.preference_pair REQUIRED
            return np.array([
                s["prompt"]["preference_pair"]["short_term"]["reward"]["value"]
                for s in self._samples
            ])
        elif color_by == "short_term_time":
            # STRICT: prompt.preference_pair REQUIRED
            vals = []
            for s in self._samples:
                short_time = s["prompt"]["preference_pair"]["short_term"]["time"]
                vals.append(self._convert_to_months(short_time["value"], short_time["unit"]))
            return np.array(vals)
        elif color_by == "long_term_reward":
            # STRICT: prompt.preference_pair REQUIRED
            return np.array([
                s["prompt"]["preference_pair"]["long_term"]["reward"]["value"]
                for s in self._samples
            ])
        elif color_by == "long_term_time":
            vals = []
            for s in self._samples:
                long_time = s["prompt"]["preference_pair"]["long_term"]["time"]
                vals.append(self._convert_to_months(long_time["value"], long_time["unit"]))
            return np.array(vals)
        # Prompt metadata (at root level of prompt_sample.json)
        elif color_by == "short_term_first":
            return np.array([s.get("short_term_first", False) for s in self._samples])
        elif color_by == "context_id":
            return np.array([s.get("context_id", 0) for s in self._samples])
        elif color_by == "sample_idx":
            return np.arange(len(self._samples))
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")

    def get_no_horizon_mask(self) -> np.ndarray:
        """Get boolean mask for samples without time horizon."""
        return np.array([
            self._extract_nested(s, "prompt.time_horizon.value") is None
            for s in self._samples
        ])

    def get_color_options(self) -> list[str]:
        """Get available color-by options in priority order for UI display."""
        return [
            # Primary time horizon metrics
            "time_horizon",  # Time Horizon (months)
            "time_scale",  # Horizon Time Scale: Weeks, Months, Years, Decades
            # Model's choice
            "term_chosen",  # Term Chosen: "Long" or "Short"
            "chosen_time",  # Chosen Time (months)
            "chosen_time_scale",  # Chosen Time Scale: computed category
            "chosen_reward",  # Chosen Reward
            # Behavioral match metrics
            "matches_rational",  # Chose Rational (chose long when horizon >= long_time)
            "matches_associated",  # Matches Associated (chose option closer to horizon)
            "matches_largest_reward",  # Chose Largest Reward
            # Confidence
            "choice_confidence",  # Choice Confidence (probability)
            "option_confidence_delta",  # Confidence Delta (0 = uncertain, 1 = confident)
            # Time horizon variants
            "log_time_horizon",  # Log Time Horizon
            "time_horizon_days",  # Time Horizon (days)
            "time_horizon_years",  # Time Horizon (years)
            "has_horizon",  # Has Horizon (boolean)
            # Option deltas
            "option_time_delta",  # Option Time Delta (long - short months)
            "option_reward_delta",  # Option Reward Delta (long - short)
            # Ratios
            "reward_ratio",  # Reward Ratio (long / short)
            "time_ratio",  # Time Ratio (long / short)
            # Individual option values
            "short_term_reward",  # Short Term Reward
            "short_term_time",  # Short Term Time (months)
            "long_term_reward",  # Long Term Reward
            "long_term_time",  # Long Term Time (months)
            # Alternative (unchosen) option
            "alt_time",  # Alt Time (months)
            "alt_reward",  # Alt Reward
            "alt_prob",  # Alt Probability
            # Prompt metadata
            "short_term_first",  # Short Term Listed First
            "context_id",  # Context ID
            "sample_idx",  # Sample Index
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
            "objective_marker": "CONSIDER:",
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
            "chat_suffix": "Chat Suffix",
            "situation_content": "Situation Content",
            "task_content": "Task Content",
            "objective_content": "Consider Content",
            "action_content": "Action Content",
            "format_content": "Format Content",
            "prompt_other": "Other Prompt Tokens",
            "response_other": "Other Response Tokens",
            # Section tail positions (last token of each section)
            "situation_tail": "Situation Tail",
            "task_tail": "Task Tail",
            "objective_tail": "Consider Tail",
            "action_tail": "Action Tail",
            "format_tail": "Format Tail",
            "options_tail": "Options Tail (end of right_time)",
        }
        # Add any missing positions
        for pos in self._semantic_positions:
            if pos not in labels:
                labels[pos] = pos.replace("_", " ").title()
        return labels

    def get_markers(self) -> dict[str, str]:
        """Get section markers from DefaultPromptFormat."""
        from ..formatting.configs.default_prompt_format import DefaultPromptFormat

        fmt = DefaultPromptFormat()
        markers = {}
        marker_keys = [
            "situation_marker", "task_marker", "objective_marker",
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
        return ""

    def get_prompt_template_structure(self) -> list[dict]:
        """Get prompt template structure for UI display.

        Returns ALL known format positions with availability based on
        precomputed embeddings.
        """
        # Use precomputed positions as the source of availability
        precomputed = set(self.get_precomputed_positions())
        labels = self.get_position_labels()

        # All known format positions in canonical order
        all_positions = [
            "chat_prefix", "chat_prefix_tail",
            "situation_marker", "situation", "situation_content", "situation_tail",
            "task_marker", "task_content", "role", "task_in_question", "task_tail",
            "left_label", "option_content", "left_reward", "left_reward_units", "left_time",
            "right_label", "right_reward", "right_reward_units", "right_time", "options_tail",
            "objective_marker", "objective_content", "objective_tail",
            "time_horizon", "post_time_horizon",
            "constraint_marker", "constraint_prefix", "constraint_content", "constraint_tail",
            "action_marker", "action_content", "action_tail", "reasoning_ask",
            "format_marker", "format_content", "format_choice_prefix", "format_reasoning_prefix", "format_tail",
            "chat_suffix", "chat_suffix_tail",
            "response_choice_prefix", "response_choice", "response_reasoning_prefix", "response_reasoning", "response_other",
        ]

        elements = []
        for name in all_positions:
            elements.append({
                "name": name,
                "label": labels.get(name, name.replace("_", " ").title()),
                "type": "semantic",
                "available": name in precomputed,
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

    def load_pca_metrics_legacy(
        self, layer: int, component: str, position: str
    ) -> dict | None:
        """Load PCA metrics for a target (legacy path)."""
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
        """Pre-load embeddings into memory cache.

        Loads pre-computed embeddings from disk into memory for faster access.
        Does NOT compute any embeddings - they must already exist on disk.
        """
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

    def get_position_mapping_for_sample(self, sample_idx: int) -> dict:
        """Get the full position mapping for a sample.

        Raises:
            ValueError: If position_mapping not loaded.
            KeyError: If sample_idx not found in mappings.
        """
        if not self._position_mapping:
            raise ValueError("position_mapping not loaded")
        for mapping in self._position_mapping["mappings"]:
            if mapping["sample_idx"] == sample_idx:
                return mapping
        raise KeyError(f"Sample index {sample_idx} not found in position mappings")

    def get_tokens_for_sample(self, sample_idx: int) -> list[dict]:
        """Get token info for a sample including format_pos labels.

        Raises:
            ValueError: If position_mapping not loaded.
            KeyError: If sample_idx not found or 'positions' key missing.
        """
        mapping = self.get_position_mapping_for_sample(sample_idx)
        return mapping["positions"]

    def get_example_sample_with_horizon(self) -> dict | None:
        """Get a sample that has time_horizon for use as an illustration.

        Returns the position mapping with decoded tokens grouped by format_pos,
        suitable for displaying actual sample text in the UI.
        """
        # Find a sample with time_horizon
        has_horizon_values = self.get_sample_metadata("has_horizon")
        example_idx = None
        for idx, has_horizon in enumerate(has_horizon_values):
            if has_horizon:
                example_idx = idx
                break

        if example_idx is None:
            # Fallback to first sample
            example_idx = 0

        mapping = self.get_position_mapping_for_sample(example_idx)
        # Group tokens by format_pos and concatenate decoded tokens
        format_texts: dict[str, str] = {}
        for pos_info in mapping["positions"]:
            format_pos = pos_info["format_pos"]
            decoded = pos_info["decoded_token"]
            if format_pos not in format_texts:
                format_texts[format_pos] = ""
            format_texts[format_pos] += decoded

        return {
            "sample_idx": example_idx,
            "format_texts": format_texts,
        }

    def preload_all_metadata(self) -> int:
        """Preload all metadata/color values into memory cache.

        Returns the number of color options loaded.
        """
        color_options = self.get_color_options()
        loaded = 0
        for color_by in color_options:
            # STRICT: crash on any failure - metadata MUST be valid
            cache_key = f"{self._cache_prefix}|metadata_{color_by}"
            if cache_key not in self._metadata_cache:
                values = self.get_sample_metadata(color_by)
                self._metadata_cache[cache_key] = values
            loaded += 1
        return loaded

    def preload_all_tokens(self) -> int:
        """Preload token mappings for all samples.

        Token mappings are already loaded in _position_mapping during _load_data(),
        so this just validates they're accessible. Returns the number of samples
        with valid token mappings.
        """
        if not self._position_mapping:
            return 0
        # Iterate over actual mappings, not assumed 0 to n-1 indices
        loaded = 0
        for mapping in self._position_mapping["mappings"]:
            if mapping.get("positions"):
                loaded += 1
        return loaded

    def warmup_all(
        self,
        methods: list[str] | None = None,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        positions: list[str] | None = None,
        include_metadata: bool = True,
        include_tokens: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Comprehensive preload of ALL data into memory.

        Loads:
        - All embeddings (PCA, UMAP, t-SNE) in parallel
        - All metadata/color values
        - All token mappings
        - All valid sample indices

        Args:
            methods: Methods to load (default: pca, umap, tsne)
            layers: Layers to load (default: all from data)
            components: Components to load (default: all)
            positions: Positions to load (default: all from get_positions())
            include_metadata: Whether to preload metadata
            include_tokens: Whether to preload token mappings
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with counts of loaded items
        """
        if methods is None:
            methods = ["pca", "umap", "tsne"]
        if layers is None:
            layers = self._layers
        if components is None:
            components = self.get_components()
        if positions is None:
            positions = self.get_positions()

        results = {
            "embeddings": 0,
            "metadata": 0,
            "tokens": 0,
            "sample_indices": 0,
        }

        # Count total tasks for progress
        embedding_tasks = len(methods) * len(layers) * len(components) * len(positions)
        metadata_tasks = len(self.get_color_options()) if include_metadata else 0
        token_tasks = 1 if include_tokens else 0  # Tokens are one bulk operation
        total_tasks = embedding_tasks + metadata_tasks + token_tasks

        current_task = [0]  # Use list for mutable reference in closure

        # Helper to report progress
        def report(desc: str):
            current_task[0] += 1
            if progress_callback:
                progress_callback(current_task[0], total_tasks, desc)

        # Load all embeddings using parallel loading
        def load_single_embedding(args):
            method, layer, component, position = args
            result = None
            if method == "pca":
                result = self.load_pca(layer, component, position)
            elif method == "umap":
                result = self.load_umap(layer, component, position)
            elif method == "tsne":
                result = self.load_tsne(layer, component, position)
            # Also load sample indices
            self.get_valid_sample_indices(layer, component, position)
            return result is not None

        # Build list of all embedding tasks
        embedding_args = [
            (method, layer, component, position)
            for method in methods
            for layer in layers
            for component in components
            for position in positions
        ]

        # Parallel embedding loading
        _log("warmup_all", f"Loading {len(embedding_args)} embeddings in parallel...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(load_single_embedding, args): args for args in embedding_args}
            for future in as_completed(futures):
                if future.result():
                    results["embeddings"] += 1
                report(f"Embedding {results['embeddings']}/{len(embedding_args)}")

        # Preload metadata - STRICT: crash on any failure
        if include_metadata:
            _log("warmup_all", "Loading all metadata/color values...")
            for color_by in self.get_color_options():
                # No try/except - let exceptions propagate
                cache_key = f"{self._cache_prefix}|metadata_{color_by}"
                if cache_key not in self._metadata_cache:
                    values = self.get_sample_metadata(color_by)
                    self._metadata_cache[cache_key] = values
                results["metadata"] += 1
                report(f"Metadata: {color_by}")

        # Preload tokens (validate they're accessible)
        if include_tokens:
            _log("warmup_all", "Validating token mappings...")
            results["tokens"] = self.preload_all_tokens()
            report("Token mappings validated")

        return results

    def load_layer_trajectory(
        self,
        component: str,
        position: str,
        include_pc2: bool = False,
        mode: str = "aligned",
    ) -> tuple[list[int], np.ndarray, list[int]] | tuple[list[int], np.ndarray, np.ndarray, list[int]]:
        """Load pre-cached layer trajectory data (PC1/PC2 across all layers).

        Args:
            component: Model component (resid_pre, attn_out, etc.)
            position: Semantic position name (may include :rel_pos suffix)
            include_pc2: If True, also return PC2 values
            mode: "aligned" (per-target PCA + sign alignment) or "shared" (single PCA subspace)

        Returns:
            If include_pc2=False: Tuple of (layers, pc1_values, sample_indices).
            If include_pc2=True: Tuple of (layers, pc1_values, pc2_values, sample_indices).
            pc1_values/pc2_values have shape (n_layers, n_samples).

        Raises:
            ValueError: If trajectory cache file does not exist (STRICT mode).
        """
        # Handle per-rel_pos positions (e.g., "chat_suffix:0" -> "chat_suffix_r0")
        if ":" in position:
            parts = position.rsplit(":", 1)
            base_pos = parts[0]
            try:
                rel_pos = int(parts[1])
                file_position = f"{base_pos}_r{rel_pos}"
            except ValueError:
                file_position = position
        else:
            file_position = position
            base_pos = position

        # Add _shared suffix for shared mode
        suffix = "_shared" if mode == "shared" else ""
        cache_file = self.data_dir / "analysis" / "trajectories" / f"layers_{component}_{file_position}{suffix}.npz"

        if not cache_file.exists():
            raise ValueError(
                f"Layer trajectory cache not found: {cache_file}\n"
                "Run compute_geometry_analysis.py to generate trajectory data first."
            )

        data = np.load(cache_file)
        layers = data["layers"].tolist()
        pc1_values = data["pc1_values"]  # shape: (n_layers, n_samples)
        sample_indices = data["sample_indices"].tolist()
        _log("trajectory", "Loaded cached layer trajectory", comp=component, position=position, mode=mode)

        if include_pc2:
            if "pc2_values" not in data:
                raise ValueError(
                    f"pc2_values not found in trajectory cache: {cache_file}\n"
                    "Trajectory was computed without PC2. Re-run compute_geometry_analysis.py."
                )
            pc2_values = data["pc2_values"]
            return layers, pc1_values, pc2_values, sample_indices
        return layers, pc1_values, sample_indices

    def load_position_trajectory(
        self,
        layer: int,
        component: str,
        include_pc2: bool = False,
        mode: str = "aligned",
    ) -> tuple[list[str], list[np.ndarray], list[list[int]]] | tuple[list[str], list[np.ndarray], list[np.ndarray], list[list[int]]]:
        """Load pre-cached position trajectory data (PC1/PC2 across all positions).

        Args:
            layer: Layer number
            component: Model component (resid_pre, attn_out, etc.)
            include_pc2: If True, also return PC2 values
            mode: "aligned" (per-target PCA + sign alignment) or "shared" (single PCA subspace)

        Returns:
            If include_pc2=False: Tuple of (positions, pc1_values_list, sample_indices_per_position).
            If include_pc2=True: Tuple of (positions, pc1_values_list, pc2_values_list, sample_indices_per_position).
            pc1/pc2_values_list is a list of arrays (one per position, may have different lengths).
            sample_indices_per_position is a list of sample index lists (one per position).

        Raises:
            ValueError: If trajectory cache file does not exist (STRICT mode).
        """
        # Add _shared suffix for shared mode
        suffix = "_shared" if mode == "shared" else ""
        cache_file = self.data_dir / "analysis" / "trajectories" / f"positions_L{layer}_{component}{suffix}.npz"
        if not cache_file.exists():
            raise ValueError(
                f"Position trajectory cache not found: {cache_file}\n"
                "Run compute_geometry_analysis.py to generate trajectory data first."
            )

        data = np.load(cache_file, allow_pickle=True)
        positions = data["positions"].tolist()
        # pc1_values is stored as object array - each element is a 1D array
        pc1_values_list = [arr for arr in data["pc1_values"]]
        sample_indices_list = [list(arr) for arr in data["sample_indices_list"]]
        _log("trajectory", "Loaded cached position trajectory", layer=layer, comp=component, mode=mode)

        if include_pc2:
            if "pc2_values" not in data:
                raise ValueError(
                    f"pc2_values not found in trajectory cache: {cache_file}\n"
                    "Trajectory was computed without PC2. Re-run compute_geometry_analysis.py."
                )
            pc2_values_list = [arr for arr in data["pc2_values"]]
            return positions, pc1_values_list, pc2_values_list, sample_indices_list
        return positions, pc1_values_list, sample_indices_list

    def _resolve_position_for_pca_analysis(self, position: str) -> str:
        """Resolve position for PCA analysis lookup (scree, alignment).

        Converts "response_choice:0" to "response_choice_r0" for file lookup.
        Per-rel_pos PCA metrics are computed during embedding generation.
        """
        if ":" in position:
            base, rel_pos = position.rsplit(":", 1)
            return f"{base}_r{rel_pos}"
        return position

    def load_pca_metrics(
        self,
        layer: int,
        component: str,
        position: str,
    ) -> dict | None:
        """Load PCA metrics including variance explained.

        Uses COMBINED position data (not per-rel_pos) because variance explained
        is a property of the full position's data distribution.

        Returns:
            Dict with 'explained_variance' (list of floats) and other metrics,
            or None if not found.
        """
        resolved_position = self._resolve_position_for_pca_analysis(position)
        metrics_file = self.data_dir / "analysis" / "pca" / f"L{layer}_{component}_{resolved_position}" / "metrics.json"
        if not metrics_file.exists():
            return None

        with open(metrics_file) as f:
            return json.load(f)

    def load_pca_components(
        self,
        layer: int,
        component: str,
        position: str,
    ) -> np.ndarray | None:
        """Load PCA components (directions) for direction alignment analysis.

        Uses COMBINED position data (not per-rel_pos) because PC directions
        are computed on the full position's data distribution.

        Returns:
            Array of shape (n_components, n_features) or None if not found.
        """
        resolved_position = self._resolve_position_for_pca_analysis(position)
        components_file = self.data_dir / "analysis" / "pca" / f"L{layer}_{component}_{resolved_position}" / "components.npy"
        if not components_file.exists():
            return None

        return np.load(components_file)

    def get_scree_data(
        self,
        position: str,
        n_components: int = 10,
    ) -> dict[str, list[dict]]:
        """Get Scree plot data (variance explained) for all layers and components.

        Returns:
            Dict with 'series': list of {label, values} where values is cumulative variance.
        """
        series = []
        components = self.get_components()

        for layer in self._layers:
            for comp in components:
                metrics = self.load_pca_metrics(layer, comp, position)
                if metrics and "explained_variance" in metrics:
                    var_explained = metrics["explained_variance"][:n_components]
                    # Convert to cumulative
                    cumulative = []
                    total = 0.0
                    for v in var_explained:
                        total += v
                        cumulative.append(total)

                    series.append({
                        "label": f"L{layer}_{comp}",
                        "layer": layer,
                        "component": comp,
                        "values": cumulative,
                    })

        return {"series": series}

    def get_direction_alignment(
        self,
        position: str,
        pc_index: int = 0,
    ) -> dict:
        """Compute direction alignment (cosine similarity) matrix for PC directions.

        Args:
            position: Semantic position to analyze.
            pc_index: Which PC to use (0 = PC1, 1 = PC2, etc.)

        Returns:
            Dict with 'labels', 'matrix' (2D cosine similarity), 'layers', 'components'.
        """
        components = self.get_components()
        labels = []
        directions = []

        # Collect all PC directions - reverse layer order so L0 is at bottom, L35 at top
        for layer in reversed(self._layers):
            for comp in components:
                pca_components = self.load_pca_components(layer, comp, position)
                if pca_components is not None and pca_components.shape[0] > pc_index:
                    labels.append(f"L{layer}_{comp[:4]}")  # Abbreviated
                    directions.append(pca_components[pc_index])

        if len(directions) == 0:
            return {"labels": [], "matrix": [], "error": "No PCA components found"}

        # Stack directions and normalize
        directions = np.array(directions)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = directions / norms

        # Compute cosine similarity matrix
        similarity = normalized @ normalized.T

        # Flip matrix horizontally so L0 is at left (origin at bottom-left)
        # Row labels stay as-is (L35 at top, L0 at bottom)
        # Column labels need to be reversed (L0 at left, L35 at right)
        similarity_flipped = np.fliplr(similarity)
        col_labels = list(reversed(labels))

        return {
            "labels": labels,  # For Y-axis (rows): L35 at top, L0 at bottom
            "col_labels": col_labels,  # For X-axis (cols): L0 at left, L35 at right
            "matrix": similarity_flipped.tolist(),
            "n_targets": len(labels),
        }
