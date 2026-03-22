"""Data collection and caching for geometric visualization.

Memory-optimized implementation:
- Activations stored as float32 (half the memory of float64)
- Each target saved to separate file during extraction
- StreamingActivationData loads targets on-demand
- Explicit memory clearing after each sample
- Configurable buffer sizes and compression
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...common.time_value import TimeValue
from ..common.preference_types import PreferenceSample, PromptSample, RewardValue
from .geo_viz_config import (
    GeoVizConfig,
    TargetSpec,
    is_absolute_position,
    parse_absolute_position,
    ACTIVATION_DTYPE,
    EXTRACTION_BUFFER_SIZE,
    USE_COMPRESSED_STORAGE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Position Resolution
# =============================================================================


@dataclass(slots=True)
class ResolvedPositions:
    """Resolved token positions for a specific sample.

    Uses __slots__ for reduced memory overhead.
    """

    named_positions: dict[str, list[int]]
    prompt_len: int
    full_len: int
    source: list[int]
    dest: list[int]

    def get(self, pos_name: str) -> int | None:
        """Get first position index by name or absolute index.

        Returns None if position doesn't exist for this sample.
        """
        if is_absolute_position(pos_name):
            idx = parse_absolute_position(pos_name)
            if idx >= self.full_len:
                return None
            return idx

        if pos_name in self.named_positions:
            positions = self.named_positions[pos_name]
            return positions[0] if positions else None

        if pos_name == "source":
            return self.source[0] if self.source else 0
        elif pos_name in ("dest", "response"):
            return self.dest[0] if self.dest else self.prompt_len
        return None

    def get_all(self, pos_name: str) -> list[int]:
        """Get all position indices for a named position or absolute index."""
        if is_absolute_position(pos_name):
            idx = parse_absolute_position(pos_name)
            if idx >= self.full_len:
                return []
            return [idx]

        if pos_name in self.named_positions:
            return self.named_positions[pos_name]
        elif pos_name == "source":
            return self.source
        elif pos_name == "dest":
            return self.dest
        raise ValueError(f"Unknown position name: {pos_name}")


def _find_substring_token_range(
    tokens: list[str], text: str, substring: str
) -> list[int]:
    """Find all token positions spanning a substring in text."""
    char_idx = text.find(substring)
    if char_idx == -1:
        return []

    char_end = char_idx + len(substring)
    positions = []

    char_count = 0
    for i, tok in enumerate(tokens):
        tok_start = char_count
        tok_end = char_count + len(tok)
        char_count = tok_end

        if tok_end > char_idx and tok_start < char_end:
            positions.append(i)

        if tok_end >= char_end:
            break

    return positions


def _find_time_value_positions(
    tokens: list[str], text: str, time_val: TimeValue
) -> list[int]:
    """Find token positions for a TimeValue's numeric value and unit."""
    positions = []

    formatted = str(time_val)
    parts = formatted.split(" ", 1)
    value_str = parts[0]
    unit_str = parts[1] if len(parts) > 1 else time_val.unit

    positions.extend(_find_substring_token_range(tokens, text, value_str))
    positions.extend(_find_substring_token_range(tokens, text, unit_str))

    return positions


def _find_reward_value_positions(
    tokens: list[str], text: str, reward_val: RewardValue
) -> list[int]:
    """Find token positions for a RewardValue's numeric value and unit."""
    positions = []

    value_str = str(RewardValue(reward_val.value))
    positions.extend(_find_substring_token_range(tokens, text, value_str))

    if reward_val.unit:
        positions.extend(_find_substring_token_range(tokens, text, reward_val.unit))

    return positions


def resolve_positions(
    sample: PromptSample,
    pref: PreferenceSample,
    runner,
) -> ResolvedPositions:
    """Resolve token positions using exact sample structure."""
    prompt_len = pref.prompt_token_count
    full_tokens = pref.chosen_traj.token_ids
    tokens = [runner._tokenizer.decode([t]) for t in full_tokens]

    full_len = len(tokens)
    prompt_tokens_decoded = tokens[:prompt_len]
    prompt_text = "".join(prompt_tokens_decoded)

    pair = sample.prompt.preference_pair
    named_positions = {}

    if sample.prompt.time_horizon is not None:
        named_positions["time_horizon"] = _find_time_value_positions(
            prompt_tokens_decoded, prompt_text, sample.prompt.time_horizon
        )

    named_positions["short_term_time"] = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, pair.short_term.time
    )
    named_positions["short_term_reward"] = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, pair.short_term.reward
    )
    named_positions["long_term_time"] = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, pair.long_term.time
    )
    named_positions["long_term_reward"] = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, pair.long_term.reward
    )
    named_positions["response"] = list(range(prompt_len, full_len))

    # Clamp all positions
    for key in named_positions:
        named_positions[key] = [max(0, min(p, full_len - 1)) for p in named_positions[key]]

    # Build legacy source/dest
    source_positions = []
    for key in ["time_horizon", "short_term_time", "short_term_reward",
                "long_term_time", "long_term_reward"]:
        if key in named_positions:
            source_positions.extend(named_positions[key])
    source_positions = sorted(set(source_positions))

    if not source_positions:
        source_positions.append(int(prompt_len * 0.6))

    dest_positions = named_positions.get("response", [])

    return ResolvedPositions(
        named_positions=named_positions,
        source=source_positions,
        dest=dest_positions,
        prompt_len=prompt_len,
        full_len=full_len,
    )


# =============================================================================
# Choice Info
# =============================================================================


@dataclass(slots=True)
class ChoiceInfo:
    """Choice information for a single sample. Uses __slots__."""

    chose_long_term: bool
    chosen_time_months: float
    chosen_reward: float
    choice_prob: float

    def to_dict(self) -> dict:
        return {
            "chose_long_term": self.chose_long_term,
            "chosen_time_months": self.chosen_time_months,
            "chosen_reward": self.chosen_reward,
            "choice_prob": self.choice_prob,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChoiceInfo":
        return cls(
            chose_long_term=data["chose_long_term"],
            chosen_time_months=data["chosen_time_months"],
            chosen_reward=data["chosen_reward"],
            choice_prob=data["choice_prob"],
        )


# =============================================================================
# File I/O Utilities
# =============================================================================


def _save_array(path: Path, arr: np.ndarray, compressed: bool = False):
    """Save numpy array, optionally compressed."""
    if compressed:
        np.savez_compressed(path.with_suffix(".npz"), data=arr)
    else:
        np.save(path.with_suffix(".npy"), arr)


def _load_array(path: Path) -> np.ndarray:
    """Load numpy array (handles both .npy and .npz)."""
    npz_path = path.with_suffix(".npz")
    npy_path = path.with_suffix(".npy")

    if npz_path.exists():
        with np.load(npz_path) as f:
            return f["data"]
    elif npy_path.exists():
        return np.load(npy_path)
    else:
        raise FileNotFoundError(f"No array file found: {path}")


def _array_exists(path: Path) -> bool:
    """Check if array file exists (either .npy or .npz)."""
    return path.with_suffix(".npy").exists() or path.with_suffix(".npz").exists()


def _safe_filename(key: str) -> str:
    """Convert target key to safe filename."""
    return key.replace("/", "_").replace("\\", "_")


# =============================================================================
# Streaming Activation Data (Memory-Efficient)
# =============================================================================


@dataclass
class ActivationData:
    """Memory-efficient container for activations.

    Activations are stored on disk in per-target files.
    Load targets on-demand using load_target() or iterate with iter_targets().
    """

    samples: list[PromptSample]
    choices: list[ChoiceInfo] | None = None
    _data_dir: Path | None = None
    _target_keys: list[str] = field(default_factory=list)
    _sample_counts: dict[str, int] = field(default_factory=dict)
    _cache: dict[str, np.ndarray] = field(default_factory=dict)
    _compressed: bool = False

    @property
    def activations(self) -> dict[str, np.ndarray]:
        """Legacy property - loads ALL activations. Use load_target() instead."""
        if not self._cache or len(self._cache) < len(self._target_keys):
            logger.warning("Loading ALL activations - consider using load_target()")
            for key in self._target_keys:
                if key not in self._cache:
                    self.load_target(key)
        return self._cache

    def get_target_keys(self) -> list[str]:
        """Get list of available target keys."""
        return self._target_keys.copy()

    def get_sample_count(self, target_key: str) -> int:
        """Get sample count for a specific target."""
        return self._sample_counts.get(target_key, len(self.samples))

    def load_target(self, target_key: str) -> np.ndarray:
        """Load a single target's activations from disk."""
        if target_key in self._cache:
            return self._cache[target_key]

        if self._data_dir is None:
            raise ValueError("Data directory not set")

        target_path = self._data_dir / "targets" / _safe_filename(target_key)
        self._cache[target_key] = _load_array(target_path)
        return self._cache[target_key]

    def unload_target(self, target_key: str):
        """Remove a target from the cache to free memory."""
        if target_key in self._cache:
            del self._cache[target_key]

    def clear_cache(self):
        """Clear all cached activations."""
        self._cache.clear()
        gc.collect()

    def iter_targets(self):
        """Iterate over targets, loading one at a time. Memory efficient."""
        for key in self._target_keys:
            activations = self.load_target(key)
            yield key, activations
            self.unload_target(key)
        gc.collect()

    def save(self, path: Path, compressed: bool | None = None):
        """Save to disk with per-target files."""
        if compressed is None:
            compressed = self._compressed

        path.mkdir(parents=True, exist_ok=True)
        targets_dir = path / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)

        # Save samples (compact JSON)
        samples_data = [s.to_dict() for s in self.samples]
        with open(path / "samples.json", "w") as f:
            json.dump(samples_data, f, separators=(",", ":"))

        # Save choices
        if self.choices:
            choices_data = [c.to_dict() for c in self.choices]
            with open(path / "choices.json", "w") as f:
                json.dump(choices_data, f, separators=(",", ":"))

        # Save metadata
        metadata = {
            "target_keys": self._target_keys,
            "sample_counts": self._sample_counts,
            "compressed": compressed,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save targets (if in cache)
        for key in self._target_keys:
            if key in self._cache:
                target_path = targets_dir / _safe_filename(key)
                _save_array(target_path, self._cache[key], compressed=compressed)

        logger.info(f"Saved {len(self.samples)} samples, {len(self._target_keys)} targets to {path}")

    @classmethod
    def load(cls, path: Path) -> "ActivationData":
        """Load from disk (lazy - doesn't load activations until requested)."""
        with open(path / "samples.json") as f:
            samples_data = json.load(f)
        samples = [PromptSample.from_dict(s) for s in samples_data]

        choices = None
        choices_path = path / "choices.json"
        if choices_path.exists():
            with open(choices_path) as f:
                choices_data = json.load(f)
            choices = [ChoiceInfo.from_dict(c) for c in choices_data]

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            target_keys = metadata.get("target_keys", [])
            sample_counts = metadata.get("sample_counts", {})
            compressed = metadata.get("compressed", False)
        else:
            # Legacy format
            target_keys, sample_counts = _scan_legacy_format(path)
            compressed = False

        data = cls(
            samples=samples,
            choices=choices,
            _data_dir=path,
            _target_keys=target_keys,
            _sample_counts=sample_counts,
            _compressed=compressed,
        )

        logger.info(f"Loaded metadata: {len(samples)} samples, {len(target_keys)} targets")
        return data


def _scan_legacy_format(path: Path) -> tuple[list[str], dict[str, int]]:
    """Scan legacy format and return target info."""
    activations_file = path / "activations.npz"
    if activations_file.exists():
        with np.load(activations_file) as f:
            return list(f.files), {k: f[k].shape[0] for k in f.files}

    targets_dir = path / "targets"
    if targets_dir.exists():
        target_keys = []
        for f in targets_dir.glob("*.npy"):
            target_keys.append(f.stem)
        for f in targets_dir.glob("*.npz"):
            target_keys.append(f.stem)
        return target_keys, {}

    return [], {}


# =============================================================================
# Sample Collection
# =============================================================================


def get_time_horizon_months(sample: PromptSample) -> float:
    """Get time horizon in months from a PromptSample."""
    if sample.prompt.time_horizon is None:
        return 60.0
    return sample.prompt.time_horizon.to_months()


def collect_samples() -> "PromptDataset":
    """Generate samples with diverse time horizons using GEO_VIZ_CFG."""
    from ..data.default_datasets import GEO_VIZ_CFG
    from ..prompt import PromptDatasetConfig, PromptDatasetGenerator

    dataset_config = PromptDatasetConfig.from_dict(GEO_VIZ_CFG)
    dataset = PromptDatasetGenerator(dataset_config).generate()

    logger.info(f"Generated {len(dataset.samples)} samples")
    return dataset


# =============================================================================
# Streaming Activation Extraction
# =============================================================================


def extract_activations(
    dataset: "PromptDataset", targets: list[TargetSpec], config: GeoVizConfig
) -> ActivationData:
    """Extract activations with minimal memory footprint.

    Memory optimizations:
    1. Each target saved to separate file as samples are processed
    2. Uses float32 (half memory)
    3. Clears model internals immediately
    4. Periodic garbage collection
    5. Configurable buffer size
    """
    from ..formatting.prompt_formats import find_prompt_format_config
    from ..preference import PreferenceQuerier, PreferenceQueryConfig

    logger.info(f"Loading model {config.model}...")

    query_config = PreferenceQueryConfig(skip_generation=True)
    querier = PreferenceQuerier(query_config)
    runner = querier._load_model(config.model)

    samples = dataset.samples
    if config.max_samples is not None and len(samples) > config.max_samples:
        samples = samples[: config.max_samples]
        logger.info(f"Limited to {config.max_samples} samples")

    hook_names = list({t.hook_name for t in targets})

    # Setup streaming save
    data_dir = config.output_dir / "data"
    targets_dir = data_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Get buffer size from config
    buffer_size = config.extraction_buffer_size
    compressed = config.use_compressed_storage

    # Preallocate buffers (more efficient than list append)
    target_buffers: dict[str, list[np.ndarray]] = {t.key: [] for t in targets}
    target_counts: dict[str, int] = {t.key: 0 for t in targets}

    valid_samples = []
    choices = []
    skipped = 0

    logger.info(f"Extracting activations (buffer_size={buffer_size}, compressed={compressed})...")

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            logger.info(f"  Sample {i}/{len(samples)} | valid: {len(valid_samples)} | skipped: {skipped}")

        prompt_format = find_prompt_format_config(sample.formatting_id)
        choice_prefix = prompt_format.get_response_prefix_before_choice()

        try:
            pref = querier.query_sample(
                sample, runner, choice_prefix, activation_names=hook_names
            )

            if pref.chosen_traj is None:
                skipped += 1
                continue

            positions = resolve_positions(sample, pref, runner)

            sample_has_data = False
            for target in targets:
                try:
                    pos_idx = positions.get(target.position)
                    if pos_idx is None:
                        continue

                    # Extract and convert to float32
                    act = pref.internals.activations[target.hook_name][pos_idx, :]
                    act_np = act.numpy().astype(ACTIVATION_DTYPE)

                    target_buffers[target.key].append(act_np)
                    sample_has_data = True

                except (ValueError, KeyError, IndexError):
                    continue

            if not sample_has_data:
                skipped += 1
                pref.internals = None
                del pref
                continue

            # Capture choice info
            pair = sample.prompt.preference_pair
            chose_long = pref.chose_long_term
            if chose_long:
                chosen_time = pair.long_term.time.to_months()
                chosen_reward = pair.long_term.reward.value
            else:
                chosen_time = pair.short_term.time.to_months()
                chosen_reward = pair.short_term.reward.value

            choices.append(
                ChoiceInfo(
                    chose_long_term=chose_long,
                    chosen_time_months=chosen_time,
                    chosen_reward=chosen_reward,
                    choice_prob=pref.choice_prob,
                )
            )
            valid_samples.append(sample)

            # Clear internals immediately
            pref.internals = None
            del pref

            # Flush buffers periodically
            if len(valid_samples) % buffer_size == 0:
                _flush_buffers(target_buffers, target_counts, targets_dir, compressed)
                gc.collect()

        except Exception as e:
            logger.warning(f"  Skipping sample {i}: {e}")
            skipped += 1
            continue

    # Final flush
    _flush_buffers(target_buffers, target_counts, targets_dir, compressed)
    del target_buffers
    gc.collect()

    logger.info(f"Extracted {len(valid_samples)} valid samples (skipped {skipped})")

    # Build sample counts
    sample_counts = {k: v for k, v in target_counts.items() if v > 0}
    for key, count in list(sample_counts.items())[:5]:
        logger.info(f"  {key}: {count} samples")
    if len(sample_counts) > 5:
        logger.info(f"  ... and {len(sample_counts) - 5} more targets")

    # Create ActivationData
    data = ActivationData(
        samples=valid_samples,
        choices=choices,
        _data_dir=data_dir,
        _target_keys=list(sample_counts.keys()),
        _sample_counts=sample_counts,
        _compressed=compressed,
    )

    # Save metadata
    data.save(data_dir, compressed=compressed)

    return data


def _flush_buffers(
    buffers: dict[str, list[np.ndarray]],
    counts: dict[str, int],
    targets_dir: Path,
    compressed: bool,
):
    """Flush activation buffers to disk."""
    for key, buffer in buffers.items():
        if not buffer:
            continue

        target_path = targets_dir / _safe_filename(key)
        new_data = np.stack(buffer)

        # Check if file exists and append
        if _array_exists(target_path):
            existing = _load_array(target_path)
            combined = np.concatenate([existing, new_data], axis=0)
            _save_array(target_path, combined, compressed=compressed)
            del existing, combined
        else:
            _save_array(target_path, new_data, compressed=compressed)

        counts[key] += len(buffer)
        buffer.clear()
        del new_data


# =============================================================================
# Caching
# =============================================================================


def load_cached_data(config: GeoVizConfig) -> ActivationData | None:
    """Load cached data if available."""
    cache_path = config.output_dir / "data"

    # Check for new format
    if (cache_path / "samples.json").exists() and (cache_path / "metadata.json").exists():
        try:
            return ActivationData.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    # Check for legacy format
    if (cache_path / "samples.json").exists() and (cache_path / "activations.npz").exists():
        try:
            return _convert_legacy_cache(cache_path, config.use_compressed_storage)
        except Exception as e:
            logger.warning(f"Failed to convert legacy cache: {e}")
            return None

    return None


def _convert_legacy_cache(path: Path, compressed: bool) -> ActivationData:
    """Convert legacy activations.npz to per-target files."""
    logger.info("Converting legacy cache to streaming format...")

    with open(path / "samples.json") as f:
        samples_data = json.load(f)
    samples = [PromptSample.from_dict(s) for s in samples_data]

    choices = None
    choices_path = path / "choices.json"
    if choices_path.exists():
        with open(choices_path) as f:
            choices_data = json.load(f)
        choices = [ChoiceInfo.from_dict(c) for c in choices_data]

    # Convert activations
    targets_dir = path / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    target_keys = []
    sample_counts = {}

    with np.load(path / "activations.npz") as activations_file:
        for key in activations_file.files:
            arr = activations_file[key].astype(ACTIVATION_DTYPE)
            target_path = targets_dir / _safe_filename(key)
            _save_array(target_path, arr, compressed=compressed)
            target_keys.append(key)
            sample_counts[key] = arr.shape[0]
            logger.info(f"  Converted {key}: {arr.shape}")
            del arr
            gc.collect()

    # Save metadata
    metadata = {
        "target_keys": target_keys,
        "sample_counts": sample_counts,
        "compressed": compressed,
    }
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return ActivationData(
        samples=samples,
        choices=choices,
        _data_dir=path,
        _target_keys=target_keys,
        _sample_counts=sample_counts,
        _compressed=compressed,
    )


def save_data(data: ActivationData, config: GeoVizConfig):
    """Save data to cache."""
    cache_path = config.output_dir / "data"
    data.save(cache_path, compressed=config.use_compressed_storage)
