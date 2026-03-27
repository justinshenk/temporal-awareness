"""Data collection and extraction for geometric visualization.

Structure:
    data/
        metadata.json             - Dataset metadata
        prompt_dataset.json       - Full PromptDataset (if generated)
        samples/
            sample_0/
                position_mapping.json   - Maps abs_pos -> format_pos for this sample
                preference_sample.json  - PreferenceSample with choice info
                choice.json             - ChoiceInfo (quick access)
                L35_resid_post_129.npy  - Activation at position 129
                L35_resid_post_130.npy
            sample_1/
                ...

    preference_datasets/          - Global preference dataset location
        {prompt_dataset_id}_{model}_{name}.json
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common.preference_types import PromptSample
from ..common.sample_position_mapping import (
    DatasetPositionMapping,
    SamplePositionMapping,
)
from .geometry_config import GeometryConfig, TargetSpec, ACTIVATION_DTYPE

logger = logging.getLogger(__name__)


# =============================================================================
# Choice Info
# =============================================================================


@dataclass(slots=True)
class ChoiceInfo:
    """Choice information for a single sample."""

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
# File I/O
# =============================================================================


def _save_array(path: Path, arr: np.ndarray, compressed: bool = False):
    """Save numpy array."""
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


# =============================================================================
# Activation Data Container
# =============================================================================


@dataclass
class ActivationData:
    """Container for extracted activations.

    New structure: per-sample folders with absolute position filenames.
    Use sample_position_mapping.json to map abs_pos -> format_pos.
    """

    samples: list[PromptSample]
    choices: list[ChoiceInfo] | None = None
    position_mappings: DatasetPositionMapping | None = None
    n_samples: int = 0
    _data_dir: Path | None = None
    _compressed: bool = False
    _target_keys: list[str] = field(default_factory=list)
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    def get_sample_dir(self, sample_idx: int) -> Path:
        """Get path to sample's activation folder."""
        if self._data_dir is None:
            raise ValueError("Data directory not set")
        return self._data_dir / "samples" / f"sample_{sample_idx}"

    def load_activation(
        self, sample_idx: int, layer: int, component: str, abs_pos: int
    ) -> np.ndarray:
        """Load activation for a specific (sample, layer, component, position)."""
        sample_dir = self.get_sample_dir(sample_idx)
        filename = f"L{layer}_{component}_{abs_pos}"
        return _load_array(sample_dir / filename)

    def load_activations_by_format_pos(
        self, layer: int, component: str, format_pos: str
    ) -> np.ndarray:
        """Load activations for all samples at a given format position.

        Uses position mappings to find the absolute position for each sample.
        Returns: (n_samples, hidden_dim) array
        """
        if self.position_mappings is None:
            raise ValueError("Position mappings not loaded")

        activations = []
        for sample_idx in range(self.n_samples):
            mapping = self.position_mappings.get(sample_idx)
            if mapping is None:
                continue

            # Find abs_pos for this format_pos
            abs_positions = mapping.named_positions.get(format_pos, [])
            if not abs_positions:
                continue

            abs_pos = abs_positions[0]  # Use first position
            try:
                act = self.load_activation(sample_idx, layer, component, abs_pos)
                activations.append(act)
            except FileNotFoundError:
                continue

        if not activations:
            raise ValueError(f"No activations found for {format_pos}")

        return np.stack(activations)

    # =========================================================================
    # Compatibility layer for analysis pipeline
    # =========================================================================

    def get_target_keys(self) -> list[str]:
        """Get list of available target keys (for analysis compatibility).

        Target key format: L{layer}_{component}_{format_pos}
        """
        if self._target_keys:
            return self._target_keys.copy()

        # Build target keys from what's actually available
        if self.position_mappings is None or self.n_samples == 0:
            return []

        # Get all format positions from first sample's mapping
        first_mapping = self.position_mappings.get(0)
        if first_mapping is None:
            return []

        # Scan sample_0 folder for available (layer, component) combinations
        sample_dir = self.get_sample_dir(0)
        if not sample_dir.exists():
            return []

        layer_components = set()
        for f in sample_dir.glob("*.npy"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                layer = parts[0]  # L35
                component = "_".join(parts[1:-1])  # resid_post
                layer_components.add((layer, component))

        # Build target keys for each (layer, component, format_pos)
        target_keys = []
        for layer, component in sorted(layer_components):
            for format_pos in first_mapping.named_positions.keys():
                target_keys.append(f"{layer}_{component}_{format_pos}")

        self._target_keys = target_keys
        return target_keys.copy()

    def load_target(self, target_key: str) -> np.ndarray:
        """Load activations for a target key (for analysis compatibility).

        Target key format: L{layer}_{component}_{format_pos}
        Returns: (n_samples, hidden_dim) array
        """
        if target_key in self._cache:
            return self._cache[target_key]

        # Parse target key
        parts = target_key.split("_")
        layer = int(parts[0][1:])  # L35 -> 35
        component = "_".join(parts[1:-1])  # resid_post, mlp_out, etc.
        format_pos = parts[-1]  # response_choice, time_horizon, etc.

        # Handle multi-word format_pos
        # Target key might be L35_resid_post_response_choice_prefix
        # We need to figure out where component ends and format_pos starts
        # Try progressively longer format_pos
        for i in range(2, len(parts)):
            potential_component = "_".join(parts[1:i])
            potential_format_pos = "_".join(parts[i:])
            if self.position_mappings and self.position_mappings.get(0):
                if (
                    potential_format_pos
                    in self.position_mappings.get(0).named_positions
                ):
                    component = potential_component
                    format_pos = potential_format_pos
                    break

        activations = self.load_activations_by_format_pos(layer, component, format_pos)
        self._cache[target_key] = activations
        return activations

    def get_sample_count(self, target_key: str) -> int:
        """Get sample count for a target (for analysis compatibility)."""
        return self.n_samples

    def unload_target(self, target_key: str):
        """Remove target from cache."""
        if target_key in self._cache:
            del self._cache[target_key]

    def clear_cache(self):
        """Clear all cached activations."""
        self._cache.clear()
        gc.collect()

    def iter_targets(self):
        """Iterate over targets (for analysis compatibility)."""
        for key in self.get_target_keys():
            try:
                activations = self.load_target(key)
                yield key, activations
                self.unload_target(key)
            except (ValueError, FileNotFoundError):
                continue
        gc.collect()

    def save(self, path: Path):
        """Save metadata only.

        Per-sample data (preference_sample.json, choice.json, position_mapping.json)
        is saved during extraction in extract_activations().
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "n_samples": self.n_samples,
            "compressed": self._compressed,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata for {self.n_samples} samples to {path}")

    @classmethod
    def load(cls, path: Path) -> "ActivationData":
        """Load from disk.

        Data is loaded from per-sample files in samples/sample_*/
        """
        # Load metadata first to get n_samples
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            n_samples = metadata.get("n_samples", 0)
            compressed = metadata.get("compressed", False)
        else:
            # Fall back to counting sample directories
            samples_dir = path / "samples"
            if samples_dir.exists():
                n_samples = len(list(samples_dir.glob("sample_*")))
            else:
                n_samples = 0
            compressed = False

        # Load from per-sample files
        samples_dir = path / "samples"
        samples = []
        choices = []
        position_mappings = DatasetPositionMapping()

        for sample_idx in range(n_samples):
            sample_dir = samples_dir / f"sample_{sample_idx}"
            if not sample_dir.exists():
                continue

            # Load prompt sample
            prompt_path = sample_dir / "prompt_sample.json"
            if prompt_path.exists():
                with open(prompt_path) as f:
                    prompt_data = json.load(f)
                samples.append(PromptSample.from_dict(prompt_data))

            # Load choice info
            choice_path = sample_dir / "choice.json"
            if choice_path.exists():
                with open(choice_path) as f:
                    choice_data = json.load(f)
                choices.append(ChoiceInfo.from_dict(choice_data))

            # Load position mapping
            mapping_path = sample_dir / "position_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    mapping_data = json.load(f)
                position_mappings.add(SamplePositionMapping.from_dict(mapping_data))

        data = cls(
            samples=samples,
            choices=choices if choices else None,
            position_mappings=position_mappings,
            n_samples=n_samples,
            _data_dir=path,
            _compressed=compressed,
        )

        logger.info(f"Loaded {n_samples} samples from {path}")
        return data


# =============================================================================
# Sample Collection
# =============================================================================


def get_time_horizon_months(sample: PromptSample) -> float:
    """Get time horizon in months from a PromptSample."""
    if sample.prompt.time_horizon is None:
        return 60.0
    return sample.prompt.time_horizon.to_months()


def _format_prompt_sample(sample: PromptSample) -> str:
    """Format a prompt sample for logging."""
    pair = sample.prompt.preference_pair
    horizon = sample.prompt.time_horizon
    horizon_str = str(horizon) if horizon else "None"
    return (
        f"  idx={sample.sample_idx} | "
        f"short={pair.short_term.reward.value:,} in {pair.short_term.time} | "
        f"long={pair.long_term.reward.value:,} in {pair.long_term.time} | "
        f"horizon={horizon_str}"
    )


def collect_samples(
    output_dir: Path | None = None, try_load: bool = True
) -> "PromptDataset":
    """Load or generate samples with diverse time horizons.

    Args:
        output_dir: Output directory. If provided and try_load=True, will attempt
            to load existing prompt_dataset.json from here first.
        try_load: If True, try to load existing data before generating.

    Returns:
        PromptDataset with samples.
    """
    from ..data.default_datasets import GEOMETRY_CFG
    from ..prompt import PromptDataset, PromptDatasetConfig, PromptDatasetGenerator

    dataset = None

    # Try to load existing dataset
    if try_load and output_dir is not None:
        prompt_dataset_path = output_dir / "data" / "prompt_dataset.json"
        if prompt_dataset_path.exists():
            logger.info(f"Loading existing prompt dataset from {prompt_dataset_path}")
            dataset = PromptDataset.from_json(prompt_dataset_path)
            logger.info(f"Loaded {len(dataset.samples)} samples")

    # Generate if not loaded
    if dataset is None:
        logger.info("Generating new prompt dataset...")
        dataset_config = PromptDatasetConfig.from_dict(GEOMETRY_CFG)
        dataset = PromptDatasetGenerator(dataset_config).generate()
        logger.info(f"Generated {len(dataset.samples)} samples")

        # Save generated dataset
        if output_dir is not None:
            data_dir = output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_as_json(data_dir / "prompt_dataset.json")
            logger.info(f"Saved prompt dataset to {data_dir / 'prompt_dataset.json'}")

    # Print first few prompt samples
    n_preview = min(5, len(dataset.samples))
    logger.info(f"First {n_preview} prompt samples:")
    for sample in dataset.samples[:n_preview]:
        logger.info(_format_prompt_sample(sample))
    if len(dataset.samples) > n_preview:
        logger.info(f"  ... and {len(dataset.samples) - n_preview} more")

    # Print full text of first sample
    if dataset.samples:
        logger.info("First sample prompt text:")
        for line in dataset.samples[0].text.split("\n"):
            logger.info(f"  | {line}")

    return dataset


# =============================================================================
# Activation Extraction
# =============================================================================


def extract_activations(
    dataset: "PromptDataset", targets: list[TargetSpec], config: GeometryConfig
) -> ActivationData:
    """Extract activations organized by sample with absolute positions.

    Output structure:
        samples/sample_{idx}/
            position_mapping.json     - Maps abs_pos -> format_pos for this sample
            prompt_sample.json        - Original PromptSample
            preference_sample.json    - PreferenceSample with choice
            choice.json               - Quick access to choice info
            L{layer}_{component}_{abs_pos}.npy - Activations
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

    # Setup output
    data_dir = config.output_dir / "data"
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    compressed = config.use_compressed_storage

    valid_samples = []
    valid_preferences = []
    choices = []
    position_mappings = DatasetPositionMapping()
    skipped = 0
    valid_idx = 0

    logger.info(f"Extracting activations (per-sample, compressed={compressed})...")

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            logger.info(
                f"  Sample {i}/{len(samples)} | valid: {valid_idx} | skipped: {skipped}"
            )

        prompt_format = find_prompt_format_config(sample.formatting_id)
        choice_prefix = prompt_format.get_response_prefix_before_choice()

        try:
            pref = querier.query_sample(
                sample, runner, choice_prefix, activation_names=hook_names
            )

            if pref.chosen_traj is None:
                skipped += 1
                continue

            # Build position mapping (this gives us abs_pos -> format_pos)
            pos_mapping = SamplePositionMapping.build(sample, pref, runner)

            # Create sample folder
            sample_dir = samples_dir / f"sample_{valid_idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Extract and save activations
            sample_has_data = False
            for target in targets:
                abs_positions = pos_mapping.named_positions.get(target.position, [])
                if not abs_positions:
                    continue

                abs_pos = abs_positions[0]  # Use first position

                try:
                    act = pref.internals.activations[target.hook_name][abs_pos, :]
                    act_np = act.numpy().astype(ACTIVATION_DTYPE)

                    filename = f"L{target.layer}_{target.component}_{abs_pos}"
                    _save_array(sample_dir / filename, act_np, compressed=compressed)
                    sample_has_data = True

                except (ValueError, KeyError, IndexError):
                    continue

            if not sample_has_data:
                sample_dir.rmdir()
                skipped += 1
                pref.internals = None
                continue

            # Save per-sample position mapping
            with open(sample_dir / "position_mapping.json", "w") as f:
                json.dump(pos_mapping.to_dict(), f)

            # Save per-sample prompt sample
            with open(sample_dir / "prompt_sample.json", "w") as f:
                json.dump(sample.to_dict(), f)

            # Save per-sample preference sample
            with open(sample_dir / "preference_sample.json", "w") as f:
                json.dump(pref.to_dict(), f)

            # Record and save choice info
            pair = sample.prompt.preference_pair
            chose_long = pref.chose_long_term
            if chose_long:
                chosen_time = pair.long_term.time.to_months()
                chosen_reward = pair.long_term.reward.value
            else:
                chosen_time = pair.short_term.time.to_months()
                chosen_reward = pair.short_term.reward.value

            choice_info = ChoiceInfo(
                chose_long_term=chose_long,
                chosen_time_months=chosen_time,
                chosen_reward=chosen_reward,
                choice_prob=pref.choice_prob,
            )
            choices.append(choice_info)

            # Save per-sample choice info
            with open(sample_dir / "choice.json", "w") as f:
                json.dump(choice_info.to_dict(), f)

            valid_samples.append(sample)
            position_mappings.add(pos_mapping)
            pref.internals = None
            valid_preferences.append(pref)

            # Log first few preference samples
            if valid_idx < 5:
                choice_str = "long_term" if chose_long else "short_term"
                logger.info(
                    f"  Preference sample {valid_idx}: "
                    f"chose={choice_str} ({pref.choice_prob:.2%}) | "
                    f"reward={chosen_reward:,.0f} in {chosen_time:.1f}mo"
                )

            valid_idx += 1

            if valid_idx % 100 == 0:
                gc.collect()

        except Exception as e:
            logger.warning(f"  Skipping sample {i}: {e}")
            skipped += 1
            continue

    gc.collect()
    logger.info(f"Extracted {valid_idx} valid samples (skipped {skipped})")

    # Create data container
    data = ActivationData(
        samples=valid_samples,
        choices=choices,
        position_mappings=position_mappings,
        n_samples=valid_idx,
        _data_dir=data_dir,
        _compressed=compressed,
    )

    # Save metadata
    data.save(data_dir)

    return data


# =============================================================================
# Cache Loading
# =============================================================================


def load_cached_data(config: GeometryConfig) -> ActivationData | None:
    """Load cached data if available."""
    cache_path = config.output_dir / "data"

    # Check for metadata.json or samples/ directory
    if not (cache_path / "metadata.json").exists() and not (cache_path / "samples").exists():
        return None

    try:
        return ActivationData.load(cache_path)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None
