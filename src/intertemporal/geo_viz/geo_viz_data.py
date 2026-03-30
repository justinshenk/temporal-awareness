"""Data collection and caching for geometric visualization."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...common.time_value import TimeValue
from ..common.preference_types import PreferenceSample, PromptSample, RewardValue
from .geo_viz_config import GeoVizConfig, TargetSpec, is_absolute_position, parse_absolute_position

logger = logging.getLogger(__name__)


# =============================================================================
# Position Resolution
# =============================================================================


@dataclass
class ResolvedPositions:
    """Resolved token positions for a specific sample.

    Supports named position groups for different analysis needs.
    """

    # Named position groups
    named_positions: dict[str, list[int]] = field(default_factory=dict)
    prompt_len: int = 0
    full_len: int = 0

    # Legacy fields for backwards compatibility
    source: list[int] = field(default_factory=list)
    dest: list[int] = field(default_factory=list)

    def get(self, pos_name: str) -> int | None:
        """Get first position index by name or absolute index.

        Returns None if position doesn't exist for this sample.
        """
        # Handle absolute positions (P86, P145, etc.)
        if is_absolute_position(pos_name):
            idx = parse_absolute_position(pos_name)
            # Clamp to valid range
            return max(0, min(idx, self.full_len - 1))

        # Check named positions first
        if pos_name in self.named_positions:
            positions = self.named_positions[pos_name]
            return positions[0] if positions else None

        # Legacy fallbacks
        if pos_name == "source":
            return self.source[0] if self.source else 0
        elif pos_name == "dest":
            return self.dest[0] if self.dest else self.prompt_len
        elif pos_name == "response":
            # response is same as dest
            return self.dest[0] if self.dest else self.prompt_len
        else:
            # Position doesn't exist for this sample
            return None

    def get_all(self, pos_name: str) -> list[int]:
        """Get all position indices for a named position or absolute index."""
        # Handle absolute positions (P86, P145, etc.)
        if is_absolute_position(pos_name):
            idx = parse_absolute_position(pos_name)
            # Clamp to valid range and return single position
            return [max(0, min(idx, self.full_len - 1))]

        if pos_name in self.named_positions:
            return self.named_positions[pos_name]
        elif pos_name == "source":
            return self.source
        elif pos_name == "dest":
            return self.dest
        else:
            raise ValueError(f"Unknown position name: {pos_name}")


def _find_substring_token_range(
    tokens: list[str], text: str, substring: str
) -> list[int]:
    """Find all token positions spanning a substring in text.

    Returns list of token indices that cover the substring.
    """
    char_idx = text.find(substring)
    if char_idx == -1:
        return []

    char_end = char_idx + len(substring)
    positions = []

    # Map character range to token indices
    char_count = 0
    for i, tok in enumerate(tokens):
        tok_start = char_count
        tok_end = char_count + len(tok)
        char_count = tok_end

        # Token overlaps with substring
        if tok_end > char_idx and tok_start < char_end:
            positions.append(i)

        if tok_end >= char_end:
            break

    return positions


def _print_positions(
    label: str, search_str: str, positions: list[int], tokens: list[str]
) -> None:
    """Print position info in a readable format."""
    if not positions:
        print(f"       {label:12} '{search_str}' -> (not found)")
        return
    tok_strs = [repr(tokens[p]) for p in positions]
    print(f"       {label:12} '{search_str}' -> pos {positions} = {tok_strs}")


def _find_time_value_positions(
    tokens: list[str], text: str, time_val: TimeValue, verbose: bool = False
) -> list[int]:
    """Find token positions for a TimeValue's numeric value and unit separately."""
    positions = []
    prompt_tokens = tokens

    # Use TimeValue's own __str__ formatting, then split to get value and unit
    formatted = str(time_val)
    parts = formatted.split(" ", 1)
    value_str = parts[0]
    unit_str = parts[1] if len(parts) > 1 else time_val.unit

    value_positions = _find_substring_token_range(prompt_tokens, text, value_str)
    positions.extend(value_positions)
    if verbose:
        _print_positions("value", value_str, value_positions, tokens)

    unit_positions = _find_substring_token_range(prompt_tokens, text, unit_str)
    positions.extend(unit_positions)
    if verbose:
        _print_positions("unit", unit_str, unit_positions, tokens)

    return positions


def _find_reward_value_positions(
    tokens: list[str], text: str, reward_val: RewardValue, verbose: bool = False
) -> list[int]:
    """Find token positions for a RewardValue's numeric value and unit separately."""
    positions = []
    prompt_tokens = tokens

    # Format numeric value with commas (e.g., "1,750") - use RewardValue's own formatting
    value_str = str(RewardValue(reward_val.value))
    value_positions = _find_substring_token_range(prompt_tokens, text, value_str)
    positions.extend(value_positions)
    if verbose:
        _print_positions("value", value_str, value_positions, tokens)

    # Unit (if present)
    if reward_val.unit:
        unit_positions = _find_substring_token_range(
            prompt_tokens, text, reward_val.unit
        )
        positions.extend(unit_positions)
        if verbose:
            _print_positions("unit", reward_val.unit, unit_positions, tokens)

    return positions


def resolve_positions(
    sample: PromptSample,
    pref: PreferenceSample,
    runner,
    verbose: bool = True,
) -> ResolvedPositions:
    """Resolve token positions using exact sample structure.

    Args:
        sample: PromptSample with structured TimeValue/RewardValue data
        pref: PreferenceSample with token info from model query
        runner: Model runner with tokenizer

    Returns named positions:
    - time_horizon: time horizon tokens
    - short_term_time: short-term delivery time tokens
    - short_term_reward: short-term reward tokens
    - long_term_time: long-term delivery time tokens
    - long_term_reward: long-term reward tokens
    - response: all response tokens
    """
    # Get token info from PreferenceSample
    prompt_len = pref.prompt_token_count
    full_tokens = pref.chosen_traj.token_ids
    tokens = [runner._tokenizer.decode([t]) for t in full_tokens]

    full_len = len(tokens)
    # Reconstruct text from tokens (matches token positions)
    prompt_tokens_decoded = tokens[:prompt_len]
    prompt_text = "".join(prompt_tokens_decoded)

    pair = sample.prompt.preference_pair
    named_positions = {}

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Sample {sample.sample_idx} | prompt_len={prompt_len} | full_len={full_len}"
        )
        print(f"{'=' * 60}")

    # Time horizon (numeric value + unit)
    if sample.prompt.time_horizon is not None:
        if verbose:
            print(f"  time_horizon: {sample.prompt.time_horizon}")
        named_positions["time_horizon"] = _find_time_value_positions(
            prompt_tokens_decoded, prompt_text, sample.prompt.time_horizon, verbose
        )

    # Short-term option: time and reward
    if verbose:
        print(f"  short_term.time: {pair.short_term.time}")
    named_positions["short_term_time"] = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, pair.short_term.time, verbose
    )

    if verbose:
        print(f"  short_term.reward: {pair.short_term.reward}")
    named_positions["short_term_reward"] = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, pair.short_term.reward, verbose
    )

    # Long-term option: time and reward
    if verbose:
        print(f"  long_term.time: {pair.long_term.time}")
    named_positions["long_term_time"] = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, pair.long_term.time, verbose
    )

    if verbose:
        print(f"  long_term.reward: {pair.long_term.reward}")
    named_positions["long_term_reward"] = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, pair.long_term.reward, verbose
    )

    # Response tokens
    named_positions["response"] = list(range(prompt_len, full_len))

    # Clamp all positions to valid range
    for key in named_positions:
        named_positions[key] = [max(0, min(p, full_len - 1)) for p in named_positions[key]]

    # Build legacy source/dest for backwards compat
    source_positions = []
    for key in ["time_horizon", "short_term_time", "short_term_reward",
                "long_term_time", "long_term_reward"]:
        if key in named_positions:
            source_positions.extend(named_positions[key])
    source_positions = sorted(set(source_positions))

    if not source_positions:
        source_positions.append(int(prompt_len * 0.6))

    dest_positions = named_positions.get("response", [])

    if verbose:
        print("\n  Named positions:")
        for key, positions in named_positions.items():
            if positions:
                pos_toks = [repr(tokens[p]) for p in positions[:3]]
                print(f"    {key}: {positions[:5]}{'...' if len(positions) > 5 else ''} = {pos_toks}")

    return ResolvedPositions(
        named_positions=named_positions,
        source=source_positions,
        dest=dest_positions,
        prompt_len=prompt_len,
        full_len=full_len,
    )


# =============================================================================
# Activation Data
# =============================================================================


@dataclass
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


@dataclass
class ActivationData:
    """Container for activations and metadata."""

    samples: list[PromptSample]
    activations: dict[str, np.ndarray]  # target_key -> (n_samples, d_model)
    choices: list[ChoiceInfo] | None = None  # Choice info per sample

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)

        samples_data = [s.to_dict() for s in self.samples]
        with open(path / "samples.json", "w") as f:
            json.dump(samples_data, f, indent=2)

        if self.choices:
            choices_data = [c.to_dict() for c in self.choices]
            with open(path / "choices.json", "w") as f:
                json.dump(choices_data, f, indent=2)

        np.savez_compressed(path / "activations.npz", **self.activations)
        logger.info(f"Saved {len(self.samples)} samples to {path}")

    @classmethod
    def load(cls, path: Path) -> "ActivationData":
        """Load from disk."""
        with open(path / "samples.json") as f:
            samples_data = json.load(f)
        samples = [PromptSample.from_dict(s) for s in samples_data]

        choices = None
        choices_path = path / "choices.json"
        if choices_path.exists():
            with open(choices_path) as f:
                choices_data = json.load(f)
            choices = [ChoiceInfo.from_dict(c) for c in choices_data]

        activations_file = np.load(path / "activations.npz")
        activations = {k: activations_file[k] for k in activations_file.files}

        logger.info(f"Loaded {len(samples)} samples from {path}")
        return cls(samples=samples, activations=activations, choices=choices)


# =============================================================================
# Sample Collection
# =============================================================================


def get_time_horizon_months(sample: PromptSample) -> float:
    """Get time horizon in months from a PromptSample."""
    if sample.prompt.time_horizon is None:
        return 60.0  # Default to 5 years for None
    return sample.prompt.time_horizon.to_months()


def collect_samples() -> "PromptDataset":
    """Generate samples with diverse time horizons using GEO_VIZ_CFG.

    Returns PromptDataset with samples and config (including format info).
    """
    from ..data.default_datasets import GEO_VIZ_CFG
    from ..prompt import PromptDatasetConfig, PromptDatasetGenerator

    dataset_config = PromptDatasetConfig.from_dict(GEO_VIZ_CFG)
    dataset = PromptDatasetGenerator(dataset_config).generate()

    logger.info(f"Generated {len(dataset.samples)} samples")
    return dataset


# =============================================================================
# Activation Extraction
# =============================================================================


def extract_activations(
    dataset: "PromptDataset", targets: list[TargetSpec], config: GeoVizConfig
) -> ActivationData:
    """Extract activations at specified positions for all samples.

    Memory-efficient: saves activations incrementally to disk and offloads.
    """
    import gc
    from ..formatting.prompt_formats import find_prompt_format_config
    from ..preference import PreferenceQuerier, PreferenceQueryConfig

    logger.info(f"Loading model {config.model}...")

    # Use PreferenceQuerier for consistent model handling
    query_config = PreferenceQueryConfig(skip_generation=True)
    querier = PreferenceQuerier(query_config)
    runner = querier._load_model(config.model)

    samples = dataset.samples
    if config.max_samples is not None and len(samples) > config.max_samples:
        samples = samples[: config.max_samples]
        logger.info(f"Limited to {config.max_samples} samples")

    hook_names = list({t.hook_name for t in targets})

    # Setup incremental save directory
    cache_dir = config.output_dir / "data" / "incremental"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Track which targets we can actually extract (some positions may not exist)
    valid_targets = []
    activations = {}
    choices = []
    valid_samples = []
    skipped = 0

    # Batch size for incremental saves
    SAVE_BATCH_SIZE = 50

    logger.info(f"Extracting activations for {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if i % 20 == 0:
            logger.info(f"  Processing sample {i}/{len(samples)} (skipped: {skipped}, valid: {len(valid_samples)})")

        # Get choice_prefix from sample's prompt format config
        prompt_format = find_prompt_format_config(sample.formatting_id)
        choice_prefix = prompt_format.get_response_prefix_before_choice()

        # Query sample with activation capture
        try:
            pref = querier.query_sample(
                sample, runner, choice_prefix, activation_names=hook_names
            )

            # Skip samples with invalid trajectories
            if pref.chosen_traj is None:
                skipped += 1
                continue

            # Resolve positions using PreferenceSample's token info
            positions = resolve_positions(sample, pref, runner, verbose=False)

            # Extract activations for each target
            sample_activations = {}
            for target in targets:
                try:
                    pos_idx = positions.get(target.position)
                    if pos_idx is None:
                        # Position doesn't exist for this sample
                        continue
                    # CapturedInternals already removes batch dim and moves to CPU
                    act = pref.internals.activations[target.hook_name][pos_idx, :].numpy()
                    sample_activations[target.key] = act
                except (ValueError, KeyError, IndexError) as e:
                    # Error extracting - skip this target
                    continue

            # Only include sample if we got at least some activations
            if not sample_activations:
                skipped += 1
                continue

            # Add to running collection
            for key, act in sample_activations.items():
                if key not in activations:
                    activations[key] = []
                activations[key].append(act)

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

            # Clear internals immediately to free memory
            pref.internals = None

            # Incremental save and offload
            if len(valid_samples) % SAVE_BATCH_SIZE == 0 and len(valid_samples) > 0:
                batch_num = len(valid_samples) // SAVE_BATCH_SIZE
                logger.info(f"  Saving batch {batch_num} to disk...")

                # Save current batch
                batch_file = cache_dir / f"batch_{batch_num:04d}.npz"
                batch_acts = {k: np.stack(v[-SAVE_BATCH_SIZE:]) for k, v in activations.items() if len(v) >= SAVE_BATCH_SIZE}
                if batch_acts:
                    np.savez_compressed(batch_file, **batch_acts)

                # Force garbage collection
                gc.collect()

        except Exception as e:
            logger.warning(f"  Skipping sample {i}: {e}")
            skipped += 1
            continue

    logger.info(f"Processed {len(valid_samples)} valid samples (skipped {skipped})")

    # Stack all activations
    if not activations:
        raise ValueError("No activations were extracted. Check position names.")

    final_activations = {}
    for k, v in activations.items():
        if v:  # Only include targets with data
            final_activations[k] = np.stack(v)
            logger.info(f"  {k}: {final_activations[k].shape}")

    # Clear intermediate data
    activations = None
    gc.collect()

    return ActivationData(samples=valid_samples, activations=final_activations, choices=choices)


# =============================================================================
# Caching
# =============================================================================


def load_cached_data(config: GeoVizConfig) -> ActivationData | None:
    """Load cached data if available."""
    cache_path = config.output_dir / "data"
    if (cache_path / "samples.json").exists() and (
        cache_path / "activations.npz"
    ).exists():
        try:
            return ActivationData.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def save_data(data: ActivationData, config: GeoVizConfig):
    """Save data to cache."""
    cache_path = config.output_dir / "data"
    data.save(cache_path)
