"""Preference data loading utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..common.paths import get_pref_dataset_dir, get_prompt_dataset_dir
from ..models.preference_dataset import (
    PreferenceSample,
    PreferenceDataset,
)
from ..prompt_datasets import PromptDataset


def find_preference_files(prefix: str, directory: Optional[Path] = None) -> list[Path]:
    """Find all preference data files matching a prefix.

    Args:
        prefix: Prefix to match (e.g., "{dataset_id}_{model_name}")
        directory: Directory to search in (default: get_pref_dataset_dir())

    Returns:
        List of matching file paths, sorted by modification time (newest first)
    """
    if directory is None:
        directory = get_pref_dataset_dir()
    directory = Path(directory)
    matches = list(directory.glob(f"{prefix}*.json"))
    # Sort by modification time, newest first
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def find_preference_data(name: str, directory: Optional[Path] = None) -> Optional[Path]:
    """Find preference data file by name or prefix.

    First tries exact match, then falls back to glob pattern for backwards
    compatibility with new naming convention.

    Args:
        name: Preference dataset name (e.g., "{dataset_id}_{model_name}")
        directory: Directory to search in (default: get_pref_dataset_dir())

    Returns:
        Path to the preference file if found, None otherwise
    """
    if directory is None:
        directory = get_pref_dataset_dir()
    directory = Path(directory)

    # Try exact match first
    exact_path = directory / f"{name}.json"
    if exact_path.exists():
        return exact_path

    # Fall back to glob pattern (finds files with prompt_dataset_name suffix)
    matches = find_preference_files(name, directory)
    if matches:
        return matches[0]  # Return newest match

    return None


def load_and_merge_preference_data(
    prefix: str, directory: Optional[Path] = None
) -> Optional[PreferenceDataset]:
    """Load all preference files matching a prefix and merge them.

    Args:
        prefix: Prefix to match (e.g., "{dataset_id}_{model_name}")
        directory: Directory to search in (default: get_pref_dataset_dir())

    Returns:
        Merged PreferenceDataset, or None if no files found
    """
    files = find_preference_files(prefix, directory)
    if not files:
        return None

    datasets = [PreferenceDataset.load_from_json(f) for f in files]
    if len(datasets) == 1:
        return datasets[0]

    print("\n\n")
    print(
        "WARNING: load_and_merge_preference_data found more than 1 match!. Merging matches."
    )
    print("\n\n")

    return PreferenceDataset.merge_all(datasets)


def fill_missing_prompt_text(
    pref_data: PreferenceDataset, prompt_dir: Optional[Path] = None
) -> int:
    """Fill empty prompt_text fields from the source PromptDataset.

    Args:
        pref_data: PreferenceDataset to update (modified in place)
        prompt_dir: Directory containing prompt datasets (default: get_prompt_dataset_dir())

    Returns:
        Count of preferences that were updated
    """
    if prompt_dir is None:
        prompt_dir = get_prompt_dataset_dir()

    # Find preferences that need prompt_text filled
    needs_fill = [p for p in pref_data.preferences if not p.prompt_text]
    if not needs_fill:
        return 0

    # Load the prompt dataset
    try:
        prompt_dataset = PromptDataset.load_from_id(pref_data.dataset_id, prompt_dir)
    except FileNotFoundError:
        return 0

    prompts_by_id = prompt_dataset.get_prompts_by_id()

    # Fill missing prompt_text
    count = 0
    for pref in needs_fill:
        if pref.sample_idx in prompts_by_id:
            pref.prompt_text = prompts_by_id[pref.sample_idx]
            count += 1

    return count


def get_full_text(pref: PreferenceSample, include_response: bool = True) -> str:
    """Get full text for a preference item.

    Args:
        pref: PreferenceSample with prompt_text and response
        include_response: Whether to include the model response

    Returns:
        Combined text (prompt + optional response)
    """
    if include_response and pref.response:
        return pref.prompt_text + pref.response
    return pref.prompt_text


def build_prompt_pairs(
    pref_data: PreferenceDataset,
    max_pairs: int,
    include_response: bool = True,
    same_labels: bool = True,
) -> list[tuple[str, str, PreferenceSample, PreferenceSample]]:
    """Build clean/corrupted text pairs from short_term and long_term samples.

    For activation patching, we need pairs of prompts where one leads to
    short_term choice and the other to long_term choice.

    Args:
        pref_data: PreferenceDataset with preferences
        max_pairs: Maximum number of pairs to generate
        include_response: Whether to include model response in text
        same_labels: If True (default), only pair samples that share the
            same short_term_label and long_term_label strings so the
            label token IDs match between clean and corrupted.

    Returns:
        List of (clean_text, corrupted_text, clean_sample, corrupted_sample)
    """
    short_term, long_term = pref_data.split_by_choice()

    if same_labels:
        # Group by (short_term_label, long_term_label) and pair within groups
        short_by_labels = defaultdict(list)
        long_by_labels = defaultdict(list)
        for s in short_term:
            short_by_labels[(s.short_term_label, s.long_term_label)].append(s)
        for l in long_term:
            long_by_labels[(l.short_term_label, l.long_term_label)].append(l)

        pairs = []
        for key in short_by_labels:
            if key not in long_by_labels:
                continue
            s_list = short_by_labels[key]
            l_list = long_by_labels[key]
            n = min(len(s_list), len(l_list))
            for i in range(n):
                clean_text = get_full_text(s_list[i], include_response)
                corrupted_text = get_full_text(l_list[i], include_response)
                if clean_text and corrupted_text:
                    pairs.append((clean_text, corrupted_text, s_list[i], l_list[i]))
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        return pairs[:max_pairs]

    n = min(len(short_term), len(long_term), max_pairs)

    pairs = []
    for i in range(n):
        clean = short_term[i]
        corrupted = long_term[i]
        clean_text = get_full_text(clean, include_response)
        corrupted_text = get_full_text(corrupted, include_response)

        if clean_text and corrupted_text:
            pairs.append((clean_text, corrupted_text, clean, corrupted))

    return pairs
