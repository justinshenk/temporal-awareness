"""Preference data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..common.file_io import load_json
from ..common.base_schema import BaseSchema


@dataclass
class PreferenceItem(BaseSchema):
    """Single preference record from query output."""

    sample_id: int
    time_horizon: Optional[dict]
    short_term_label: str
    long_term_label: str
    choice: str
    choice_prob: float
    alt_prob: float
    response: str
    internals: Optional[dict] = None
    prompt_text: str = ""  # Merged from dataset


@dataclass
class PreferenceData(BaseSchema):
    """Loaded preference data with metadata."""

    dataset_id: str
    model: str
    preferences: list[PreferenceItem] = field(default_factory=list)

    def split_by_choice(self) -> tuple[list[PreferenceItem], list[PreferenceItem]]:
        """Split preferences into short_term and long_term lists."""
        short_term = [p for p in self.preferences if p.choice == "short_term"]
        long_term = [p for p in self.preferences if p.choice == "long_term"]
        return short_term, long_term

    def filter_valid(self) -> list[PreferenceItem]:
        """Return preferences with known choice."""
        return [p for p in self.preferences if p.choice in ("short_term", "long_term")]


def load_preference_data(
    path_or_id: str | Path,
    preference_dir: Optional[Path] = None,
) -> PreferenceData:
    """
    Load preference data from JSON file.

    Args:
        path_or_id: Full path to JSON file, or a prefix to match against filenames
        preference_dir: Directory to search if path_or_id is a prefix

    Returns:
        PreferenceData with loaded preferences
    """
    path = Path(path_or_id)

    # If not a direct path, search for matching file
    if not path.exists() and preference_dir is not None:
        matches = [
            f for f in preference_dir.glob("*.json") if str(path_or_id) in f.name
        ]
        if not matches:
            raise FileNotFoundError(f"No preference data matching: {path_or_id}")
        path = matches[0]

    if not path.exists():
        raise FileNotFoundError(f"Preference data not found: {path}")

    data = load_json(path)

    preferences = []
    for p in data.get("preferences", []):
        preferences.append(
            PreferenceItem(
                sample_id=p["sample_id"],
                time_horizon=p.get("time_horizon"),
                short_term_label=p["short_term_label"],
                long_term_label=p["long_term_label"],
                choice=p["choice"],
                choice_prob=p["choice_prob"],
                alt_prob=p["alt_prob"],
                response=p["response"],
                internals=p.get("internals"),
            )
        )

    return PreferenceData(
        dataset_id=data["dataset_id"],
        model=data["model"],
        preferences=preferences,
    )


def load_dataset(
    dataset_id: str,
    datasets_dir: Path,
) -> dict:
    """
    Load dataset by ID.

    Args:
        dataset_id: Dataset ID to search for
        datasets_dir: Directory containing dataset JSON files

    Returns:
        Dataset dictionary with samples
    """
    matches = list(datasets_dir.glob(f"*{dataset_id}*.json"))
    if not matches:
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return load_json(matches[0])


def get_prompt_text(sample: dict) -> str:
    """Extract prompt text from dataset sample."""
    prompt = sample.get("prompt", {})
    if isinstance(prompt, dict):
        text = prompt.get("text", "")
        return "\n".join(text) if isinstance(text, list) else text
    return "\n".join(prompt) if isinstance(prompt, list) else str(prompt)


def merge_prompt_text(
    pref_data: PreferenceData,
    dataset: dict,
) -> None:
    """
    Merge prompt text from dataset into preference items.

    Modifies pref_data.preferences in place.
    """
    samples = dataset.get("samples", [])
    prompts_by_id = {s["sample_id"]: get_prompt_text(s) for s in samples}

    for pref in pref_data.preferences:
        pref.prompt_text = prompts_by_id.get(pref.sample_id, "")


def load_pref_data_with_prompts(
    preference_id: str,
    preference_dir: Path,
    datasets_dir: Path,
) -> PreferenceData:
    """
    Load preference data and merge prompt text from dataset.

    Convenience function that combines load_preference_data, load_dataset,
    and merge_prompt_text.

    Args:
        preference_id: Preference data ID or path
        preference_dir: Directory containing preference JSON files
        datasets_dir: Directory containing dataset JSON files

    Returns:
        PreferenceData with prompt_text populated
    """
    pref_data = load_preference_data(preference_id, preference_dir)
    dataset = load_dataset(pref_data.dataset_id, datasets_dir)
    merge_prompt_text(pref_data, dataset)
    return pref_data


def get_full_text(pref: PreferenceItem, include_response: bool = True) -> str:
    """Get full text for a preference item.

    Args:
        pref: PreferenceItem with prompt_text and response
        include_response: Whether to include the model response

    Returns:
        Combined text (prompt + optional response)
    """
    if include_response and pref.response:
        return pref.prompt_text + pref.response
    return pref.prompt_text


def build_prompt_pairs(
    pref_data: PreferenceData,
    max_pairs: int,
    include_response: bool = True,
    same_labels: bool = True,
) -> list[tuple[str, str, PreferenceItem, PreferenceItem]]:
    """Build clean/corrupted text pairs from short_term and long_term samples.

    For activation patching, we need pairs of prompts where one leads to
    short_term choice and the other to long_term choice.

    Args:
        pref_data: PreferenceData with preferences
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
        from collections import defaultdict

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


def find_preference_data(
    preference_dir: Path,
    preference_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Find preference data file.

    Args:
        preference_dir: Directory containing preference JSON files
        preference_id: Optional ID to search for. If None, returns most recent.

    Returns:
        Path to preference data file, or None if not found
    """
    if not preference_dir.exists():
        return None

    files = list(preference_dir.glob("*.json"))
    if not files:
        return None

    if preference_id:
        # Search for matching ID
        matches = [f for f in files if preference_id in f.name]
        return matches[0] if matches else None

    # Return most recent by modification time
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]


def get_preference_data_id(path: Path) -> str:
    """Extract preference data ID from file path."""
    # Filename format: <dataset_id>_<model>.json
    # The ID is the first segment (hash)
    return path.stem.split("_")[0]
