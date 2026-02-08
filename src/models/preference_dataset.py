"""Preference dataset classes for storing model preferences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..common.io import load_json, save_json, ensure_dir
from ..common.types import SchemaClass
from ..common.paths import get_internals_dir
from ..common.types import PreferenceSample

import json


@dataclass
class PreferenceDataset(SchemaClass):
    """Preference dataset with metadata."""

    prompt_dataset_id: str
    model: str
    preferences: list[PreferenceSample] = field(default_factory=list)
    prompt_dataset_name: str = ""  # Name of the source prompt dataset

    @property
    def dataset_id(self) -> str:
        """Alias for prompt_dataset_id (for backwards compatibility)."""
        return self.prompt_dataset_id

    @staticmethod
    def extract_model_name(model: str) -> str:
        """Extract model name from full model path (e.g., 'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen2.5-1.5B-Instruct')."""
        return model.split("/")[-1]

    @staticmethod
    def make_prefix(prompt_dataset_id: str, model: str) -> str:
        """Build prefix from components: {prompt_dataset_id}_{model_name}."""
        model_name = PreferenceDataset.extract_model_name(model)
        return f"{prompt_dataset_id}_{model_name}"

    @property
    def model_name(self) -> str:
        return self.extract_model_name(self.model)

    def get_prefix(self) -> str:
        """Get the prefix for preference dataset files: {prompt_dataset_id}_{model_name}."""
        return self.make_prefix(self.prompt_dataset_id, self.model)

    def get_filename(self) -> str:
        """Get the filename for this preference dataset.

        Returns {prefix}_{prompt_dataset_name} if prompt_dataset_name is set,
        otherwise falls back to {prefix} for compatibility.
        """
        prefix = self.get_prefix()
        if self.prompt_dataset_name:
            return f"{prefix}_{self.prompt_dataset_name}.json"
        return f"{prefix}.json"

    def get_internals_filename(self, sample_idx: int) -> str | None:
        """Get filename for internals .pt file at given index."""
        if len(self.preferences) <= sample_idx:
            return None
        return f"{self.get_prefix()}_sample_{self.preferences[sample_idx].sample_idx}.pt"

    def split_by_choice(
        self,
    ) -> tuple[list[PreferenceSample], list[PreferenceSample]]:
        """Split preferences into short_term and long_term lists."""
        short_term = [p for p in self.preferences if p.choice == "short_term"]
        long_term = [p for p in self.preferences if p.choice == "long_term"]
        return short_term, long_term

    def filter_valid(self) -> list[PreferenceSample]:
        """Return preferences with known choice."""
        return [p for p in self.preferences if p.choice in ("short_term", "long_term")]

    def _to_dict(self) -> dict:
        preferences_data = []
        for pref in self.preferences:
            pref_dict = pref.to_dict()
            pref_dict["internals"] = None
            preferences_data.append(pref_dict)

        data = {
            "dataset_id": self.prompt_dataset_id,
            "model": self.model,
            "prompt_dataset_name": self.prompt_dataset_name,
            "preferences": preferences_data,
        }
        return data

    def to_string(
        self, without_internals: bool = True, without_texts: bool = False
    ) -> str:
        d = self._to_dict()
        for p in d["preferences"]:
            if without_internals:
                p.pop("internals", None)
                p.pop("internals_paths", None)
            if without_texts:
                p.pop("promot_text", None)
                p.pop("response_text", None)
        return json.dumps(d, indent=4)

    def save_as_json(self, path: Path, with_internals: bool = True) -> None:
        """Save the full preference dataset to JSON.

        If preferences have CapturedInternals with tensors, they are saved
        to separate .pt files and replaced with file path references.

        Args:
            path: Output path for the JSON file
        """
        path = Path(path)
        ensure_dir(path.parent)
        internals_dir = get_internals_dir(path.parent)
        ensure_dir(internals_dir)

        data = self._to_dict()

        if with_internals:
            for idx, pref in enumerate(self.preferences):
                if pref.internals:
                    file_path = internals_dir / self.get_internals_filename(idx)
                    ip = {
                        "file_path": str(file_path),
                        "activations": pref.internals.activation_names,
                    }
                    pref.internals_paths = ip
                    data["preferences"][idx]["internals_paths"] = ip
                    torch.save(pref.internals.activations, file_path)

        save_json(data, path)

    @classmethod
    def load_from_json(
        cls, path: str, with_internals: bool = True
    ) -> PreferenceDataset:
        """Load preference dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            PreferenceDataset with loaded preferences
        """
        path = Path(path)
        data = load_json(path)

        preferences = []
        for p in data.get("preferences", []):
            # Backward compatibility: rename sample_id -> sample_idx
            if "sample_id" in p and "sample_idx" not in p:
                p["sample_idx"] = p.pop("sample_id")
            preferences.append(PreferenceSample(**p))

        if with_internals:
            for p in preferences:
                if p.internals_paths:
                    # TODO(claude, implement load_internals)
                    break  # Remove after load_internals is implemented
                    p.internals = load_internals(p.internals_paths)

        return cls(
            prompt_dataset_id=data["dataset_id"],
            model=data["model"],
            preferences=preferences,
            prompt_dataset_name=data.get("prompt_dataset_name", ""),
        )

    def merge(self, other: "PreferenceDataset") -> "PreferenceDataset":
        """Merge another dataset into this one.

        Combines all samples from both datasets and re-indexes them.
        Combines prompt_dataset_name values with '+' separator.

        Args:
            other: Another PreferenceDataset to merge

        Returns:
            New PreferenceDataset with combined preferences
        """
        if self.prompt_dataset_id != other.prompt_dataset_id:
            raise ValueError(
                f"Cannot merge datasets with different prompt_dataset_ids: "
                f"{self.prompt_dataset_id} vs {other.prompt_dataset_id}"
            )
        if self.model != other.model:
            raise ValueError(
                f"Cannot merge datasets with different models: "
                f"{self.model} vs {other.model}"
            )

        # Combine all samples and re-index
        merged_prefs = []
        for idx, pref in enumerate(self.preferences):
            new_pref = PreferenceSample(
                sample_idx=idx,
                choice=pref.choice,
                choice_prob=pref.choice_prob,
                alt_prob=pref.alt_prob,
                short_term_label=pref.short_term_label,
                long_term_label=pref.long_term_label,
                short_term_reward=pref.short_term_reward,
                long_term_reward=pref.long_term_reward,
                short_term_time=pref.short_term_time,
                long_term_time=pref.long_term_time,
                time_horizon=pref.time_horizon,
                response_text=pref.response_text,
                prompt_text=pref.prompt_text,
                internals=pref.internals,
                internals_paths=pref.internals_paths,
            )
            merged_prefs.append(new_pref)

        offset = len(self.preferences)
        for idx, pref in enumerate(other.preferences):
            new_pref = PreferenceSample(
                sample_idx=offset + idx,
                choice=pref.choice,
                choice_prob=pref.choice_prob,
                alt_prob=pref.alt_prob,
                short_term_label=pref.short_term_label,
                long_term_label=pref.long_term_label,
                short_term_reward=pref.short_term_reward,
                long_term_reward=pref.long_term_reward,
                short_term_time=pref.short_term_time,
                long_term_time=pref.long_term_time,
                time_horizon=pref.time_horizon,
                response_text=pref.response_text,
                prompt_text=pref.prompt_text,
                internals=pref.internals,
                internals_paths=pref.internals_paths,
            )
            merged_prefs.append(new_pref)

        # Combine prompt_dataset_name values
        names = []
        if self.prompt_dataset_name:
            names.append(self.prompt_dataset_name)
        if other.prompt_dataset_name and other.prompt_dataset_name not in names:
            names.append(other.prompt_dataset_name)
        combined_name = "+".join(names) if names else ""

        return PreferenceDataset(
            prompt_dataset_id=self.prompt_dataset_id,
            model=self.model,
            preferences=merged_prefs,
            prompt_dataset_name=combined_name,
        )

    @classmethod
    def merge_all(cls, datasets: list["PreferenceDataset"]) -> "PreferenceDataset":
        """Merge a list of datasets into one.

        Args:
            datasets: List of PreferenceDataset objects to merge

        Returns:
            Single merged PreferenceDataset

        Raises:
            ValueError: If datasets list is empty
        """
        if not datasets:
            raise ValueError("Cannot merge empty list of datasets")

        result = datasets[0]
        for ds in datasets[1:]:
            result = result.merge(ds)
        return result
