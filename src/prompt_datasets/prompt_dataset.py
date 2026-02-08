"""Prompt dataset class for storing generated prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..common.io import load_json, save_json, ensure_dir, get_timestamp
from ..common.paths import get_prompt_dataset_dir

if TYPE_CHECKING:
    from .prompt_dataset_config import PromptDatasetConfig
    from ..common.types import PromptSample


@dataclass
class PromptDataset:
    """Prompt dataset with samples and config."""

    dataset_id: str
    config: PromptDatasetConfig
    samples: list[PromptSample] = field(default_factory=list)

    def get_prompts_by_id(self) -> dict[int, str]:
        """Build a lookup of sample_idx -> prompt text."""
        return {s.sample_idx: s.prompt.text for s in self.samples}

    def save_as_json(self, path: Optional[Path] = None) -> None:
        """Save the prompt dataset to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        if path is None:
            path = get_prompt_dataset_dir() / self.config.get_filename()

        path = Path(path)
        ensure_dir(path.parent)

        data = {
            "dataset_id": self.dataset_id,
            "timestamp": get_timestamp(),
            "config": self.config.to_dict(),
            "samples": [asdict(s) for s in self.samples],
        }
        save_json(data, path)

    @classmethod
    def from_json(cls, path: str) -> "PromptDataset":
        """Load prompt dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            PromptDataset with loaded samples
        """
        from .prompt_dataset_config import PromptDatasetConfig
        from ..common.types import (
            PromptSample,
            Prompt,
            PreferencePair,
            IntertemporalOption,
            RewardValue,
            TimeValue,
        )

        path = Path(path)
        data = load_json(path)

        config = PromptDatasetConfig.from_dict(data["config"])

        samples = []
        for s in data.get("samples", []):
            # Parse the prompt
            prompt_data = s["prompt"]
            pair_data = prompt_data["preference_pair"]

            short_term = IntertemporalOption(
                label=pair_data["short_term"]["label"],
                time=TimeValue.parse(pair_data["short_term"]["time"]),
                reward=RewardValue(
                    value=pair_data["short_term"]["reward"]["value"],
                    unit=pair_data["short_term"]["reward"]["unit"],
                ),
            )
            long_term = IntertemporalOption(
                label=pair_data["long_term"]["label"],
                time=TimeValue.parse(pair_data["long_term"]["time"]),
                reward=RewardValue(
                    value=pair_data["long_term"]["reward"]["value"],
                    unit=pair_data["long_term"]["reward"]["unit"],
                ),
            )

            time_horizon = None
            if prompt_data.get("time_horizon"):
                time_horizon = TimeValue.parse(prompt_data["time_horizon"])

            prompt = Prompt(
                preference_pair=PreferencePair(
                    short_term=short_term, long_term=long_term
                ),
                time_horizon=time_horizon,
                text=prompt_data["text"],
            )

            samples.append(
                PromptSample(
                    sample_idx=s["sample_idx"],
                    prompt=prompt,
                )
            )

        return cls(
            dataset_id=data["dataset_id"],
            config=config,
            samples=samples,
        )

    @classmethod
    def load_from_id(
        cls,
        dataset_id: str,
        directory: Optional[Path] = None,
    ) -> "PromptDataset":
        """Load prompt dataset by its ID.

        Searches for a file matching *_{dataset_id}.json or dataset_{dataset_id}.json
        in the specified directory.

        Args:
            dataset_id: The dataset ID to search for
            directory: Directory to search in (default: get_prompt_dataset_dir())

        Returns:
            PromptDataset loaded from the matching file

        Raises:
            FileNotFoundError: If no matching dataset file is found
        """
        if directory is None:
            directory = get_prompt_dataset_dir()
        directory = Path(directory)

        # Search for files matching the dataset_id pattern: {name}_{dataset_id}.json
        pattern = f"*_{dataset_id}.json"
        matches = list(directory.glob(pattern))
        if matches:
            return cls.from_json(matches[0])

        raise FileNotFoundError(
            f"No prompt dataset found with ID '{dataset_id}' in {directory}"
        )
