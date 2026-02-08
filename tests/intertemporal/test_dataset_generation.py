"""Tests for prompt dataset generation functionality.

These tests verify PromptDatasetGenerator and PromptDatasetConfig work correctly.
Fast tests - no model loading required.
"""

from src.prompt_datasets import PromptDatasetGenerator, PromptDatasetConfig
from src.common.paths import get_prompt_dataset_configs_dir


class TestDatasetGeneration:
    """Tests for dataset generation functionality."""

    def test_load_test_minimal_config(self):
        """test_minimal config loads correctly."""
        config_path = get_prompt_dataset_configs_dir() / "test_minimal.json"
        cfg = PromptDatasetConfig.from_json(config_path)

        assert cfg.name == "test_minimal"
        assert cfg.time_horizons == [None]

    def test_generates_samples(self):
        """PromptDatasetGenerator produces samples."""
        config_path = get_prompt_dataset_configs_dir() / "test_minimal.json"
        cfg = PromptDatasetConfig.from_json(config_path)

        generator = PromptDatasetGenerator(cfg)
        dataset = generator.generate()

        assert len(dataset.samples) > 0

    def test_sample_structure(self):
        """Generated samples have expected structure."""
        config_path = get_prompt_dataset_configs_dir() / "test_minimal.json"
        cfg = PromptDatasetConfig.from_json(config_path)

        generator = PromptDatasetGenerator(cfg)
        dataset = generator.generate()
        sample = dataset.samples[0]

        assert hasattr(sample, "sample_idx")
        assert hasattr(sample, "prompt")
        # Prompt is a Prompt object, not a dict
        assert hasattr(sample.prompt, "text")
        assert hasattr(sample.prompt, "preference_pair")
        assert hasattr(sample.prompt.preference_pair, "short_term")
        assert hasattr(sample.prompt.preference_pair, "long_term")
        assert hasattr(sample.prompt.preference_pair.short_term, "label")

    def test_dataset_id_generated(self):
        """Config generates a dataset ID."""
        config_path = get_prompt_dataset_configs_dir() / "test_minimal.json"
        cfg = PromptDatasetConfig.from_json(config_path)

        dataset_id = cfg.get_id()
        assert isinstance(dataset_id, str)
        assert len(dataset_id) > 0
