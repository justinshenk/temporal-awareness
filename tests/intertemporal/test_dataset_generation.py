"""Tests for dataset generation functionality.

These tests verify DatasetGenerator and DatasetConfig work correctly.
Fast tests - no model loading required.
"""

from pathlib import Path

from src.datasets import DatasetGenerator


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class TestDatasetGeneration:
    """Tests for dataset generation functionality."""

    def test_load_test_minimal_config(self):
        """test_minimal config loads correctly."""
        config_path = SCRIPTS_DIR / "data" / "configs" / "test_minimal.json"
        cfg = DatasetGenerator.load_dataset_config(config_path)

        assert cfg.name == "test_minimal"
        assert cfg.time_horizons == [None]

    def test_generates_samples(self):
        """DatasetGenerator produces samples."""
        config_path = SCRIPTS_DIR / "data" / "configs" / "test_minimal.json"
        cfg = DatasetGenerator.load_dataset_config(config_path)

        generator = DatasetGenerator(cfg)
        samples = generator.generate()

        assert len(samples) > 0

    def test_sample_structure(self):
        """Generated samples have expected structure."""
        config_path = SCRIPTS_DIR / "data" / "configs" / "test_minimal.json"
        cfg = DatasetGenerator.load_dataset_config(config_path)

        generator = DatasetGenerator(cfg)
        samples = generator.generate()
        sample = samples[0]

        assert hasattr(sample, "sample_id")
        assert hasattr(sample, "prompt")
        # Prompt is a Prompt object, not a dict
        assert hasattr(sample.prompt, "text")
        assert hasattr(sample.prompt, "preference_pair")
        assert hasattr(sample.prompt.preference_pair, "short_term")
        assert hasattr(sample.prompt.preference_pair, "long_term")
        assert hasattr(sample.prompt.preference_pair.short_term, "label")

    def test_dataset_id_generated(self):
        """Config generates a dataset ID."""
        config_path = SCRIPTS_DIR / "data" / "configs" / "test_minimal.json"
        cfg = DatasetGenerator.load_dataset_config(config_path)

        dataset_id = cfg.get_id()
        assert isinstance(dataset_id, str)
        assert len(dataset_id) > 0
