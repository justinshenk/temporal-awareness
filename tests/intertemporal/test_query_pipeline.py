"""Tests for QueryRunner pipeline with production model.

These tests verify QueryRunner works end-to-end with Qwen2.5-1.5B.
Marked slow because they require model loading.
"""

import pytest

from src.prompt import PromptDatasetGenerator, PromptDatasetConfig
from src.models import QueryRunner, QueryConfig
from src.common.paths import get_prompt_dataset_configs_dir


TEST_MODEL = "Qwen/Qwen2.5-1.5B"


@pytest.mark.slow
class TestQueryPipeline:
    """Tests for QueryRunner with production model."""

    @pytest.fixture
    def setup_dataset(self, tmp_path):
        """Generate a minimal dataset for testing."""
        config_path = get_prompt_dataset_configs_dir() / "test_minimal.json"
        cfg = PromptDatasetConfig.from_json(config_path)

        generator = PromptDatasetGenerator(cfg)
        dataset = generator.generate()

        # Save dataset
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()

        output_path = datasets_dir / f"{cfg.name}_{cfg.get_id()}.json"
        dataset.save_as_json(output_path)

        return datasets_dir, cfg.get_id()

    def test_query_config_loads(self):
        """QueryConfig loads correctly."""
        config = QueryConfig(
            models=[TEST_MODEL],
            datasets=["test"],
            internals=None,
            subsample=1.0,
        )
        assert config.models == [TEST_MODEL]
        assert config.subsample == 1.0

    def test_query_runner_queries_model(self, setup_dataset):
        """QueryRunner queries model successfully."""
        datasets_dir, dataset_id = setup_dataset

        config = QueryConfig(
            models=[TEST_MODEL],
            datasets=[dataset_id],
            internals=None,
            subsample=1.0,
        )

        runner = QueryRunner(config, datasets_dir)
        output = runner.query_dataset(dataset_id, TEST_MODEL)

        assert output.model == TEST_MODEL
        assert output.dataset_id == dataset_id
        assert len(output.preferences) > 0

    def test_preferences_have_choices(self, setup_dataset):
        """Query output preferences have choice field."""
        datasets_dir, dataset_id = setup_dataset

        config = QueryConfig(
            models=[TEST_MODEL],
            datasets=[dataset_id],
            internals=None,
            subsample=1.0,
        )

        runner = QueryRunner(config, datasets_dir)
        output = runner.query_dataset(dataset_id, TEST_MODEL)

        for pref in output.preferences:
            assert pref.choice in ("short_term", "long_term", "unknown")
            assert isinstance(pref.choice_prob, float)
            assert isinstance(pref.alt_prob, float)
