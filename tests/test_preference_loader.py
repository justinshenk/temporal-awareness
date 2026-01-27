"""Tests for preference data loading."""

import json
import pytest
from pathlib import Path

from src.data.preference_loader import (
    PreferenceData,
    PreferenceItem,
    load_preference_data,
    load_dataset,
    get_prompt_text,
    merge_prompt_text,
)


# Sample preference data for testing
SAMPLE_PREFERENCE_DATA = {
    "dataset_id": "test123",
    "model": "test-model",
    "preferences": [
        {
            "sample_id": 0,
            "time_horizon": None,
            "short_term_label": "A",
            "long_term_label": "B",
            "choice": "short_term",
            "choice_prob": 0.7,
            "alt_prob": 0.3,
            "response": "I select: A",
        },
        {
            "sample_id": 1,
            "time_horizon": {"value": 2, "unit": "years"},
            "short_term_label": "A",
            "long_term_label": "B",
            "choice": "long_term",
            "choice_prob": 0.6,
            "alt_prob": 0.4,
            "response": "I select: B",
        },
        {
            "sample_id": 2,
            "time_horizon": None,
            "short_term_label": "A",
            "long_term_label": "B",
            "choice": "unknown",
            "choice_prob": 0.5,
            "alt_prob": 0.5,
            "response": "I cannot decide",
        },
    ],
}

SAMPLE_DATASET = {
    "dataset_id": "test123",
    "samples": [
        {
            "sample_id": 0,
            "prompt": {"text": ["Line 1", "Line 2"]},
        },
        {
            "sample_id": 1,
            "prompt": {"text": "Single line prompt"},
        },
        {
            "sample_id": 2,
            "prompt": {"text": ["Another", "Multi", "Line"]},
        },
    ],
}


class TestPreferenceItem:
    """Test PreferenceItem dataclass."""

    def test_create(self):
        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            short_term_label="A",
            long_term_label="B",
            choice="short_term",
            choice_prob=0.7,
            alt_prob=0.3,
            response="I select: A",
        )
        assert item.sample_id == 0
        assert item.choice == "short_term"
        assert item.prompt_text == ""

    def test_with_internals(self):
        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            short_term_label="A",
            long_term_label="B",
            choice="short_term",
            choice_prob=0.7,
            alt_prob=0.3,
            response="I select: A",
            internals={"file_path": "/path/to/file.pt"},
        )
        assert item.internals is not None
        assert "file_path" in item.internals


class TestPreferenceData:
    """Test PreferenceData dataclass."""

    def test_split_by_choice(self):
        data = PreferenceData(
            dataset_id="test",
            model="model",
            preferences=[
                PreferenceItem(
                    sample_id=0,
                    time_horizon=None,
                    short_term_label="A",
                    long_term_label="B",
                    choice="short_term",
                    choice_prob=0.7,
                    alt_prob=0.3,
                    response="A",
                ),
                PreferenceItem(
                    sample_id=1,
                    time_horizon=None,
                    short_term_label="A",
                    long_term_label="B",
                    choice="long_term",
                    choice_prob=0.6,
                    alt_prob=0.4,
                    response="B",
                ),
            ],
        )
        short, long = data.split_by_choice()
        assert len(short) == 1
        assert len(long) == 1
        assert short[0].sample_id == 0
        assert long[0].sample_id == 1

    def test_filter_valid(self):
        data = PreferenceData(
            dataset_id="test",
            model="model",
            preferences=[
                PreferenceItem(
                    sample_id=0,
                    time_horizon=None,
                    short_term_label="A",
                    long_term_label="B",
                    choice="short_term",
                    choice_prob=0.7,
                    alt_prob=0.3,
                    response="A",
                ),
                PreferenceItem(
                    sample_id=1,
                    time_horizon=None,
                    short_term_label="A",
                    long_term_label="B",
                    choice="unknown",
                    choice_prob=0.5,
                    alt_prob=0.5,
                    response="?",
                ),
            ],
        )
        valid = data.filter_valid()
        assert len(valid) == 1
        assert valid[0].sample_id == 0


class TestLoadPreferenceData:
    """Test load_preference_data function."""

    def test_load_from_path(self, tmp_path):
        pref_file = tmp_path / "prefs.json"
        pref_file.write_text(json.dumps(SAMPLE_PREFERENCE_DATA))

        data = load_preference_data(pref_file)
        assert data.dataset_id == "test123"
        assert data.model == "test-model"
        assert len(data.preferences) == 3

    def test_load_by_id(self, tmp_path):
        pref_file = tmp_path / "test123_model.json"
        pref_file.write_text(json.dumps(SAMPLE_PREFERENCE_DATA))

        data = load_preference_data("test123", preference_dir=tmp_path)
        assert data.dataset_id == "test123"

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_preference_data("nonexistent", preference_dir=tmp_path)

    def test_preferences_parsed(self, tmp_path):
        pref_file = tmp_path / "prefs.json"
        pref_file.write_text(json.dumps(SAMPLE_PREFERENCE_DATA))

        data = load_preference_data(pref_file)
        assert data.preferences[0].choice == "short_term"
        assert data.preferences[1].time_horizon == {"value": 2, "unit": "years"}


class TestLoadDataset:
    """Test load_dataset function."""

    def test_load(self, tmp_path):
        ds_file = tmp_path / "config_test123.json"
        ds_file.write_text(json.dumps(SAMPLE_DATASET))

        dataset = load_dataset("test123", tmp_path)
        assert dataset["dataset_id"] == "test123"
        assert len(dataset["samples"]) == 3

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent", tmp_path)


class TestGetPromptText:
    """Test get_prompt_text function."""

    def test_list_text(self):
        sample = {"prompt": {"text": ["Line 1", "Line 2"]}}
        text = get_prompt_text(sample)
        assert text == "Line 1\nLine 2"

    def test_string_text(self):
        sample = {"prompt": {"text": "Single line"}}
        text = get_prompt_text(sample)
        assert text == "Single line"

    def test_empty(self):
        sample = {"prompt": {}}
        text = get_prompt_text(sample)
        assert text == ""


class TestMergePromptText:
    """Test merge_prompt_text function."""

    def test_merge(self, tmp_path):
        pref_file = tmp_path / "prefs.json"
        pref_file.write_text(json.dumps(SAMPLE_PREFERENCE_DATA))

        data = load_preference_data(pref_file)
        merge_prompt_text(data, SAMPLE_DATASET)

        assert data.preferences[0].prompt_text == "Line 1\nLine 2"
        assert data.preferences[1].prompt_text == "Single line prompt"
