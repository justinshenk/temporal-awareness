"""Tests for query_runner module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import json
import tempfile

import pytest
import torch

from src.models.query_runner import (
    ActivationSpec,
    parse_choice,
    QueryRunner,
    QueryConfig,
    InternalsConfig,
)


# =============================================================================
# parse_choice Tests
# =============================================================================


class TestParseChoice:
    @pytest.mark.parametrize(
        "response,short,long,prefix,expected",
        [
            # Basic
            ("I select: a)", "a)", "b)", "I select:", "short_term"),
            ("I select: b)", "a)", "b)", "I select:", "long_term"),
            ("I select: a", "a)", "b)", "I select:", "short_term"),  # stripped
            # Case insensitive
            ("i select: A)", "a)", "b)", "I select:", "short_term"),
            ("I SELECT: B)", "a)", "b)", "I select:", "long_term"),
            # With context
            ("Thinking... I select: a) because", "a)", "b)", "I select:", "short_term"),
            # Whitespace
            ("I select:a)", "a)", "b)", "I select:", "short_term"),
            ("I select:   b)", "a)", "b)", "I select:", "long_term"),
            # Different labels
            ("I select: option_one", "option_one", "option_two", "I select:", "short_term"),
            ("My choice: X", "X", "Y", "My choice:", "short_term"),
            # Unknown
            ("I'm not sure", "a)", "b)", "I select:", "unknown"),
            ("I select: c)", "a)", "b)", "I select:", "unknown"),
            ("a) is better", "a)", "b)", "I select:", "unknown"),  # no prefix
            ("", "a)", "b)", "I select:", "unknown"),
        ],
    )
    def test_parse_choice(self, response, short, long, prefix, expected):
        assert parse_choice(response, short, long, prefix) == expected

    def test_regex_special_chars_escaped(self):
        assert parse_choice("I select: a.)", "a.)", "b.)", "I select:") == "short_term"

    def test_longer_label_priority(self):
        assert parse_choice("I select: option_a", "option_a", "option_b", "I select:") == "short_term"


# =============================================================================
# names_filter Tests
# =============================================================================


class TestNamesFilter:
    def test_filter_matches_exact_names(self):
        names = ["blocks.8.hook_resid_post", "blocks.14.hook_resid_post"]
        f = lambda name: name in names

        assert f("blocks.8.hook_resid_post")
        assert f("blocks.14.hook_resid_post")
        assert not f("blocks.0.hook_resid_post")
        assert not f("blocks.8.hook_attn_out")

    def test_filter_with_negative_indices(self):
        names = ["blocks.-2.hook_resid_post"]
        f = lambda name: name in names

        assert f("blocks.-2.hook_resid_post")
        assert not f("blocks.2.hook_resid_post")


# =============================================================================
# Mock Backend for Testing
# =============================================================================


class MockBackend:
    """Mock that captures calls for verification."""

    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.calls = []

    def generate(self, prompt, max_new_tokens, temperature, intervention, past_kv_cache):
        self.calls.append(("generate", prompt))
        return self.responses.get("generate", "I select: a)")

    def run_with_cache(self, input_ids, names_filter, past_kv_cache):
        self.calls.append(("run_with_cache", names_filter))
        logits = torch.zeros(1, input_ids.shape[1], 100)
        cache = {}
        if names_filter:
            for name in ["blocks.8.hook_resid_post", "blocks.14.hook_resid_post"]:
                if names_filter(name):
                    cache[name] = [torch.zeros(1, input_ids.shape[1], 64)]
        return logits, cache

    def get_next_token_probs_by_id(self, prompt, token_ids, past_kv_cache):
        return {tid: 0.5 for tid in token_ids if tid is not None}

    def tokenize(self, text):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids):
        return "decoded"

    def get_tokenizer(self):
        tok = MagicMock()
        tok.encode = lambda x, add_special_tokens=False: [ord(x[0]) if x else 0]
        tok.eos_token_id = 0
        return tok

    def init_kv_cache(self):
        cache = MagicMock()
        cache.freeze = MagicMock()
        return cache

    def generate_from_cache(self, prefill_logits, frozen_kv_cache, max_new_tokens, temperature):
        return self.responses.get("generate", "I select: a)")


class MockModelRunner:
    """Mock ModelRunner that uses MockBackend."""

    def __init__(self, backend: MockBackend):
        self._backend = backend
        self.device = "cpu"
        self.dtype = torch.float32
        self._is_chat_model = False

    @property
    def tokenizer(self):
        return self._backend.get_tokenizer()

    def tokenize(self, text):
        return self._backend.tokenize(text)

    def decode(self, ids):
        return self._backend.decode(ids)

    def run_with_cache(self, prompt, names_filter=None, past_kv_cache=None):
        input_ids = self.tokenize(prompt)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    def generate_from_cache(self, prefill_logits, frozen_kv_cache, max_new_tokens=256, temperature=0.0):
        return self._backend.generate_from_cache(prefill_logits, frozen_kv_cache, max_new_tokens, temperature)

    def generate(self, prompt, max_new_tokens=256, temperature=0.0, intervention=None):
        return self._backend.generate(prompt, max_new_tokens, temperature, intervention, None)

    def get_label_probs(self, prompt, choice_prefix, labels, past_kv_cache=None):
        return (0.6, 0.4)

    def init_kv_cache(self):
        return self._backend.init_kv_cache()


# =============================================================================
# query_dataset Tests
# =============================================================================


@pytest.fixture
def sample_dataset_dict():
    """Create minimal dataset dict for testing."""
    return {
        "dataset_id": "001",
        "config": {
            "prompt_format": "default_prompt_format",
            "name": "test",
            "context": {
                "reward_unit": "dollars",
                "role": "a person",
                "situation": "Test situation",
                "labels": ["a)", "b)"],
                "method": "grid",
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 100],
                    "time_range": [[1, "months"], [1, "months"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
                "long_term": {
                    "reward_range": [200, 200],
                    "time_range": [[12, "months"], [12, "months"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
            },
            "time_horizons": [None],
        },
        "samples": [
            {
                "sample_idx": 1,
                "prompt": {
                    "text": "Choose between options:",
                    "preference_pair": {
                        "short_term": {
                            "label": "a)",
                            "time": {"value": 1, "unit": "months"},
                            "reward": {"value": 100, "unit": "dollars"},
                        },
                        "long_term": {
                            "label": "b)",
                            "time": {"value": 12, "unit": "months"},
                            "reward": {"value": 200, "unit": "dollars"},
                        },
                    },
                    "time_horizon": {"value": 6, "unit": "months"},
                }
            },
            {
                "sample_idx": 2,
                "prompt": {
                    "text": "Another choice:",
                    "preference_pair": {
                        "short_term": {
                            "label": "X",
                            "time": {"value": 1, "unit": "months"},
                            "reward": {"value": 100, "unit": "dollars"},
                        },
                        "long_term": {
                            "label": "Y",
                            "time": {"value": 12, "unit": "months"},
                            "reward": {"value": 200, "unit": "dollars"},
                        },
                    },
                    "time_horizon": None,
                }
            },
        ],
    }


@pytest.fixture
def sample_prompt_dataset(sample_dataset_dict):
    """Create a PromptDataset for testing."""
    from src.prompt import PromptDataset
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dataset_001.json"
        with open(path, "w") as f:
            json.dump(sample_dataset_dict, f)
        yield PromptDataset.from_json(path)


class TestQueryDataset:
    def test_loads_dataset_and_queries_all_samples(self, sample_prompt_dataset):
        """query_dataset processes all samples in dataset."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend({"generate": "I select: a)"})
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        assert output.dataset_id == "001"
        assert output.model == "test"
        assert len(output.preferences) == 2

    def test_extracts_choice_correctly(self, sample_prompt_dataset):
        """Choices are parsed from model responses."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # Both should be short_term since mock returns "I select: a)"
        assert output.preferences[0].choice == "short_term"

    def test_captures_internals_with_names_filter(self, sample_prompt_dataset):
        """Activations are captured using names_filter."""
        internals = InternalsConfig(
            activations=[ActivationSpec(component="resid_post", layers=[8, 14])],
        )
        config = QueryConfig(internals=internals)
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # Check that run_with_cache was called with a names_filter
        rwc_calls = [c for c in backend.calls if c[0] == "run_with_cache"]
        assert len(rwc_calls) > 0
        # Verify filter accepts configured names
        names_filter = rwc_calls[0][1]
        if names_filter:
            assert names_filter("blocks.8.hook_resid_post")
            assert names_filter("blocks.14.hook_resid_post")
            assert not names_filter("blocks.0.hook_resid_post")

    def test_subsample_reduces_samples(self, sample_prompt_dataset):
        """subsample < 1.0 processes fewer samples."""
        config = QueryConfig(subsample=0.5, internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # With 2 samples and 0.5 subsample, should get 1
        assert len(output.preferences) == 1

    def test_uses_choice_prefix_from_dataset(self, sample_prompt_dataset):
        """Choice prefix is extracted from dataset config."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        # Response has correct label but wrong prefix - should be unknown
        backend = MockBackend({"generate": "My choice: a)"})
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # Should be unknown since "My choice:" != "I select:"
        assert output.preferences[0].choice == "unknown"

    def test_long_term_choice_extracted(self, sample_prompt_dataset):
        """Long term choices are correctly identified."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend({"generate": "I select: b)"})
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        assert output.preferences[0].choice == "long_term"

    def test_probabilities_captured(self, sample_prompt_dataset):
        """Choice probabilities are stored in output."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        pref = output.preferences[0]
        # MockModelRunner.get_label_probs returns (0.6, 0.4)
        assert pref.choice_prob == 0.6
        assert pref.alt_prob == 0.4

    def test_response_stored(self, sample_prompt_dataset):
        """Raw response is stored in preference item."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend({"generate": "I select: a) for good reasons"})
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        assert output.preferences[0].response_text == "I select: a) for good reasons"

    def test_time_horizon_extracted(self, sample_prompt_dataset):
        """Time horizon from sample is preserved."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # First sample has time_horizon
        assert output.preferences[0].time_horizon == {"value": 6, "unit": "months"}
        # Second sample has no time_horizon
        assert output.preferences[1].time_horizon is None

    def test_labels_stored(self, sample_prompt_dataset):
        """Short and long term labels are stored."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        pref = output.preferences[0]
        assert pref.short_term_label == "a)"
        assert pref.long_term_label == "b)"

    def test_sample_idx_preserved(self, sample_prompt_dataset):
        """Sample IDs from dataset are preserved."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        assert output.preferences[0].sample_idx == 1
        assert output.preferences[1].sample_idx == 2

    def test_unknown_choice_probability_handling(self, sample_prompt_dataset):
        """Unknown choices use max/min of probabilities."""
        config = QueryConfig(internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        # Response that won't match any label
        backend = MockBackend({"generate": "I cannot decide"})
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        pref = output.preferences[0]
        assert pref.choice == "unknown"
        # For unknown, choice_prob = max, alt_prob = min
        assert pref.choice_prob == 0.6
        assert pref.alt_prob == 0.4

    def test_skip_generation_infers_choice_from_probs(self, sample_prompt_dataset):
        """skip_generation=True skips generate() and infers choice from probs."""
        config = QueryConfig(skip_generation=True, internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        backend = MockBackend()
        runner._model = MockModelRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        # generate should NOT have been called
        gen_calls = [c for c in backend.calls if c[0] == "generate"]
        assert len(gen_calls) == 0

        # Choice should be inferred from probs (0.6, 0.4) -> short_term wins
        pref = output.preferences[0]
        assert pref.choice == "short_term"
        assert pref.response_text == ""  # No response when skipping generation

    def test_skip_generation_long_term_when_higher_prob(self, sample_prompt_dataset):
        """skip_generation correctly picks long_term when it has higher prob."""
        config = QueryConfig(skip_generation=True, internals=InternalsConfig.empty())
        runner = QueryRunner(config)

        # Mock returns (0.3, 0.7) - long_term has higher prob
        class LongTermBackend(MockBackend):
            pass

        class LongTermMockRunner(MockModelRunner):
            def get_label_probs(self, prompt, choice_prefix, labels, past_kv_cache=None):
                return (0.3, 0.7)  # long_term wins

        backend = LongTermBackend()
        runner._model = LongTermMockRunner(backend)
        runner._model_name = "test"

        output = runner.query_dataset(sample_prompt_dataset, "test")

        pref = output.preferences[0]
        assert pref.choice == "long_term"
        assert pref.choice_prob == 0.7
        assert pref.alt_prob == 0.3


# =============================================================================
# QueryRunner Config Tests
# =============================================================================


class TestQueryConfig:
    def test_load_config_from_json(self):
        """Config loads correctly from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "max_new_tokens": 128,
                "temperature": 0.5,
                "subsample": 0.8,
                "internals": {"resid_post": {"layers": [8]}},
            }, f)
            f.flush()

            config = QueryConfig.from_json(Path(f.name))

            assert config.max_new_tokens == 128
            assert config.temperature == 0.5
            assert config.subsample == 0.8
            assert config.internals is not None

    def test_activation_names_from_config(self):
        """_get_activation_names builds correct hook names."""
        internals = InternalsConfig(
            activations=[ActivationSpec(component="resid_post", layers=[0, 5, -1])],
        )
        config = QueryConfig(internals=internals)

        runner = QueryRunner(config)
        # Create a mock model runner for the test
        mock_runner = MockModelRunner(MockBackend())
        names = runner._get_activation_names(mock_runner)

        assert "blocks.0.hook_resid_post" in names
        assert "blocks.5.hook_resid_post" in names
        assert "blocks.-1.hook_resid_post" in names
