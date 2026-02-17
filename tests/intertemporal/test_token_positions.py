"""Tests for token position resolution."""

import pytest

from src.common.token_positions import (
    ResolvedPosition,
    resolve_position,
    resolve_positions,
)


# Sample tokens mimicking a typical prompt (tokens as they appear from tokenizer)
SAMPLE_TOKENS = [
    "<bos>", "SITUATION", ":", " Plan", " for", " housing",
    " TASK", ":", " You", ",", " the", " city",
    " CONSIDER", ":", " Think", " deeply",
    " ACTION", ":", " Select",
    " FORMAT", ":", " Respond",
    " I", " select", ":", " OPTION", "_ONE",
]


class TestResolvePositionAbsolute:
    """Test absolute position resolution."""

    def test_valid_position(self):
        result = resolve_position(5, SAMPLE_TOKENS)
        assert result.index == 5
        assert result.found is True
        assert "pos_5" in result.label

    def test_zero_position(self):
        result = resolve_position(0, SAMPLE_TOKENS)
        assert result.index == 0
        assert result.found is True

    def test_last_position(self):
        result = resolve_position(len(SAMPLE_TOKENS) - 1, SAMPLE_TOKENS)
        assert result.index == len(SAMPLE_TOKENS) - 1
        assert result.found is True

    def test_out_of_bounds_positive(self):
        result = resolve_position(100, SAMPLE_TOKENS)
        assert result.index == -1
        assert result.found is False

    def test_negative_position(self):
        result = resolve_position(-1, SAMPLE_TOKENS)
        assert result.found is False


class TestResolvePositionTextSearch:
    """Test text search resolution."""

    def test_exact_match(self):
        result = resolve_position({"text": "CONSIDER"}, SAMPLE_TOKENS)
        assert result.found is True
        assert SAMPLE_TOKENS[result.index].strip().upper() == "CONSIDER"

    def test_substring_match(self):
        result = resolve_position({"text": "housing"}, SAMPLE_TOKENS)
        assert result.found is True
        assert "housing" in SAMPLE_TOKENS[result.index].lower()

    def test_case_insensitive(self):
        result = resolve_position({"text": "consider"}, SAMPLE_TOKENS)
        assert result.found is True

    def test_not_found(self):
        result = resolve_position({"text": "NONEXISTENT"}, SAMPLE_TOKENS)
        assert result.found is False
        assert result.index == -1

    def test_string_shorthand(self):
        result = resolve_position("TASK", SAMPLE_TOKENS)
        assert result.found is True

    def test_text_last_occurrence(self):
        """{"text": ..., "last": True} finds last occurrence."""
        # ":" appears multiple times in SAMPLE_TOKENS (positions 2, 7, 17, 24)
        first = resolve_position({"text": ":"}, SAMPLE_TOKENS)
        last = resolve_position({"text": ":", "last": True}, SAMPLE_TOKENS)
        assert first.found is True
        assert last.found is True
        assert last.index > first.index

    def test_text_last_single_occurrence(self):
        """{"text": ..., "last": True} works when only one occurrence exists."""
        result = resolve_position({"text": "housing", "last": True}, SAMPLE_TOKENS)
        assert result.found is True
        assert "housing" in SAMPLE_TOKENS[result.index].lower()


class TestResolvePositionRelative:
    """Test relative position resolution."""

    def test_relative_to_end(self):
        result = resolve_position({"relative_to": "end", "offset": -1}, SAMPLE_TOKENS)
        assert result.index == len(SAMPLE_TOKENS) - 1
        assert result.found is True

    def test_relative_to_end_offset_minus_5(self):
        result = resolve_position({"relative_to": "end", "offset": -5}, SAMPLE_TOKENS)
        assert result.index == len(SAMPLE_TOKENS) - 5
        assert result.found is True

    def test_relative_to_prompt_end(self):
        prompt_len = 15
        result = resolve_position(
            {"relative_to": "prompt_end", "offset": 0},
            SAMPLE_TOKENS,
            prompt_len=prompt_len,
        )
        assert result.index == prompt_len
        assert result.found is True

    def test_relative_to_start(self):
        result = resolve_position({"relative_to": "start", "offset": 5}, SAMPLE_TOKENS)
        assert result.index == 5
        assert result.found is True

    def test_relative_out_of_bounds(self):
        result = resolve_position({"relative_to": "end", "offset": 10}, SAMPLE_TOKENS)
        assert result.found is False


class TestResolvePositions:
    """Test batch resolution."""

    def test_multiple_specs(self):
        specs = [
            {"text": "CONSIDER"},
            {"relative_to": "end", "offset": -1},
            5,
        ]
        results = resolve_positions(specs, SAMPLE_TOKENS)
        assert len(results) == 3
        assert all(r.found for r in results)

    def test_mixed_found_not_found(self):
        specs = [
            {"text": "CONSIDER"},
            {"text": "NONEXISTENT"},
        ]
        results = resolve_positions(specs, SAMPLE_TOKENS)
        assert results[0].found is True
        assert results[1].found is False


class TestGetInterestingPositions:
    """Test DefaultPromptFormat.get_interesting_positions()."""

    def test_returns_list_of_dicts(self):
        from src.intertemporal.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        assert isinstance(positions, list)
        assert len(positions) > 0
        assert all(isinstance(p, dict) for p in positions)

    def test_contains_prompt_markers(self):
        from src.intertemporal.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        texts = [p["text"] for p in positions]
        assert fmt.prompt_const_keywords["situation_marker"] in texts
        assert fmt.prompt_const_keywords["task_marker"] in texts
        assert fmt.prompt_const_keywords["consider_marker"] in texts
        assert fmt.prompt_const_keywords["action_marker"] in texts

    def test_response_markers_use_last(self):
        from src.intertemporal.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        response_positions = [p for p in positions if p.get("section") == "response"]
        assert len(response_positions) > 0
        # Response markers use last occurrence
        response_texts = [p["text"] for p in response_positions]
        assert fmt.response_const_keywords["response_choice_prefix"] in response_texts
        assert fmt.response_const_keywords["response_reasoning_prefix"] in response_texts

    def test_positions_resolve_against_tokens(self):
        """Interesting positions can resolve against sample tokens."""
        from src.intertemporal.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        # Use a token list that mimics a rendered prompt with response
        tokens = [
            "SITUATION", ":", " Choose", ".",
            " TASK", ":", " You",
            " CONSIDER", ":", " Think",
            " ACTION", ":", " Select",
            " FORMAT", ":", " Respond",
            " I", " select", ":",  # FORMAT section occurrence
            " A", ".",
            " I", " select", ":",  # Response occurrence
            " B", ".",
            " My", " reasoning", ":",  # Response occurrence
            " Because",
        ]
        for pos_spec in positions:
            result = resolve_position(pos_spec, tokens)
            # All markers should be found in this token list
            assert result.found, f"Position {pos_spec} not found in tokens"
