"""Tests for token position resolution."""

from src.common.token_positions import (
    resolve_position,
    resolve_positions,
)


# Sample tokens mimicking a typical prompt (tokens as they appear from tokenizer)
SAMPLE_TOKENS = [
    "<bos>",
    "SITUATION",
    ":",
    " Plan",
    " for",
    " housing",
    " TASK",
    ":",
    " You",
    ",",
    " the",
    " city",
    " CONSIDER",
    ":",
    " Think",
    " deeply",
    " ACTION",
    ":",
    " Select",
    " FORMAT",
    ":",
    " Respond",
    " I",
    " select",
    ":",
    " OPTION",
    "_ONE",
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
