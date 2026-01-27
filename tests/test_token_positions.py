"""Tests for token position resolution."""

import pytest

from src.common.token_positions import (
    TokenPositionSpec,
    ResolvedPosition,
    resolve_position,
    resolve_positions,
    get_position_label,
    PROMPT_KEYWORDS,
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


class TestResolvePositionKeyword:
    """Test keyword-based resolution."""

    def test_keyword_consider(self):
        result = resolve_position({"keyword": "consider"}, SAMPLE_TOKENS)
        assert result.found is True
        assert "CONSIDER" in SAMPLE_TOKENS[result.index].upper()

    def test_keyword_task(self):
        result = resolve_position({"keyword": "task"}, SAMPLE_TOKENS)
        assert result.found is True

    def test_keyword_string_shorthand(self):
        # String that matches a keyword name uses PROMPT_KEYWORDS
        result = resolve_position("consider", SAMPLE_TOKENS)
        # Should search for "CONSIDER:" not just "consider"
        assert result.found is True

    def test_unknown_keyword(self):
        result = resolve_position({"keyword": "unknown"}, SAMPLE_TOKENS)
        assert result.found is False


class TestTokenPositionSpec:
    """Test TokenPositionSpec dataclass."""

    def test_from_dict(self):
        spec = TokenPositionSpec.from_dict({"text": "test"})
        assert spec.spec == {"text": "test"}

    def test_from_int(self):
        spec = TokenPositionSpec.from_dict(5)
        assert spec.spec == 5

    def test_resolve_with_spec_object(self):
        spec = TokenPositionSpec(spec={"text": "CONSIDER"})
        result = resolve_position(spec, SAMPLE_TOKENS)
        assert result.found is True


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


class TestGetPositionLabel:
    """Test label generation."""

    def test_int_label(self):
        assert get_position_label(5) == "pos 5"

    def test_text_label_short(self):
        label = get_position_label({"text": "test"})
        assert label == "test"

    def test_text_label_long(self):
        label = get_position_label({"text": "very long text that exceeds limit"})
        assert ".." in label
        assert len(label) <= 12  # 10 chars + ".."

    def test_relative_label(self):
        label = get_position_label({"relative_to": "end", "offset": -3})
        assert "end" in label
        assert "-3" in label

    def test_keyword_label(self):
        label = get_position_label({"keyword": "consider"})
        assert "consider" in label.lower()


class TestPromptKeywords:
    """Test PROMPT_KEYWORDS mapping."""

    def test_keywords_defined(self):
        assert "situation" in PROMPT_KEYWORDS
        assert "task" in PROMPT_KEYWORDS
        assert "consider" in PROMPT_KEYWORDS
        assert "action" in PROMPT_KEYWORDS
        assert "choice_prefix" in PROMPT_KEYWORDS

    def test_keyword_values(self):
        assert "SITUATION:" in PROMPT_KEYWORDS["situation"]
        assert "I select:" in PROMPT_KEYWORDS["choice_prefix"]
