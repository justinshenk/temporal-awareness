"""Tests for token position resolution."""

import pytest

from src.common.token_positions import (
    TokenPositionSpec,
    ResolvedPosition,
    resolve_position,
    resolve_positions,
    get_position_label,
    PROMPT_KEYWORDS,
    LAST_OCCURRENCE_KEYWORDS,
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
        assert "format" in PROMPT_KEYWORDS
        assert "choice_prefix" in PROMPT_KEYWORDS
        assert "reasoning_prefix" in PROMPT_KEYWORDS

    def test_keyword_values(self):
        assert PROMPT_KEYWORDS["situation"] == "SITUATION:"
        assert PROMPT_KEYWORDS["task"] == "TASK:"
        assert PROMPT_KEYWORDS["consider"] == "CONSIDER:"
        assert PROMPT_KEYWORDS["action"] == "ACTION:"
        assert PROMPT_KEYWORDS["choice_prefix"] == "I select:"
        assert PROMPT_KEYWORDS["reasoning_prefix"] == "My reasoning:"

    def test_no_stale_keywords(self):
        """Ensure removed keywords are not present."""
        assert "option_one" not in PROMPT_KEYWORDS
        assert "option_two" not in PROMPT_KEYWORDS

    def test_derived_from_default_prompt_format(self):
        """PROMPT_KEYWORDS values come from DefaultPromptFormat helper methods."""
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        keyword_map = fmt.get_keyword_map()
        assert PROMPT_KEYWORDS == keyword_map
        assert PROMPT_KEYWORDS["situation"] == fmt.const_keywords["situation_marker"]
        assert PROMPT_KEYWORDS["choice_prefix"] == fmt.const_keywords["format_choice_prefix"]

    def test_last_occurrence_keywords(self):
        """LAST_OCCURRENCE_KEYWORDS matches DefaultPromptFormat helper."""
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        assert LAST_OCCURRENCE_KEYWORDS == fmt.get_last_occurrence_keyword_names()
        assert "choice_prefix" in LAST_OCCURRENCE_KEYWORDS
        assert "reasoning_prefix" in LAST_OCCURRENCE_KEYWORDS
        assert "situation" not in LAST_OCCURRENCE_KEYWORDS


class TestGetInterestingPositions:
    """Test DefaultPromptFormat.get_interesting_positions()."""

    def test_returns_list_of_dicts(self):
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        assert isinstance(positions, list)
        assert len(positions) > 0
        assert all(isinstance(p, dict) for p in positions)

    def test_contains_prompt_markers(self):
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        texts = [p["text"] for p in positions]
        assert "SITUATION:" in texts
        assert "TASK:" in texts
        assert "CONSIDER:" in texts
        assert "ACTION:" in texts

    def test_response_markers_use_last(self):
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
        fmt = DefaultPromptFormat()
        positions = fmt.get_interesting_positions()
        last_positions = [p for p in positions if p.get("last", False)]
        assert len(last_positions) > 0
        # Response markers search for "I select:" and "My reasoning:"
        last_texts = [p["text"] for p in last_positions]
        assert "I select:" in last_texts
        assert "My reasoning:" in last_texts

    def test_positions_resolve_against_tokens(self):
        """Interesting positions can resolve against sample tokens."""
        from src.formatting.configs.default_prompt_format import DefaultPromptFormat
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
