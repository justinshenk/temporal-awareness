"""Unit tests for the refusal-decision token slice."""

from __future__ import annotations

import pytest

from src.probes.attribution.refusal_collection import refusal_token_slice


def test_includes_last_prompt_token_and_decode():
    # prompt_len=5, 8 decoded tokens -> positions 4..12 (decision + 8 decode) = 9 rows
    sl = refusal_token_slice(5, 13)
    assert sl == slice(4, 13)
    assert len(range(*sl.indices(13))) == 9


def test_no_tokens_generated_keeps_decision_position():
    # even if the model emits nothing, the last prompt token (the decision) is kept
    sl = refusal_token_slice(5, 5)
    assert sl == slice(4, 5)
    assert len(range(*sl.indices(5))) == 1


def test_start_is_one_before_prompt_len():
    assert refusal_token_slice(1, 4).start == 0
    assert refusal_token_slice(10, 20).start == 9


@pytest.mark.parametrize("prompt_len,full_len", [(0, 5), (6, 5), (-1, 3), (0, 0)])
def test_invalid_bounds_raise(prompt_len, full_len):
    with pytest.raises(ValueError):
        refusal_token_slice(prompt_len, full_len)
