"""Unit tests for the steered-generation coherence check."""

from __future__ import annotations

import pytest

from src.probes.safety.text_coherence import is_coherent


@pytest.mark.parametrize("text", [
    "I cannot fulfill your request. I'm just an AI, and I don't have the expertise to help.",
    "Sure, here is a plan. The target should be chosen carefully and the steps are as follows.",
    "I cannot.",
    "No.",
])
def test_coherent_text_passes(text):
    assert is_coherent(text) is True


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "ЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪ",                       # non-ASCII charset spam
    "Sure Sure Sure Sure Sure Sure Sure Sure",   # single-token loop
    ". . . . . . . . . . . . . . .",             # punctuation loop
    "double double double double double double",
])
def test_degenerate_text_fails(text):
    assert is_coherent(text) is False


def test_short_distinct_text_not_penalized_for_repetition():
    # fewer than 5 tokens: skip the diversity check (don't reject legitimate short replies)
    assert is_coherent("I will not.") is True
