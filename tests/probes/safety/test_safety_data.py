"""Tests for pure field-extraction from safety datasets (no network)."""

from src.probes.safety.safety_data import harmful_texts, harmless_instructions


def test_harmful_texts_extracts_and_strips():
    rows = [{"text": "  do bad thing  "}, {"text": ""}, {"text": "another"}]
    assert harmful_texts(rows) == ["do bad thing", "another"]


def test_harmless_instructions_filters_inputful_and_empty():
    rows = [
        {"instruction": "Summarize this", "input": "some passage"},  # has input -> drop
        {"instruction": "Give three tips", "input": ""},              # keep
        {"instruction": "  ", "input": ""},                          # empty -> drop
        {"instruction": "Name a color", "input": "   "},             # whitespace input -> keep
    ]
    assert harmless_instructions(rows) == ["Give three tips", "Name a color"]
