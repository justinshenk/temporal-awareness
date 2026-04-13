"""Tests for src/probes/ddxplus.py helpers."""

from src.probes.ddxplus import extract_mcq_answer


def test_extract_letter_only():
    assert extract_mcq_answer("B") == "B"


def test_extract_letter_with_leading_space():
    assert extract_mcq_answer(" B") == "B"


def test_extract_letter_with_trailing_punctuation():
    assert extract_mcq_answer("B.") == "B"
    assert extract_mcq_answer("B)") == "B"


def test_extract_letter_with_continuation():
    assert extract_mcq_answer("B because the symptoms suggest...") == "B"


def test_extract_answer_colon_form():
    """Regression: previously parsed as 'A' due to 'ANSWER' upper-case bug."""
    assert extract_mcq_answer("Answer: B") == "B"


def test_extract_the_answer_is_form():
    assert extract_mcq_answer("The answer is C.") == "C"


def test_extract_returns_none_on_garbage():
    assert extract_mcq_answer("wrong format") is None
    assert extract_mcq_answer("") is None
    assert extract_mcq_answer("   ") is None


def test_extract_lowercase():
    assert extract_mcq_answer("b") == "B"
