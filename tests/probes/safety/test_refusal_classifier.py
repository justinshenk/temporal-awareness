"""Tests for the substring refusal classifier."""

import pytest

from src.probes.safety.refusal_classifier import is_refusal, refusal_rate


@pytest.mark.parametrize("text", [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with this request.",
    "As an AI, I am unable to provide that information.",
    "It is not appropriate for me to do that.",
])
def test_refusals_detected(text):
    assert is_refusal(text) is True


@pytest.mark.parametrize("text", [
    "Sure! Here is how you can do that: step 1...",
    "The most likely diagnosis is B) Pneumonia.",
    "Certainly, the steps are as follows.",
])
def test_compliance_detected(text):
    assert is_refusal(text) is False


def test_empty_is_not_refusal():
    assert is_refusal("") is False


def test_refusal_rate():
    responses = ["I'm sorry, I can't.", "Sure, here you go.", "I cannot help."]
    assert refusal_rate(responses) == pytest.approx(2 / 3)


def test_refusal_rate_empty_raises():
    with pytest.raises(ValueError, match="no responses"):
        refusal_rate([])
