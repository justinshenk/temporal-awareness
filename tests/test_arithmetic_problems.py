"""Tests for the short-output arithmetic problem generator (CPU, fast)."""

from __future__ import annotations

import re

from src.probes.attribution.arithmetic_problems import (
    DEFAULT_TIERS,
    generate_arithmetic_problems,
)

_WORD_OP = {"times": lambda a, b: a * b, "plus": lambda a, b: a + b, "minus": lambda a, b: a - b}
_Q_RE = re.compile(r"What is (\d+) (times|plus|minus) (\d+)\?")


def _recompute(question: str) -> float:
    a, word, b = _Q_RE.match(question).groups()
    return float(_WORD_OP[word](int(a), int(b)))


def test_deterministic_same_seed():
    assert generate_arithmetic_problems(30, 42) == generate_arithmetic_problems(30, 42)


def test_different_seed_differs():
    assert generate_arithmetic_problems(30, 42) != generate_arithmetic_problems(30, 7)


def test_gold_matches_question_arithmetic():
    for question, gold, _ in generate_arithmetic_problems(60, 42):
        assert gold == _recompute(question)


def test_format_and_count():
    probs = generate_arithmetic_problems(12, 1)
    assert len(probs) == 12
    for question, gold, tier in probs:
        assert question.startswith("What is ") and question.endswith("?")
        assert tier in DEFAULT_TIERS
        assert isinstance(gold, float)


def test_tiers_round_robin_balanced():
    counts = {}
    for _, _, tier in generate_arithmetic_problems(30, 42):
        counts[tier] = counts.get(tier, 0) + 1
    assert counts == {t: 10 for t in DEFAULT_TIERS}


def test_subtraction_stays_positive():
    for _, gold, tier in generate_arithmetic_problems(90, 99):
        if tier == "sub3":
            assert gold >= 0
