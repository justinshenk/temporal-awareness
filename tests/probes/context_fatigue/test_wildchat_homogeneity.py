"""Tests for TF-IDF conversation homogeneity scoring (GPU-free)."""

import math

import pytest

from src.probes.context_fatigue.wildchat_homogeneity import (
    consecutive_homogeneity,
    homogeneity_score,
    user_messages,
)


def test_user_messages_picks_user_turns():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    assert user_messages(msgs) == ["a", "c"]


def test_identical_turns_score_one():
    texts = ["translate hello into french"] * 4
    assert homogeneity_score(texts) == pytest.approx(1.0, abs=1e-6)


def test_disjoint_vocab_scores_zero():
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    assert homogeneity_score(texts) == pytest.approx(0.0, abs=1e-6)


def test_homogeneous_above_heterogeneous():
    homo = ["translate cat into french", "translate dog into french",
            "translate bird into french", "translate fish into french"]
    hetero = ["translate cat into french", "write a python sorting function",
              "what is the capital of peru", "explain photosynthesis briefly"]
    assert homogeneity_score(homo) > homogeneity_score(hetero)


def test_too_few_or_empty_turns_return_nan():
    assert math.isnan(homogeneity_score(["only one"]))
    assert math.isnan(homogeneity_score([]))
    assert math.isnan(homogeneity_score(["", "   "]))


def test_consecutive_homogeneity_runs_and_bounds():
    texts = ["translate a", "translate b", "translate c"]
    v = consecutive_homogeneity(texts)
    assert 0.0 <= v <= 1.0
