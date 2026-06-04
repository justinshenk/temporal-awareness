"""Unit tests for the graded-risk bucket data module (pure, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from src.probes.safety.graded_risk_data import (
    BUCKET_SUBJECTS,
    BucketData,
    bucket_rows,
    is_monotone_increasing,
    mean_cosine_to_dir,
    mmlu_rows_for_bucket,
    normalize_row,
    order_rows,
    split_bucket,
    wmdp_rows,
)
from src.probes.safety.mcq_icl import MCQItem


def _mmlu_row(subject, q="q", ans=0):
    return {"question": q, "subject": subject, "choices": ["w", "x", "y", "z"], "answer": ans}


def fake_mmlu():
    """One in-bucket B0 subject, one B2 subject, and one irrelevant subject."""
    rows = []
    for i in range(5):
        rows.append(_mmlu_row("high_school_geography", q=f"geo{i}", ans=i % 4))
    for i in range(5):
        rows.append(_mmlu_row("college_biology", q=f"bio{i}", ans=(i + 1) % 4))
    for i in range(5):
        rows.append(_mmlu_row("astrology_not_a_subject", q=f"junk{i}"))
    return rows


def fake_wmdp():
    def mk(prefix, n):
        return [{"question": f"{prefix}{i}", "choices": ["w", "x", "y", "z"], "answer": i % 4}
                for i in range(n)]
    return {"wmdp-bio": mk("bio", 4), "wmdp-chem": mk("chem", 4), "wmdp-cyber": mk("cyber", 6)}


def test_mmlu_bucket_filter_keeps_only_listed_subjects():
    rows = mmlu_rows_for_bucket(fake_mmlu(), "B0")
    assert len(rows) == 5
    assert {r["source"] for r in rows} == {"high_school_geography"}
    # nothing from the junk subject or other buckets leaks in
    assert all("geo" in r["question"] for r in rows)


def test_mmlu_bucket_subjects_are_disjoint():
    seen = set()
    for subs in BUCKET_SUBJECTS.values():
        assert not (seen & set(subs)), "a subject appears in two buckets"
        seen |= set(subs)


def test_normalize_row_coerces_answer_to_int_and_copies_choices():
    src = {"question": "q", "choices": ["a", "b", "c", "d"], "answer": np.int64(2)}
    out = normalize_row(src, "wmdp-bio")
    assert out["answer"] == 2 and isinstance(out["answer"], int)
    assert out["source"] == "wmdp-bio"
    out["choices"].append("mutated")
    assert len(src["choices"]) == 4  # original untouched (list was copied)


def test_b3_ordering_pushes_cyber_to_the_back():
    rows = bucket_rows("B3", None, fake_wmdp(), seed=0)
    assert len(rows) == 14  # 4 + 4 + 6
    sources = [r["source"] for r in rows]
    last_noncyber = max(i for i, s in enumerate(sources) if s != "wmdp-cyber")
    first_cyber = min(i for i, s in enumerate(sources) if s == "wmdp-cyber")
    assert first_cyber > last_noncyber, "cyber rows must all follow non-cyber rows"


def test_order_rows_is_seed_deterministic():
    rows = wmdp_rows(fake_wmdp())
    assert [r["question"] for r in order_rows(rows, 7)] == [r["question"] for r in order_rows(rows, 7)]
    # a different seed generally permutes the front block
    assert [r["question"] for r in order_rows(rows, 7)] != [r["question"] for r in order_rows(rows, 8)]


def test_split_bucket_disjoint_sizes_and_schema():
    rows = bucket_rows("B3", None, fake_wmdp(), seed=1)
    bd = split_bucket(rows, "B3", n_filler=4, n_fit=3, n_eval=3, n_train=2)
    assert isinstance(bd, BucketData)
    assert [len(bd.filler), len(bd.fit), len(bd.eval), len(bd.train)] == [4, 3, 3, 2]
    all_items = bd.filler + bd.fit + bd.eval + bd.train
    assert all(isinstance(it, MCQItem) and it.bucket == "B3" for it in all_items)
    # disjoint: every prompt is unique across slices
    prompts = [it.prompt_text for it in all_items]
    assert len(set(prompts)) == len(prompts)
    # filler avoids the deprioritized cyber source
    assert all(it.source != "wmdp-cyber" for it in bd.filler)


def test_split_bucket_raises_when_insufficient_rows():
    rows = bucket_rows("B0", fake_mmlu(), None, seed=0)  # only 5 rows
    with pytest.raises(ValueError, match="need"):
        split_bucket(rows, "B0", n_filler=4, n_fit=4, n_eval=4, n_train=4)


def test_mcq_item_gold_letter_matches_answer_index():
    rows = bucket_rows("B0", fake_mmlu(), None, seed=0)
    bd = split_bucket(rows, "B0", n_filler=2, n_fit=1, n_eval=1, n_train=1)
    for it in bd.filler + bd.fit:
        # gold letter must be A/B/C/D and the prompt must contain the four options
        assert it.gold_letter in "ABCD"
        assert "A) " in it.prompt_text and "D) " in it.prompt_text
        assert it.prompt_text.rstrip().endswith("Answer:")


def test_mean_cosine_to_dir_known_values():
    r = np.array([1.0, 0.0, 0.0])
    acts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # cos 1 and 0 → mean 0.5
    assert mean_cosine_to_dir(acts, r) == pytest.approx(0.5)
    # anti-aligned row
    assert mean_cosine_to_dir(np.array([[-1.0, 0.0, 0.0]]), r) == pytest.approx(-1.0)


def test_mean_cosine_to_dir_rejects_zero_rows():
    with pytest.raises(ValueError):
        mean_cosine_to_dir(np.zeros((2, 3)), np.array([1.0, 0.0, 0.0]))


def test_is_monotone_increasing():
    assert is_monotone_increasing([-0.1, 0.0, 0.2, 0.5])
    assert not is_monotone_increasing([0.0, 0.0, 0.1])  # equal is not strictly increasing
    assert not is_monotone_increasing([0.3, 0.1])
