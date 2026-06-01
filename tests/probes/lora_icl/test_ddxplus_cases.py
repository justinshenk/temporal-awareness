"""Tests for DDXPlus case construction (model-free, fake dataset)."""

import random

import pytest

from src.probes.lora_icl.ddxplus_cases import (
    build_case,
    build_cases,
    disjoint_split,
    permute_labels,
    select_valid_indices,
)

# Minimal evidence DB + fake dataset rows mimicking aai530-group6/ddxplus fields.
_EVIDENCE_DB = {
    "E_55": {"question": "Do you have a cough?", "is_antecedent": False,
             "data_type": "B", "value_meanings": {}},
}


class _FakeDS:
    """List-of-dicts standing in for a HF dataset."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def _row(pathology, ddx):
    return {
        "AGE": 40, "SEX": "M", "INITIAL_EVIDENCE": "E_55", "EVIDENCES": "['E_55']",
        "PATHOLOGY": pathology,
        "DIFFERENTIAL_DIAGNOSIS": str([[d, 0.5] for d in ddx]),
    }


def test_select_valid_indices_filters_gold_in_topn():
    ds = _FakeDS([
        _row("Flu", ["Flu", "Cold", "RSV"]),          # gold in top-3 -> valid
        _row("Cancer", ["Flu", "Cold", "RSV"]),         # gold absent -> dropped
    ])
    assert select_valid_indices(ds, n_options=3) == [0]


def test_build_case_gold_letter_matches_shuffled_position():
    ds = _FakeDS([_row("Cold", ["Flu", "Cold", "RSV", "Asthma", "COPD"])])
    case = build_case(ds, 0, _EVIDENCE_DB, n_options=5, rng=random.Random(0))
    # gold letter must point at "Cold" in the shuffled option list
    gold_idx = "ABCDE".index(case.gold_letter)
    assert case.option_names[gold_idx] == "Cold"
    assert "Most likely diagnosis:" in case.prompt_text


def test_build_case_rejects_gold_not_in_options():
    ds = _FakeDS([_row("Cancer", ["Flu", "Cold", "RSV"])])
    with pytest.raises(ValueError, match="not in top"):
        build_case(ds, 0, _EVIDENCE_DB, n_options=3, rng=random.Random(0))


def test_build_cases_deterministic():
    ds = _FakeDS([_row("Cold", ["Flu", "Cold", "RSV", "Asthma", "COPD"])] * 4)
    a = build_cases(ds, [0, 1, 2, 3], _EVIDENCE_DB, 5, seed=42)
    b = build_cases(ds, [0, 1, 2, 3], _EVIDENCE_DB, 5, seed=42)
    assert [c.gold_letter for c in a] == [c.gold_letter for c in b]


def test_disjoint_split_no_overlap():
    train, ev = disjoint_split(list(range(100)), n_train=60, n_eval=20, seed=1)
    assert len(train) == 60 and len(ev) == 20
    assert set(train).isdisjoint(set(ev))


def test_disjoint_split_too_large_raises():
    with pytest.raises(ValueError, match="exceeds"):
        disjoint_split(list(range(10)), n_train=8, n_eval=8, seed=1)


def test_permute_labels_preserves_prompts_and_letter_multiset():
    ds = _FakeDS([_row("Cold", ["Flu", "Cold", "RSV", "Asthma", "COPD"])] * 6)
    cases = build_cases(ds, list(range(6)), _EVIDENCE_DB, 5, seed=3)
    permuted = permute_labels(cases, seed=7)
    assert [c.prompt_text for c in permuted] == [c.prompt_text for c in cases]
    assert sorted(c.gold_letter for c in permuted) == sorted(c.gold_letter for c in cases)
