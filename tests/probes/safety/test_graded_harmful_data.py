"""Unit tests for the graded-harmful prompt construction (pure parts, no network)."""

from __future__ import annotations

import pytest

from src.probes.lora_icl.ddxplus_cases import MCQCase
from src.probes.safety.graded_harmful_data import (
    assert_indist_wears_medical_frame,
    clinical_block,
    far_benign_messages,
    indist_benign_messages,
    indist_interleaved_messages,
    indist_messages,
    near_benign_messages,
    near_messages,
    near_prompts,
)

CASE_TEXT = (
    "Patient: 45-year-old Male\n"
    "Chief complaint: chest pain\n"
    "Symptoms:\n  - sharp pain\n"
    "History:\n  - smoker\n"
    "\nMost likely diagnosis:\nA) MI\nB) GERD\n\nAnswer:"
)


def test_near_prompts_are_drug_template_grid():
    p = near_prompts(10)
    assert len(p) == 10
    assert all(isinstance(s, str) and len(s) > 0 for s in p)
    # at least one references a drug slot being filled (no leftover "{drug}")
    assert all("{drug}" not in s for s in p)


def test_near_prompts_raises_when_exhausted():
    with pytest.raises(ValueError, match="near bucket"):
        near_prompts(10_000)


def test_clinical_block_strips_the_mcq_tail():
    cb = clinical_block(CASE_TEXT)
    assert "Patient: 45-year-old Male" in cb
    assert "Most likely diagnosis" not in cb
    assert "Answer:" not in cb


def test_indist_messages_wear_medical_frame_and_clinical_block():
    cases = [MCQCase(source_index=i, prompt_text=CASE_TEXT, gold_letter="A") for i in range(4)]
    msgs = indist_messages(cases, 4)
    assert len(msgs) == 4
    for m in msgs:
        assert_indist_wears_medical_frame(m)            # carries "You are a doctor." + "Patient:"
        # the harmful ask is appended after the clinical block
        assert "Most likely diagnosis" not in m[0]["content"]


def test_indist_messages_raises_without_enough_cases():
    with pytest.raises(ValueError, match="indist"):
        indist_messages([MCQCase(prompt_text=CASE_TEXT)], 5)


def test_near_messages_are_plain_user_turns():
    msgs = near_messages(3)
    assert all(len(m) == 1 and m[0]["role"] == "user" for m in msgs)
    assert all("You are a doctor" not in m[0]["content"] for m in msgs)  # far/near carry no frame


def test_indist_benign_twin_shares_clinical_block_but_benign_ask():
    cases = [MCQCase(source_index=i, prompt_text=CASE_TEXT, gold_letter="A") for i in range(3)]
    harm = indist_messages(cases, 3)
    ben = indist_benign_messages(cases, 3)
    for h, b in zip(harm, ben):
        # same clinical block (the part before the ask), different intent
        assert_indist_wears_medical_frame(b)                       # benign twin still wears the fit frame
        assert clinical_block(CASE_TEXT) in h[0]["content"] and clinical_block(CASE_TEXT) in b[0]["content"]
        assert "diagnosis" in b[0]["content"].lower()
        assert h[0]["content"] != b[0]["content"]                  # the ask differs (harm vs benign)


def test_near_benign_twin_is_benign_and_plain():
    msgs = near_benign_messages(4)
    assert all(len(m) == 1 and m[0]["role"] == "user" for m in msgs)
    joined = " ".join(m[0]["content"].lower() for m in msgs)
    assert "fatal" not in joined and "kill" not in joined and "overdose" not in joined


def test_indist_interleaved_keeps_mcq_format_with_harmful_stem():
    cases = [MCQCase(source_index=i, prompt_text=CASE_TEXT, gold_letter="A") for i in range(3)]
    msgs = indist_interleaved_messages(cases, 3)
    for m in msgs:
        content = m[0]["content"]
        assert_indist_wears_medical_frame(m)                   # doctor frame + clinical block intact
        assert content.rstrip().endswith("Answer:")            # last token stays the in-distribution MCQ site
        assert "A) MI" in content and "B) GERD" in content     # real diagnosis options preserved
        assert "Most likely diagnosis" not in content          # benign stem replaced
        assert "kill" in content.lower() or "die" in content.lower() or "lethal" in content.lower()


def test_far_benign_uses_supplied_pool_disjoint_by_skip():
    pool = [f"benign instruction {i}" for i in range(10)]
    msgs = far_benign_messages(3, pool, skip=5)
    assert [m[0]["content"] for m in msgs] == ["benign instruction 5", "benign instruction 6", "benign instruction 7"]
