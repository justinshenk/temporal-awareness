"""Tests for DDXPlus case formatting, and for the vignette/question split E1 depends on.

E1 moves the case vignette away from the question while leaving the question byte-identical. That
is only sound if the two really do detach — so these tests pin the composition
(``vignette + "\\n" + question == format_case_mcq``) and the byte-level stability of the combined
output, which several existing drivers already depend on.

Everything runs on a synthetic evidence database, so the suite needs neither the network nor the
88 MB DDXPlus test split.
"""

import pytest

from src.probes.context_fatigue.ddxplus_cases import (
    decode_evidence,
    format_case_mcq,
    format_case_question,
    format_case_vignette,
)

EVIDENCE_DB = {
    "E_cough": {"question": "Do you have a cough?", "is_antecedent": False,
                "data_type": "B", "value_meanings": {}},
    "E_pain": {"question": "How intense is the pain?", "is_antecedent": False,
               "data_type": "C", "value_meanings": {}},
    "E_site": {"question": "Do you feel pain somewhere?", "is_antecedent": False,
               "data_type": "M", "value_meanings": {"V_1": "chest", "V_2": "back"}},
    "E_obese": {"question": "Are you significantly overweight?", "is_antecedent": True,
                "data_type": "B", "value_meanings": {}},
}

EVIDENCES = "['E_cough', 'E_pain_@_7', 'E_site_@_V_1', 'E_site_@_V_2', 'E_obese']"
OPTIONS = ["Bronchitis", "GERD", "Pericarditis", "Unstable angina", "Pneumonia"]


def _case_args():
    return dict(age=49, sex="F", initial_ev="E_cough", evidence_str=EVIDENCES,
                evidence_db=EVIDENCE_DB, options=OPTIONS, n_options=5)


# ── the split E1 relies on ──────────────────────────────────────────────

def test_vignette_and_question_compose_to_the_combined_case():
    args = _case_args()
    combined = format_case_mcq(**args)
    vignette = format_case_vignette(args["age"], args["sex"], args["initial_ev"],
                                    args["evidence_str"], args["evidence_db"])
    question = format_case_question(args["options"], args["n_options"])

    assert combined == vignette + "\n" + question


def test_question_is_independent_of_the_vignette():
    """E1 moves the vignette; the question must not move with it."""
    first = format_case_question(OPTIONS, 5)
    second = format_case_question(OPTIONS, 5)
    assert first == second
    assert "Patient:" not in first
    assert "Chief complaint" not in first


def test_vignette_carries_no_options():
    """If the options leaked into the vignette, `back_k` would move the answer set too."""
    vignette = format_case_vignette(49, "F", "E_cough", EVIDENCES, EVIDENCE_DB)
    for opt in OPTIONS:
        assert opt not in vignette
    assert "Answer:" not in vignette


def test_question_lists_exactly_n_options_with_labels():
    question = format_case_question(OPTIONS, n_options=5)
    for label, opt in zip("ABCDE", OPTIONS):
        assert f"{label}) {opt}" in question
    assert question.rstrip().endswith("Answer:")


def test_n_options_truncates():
    question = format_case_question(OPTIONS, n_options=3)
    assert "C) Pericarditis" in question
    assert "D)" not in question


# ── byte-stability of the combined output ───────────────────────────────

def test_combined_case_is_byte_stable():
    """Several committed drivers format cases this way; the exact bytes are the interface."""
    expected = (
        "Patient: 49-year-old Female\n"
        "Chief complaint: a cough\n"
        "Symptoms:\n"
        "  - Yes — Has a cough\n"
        "  - How intense is the pain: 7\n"
        "  - Do you feel pain somewhere: chest, back\n"
        "History:\n"
        "  - Yes — Is significantly overweight\n"
        "\n"
        "Most likely diagnosis:\n"
        "A) Bronchitis\n"
        "B) GERD\n"
        "C) Pericarditis\n"
        "D) Unstable angina\n"
        "E) Pneumonia\n"
        "\n"
        "Answer:"
    )
    assert format_case_mcq(**_case_args()) == expected


def test_male_sex_renders_full_word():
    args = _case_args() | {"sex": "M"}
    assert format_case_mcq(**args).startswith("Patient: 49-year-old Male")


# ── evidence decoding ───────────────────────────────────────────────────

def test_decode_evidence_separates_symptoms_from_antecedents():
    symptoms, antecedents = decode_evidence(EVIDENCES, EVIDENCE_DB)
    assert any("cough" in s for s in symptoms)
    assert any("overweight" in a for a in antecedents)
    assert not any("overweight" in s for s in symptoms)


def test_decode_evidence_ignores_unknown_codes():
    symptoms, antecedents = decode_evidence("['E_cough', 'E_not_in_db']", EVIDENCE_DB)
    assert len(symptoms) == 1
    assert antecedents == []


def test_decode_evidence_joins_multi_values():
    symptoms, _ = decode_evidence("['E_site_@_V_1', 'E_site_@_V_2']", EVIDENCE_DB)
    assert symptoms == ["Do you feel pain somewhere: chest, back"]


def test_vignette_omits_empty_sections():
    """A case with no antecedents must not emit a bare 'History:' header."""
    vignette = format_case_vignette(30, "M", "E_cough", "['E_cough']", EVIDENCE_DB)
    assert "History:" not in vignette
    assert "Symptoms:" in vignette


@pytest.mark.parametrize("bad", ["not-a-list", "[unclosed"])
def test_decode_evidence_rejects_malformed_input(bad):
    with pytest.raises((ValueError, SyntaxError)):
        decode_evidence(bad, EVIDENCE_DB)


# ── the E1 referent ─────────────────────────────────────────────────────

REFERENT = "For the patient described earlier"


def test_referent_is_prepended_to_the_question():
    """In `back_k` the question arrives many turns after the vignette; the referent makes it point
    at the patient explicitly, so the arm measures retrieval at distance rather than whether the
    model noticed a patient was mentioned at all."""
    question = format_case_question(OPTIONS, 5, referent=REFERENT)
    assert question.startswith(f"\n{REFERENT}, most likely diagnosis:")
    assert question.rstrip().endswith("Answer:")


def test_referent_is_identical_across_arms():
    """It goes into *every* arm, `local` included, so question bytes stay equal across the ladder."""
    assert format_case_question(OPTIONS, 5, referent=REFERENT) == \
        format_case_question(OPTIONS, 5, referent=REFERENT)


def test_referent_changes_the_question_it_is_added_to():
    assert format_case_question(OPTIONS, 5, referent=REFERENT) != format_case_question(OPTIONS, 5)


def test_referent_defaults_off_so_the_single_turn_format_is_unchanged():
    """The committed drivers' bytes are the interface; the referent must be opt-in."""
    assert format_case_question(OPTIONS, 5) == format_case_question(OPTIONS, 5, referent=None)
    assert "described earlier" not in format_case_mcq(**_case_args())


def test_referent_never_enters_the_vignette():
    vignette = format_case_vignette(49, "F", "E_cough", EVIDENCES, EVIDENCE_DB)
    assert "described earlier" not in vignette


def test_answer_cue_can_be_suppressed_for_format_experiments():
    """The trailing 'Answer:' cue writes the first line of the reply for the model.

    Harmless when the reply is a bare letter, fatal when the experiment is about whether a
    system-prompt-specified format survives: the probe would be competing with the system prompt
    for control of the output.
    """
    from src.probes.context_fatigue.ddxplus_cases import format_case_question
    opts = ["Bronchitis", "Pneumonia", "URTI", "Asthma", "GERD"]
    assert format_case_question(opts).rstrip().endswith("Answer:")
    assert not format_case_question(opts, answer_cue=False).rstrip().endswith("Answer:")


def test_suppressing_the_cue_changes_nothing_else():
    from src.probes.context_fatigue.ddxplus_cases import format_case_question
    opts = ["Bronchitis", "Pneumonia", "URTI", "Asthma", "GERD"]
    with_cue = format_case_question(opts)
    without = format_case_question(opts, answer_cue=False)
    assert with_cue.replace("\n\nAnswer:", "").rstrip() == without.rstrip()
    for o in opts:
        assert o in without
