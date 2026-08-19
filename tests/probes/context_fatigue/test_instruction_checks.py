"""Tests for checkable instruction-adherence specs and arm builders (GPU-free)."""

import pytest

from src.probes.context_fatigue.instruction_checks import (
    INSTRUCTIONS,
    InstructionSpec,
    fill_bin_stats,
    history_assistant_for,
    pearson,
    system_prompt_for,
    user_content_for,
)

BASE_SYSTEM = "You are a doctor."

PREFIX = INSTRUCTIONS["prefix_marker"]
SUFFIX = INSTRUCTIONS["suffix_ok"]
FORBID = INSTRUCTIONS["forbid_diagnosis"]


# ── check_obeyed ─────────────────────────────────────────────────────────

def test_prefix_obeyed_when_response_starts_with_target():
    assert PREFIX.check_obeyed("◆ B")
    assert PREFIX.check_obeyed("  ◆  the answer is B")  # leading-space robust


def test_prefix_violated_when_prefix_absent():
    assert not PREFIX.check_obeyed("B")
    assert not PREFIX.check_obeyed("The answer is B")


def test_suffix_obeyed_when_last_line_has_tag():
    assert SUFFIX.check_obeyed("B\n⟦OK⟧")
    assert SUFFIX.check_obeyed("◆ B\n⟦OK⟧")


def test_suffix_violated_when_tag_missing_or_not_last():
    assert not SUFFIX.check_obeyed("B")
    assert not SUFFIX.check_obeyed("⟦OK⟧\nB")  # tag must be on the last non-empty line


def test_forbid_violated_when_word_present_as_word():
    assert not FORBID.check_obeyed("The diagnosis is flu")
    assert FORBID.check_obeyed("The answer is flu")
    # substring inside another word does not count as a violation
    assert FORBID.check_obeyed("diagnostics were ordered")


# ── make_compliant (used to build the 'forced' history) ───────────────────

def test_make_compliant_is_idempotent_on_already_compliant():
    assert PREFIX.make_compliant("◆ B") == "◆ B"
    assert SUFFIX.make_compliant("B\n⟦OK⟧") == "B\n⟦OK⟧"


def test_make_compliant_repairs_a_violation():
    out = PREFIX.make_compliant("B")
    assert PREFIX.check_obeyed(out) and "B" in out
    out = SUFFIX.make_compliant("B")
    assert SUFFIX.check_obeyed(out) and out.splitlines()[0] == "B"
    out = FORBID.make_compliant("The diagnosis is flu")
    assert FORBID.check_obeyed(out)


# ── arm builders ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("arm", ["baseline", "forced"])
def test_system_prompt_carries_instruction_except_refresh(arm):
    sp = system_prompt_for(PREFIX, arm, BASE_SYSTEM)
    assert BASE_SYSTEM in sp and PREFIX.system_text in sp


def test_refresh_moves_instruction_out_of_system_into_user():
    sp = system_prompt_for(PREFIX, "refresh", BASE_SYSTEM)
    assert sp == BASE_SYSTEM and PREFIX.system_text not in sp
    uc = user_content_for(PREFIX, "refresh", "CASE_TEXT")
    assert "CASE_TEXT" in uc and PREFIX.system_text in uc


def test_nonrefresh_user_content_is_just_the_case():
    assert user_content_for(PREFIX, "baseline", "CASE_TEXT") == "CASE_TEXT"


def test_forced_history_is_always_compliant_even_for_violating_response():
    # The model dropped the prefix; forced arm must still write a compliant turn.
    h = history_assistant_for(PREFIX, "forced", "B")
    assert PREFIX.check_obeyed(h)


@pytest.mark.parametrize("arm", ["baseline", "refresh"])
def test_nonforced_history_is_the_raw_response(arm):
    assert history_assistant_for(PREFIX, arm, "B") == "B"


# ── stats helpers ─────────────────────────────────────────────────────────

def test_pearson_monotone_signs():
    assert pearson([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert pearson([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0  # degenerate -> 0, no crash


def test_violation_rises_with_fill_gives_positive_corr():
    # synthetic: violation flips on once context is past the midpoint
    turns = [{"context_fill": f, "violation": int(f > 0.5)}
             for f in [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]]
    fills = [t["context_fill"] for t in turns]
    viol = [t["violation"] for t in turns]
    assert pearson(fills, viol) > 0


def test_fill_bin_stats_counts_and_means():
    turns = [{"context_fill": 0.1, "violation": 0},
             {"context_fill": 0.15, "violation": 1},
             {"context_fill": 0.85, "violation": 1}]
    bins = [(0.0, 0.2), (0.8, 1.0)]
    stats = fill_bin_stats(turns, "violation", bins)
    assert stats["0%-20%"]["count"] == 2
    assert stats["0%-20%"]["mean"] == pytest.approx(0.5)
    assert stats["80%-100%"]["count"] == 1
    assert stats["80%-100%"]["mean"] == pytest.approx(1.0)


def test_instruction_registry_specs_are_well_formed():
    for name, spec in INSTRUCTIONS.items():
        assert isinstance(spec, InstructionSpec)
        assert spec.name == name
        assert spec.system_text and spec.target


# ── bundled canary scoring (E5 system-mass clamp) ───────────────────────

def test_check_all_scores_every_spec_independently():
    from src.probes.context_fatigue.instruction_checks import INSTRUCTIONS, check_all
    specs = [INSTRUCTIONS["prefix_marker"], INSTRUCTIONS["suffix_ok"],
             INSTRUCTIONS["forbid_diagnosis"]]
    out = check_all("◆ The answer is B.\n⟦OK⟧", specs)
    assert out == {"prefix_marker": True, "suffix_ok": True, "forbid_diagnosis": True}


def test_check_all_isolates_a_single_violation():
    from src.probes.context_fatigue.instruction_checks import INSTRUCTIONS, check_all
    specs = [INSTRUCTIONS["prefix_marker"], INSTRUCTIONS["suffix_ok"],
             INSTRUCTIONS["forbid_diagnosis"]]
    out = check_all("◆ The diagnosis is B.\n⟦OK⟧", specs)
    assert out["forbid_diagnosis"] is False
    assert out["prefix_marker"] is True and out["suffix_ok"] is True


def test_check_all_treats_an_empty_response_as_violating_the_positive_canaries():
    """A clamp that silences the model must not read as compliance."""
    from src.probes.context_fatigue.instruction_checks import INSTRUCTIONS, check_all
    out = check_all("", [INSTRUCTIONS["prefix_marker"], INSTRUCTIONS["suffix_ok"]])
    assert out == {"prefix_marker": False, "suffix_ok": False}


def test_bundled_system_text_contains_every_canary_instruction():
    from src.probes.context_fatigue.instruction_checks import INSTRUCTIONS, bundled_system_text
    text = bundled_system_text("You are a doctor.", list(INSTRUCTIONS.values()))
    assert text.startswith("You are a doctor.")
    for spec in INSTRUCTIONS.values():
        assert spec.system_text in text
