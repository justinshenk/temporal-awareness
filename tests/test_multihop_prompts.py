"""CPU/offline contract for the multi-hop (MuSiQue) procedure seams.

Mirrors ``test_*`` conventions: pure string/logic functions, no network, no torch model. These lock
the four arithmetic-specific seams ported from ``gsm8k_prompts`` for the generality test
(``tasks/multihop_generality.md``): prompt template, gold-chain linearizer, answer scoring, and the
answer-span gate used by ``gated_lockstep_generate``.
"""

from __future__ import annotations

from src.probes.attribution.multihop_prompts import (
    answer_match,
    answer_span_gate,
    extract_pred_answer,
    format_multihop_solution,
    multihop_prompt,
    normalize_answer,
    resolve_decomposition,
)

# A real MuSiQue 2-hop shape (fields trimmed to what the seams consume).
DECOMP = [
    {"question": "The Collegian >> owned by", "answer": "Houston Baptist University"},
    {"question": "When was #1 founded?", "answer": "1960"},
]
PASSAGES = [
    ("The Collegian (Houston Baptist University)", "The Collegian is the student paper of HBU."),
    ("Houston", "Houston Baptist University was founded in 1960."),
]
QUESTION = "When was the institute that owned The Collegian founded?"


# --- prompt template -----------------------------------------------------------------------------

def test_prompt_has_fixed_cot_continuation_point():
    p = multihop_prompt(QUESTION, PASSAGES)
    # ends exactly at the CoT continuation point; no worked solution leaks past it. (Open-book: the
    # gold passages legitimately contain the answer text — the model must COMPOSE to surface it; the
    # leakage that would invalidate the run is the solution/answer marker, not the supporting facts.)
    assert p.rstrip().endswith("Let's think step by step.")
    assert "The answer is:" not in p
    assert QUESTION in p


def test_prompt_renders_all_passages():
    p = multihop_prompt(QUESTION, PASSAGES)
    for title, text in PASSAGES:
        assert title in p and text in p


# --- gold-chain linearizer -----------------------------------------------------------------------

def test_resolve_decomposition_substitutes_hop_references():
    steps = resolve_decomposition(DECOMP)
    assert steps == [
        ("The Collegian >> owned by", "Houston Baptist University"),
        ("When was Houston Baptist University founded?", "1960"),   # #1 -> hop-1 answer
    ]


def test_resolve_decomposition_preserves_out_of_range_ref_literals():
    # "#9 Dream" is a song title, not a back-reference; out-of-range #k must be kept verbatim, not crash.
    steps = resolve_decomposition([
        {"question": "#9 Dream >> performer", "answer": "John Lennon"},
        {"question": "#1 >> father", "answer": "Alfred Lennon"},
    ])
    assert steps[0][0] == "#9 Dream >> performer"          # out-of-range -> literal preserved
    assert steps[1][0] == "John Lennon >> father"          # valid back-ref still substituted


def test_format_solution_ends_in_answer_marker():
    sol = format_multihop_solution(DECOMP)
    assert sol.rstrip().endswith("The answer is: 1960")
    assert "Houston Baptist University" in sol                      # intermediate hop is shown


# --- answer scoring ------------------------------------------------------------------------------

def test_extract_pred_answer_parses_tail_and_handles_missing_marker():
    assert extract_pred_answer("Step 1: ...\nThe answer is: 1960").strip() == "1960"
    assert extract_pred_answer("no marker here") is None
    # takes the LAST marker occurrence (model may restate)
    assert extract_pred_answer("The answer is: foo\nThe answer is: 1960").strip() == "1960"


def test_normalize_answer_squad_style():
    assert normalize_answer("The  Houston, Baptist University!") == "houston baptist university"
    assert normalize_answer("1960.") == "1960"


def test_answer_match_uses_aliases_and_normalization():
    assert answer_match("1960", "1960")
    assert answer_match(" the 1960 ", "1960")                       # normalization-invariant
    assert answer_match("USA", "United States", aliases=["US", "USA"])
    assert not answer_match("1961", "1960")
    assert not answer_match(None, "1960")                           # unparsed -> no match


# --- answer-span gate (analogue of GSM8K result/planning partition) ------------------------------

def test_answer_span_gate_fires_only_after_the_marker():
    gate = answer_span_gate()      # default marker "The answer is:"
    # decoded = list of token strings emitted so far (current token not yet known -> no look-ahead)
    pre = ["Step", " 1", ":", " owned", " by"]
    assert not gate(pre, len(pre))                                  # still in the reasoning span
    post = ["...", "The", " answer", " is", ":", " 19"]             # marker now present in prefix
    assert gate(post, len(post))                                   # inside the answer span


def test_answer_gate_and_reasoning_gate_are_exact_complements():
    answer = answer_span_gate()
    reasoning = answer_span_gate(complement=True)
    streams = [
        [], ["Step", " 1"], ["The", " answer", " is", ":"], ["The", " answer", " is", ":", " 1960"],
    ]
    for s in streams:
        assert answer(s, len(s)) != reasoning(s, len(s))            # XOR == all-True (partition)
