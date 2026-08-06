"""CPU tests for the P4 gold-token role classifier (``chain_token_roles``).

Offline/pure: no torch, no ``datasets``, no tokenizer download. The multi-hop side is exercised
through the pure span/offset functions plus a fake fast tokenizer; the GSM8K side is pinned against
the legacy ``gold_token_lens_gsm8k.computed_flags`` state machine it replaces, so the refactor is
provably behaviour-preserving before any GPU run.
"""

from __future__ import annotations

import pytest

from src.probes.attribution.chain_token_roles import (
    ROLE_FINAL_ANSWER,
    ROLE_HOP_ANSWER,
    ROLE_PROMPT,
    ROLE_SCAFFOLD,
    ROLE_SUB_QUESTION,
    gsm8k_token_roles,
    multihop_chain_spans,
    multihop_token_roles,
    roles_from_offsets,
)
from src.probes.attribution.multihop_prompts import format_multihop_solution

TWO_HOP = [
    {"question": "Who is the performer of Mary's Prayer?", "answer": "Danny Wilson"},
    {"question": "What record label is #1 signed to?", "answer": "Virgin Records"},
]

FOUR_HOP = [
    {"question": "Who directed Alpha?", "answer": "Ada"},
    {"question": "Where was #1 born?", "answer": "Beta City"},
    {"question": "Which country contains #2?", "answer": "Gamma"},
    {"question": "What is the capital of #3?", "answer": "Delta"},
]


def rendered(spans, text):
    """Spans as readable ``(role, covered_text, hop)`` triples."""
    return [(s.role, text[s.start:s.end], s.hop) for s in spans]


class FakeEncoding:
    def __init__(self, input_ids, offset_mapping):
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping


class FakeFastTokenizer:
    """Character-level stand-in: one token per character, exact offsets.

    Enough to drive :func:`multihop_token_roles` end-to-end on CPU (the real Llama fast tokenizer
    supplies the same ``input_ids``/``offset_mapping`` contract).
    """

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return FakeEncoding(ids, offsets if return_offsets_mapping else None)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeSpTokenizer:
    """Decodes ids via a fixed token table — mirrors how the GSM8K state machine reads tokens."""

    is_fast = True

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


# --------------------------------------------------------------------------------------
# multi-hop chain spans
# --------------------------------------------------------------------------------------


def test_chain_text_is_the_trained_target_verbatim():
    """The span builder must render byte-identically to the supervised target (single source of truth)."""
    text, _ = multihop_chain_spans(TWO_HOP)
    assert text == "\n" + format_multihop_solution(TWO_HOP)


def test_spans_tile_the_chain_exactly():
    """Contiguous, non-overlapping, and covering every character — no token can be unlabeled."""
    text, spans = multihop_chain_spans(FOUR_HOP)
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    for prev, nxt in zip(spans, spans[1:]):
        assert prev.end == nxt.start
    assert "".join(text[s.start:s.end] for s in spans) == text


def test_two_hop_roles_are_exact():
    text, spans = multihop_chain_spans(TWO_HOP)
    assert rendered(spans, text) == [
        (ROLE_SCAFFOLD, "\n", None),
        (ROLE_SCAFFOLD, "Step 1:", 1),
        (ROLE_SUB_QUESTION, " Who is the performer of Mary's Prayer?", 1),
        (ROLE_HOP_ANSWER, " Danny Wilson", 1),
        (ROLE_SCAFFOLD, ".", 1),
        (ROLE_SCAFFOLD, "\n", None),
        (ROLE_SCAFFOLD, "Step 2:", 2),
        # '#1' resolved to hop 1's answer by resolve_decomposition
        (ROLE_SUB_QUESTION, " What record label is Danny Wilson signed to?", 2),
        (ROLE_HOP_ANSWER, " Virgin Records", 2),
        (ROLE_SCAFFOLD, ".", 2),
        (ROLE_SCAFFOLD, "\n", None),
        (ROLE_SCAFFOLD, "The answer is:", None),
        (ROLE_FINAL_ANSWER, " Virgin Records", None),
    ]


def test_separating_space_belongs_to_the_following_role():
    """Boundaries fall on the space that *starts* the new role, so ' Danny' cannot straddle two roles."""
    text, spans = multihop_chain_spans(TWO_HOP)
    for s in spans:
        if s.role in (ROLE_SUB_QUESTION, ROLE_HOP_ANSWER, ROLE_FINAL_ANSWER):
            assert text[s.start] == " "


def test_answer_repeated_inside_its_own_sub_question_takes_the_line_final_span():
    """A hop answer that also occurs in its sub-question must be labeled at the line-final occurrence."""
    decomp = [{"question": "What did Virgin Records release?", "answer": "Virgin Records"}]
    text, spans = multihop_chain_spans(decomp)
    hop = [s for s in spans if s.role == ROLE_HOP_ANSWER]
    assert len(hop) == 1
    assert text[hop[0].start:hop[0].end] == " Virgin Records"
    # the occurrence inside the sub-question stays sub_question
    sub = [s for s in spans if s.role == ROLE_SUB_QUESTION][0]
    assert "Virgin Records" in text[sub.start:sub.end]
    assert hop[0].start > sub.start


def test_out_of_range_ref_literal_is_preserved():
    """'#9 Dream' is a title literal, not a back-reference — it must survive into the sub-question."""
    decomp = [
        {"question": "Who wrote #9 Dream?", "answer": "John Lennon"},
        {"question": "Where was #1 born?", "answer": "Liverpool"},
    ]
    text, spans = multihop_chain_spans(decomp)
    subs = [text[s.start:s.end] for s in spans if s.role == ROLE_SUB_QUESTION]
    assert subs[0] == " Who wrote #9 Dream?"
    assert subs[1] == " Where was John Lennon born?"


def test_hop_indices_are_one_based_and_cover_every_hop():
    text, spans = multihop_chain_spans(FOUR_HOP)
    hops = [(s.hop, text[s.start:s.end]) for s in spans if s.role == ROLE_HOP_ANSWER]
    assert hops == [(1, " Ada"), (2, " Beta City"), (3, " Gamma"), (4, " Delta")]


def test_final_answer_span_is_the_tail_after_the_marker():
    text, spans = multihop_chain_spans(FOUR_HOP)
    final = [s for s in spans if s.role == ROLE_FINAL_ANSWER]
    assert len(final) == 1
    assert final[0].end == len(text)
    assert text[final[0].start:final[0].end] == " Delta"


def test_empty_decomposition_raises():
    with pytest.raises(ValueError):
        multihop_chain_spans([])


# --------------------------------------------------------------------------------------
# offsets -> roles
# --------------------------------------------------------------------------------------


def test_roles_from_offsets_uses_the_first_character_of_each_token():
    text, spans = multihop_chain_spans(TWO_HOP)
    a = text.index(" Danny Wilson")
    # one token covering ' Danny' (starts on the space that opens the hop_answer span)
    roles = roles_from_offsets([(a, a + 6)], spans)
    assert roles[0]["role"] == ROLE_HOP_ANSWER
    assert roles[0]["hop"] == 1


def test_roles_from_offsets_boundary_straddling_token_takes_the_earlier_role():
    """A token starting in sub_question but running into hop_answer is scored as sub_question."""
    text, spans = multihop_chain_spans(TWO_HOP)
    a = text.index(" Danny Wilson")
    roles = roles_from_offsets([(a - 1, a + 6)], spans)
    assert roles[0]["role"] == ROLE_SUB_QUESTION


def test_roles_from_offsets_labels_every_character_token():
    text, spans = multihop_chain_spans(TWO_HOP)
    roles = roles_from_offsets([(i, i + 1) for i in range(len(text))], spans)
    assert len(roles) == len(text)
    assert {r["role"] for r in roles} == {
        ROLE_SCAFFOLD, ROLE_SUB_QUESTION, ROLE_HOP_ANSWER, ROLE_FINAL_ANSWER}


def test_roles_from_offsets_rejects_out_of_range_offsets():
    _, spans = multihop_chain_spans(TWO_HOP)
    with pytest.raises(ValueError):
        roles_from_offsets([(10_000, 10_001)], spans)


# --------------------------------------------------------------------------------------
# driver-facing multi-hop wrapper
# --------------------------------------------------------------------------------------


def test_multihop_token_roles_labels_prompt_chain_and_eos():
    tok = FakeFastTokenizer()
    gold = {"answer": "Virgin Records", "decomposition": TWO_HOP}
    text, _ = multihop_chain_spans(TWO_HOP)
    prompt_ids = [1, 2, 3]
    ids = prompt_ids + [ord(c) for c in text] + [2]          # + eos
    roles = multihop_token_roles(tok, ids, len(prompt_ids), gold)

    assert len(roles) == len(ids)
    assert [r["role"] for r in roles[:3]] == [ROLE_PROMPT] * 3
    assert roles[-1]["role"] == ROLE_SCAFFOLD                # trailing eos is scaffold
    assert roles[3]["role"] == ROLE_SCAFFOLD                 # leading '\n'
    hop_tokens = [r for r in roles if r["role"] == ROLE_HOP_ANSWER]
    assert {r["hop"] for r in hop_tokens} == {1, 2}


def test_multihop_token_roles_rejects_a_chain_that_does_not_match_the_ids():
    """Guard against a silent prompt/chain join drift between the driver and training."""
    tok = FakeFastTokenizer()
    gold = {"answer": "Virgin Records", "decomposition": TWO_HOP}
    ids = [1, 2, 3] + [ord("x")] * 5
    with pytest.raises(ValueError):
        multihop_token_roles(tok, ids, 3, gold)


# --------------------------------------------------------------------------------------
# GSM8K side — parity with the state machine it replaces
# --------------------------------------------------------------------------------------

# '= 48' tokenizes as '=', ' ', '4', '8'; the restatement digits are copies, not computed results.
GSM8K_PIECES = ["<s>", "Q", ":", "\n", "2", "4", "+", "2", "4", "=", " ", "4", "8",
                "\n", "The", " answer", " is", ":", " ", "4", "8"]


def test_gsm8k_roles_split_computed_from_copied_digits():
    tok = FakeSpTokenizer(GSM8K_PIECES)
    ids = list(range(len(GSM8K_PIECES)))
    roles = [r["role"] for r in gsm8k_token_roles(tok, ids, 4, None)]

    assert roles[11] == "computed" and roles[12] == "computed"        # the '4','8' after '='
    assert roles[19] == "copied_digit" and roles[20] == "copied_digit"  # restatement digits
    assert roles[7] == "copied_digit" and roles[8] == "copied_digit"    # problem digits (pre-'=')
    assert roles[16] == "other"                                        # ' is'
    assert roles[:4] == [ROLE_PROMPT] * 4


def test_gsm8k_result_span_closes_on_a_non_digit():
    """The span opened by '=' must close on the first non-space, non-digit token."""
    pieces = ["=", " ", "4", "8", " apples", "1", "2"]
    tok = FakeSpTokenizer(pieces)
    roles = [r["role"] for r in gsm8k_token_roles(tok, list(range(len(pieces))), 0, None)]
    assert roles == ["other", "other", "computed", "computed", "other", "copied_digit", "copied_digit"]


def test_gsm8k_whitespace_keeps_the_result_span_open():
    """Pins the *committed* semantics: any whitespace token — newline included — stays in the span.

    ``temporal_gate.in_result_span`` deliberately refines this (a newline closes the span) for
    *gating*; the E1b classifier must not, or the published GSM8K numbers would move.
    """
    pieces = ["=", "4", "\n", "5", "=", "6"]
    tok = FakeSpTokenizer(pieces)
    roles = [r["role"] for r in gsm8k_token_roles(tok, list(range(len(pieces))), 0, None)]
    assert roles == ["other", "computed", "other", "computed", "other", "computed"]


def test_gsm8k_roles_have_no_hop_index():
    tok = FakeSpTokenizer(GSM8K_PIECES)
    roles = gsm8k_token_roles(tok, list(range(len(GSM8K_PIECES))), 0, None)
    assert all(r["hop"] is None for r in roles)
