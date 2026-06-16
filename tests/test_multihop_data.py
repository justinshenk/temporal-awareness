"""CPU/offline contract for the MuSiQue data seams (no network, no real model).

Locks the open-book bridging logic in ``multihop_data``: supporting-passage selection, the
driver-facing instruction string, the supervised target (leading-newline join + answer marker), and
the response-only encode boundary (``plen`` must point exactly at the first target token so
``collate_left_padded`` masks the prompt and supervises only the chain).
"""

from __future__ import annotations

from src.probes.attribution.multihop_data import (
    encode_multihop,
    gold_target,
    problem_instruction,
    supporting_passages,
)
from src.probes.attribution.multihop_prompts import (
    ANSWER_MARKER,
    multihop_prompt_from_instruction,
)

# A real MuSiQue 2-hop shape (with a distractor paragraph that must be dropped).
ITEM = {
    "question": "When was the institute that owned The Collegian founded?",
    "paragraphs": [
        {"idx": 0, "title": "The Collegian", "paragraph_text": "Owned by HBU.", "is_supporting": True},
        {"idx": 1, "title": "Distractor", "paragraph_text": "Unrelated text.", "is_supporting": False},
        {"idx": 2, "title": "HBU", "paragraph_text": "Founded in 1960.", "is_supporting": True},
    ],
    "question_decomposition": [
        {"question": "The Collegian >> owned by", "answer": "Houston Baptist University"},
        {"question": "When was #1 founded?", "answer": "1960"},
    ],
    "answer": "1960",
    "answer_aliases": ["1960 AD"],
    "answerable": True,
}


def test_supporting_passages_drops_distractors():
    sp = supporting_passages(ITEM)
    assert sp == [("The Collegian", "Owned by HBU."), ("HBU", "Founded in 1960.")]


def test_problem_instruction_inlines_passages_and_question_without_leaking_solution():
    instr = problem_instruction(ITEM)
    assert ITEM["question"] in instr
    assert "Owned by HBU." in instr and "Founded in 1960." in instr
    assert "Distractor" not in instr
    assert ANSWER_MARKER not in instr            # no answer/solution leakage in the prompt body


def test_gold_target_joins_with_newline_and_ends_in_answer_marker():
    tgt = gold_target(ITEM)
    assert tgt.startswith("\n")                  # join separator after "Let's think step by step."
    assert tgt.rstrip().endswith(f"{ANSWER_MARKER} 1960")
    assert "Houston Baptist University" in tgt   # intermediate hop shown


class _FakeTok:
    """Whitespace tokenizer with a BOS id, mimicking the add_special_tokens contract."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [self.bos_token_id] if add_special_tokens else []
        ids += [10 + (len(w) % 50) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def test_encode_multihop_plen_marks_prompt_boundary():
    tok = _FakeTok()
    prompt = multihop_prompt_from_instruction(problem_instruction(ITEM))
    expected_plen = len(tok(prompt, add_special_tokens=True).input_ids)
    enc = encode_multihop(tok, [ITEM], max_len=10_000)
    assert len(enc) == 1
    row = enc[0]
    # plen points past the BOS+prompt tokens; everything before it is masked, the chain is supervised.
    assert row["plen"] == expected_plen
    assert len(row["ids"]) > row["plen"]         # at least one supervised target token survives
    # the target ids (post-plen) end with the appended eos
    assert row["ids"][-1] == tok.eos_token_id


def test_encode_multihop_drops_when_truncation_eats_response():
    tok = _FakeTok()
    prompt = multihop_prompt_from_instruction(problem_instruction(ITEM))
    plen = len(tok(prompt, add_special_tokens=True).input_ids)
    # max_len at exactly the prompt length leaves no room for any response token -> dropped.
    assert encode_multihop(tok, [ITEM], max_len=plen) == []
