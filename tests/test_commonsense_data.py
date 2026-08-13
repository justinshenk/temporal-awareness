"""Tests for the LLM-Adapters commonsense data glue (CPU, no network, torch-free module).

Fixtures mirror the verified on-disk format of commonsense_170k / boolq / piqa / ARC-Challenge:
``{"instruction", "input"(empty), "output": "the correct answer is X", "answer": "X"}``. The
prompt template is literally ``"%s\\n" % instruction`` (pyreft commonsense task_config — no alpaca
wrapper) and answers are read as the word following the trigger ``"the correct answer is"``.
"""

from __future__ import annotations

import json

import pytest

from src.probes.attribution.commonsense_data import (
    commonsense_problems,
    extract_answer,
    format_prompt,
    format_target,
    load_commonsense_json,
    score_predictions,
    subset_examples,
)

BOOLQ_ITEM = {
    "instruction": ("Please answer the following question with true or false, question: "
                    "does ethanol take more energy make that produces?\n\nAnswer format: true/false"),
    "input": "",
    "output": "the correct answer is false",
    "answer": "false",
}
PIQA_ITEM = {
    "instruction": ("Please choose the correct solution to the question: Make outdoor pillow.\n\n"
                    "Solution1: Blow into tin can and tie with rubber band.\n\n"
                    "Solution2: Blow into trash bag and tie with rubber band.\n\n"
                    "Answer format: solution1/solution2"),
    "input": "",
    "output": "the correct answer is solution2",
    "answer": "solution2",
}
ARC_ITEM = {
    "instruction": ("Please choose the correct answer to the question: An astronomer observes that "
                    "a planet rotates faster after a meteorite impact. Which is the most likely "
                    "effect of this increase in rotation?\n\nAnswer1: Planetary density will "
                    "decrease. Answer2: Planetary years will become longer. Answer3: Planetary days "
                    "will become shorter. Answer4: Planetary gravity will become stronger.\n\n"
                    "Answer format: answer1/answer2/answer3/answer4"),
    "input": "",
    "output": "the correct answer is answer3",
    "answer": "answer3",
}


def test_load_and_format_round_trip(tmp_path):
    path = tmp_path / "boolq_test.json"
    path.write_text(json.dumps([BOOLQ_ITEM, PIQA_ITEM, ARC_ITEM]))
    data = load_commonsense_json(path)
    assert len(data) == 3
    assert format_prompt(data[0]) == BOOLQ_ITEM["instruction"] + "\n"
    assert format_target(data[0]) == "the correct answer is false"
    assert data[2]["answer"] == "answer3"


def test_extract_answer_happy_paths():
    assert extract_answer("the correct answer is false") == "false"
    assert extract_answer("the correct answer is solution2.") == "solution2"
    assert extract_answer("The correct answer is Answer3") == "answer3"          # case-insensitive
    assert extract_answer("blah blah.\nthe correct answer is true, because...") == "true"


def test_extract_answer_garbage_is_empty():
    assert extract_answer("I think the answer might be yes?") == ""
    assert extract_answer("") == ""
    assert extract_answer("the correct answer is") == ""                          # trigger, no word


def test_score_predictions():
    preds = ["true", "solution2", "answer3", "answer1", ""]
    golds = ["true", "solution2", "answer3", "answer3", "false"]
    assert score_predictions(preds, golds) == 3 / 5


def _fixture_dir(tmp_path):
    """A miniature ``data/commonsense/`` — one eval split and the train file."""
    (tmp_path / "ARC-Challenge_test.json").write_text(json.dumps([ARC_ITEM, BOOLQ_ITEM, PIQA_ITEM]))
    (tmp_path / "commonsense_170k.json").write_text(json.dumps([BOOLQ_ITEM, PIQA_ITEM]))
    return tmp_path


def test_commonsense_problems_returns_instruction_answer_pairs(tmp_path):
    """The driver contract: ``(question, gold)`` where question is what ``prompt`` consumes."""
    problems = commonsense_problems("ARC-Challenge", 3, data_dir=_fixture_dir(tmp_path))
    assert problems == [(ARC_ITEM["instruction"], "answer3"),
                        (BOOLQ_ITEM["instruction"], "false"),
                        (PIQA_ITEM["instruction"], "solution2")]


def test_commonsense_problems_honours_n_and_skip(tmp_path):
    d = _fixture_dir(tmp_path)
    assert commonsense_problems("ARC-Challenge", 1, data_dir=d) == [(ARC_ITEM["instruction"], "answer3")]
    assert commonsense_problems("ARC-Challenge", 1, skip=2, data_dir=d) == [(PIQA_ITEM["instruction"], "solution2")]
    assert len(commonsense_problems("ARC-Challenge", 99, data_dir=d)) == 3      # n > len → everything
    assert commonsense_problems("ARC-Challenge", 0, data_dir=d) == []


def test_commonsense_problems_train_split_reads_the_170k_file(tmp_path):
    """S2c fits the ridge map on train residuals, so ``train`` must resolve, not just eval splits."""
    problems = commonsense_problems("train", 2, data_dir=_fixture_dir(tmp_path))
    assert problems == [(BOOLQ_ITEM["instruction"], "false"), (PIQA_ITEM["instruction"], "solution2")]


def test_commonsense_problems_seed_is_accepted_and_ignored(tmp_path):
    """File order is already deterministic; ``seed`` exists for registry signature parity only.

    It must not reshuffle — the contrast cache stores *indices* into this scan and is rehydrated
    later by ``load_contrast``, so a seed-dependent order would silently misindex the whole set.
    """
    d = _fixture_dir(tmp_path)
    assert commonsense_problems("ARC-Challenge", 3, seed=1, data_dir=d) == \
           commonsense_problems("ARC-Challenge", 3, seed=999, data_dir=d)


def test_commonsense_problems_unknown_split_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no commonsense split"):
        commonsense_problems("nope", 1, data_dir=_fixture_dir(tmp_path))


def test_subset_examples_deterministic_and_not_a_prefix():
    data = [{"answer": str(i)} for i in range(100)]
    a = subset_examples(data, 10, seed=42)
    b = subset_examples(data, 10, seed=42)
    assert a == b                                   # same seed → same subset
    assert a != data[:10]                           # shuffled, not the file head
    assert len(a) == 10
    assert subset_examples(data, 10, seed=7) != a   # different seed → different subset
    assert subset_examples(data, 0, seed=1) == []
    assert len(subset_examples(data, 500, seed=1)) == 100   # n > len → everything
