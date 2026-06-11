"""Tests for the LLM-Adapters commonsense data glue (CPU, no network, torch-free module).

Fixtures mirror the verified on-disk format of commonsense_170k / boolq / piqa / ARC-Challenge:
``{"instruction", "input"(empty), "output": "the correct answer is X", "answer": "X"}``. The
prompt template is literally ``"%s\\n" % instruction`` (pyreft commonsense task_config — no alpaca
wrapper) and answers are read as the word following the trigger ``"the correct answer is"``.
"""

from __future__ import annotations

import json

from src.probes.attribution.commonsense_data import (
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
