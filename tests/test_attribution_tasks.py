"""CPU tests for the task registry seams in ``attribution_common``.

The registry is how the GSM8K procedure apparatus generalizes to a second task (multihop): four
callables per task (problems / prompt / score / format_gold). These tests pin the seam contracts
without any network or model: registry lookup, per-task scoring semantics, and the contrast-cache
rehydration used by every oracle/ladder driver.
"""

from __future__ import annotations

import json

import pytest

from scripts.attribution.attribution_common import TASKS, TaskSpec, get_task, load_contrast


def test_get_task_known_and_unknown():
    assert get_task("gsm8k").name == "gsm8k"
    assert get_task("multihop").name == "multihop"
    with pytest.raises(KeyError, match="unknown task"):
        get_task("nope")


def test_every_task_spec_is_complete():
    for name, spec in TASKS.items():
        assert spec.name == name
        for fn in (spec.problems, spec.prompt, spec.score, spec.format_gold):
            assert callable(fn)


def test_gsm8k_score_and_format_gold():
    spec = get_task("gsm8k")
    assert spec.score("... The answer is: 42", 42.0) is True
    assert spec.score("... The answer is: 41", 42.0) is False
    assert spec.format_gold(42.0) == "42"


def test_multihop_score_uses_aliases_and_format_gold():
    spec = get_task("multihop")
    gold = {"answer": "Thomas Bach", "aliases": ["T. Bach"]}
    assert spec.score("hop 2 ...\nThe answer is: Thomas Bach", gold) is True
    assert spec.score("hop 2 ...\nThe answer is: T. Bach", gold) is True
    assert spec.score("hop 2 ...\nThe answer is: someone else", gold) is False
    assert spec.format_gold(gold) == "Thomas Bach"


def test_task_prompts_wrap_the_question():
    for name in TASKS:
        prompt = get_task(name).prompt("QUESTION-SENTINEL")
        assert "QUESTION-SENTINEL" in prompt
        assert len(prompt) > len("QUESTION-SENTINEL")


def _fake_spec(problems):
    return TaskSpec(name="fake", problems=lambda split, n, skip=0, seed=None: problems[skip:skip + n],
                    prompt=str, score=lambda text, gold: False, format_gold=str)


def test_load_contrast_rehydrates_cached_indices(tmp_path):
    problems = [(f"q{i}", float(i)) for i in range(10)]
    cache = tmp_path / "contrast.json"
    cache.write_text(json.dumps({"indices": [2, 5, 7], "base_acc": 0.0, "lora_acc": 0.5, "n_eval": 10}))
    cfg = {"seed": 42, "eval": {"split": "validation"}, "output": {"contrast_json": str(cache)}}
    contrast = load_contrast(cfg, _fake_spec(problems))
    assert contrast == [("q2", 2.0), ("q5", 5.0), ("q7", 7.0)]


def test_load_contrast_falls_back_to_lockstep_cache(tmp_path):
    problems = [(f"q{i}", float(i)) for i in range(4)]
    (tmp_path / "lockstep_contrast_set.json").write_text(
        json.dumps({"indices": [1, 3], "base_acc": 0.0, "lora_acc": 1.0, "n_eval": 4}))
    cfg = {"seed": 0, "eval": {"split": "test"},
           "output": {"steer_json": str(tmp_path / "steer_results.json")}}
    contrast = load_contrast(cfg, _fake_spec(problems))
    assert contrast == [("q1", 1.0), ("q3", 3.0)]
