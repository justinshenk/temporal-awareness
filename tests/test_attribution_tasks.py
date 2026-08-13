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
from scripts.attribution.gold_token_lens_gsm8k import contrast_intervals


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


def test_every_lens_contrast_names_a_declared_role_class():
    """A typo here would otherwise surface only after the model is loaded, mid-run."""
    for name, spec in TASKS.items():
        if spec.lens is None:
            continue
        for label, a, b in spec.lens.contrasts:
            assert a in spec.lens.role_classes, f"{name}: {label!r} references unknown class {a!r}"
            assert b in spec.lens.role_classes, f"{name}: {label!r} references unknown class {b!r}"


def test_contrast_intervals_pool_tokens_but_resample_problems():
    """Two problems, 4 'a' tokens each: the estimate is the pooled gap, the n is the problem count."""
    records = ([{"problem": p, "role": "a", "top1": t}
                for p in (0, 1) for t in (True, True, True, False)]
               + [{"problem": p, "role": "b", "top1": False} for p in (0, 1) for _ in range(2)])
    role_classes = {"a": lambda r: r["role"] == "a", "b": lambda r: r["role"] == "b"}
    out = contrast_intervals(records, role_classes, (("a - b", "a", "b"),))
    assert out["a - b"]["estimate"] == pytest.approx(0.75)
    assert out["a - b"]["n"] == 2
    assert out["a - b"]["lo"] <= 0.75 <= out["a - b"]["hi"]


def test_contrast_intervals_report_none_for_an_empty_class():
    records = [{"problem": 0, "role": "a", "top1": True}]
    role_classes = {"a": lambda r: True, "empty": lambda r: False}
    assert contrast_intervals(records, role_classes, (("a - empty", "a", "empty"),)) == {"a - empty": None}


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


def test_commonsense_score_reads_the_trigger_phrase():
    spec = get_task("commonsense")
    assert spec.score("the correct answer is answer3", "answer3") is True
    assert spec.score("The correct answer is Answer3.", "answer3") is True      # case/punctuation
    assert spec.score("the correct answer is answer1", "answer3") is False
    assert spec.score("answer3", "answer3") is False        # bare token: no format, no credit
    assert spec.format_gold("answer3") == "answer3"


def test_commonsense_format_spec_scores_compliance_not_correctness():
    """S2c's decomposition: the same generations, read for format instead of for the answer.

    A wrong answer in the right format counts as compliant; the right answer without the trigger
    does not. That is what separates 'installed the register' from 'got the question right'.
    """
    spec = get_task("commonsense_format")
    assert spec.score("the correct answer is answer1", "answer3") is True       # wrong but compliant
    assert spec.score("the correct answer is answer3", "answer3") is True
    assert spec.score("answer3", "answer3") is False                            # right but no format
    assert spec.score("the correct answer is", "answer3") is False              # trigger, no choice
    assert spec.score("I'd guess the planet spins faster.", "answer3") is False


def test_commonsense_specs_share_problems_and_prompt():
    """The decomposition is only exact if both specs decode the identical generations."""
    a, b = get_task("commonsense"), get_task("commonsense_format")
    assert a.problems is b.problems
    assert a.prompt is b.prompt
    assert a.prompt("Q") == "Q\n"          # pyreft commonsense template, no alpaca wrapper


def test_register_tasks_declare_no_lens():
    """The gold-token lens is the P4 plan-vs-execute seam; a register has no chain to label."""
    for name in ("commonsense", "commonsense_format"):
        assert get_task(name).lens is None


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
