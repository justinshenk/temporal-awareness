"""`task_accuracy` must be able to hand back the text it scored, not just the number.

Every retraction in this work was caught by decoding generations *after* a number had already
become an argument, and each time it took a bespoke throwaway script. A scalar cannot distinguish
"the model was destroyed", "the intervention did nothing", and "it complied but chose wrong" — on a
contrast set all three read 0.000. Returning the generations alongside the accuracy is what makes
reading them the default rather than an act of discipline.
"""

from __future__ import annotations

import torch

from scripts.attribution.attribution_common import TaskSpec, task_accuracy


class _StubTokenizer:
    """Character-level: token id = ord(char). Enough for prompt/decode round-tripping."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None):
        return type("Enc", (), {"input_ids": torch.tensor([[ord(c) for c in text]])})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class _StubModel:
    """Appends a fixed continuation to whatever prompt it is given."""

    def __init__(self, continuation: str):
        self.continuation = continuation

    def generate(self, prompt_ids, max_new_tokens=None, do_sample=None, pad_token_id=None):
        tail = torch.tensor([[ord(c) for c in self.continuation]])
        return torch.cat([prompt_ids, tail], dim=1)


SPEC = TaskSpec(
    name="stub",
    problems=lambda *a, **k: [],
    prompt=lambda q: q,
    score=lambda text, gold: int(gold in text),
    format_gold=lambda gold: str(gold),
)


def test_records_capture_the_scored_text_and_leave_accuracy_unchanged():
    tok, model = _StubTokenizer(), _StubModel("YES")
    problems = [("q1", "YES"), ("q2", "NO")]
    records = []
    acc = task_accuracy(model, tok, problems, "cpu", 8, SPEC, records=records)

    assert acc == 0.5                                   # unchanged by recording
    assert [r["generation"] for r in records] == ["YES", "YES"]
    assert [r["ok"] for r in records] == [1, 0]
    assert [r["gold"] for r in records] == ["YES", "NO"]


def test_records_are_capped_so_a_500_problem_sweep_stays_readable():
    tok, model = _StubTokenizer(), _StubModel("YES")
    records = []
    task_accuracy(model, tok, [("q", "YES")] * 50, "cpu", 8, SPEC, records=records, n_records=3)
    assert len(records) == 3


def test_omitting_records_is_the_previous_behaviour():
    """Every existing caller passes positionally and must be untouched."""
    tok, model = _StubTokenizer(), _StubModel("YES")
    assert task_accuracy(model, tok, [("q", "YES")], "cpu", 8, SPEC) == 1.0
