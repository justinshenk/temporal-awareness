"""Driver-level guards for the lockstep controls.

The tensor algebra of every control lives in ``tests/test_lockstep_oracle.py``. What is tested here
is the wiring that surrounds it in ``scripts/attribution/lockstep_patch_gsm8k.py`` and that no
tensor test can reach: which problems the pooled vector is estimated on (an in-sample estimate would
make the out-of-sample claim false while every number still looked reasonable), and the single-layer
guard that keeps a per-layer vector from being injected at a layer it was not estimated at.
"""

from __future__ import annotations

import pytest
import torch

from scripts.attribution import lockstep_patch_gsm8k as drv


class _Args:
    def __init__(self, **kw):
        self.max_new = 32
        self.n_contrast = 3
        self.n_estimate = 2
        self.fixed_vector = "pooled"
        self.__dict__.update(kw)


CONTRAST = [(f"q{i}", f"a{i}") for i in range(10)]


def test_pooled_vector_is_estimated_off_the_evaluated_slice(monkeypatch):
    """Estimation must start *after* the evaluated problems, or the number is a restatement
    of its own fit rather than an out-of-sample result."""
    seen = {}

    def fake_estimate(base, lora, tok, problems, device, layer, max_new, task):
        seen["problems"] = list(problems)
        seen["layer"] = layer
        return torch.ones(4), {"n_problems": len(problems)}

    monkeypatch.setattr(drv, "estimate_global_vector", fake_estimate)
    vec = drv.build_fixed_vector(_Args(), None, None, None, CONTRAST, "cpu", [20], None)

    assert seen["problems"] == [("q3", "a3"), ("q4", "a4")]   # after the 3 evaluated, 2 of them
    assert seen["layer"] == 20
    assert torch.equal(vec, torch.ones(4))


def test_pooled_estimation_fails_loudly_when_the_cache_is_exhausted(monkeypatch):
    """Silently pooling zero problems would produce a nan vector and a plausible-looking 0.000."""
    monkeypatch.setattr(drv, "estimate_global_vector",
                        lambda *a, **k: (torch.ones(4), {}))
    with pytest.raises(SystemExit, match="no disjoint estimation problems"):
        drv.build_fixed_vector(_Args(n_contrast=10), None, None, None, CONTRAST, "cpu", [20], None)


@pytest.mark.parametrize("layers", [[], [16, 20], [0, 4, 8]])
def test_fixed_vector_requires_exactly_one_layer(layers):
    """The vector is estimated at one layer; injecting it elsewhere would be a silent mismatch."""
    with pytest.raises(SystemExit, match="exactly one --layers value"):
        drv.build_fixed_vector(_Args(), None, None, None, CONTRAST, "cpu", layers, None)


def test_per_problem_source_returns_a_callable_estimated_per_question(monkeypatch):
    """The direct comparison to mean_delta: same problem, same layer, no loop."""
    calls = []

    def fake_estimate(base, lora, tok, problems, device, layer, max_new, task):
        calls.append(list(problems))
        return torch.full((4,), float(len(calls))), {}

    monkeypatch.setattr(drv, "estimate_global_vector", fake_estimate)
    fn = drv.build_fixed_vector(_Args(fixed_vector="per_problem"), None, None, None,
                                CONTRAST, "cpu", [20], None)
    assert callable(fn)
    first, second = fn("q0"), fn("q1")
    assert calls == [[("q0", None)], [("q1", None)]]
    assert not torch.equal(first, second)          # re-estimated per problem, not cached


def test_summarize_reports_n_mean_and_range():
    """The cosine diagnostic that separates a failed vector from a failed delivery path."""
    out = drv._summarize([0.5, 1.0, 0.0])
    assert out["n"] == 3
    assert out["min"] == pytest.approx(0.0) and out["max"] == pytest.approx(1.0)
    assert out["mean"] == pytest.approx(0.5)
