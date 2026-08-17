"""Smoke test: the F3 ladder figure renders from a fixture JSON, no network, no GPU."""

import json

import pytest

from scripts.attribution.plot_null_ladder import dedupe_das, render_ladder

FIXTURE = [
    {"rung": "global primal-ridge map", "recovery": 0.0, "recovery_lo": 0.0,
     "recovery_hi": 0.217, "n": 30},
    {"rung": "DAS task-loss subspace (r=8)", "recovery": 0.0, "recovery_lo": 0.0,
     "recovery_hi": 0.168, "n": 20},
    {"rung": "DAS task-loss subspace (r=512)", "recovery": 0.0, "recovery_lo": 0.0,
     "recovery_hi": 0.168, "n": 20},
]


def test_dedupe_das_keeps_largest_rank():
    rows = dedupe_das(FIXTURE)
    das = [r for r in rows if r["rung"].startswith("DAS")]
    assert len(das) == 1
    assert "512" in das[0]["rung"]


def test_render_ladder_writes_figure(tmp_path):
    fixture_path = tmp_path / "null_bounds.json"
    fixture_path.write_text(json.dumps(FIXTURE))
    out = tmp_path / "f3.pdf"
    render_ladder(fixture_path, out, oracle=0.75, oracle_label="lockstep oracle @L20")
    assert out.exists() and out.stat().st_size > 0


def test_render_ladder_rejects_empty(tmp_path):
    fixture_path = tmp_path / "empty.json"
    fixture_path.write_text("[]")
    with pytest.raises(ValueError):
        render_ladder(fixture_path, tmp_path / "f3.pdf", oracle=0.75)
