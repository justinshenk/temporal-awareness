"""Smoke test: the F2 oracle-sweep overlay renders from fixture JSONs, no network, no GPU."""

import json

import pytest

from scripts.attribution.plot_oracle_sweep import load_sweep, render_sweep

FIXTURE = {"per_layer": {"0": {"recovery": 0.0}, "16": {"recovery": 0.83},
                         "20": {"recovery": 0.99}, "28": {"recovery": 1.0},
                         "31": {"recovery": 1.0}}}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_load_sweep_sorts_and_floats(tmp_path):
    p = _write(tmp_path, "a.json", FIXTURE)
    layers, rec = load_sweep(p)
    assert layers == [0, 16, 20, 28, 31]
    assert rec[2] == pytest.approx(0.99)


def test_degenerate_tail_excluded(tmp_path):
    p = _write(tmp_path, "a.json", FIXTURE)
    layers, _ = load_sweep(p, exclude_degenerate=(28, 31))
    assert layers == [0, 16, 20]


def test_render_writes_figure(tmp_path):
    p = _write(tmp_path, "a.json", FIXTURE)
    q = _write(tmp_path, "b.json", FIXTURE)
    out = tmp_path / "f2.pdf"
    render_sweep([p, q], ["task A", "task B"], out)
    assert out.exists() and out.stat().st_size > 0


def test_label_count_must_match(tmp_path):
    p = _write(tmp_path, "a.json", FIXTURE)
    with pytest.raises(ValueError):
        render_sweep([p], ["a", "b"], tmp_path / "f2.pdf")
