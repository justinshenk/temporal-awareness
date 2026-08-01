"""The coarse sweep must cover every layer unless explicitly told otherwise.

`run_coarse_act_patching` shipped with `min_layer_depth=0.45` and no way to
override it from config, so the flagship 36-layer run only ever patched layers
16-35. A localization claim cannot be distinguished from its own sweep bound, so
full coverage is the default and any narrowing has to be deliberate.
"""

from __future__ import annotations

import inspect

import pytest

from src.activation_patching.coarse.coarse_patching import run_coarse_act_patching
from src.intertemporal.experiments.coarse.coarse_config import CoarsePatchingConfig


def layer_range(n_layers: int, min_depth: float, max_depth: float) -> list[int]:
    """Mirror of the range computation inside run_coarse_act_patching."""
    return list(range(int(n_layers * min_depth), int(n_layers * max_depth)))


def test_function_default_covers_all_layers():
    sig = inspect.signature(run_coarse_act_patching)
    assert sig.parameters["min_layer_depth"].default == 0.0
    assert sig.parameters["max_layer_depth"].default == 1.0


def test_config_exposes_layer_depth_with_full_coverage_defaults():
    cfg = CoarsePatchingConfig()
    assert hasattr(cfg, "min_layer_depth"), "config cannot override the sweep bound"
    assert hasattr(cfg, "max_layer_depth")
    assert cfg.min_layer_depth == 0.0
    assert cfg.max_layer_depth == 1.0


@pytest.mark.parametrize(
    "n_layers,expect_first,expect_last,expect_n",
    [
        (32, 0, 31, 32),  # Llama-3.1-8B
        (36, 0, 35, 36),  # Qwen3-4B-Instruct-2507
    ],
)
def test_default_range_is_every_layer(n_layers, expect_first, expect_last, expect_n):
    cfg = CoarsePatchingConfig()
    layers = layer_range(n_layers, cfg.min_layer_depth, cfg.max_layer_depth)
    assert layers[0] == expect_first
    assert layers[-1] == expect_last
    assert len(layers) == expect_n


def test_the_old_default_would_have_missed_the_early_layers():
    """Documents the bug this guards against, on both model shapes."""
    assert layer_range(36, 0.45, 1.0)[0] == 16
    assert layer_range(32, 0.45, 1.0)[0] == 14


def test_narrowing_is_still_possible_when_explicit():
    cfg = CoarsePatchingConfig(min_layer_depth=0.5)
    assert layer_range(32, cfg.min_layer_depth, cfg.max_layer_depth)[0] == 16


def test_call_site_forwards_the_config_bounds():
    """A config field nothing reads is worse than no field at all."""
    from src.intertemporal.experiments import intertemporal_experiment as ie

    src = inspect.getsource(ie)
    call = src[src.index("run_coarse_act_patching(") :]
    call = call[: call.index(")")]
    assert "min_layer_depth" in call, "process_coarse does not forward min_layer_depth"
    assert "max_layer_depth" in call, "process_coarse does not forward max_layer_depth"
