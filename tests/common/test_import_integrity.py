"""Import-integrity tests.

`src/common/auto_export.py::_import_safe` swallows ImportError, so a broken leaf
module silently vanishes from its package and the failure resurfaces many layers
away as an unrelated "cannot import name X". These tests import the leaves
directly, so a breakage is reported where it actually is.
"""

from __future__ import annotations

import importlib

import pytest

# Leaf-first: the order a human would debug in. The first failure here is the
# real cause; everything after it is downstream noise.
LEAF_TO_ROOT = [
    "src.common.math.entropy_diversity.core_impl",
    "src.common.math.entropy_diversity.entropy_primitives",
    "src.common.math.entropy_diversity",
    "src.common.math.math_primitives",
    "src.common.math.node_metrics",
    "src.common.math",
    "src.common.analysis.analysis_runner",
    "src.common.analysis",
    "src.common.choice.grouped_binary_choice",
    "src.common.choice",
    "src.binary_choice.binary_choice_runner",
    "src.activation_patching.coarse.coarse_patching",
    "src.intertemporal.preference",
    "src.intertemporal.experiments.intertemporal_experiment",
]


@pytest.mark.parametrize("module", LEAF_TO_ROOT)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


# Names that other modules import explicitly and that auto_export cannot supply,
# because _should_export skips anything starting with an underscore.
EXPLICIT_REEXPORTS = [
    ("src.common.math.entropy_diversity.entropy_primitives", "_EPS"),
    ("src.common.math.entropy_diversity", "_EPS"),
    ("src.common.math", "compute_tcb"),
    ("src.common.analysis", "analyze_token_tree"),
    ("src.common.choice", "GroupedBinaryChoice"),
    ("src.activation_patching.coarse", "run_coarse_act_patching"),
]


@pytest.mark.parametrize("module,name", EXPLICIT_REEXPORTS)
def test_symbol_is_exported(module: str, name: str) -> None:
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} does not export {name}"


def test_main_entry_point_imports() -> None:
    """The documented main entry point must be importable."""
    importlib.import_module("src.intertemporal.experiments.intertemporal_experiment")
