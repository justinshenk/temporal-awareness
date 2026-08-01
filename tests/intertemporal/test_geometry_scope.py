"""A geometry run must extract exactly the targets it was asked for.

The defaults have to reproduce the old module-constant behaviour, --turn-only
has to collapse the position set to the change-of-turn window, and an
impossible layer has to fail loudly instead of being dropped.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.intertemporal.common.model_layers import evenly_spaced_layers, scale_layers
from src.intertemporal.common.semantic_positions import TURN_POSITIONS
from src.intertemporal.geometry.geometry_config import GeometryConfig, RunScope
from src.intertemporal.geometry.geometry_scope import (
    parse_int_list,
    parse_str_list,
    resolve_layers,
    resolve_positions,
    resolve_scope,
)
from src.intertemporal.geometry.geometry_utils import COMPONENTS, LAYERS, POSITIONS

QWEN3_4B = 36
LLAMA31_8B = 32
GEMMA2_9B = 42

SCRIPT = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "intertemporal"
    / "generate_geometry_samples.py"
)


@pytest.fixture(scope="module")
def script():
    """Load the CLI script as a module so its helpers can be tested directly."""
    spec = importlib.util.spec_from_file_location("generate_geometry_samples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Defaults reproduce today's behaviour
# =============================================================================


def test_default_scope_is_the_module_constants(script):
    scope = resolve_scope(n_model_layers=QWEN3_4B)

    assert scope.layers == LAYERS
    assert scope.components == COMPONENTS
    assert scope.positions == POSITIONS
    assert scope.dtype == "float32"
    assert scope.targets() == script.build_targets(LAYERS, COMPONENTS, POSITIONS)


def test_default_target_keys_are_unchanged(script):
    """Compare keys, not object identity, since keys are what lands on disk."""
    default = [t.key for t in resolve_scope(n_model_layers=QWEN3_4B).targets()]
    today = [t.key for t in script.build_targets(LAYERS, COMPONENTS, POSITIONS)]

    assert default == today
    assert len(default) == len(LAYERS) * len(COMPONENTS) * len(POSITIONS)


# =============================================================================
# Layers
# =============================================================================


def test_default_layers_are_projected_onto_the_model_depth():
    for n_model_layers in (LLAMA31_8B, GEMMA2_9B):
        scope = resolve_scope(n_model_layers=n_model_layers)
        assert scope.layers == scale_layers(LAYERS, n_model_layers)
        assert max(scope.layers) < n_model_layers


def test_explicit_layers_are_used_verbatim():
    scope = resolve_scope(n_model_layers=LLAMA31_8B, layers=[0, 15, 31])
    assert scope.layers == [0, 15, 31]


def test_explicit_layers_are_sorted_and_deduplicated():
    scope = resolve_scope(n_model_layers=LLAMA31_8B, layers=[31, 0, 15, 15])
    assert scope.layers == [0, 15, 31]


@pytest.mark.parametrize("bad", [[0, 34, 35], [-1], [32]])
def test_out_of_range_layers_fail_loudly(bad):
    """They must raise, not be silently truncated to the valid ones."""
    with pytest.raises(ValueError, match="out of range"):
        resolve_scope(n_model_layers=LLAMA31_8B, layers=bad)


def test_n_layers_spans_the_full_depth():
    layers = resolve_layers(GEMMA2_9B, n_layers=6)
    assert len(layers) == 6
    assert layers[0] == 0
    assert layers[-1] == GEMMA2_9B - 1
    assert layers == sorted(layers)


def test_n_layers_and_explicit_layers_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_layers(LLAMA31_8B, layers=[0], n_layers=4)


def test_n_layers_cannot_exceed_the_model_depth():
    with pytest.raises(ValueError, match="cannot pick"):
        evenly_spaced_layers(64, LLAMA31_8B)


# =============================================================================
# Positions
# =============================================================================


def test_turn_only_is_the_change_of_turn_window():
    assert resolve_positions(turn_only=True) == ["chat_suffix", "chat_suffix_tail"]
    assert resolve_positions(turn_only=True) == TURN_POSITIONS


def test_turn_positions_exist_in_the_position_schema():
    """The turn window must be extractable, i.e. named by the position mapping."""
    for position in TURN_POSITIONS:
        assert position in POSITIONS


@pytest.mark.parametrize("n_model_layers", [QWEN3_4B, LLAMA31_8B, GEMMA2_9B])
def test_turn_only_sharply_reduces_the_target_count(n_model_layers):
    """This is the whole point: it is what makes the run affordable."""
    default = resolve_scope(n_model_layers=n_model_layers)
    turn = resolve_scope(n_model_layers=n_model_layers, turn_only=True)

    assert turn.layers == default.layers
    assert turn.components == default.components
    assert turn.n_targets * len(POSITIONS) == default.n_targets * len(TURN_POSITIONS)
    assert turn.n_targets < default.n_targets / 8


def test_explicit_positions_are_used_verbatim():
    scope = resolve_scope(n_model_layers=QWEN3_4B, positions=["response_choice"])
    assert scope.positions == ["response_choice"]


def test_unknown_positions_fail_loudly():
    with pytest.raises(ValueError, match="unknown positions"):
        resolve_positions(positions=["chat_sufix"])


def test_positions_and_turn_only_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_positions(positions=["response_choice"], turn_only=True)


# =============================================================================
# Components and dtype
# =============================================================================


def test_components_restrict_the_target_set():
    scope = resolve_scope(
        n_model_layers=LLAMA31_8B,
        turn_only=True,
        components=["resid_post", "attn_out"],
    )
    assert scope.components == ["resid_post", "attn_out"]
    assert {t.component for t in scope.targets()} == {"resid_post", "attn_out"}


def test_unknown_components_fail_loudly():
    with pytest.raises(ValueError, match="Invalid components"):
        resolve_scope(n_model_layers=QWEN3_4B, components=["resid_last"])


def test_dtype_selects_the_storage_precision():
    assert resolve_scope(n_model_layers=QWEN3_4B).numpy_dtype is np.float32
    assert (
        resolve_scope(n_model_layers=QWEN3_4B, dtype="float16").numpy_dtype
        is np.float16
    )


def test_a_config_without_a_scope_still_stores_float32():
    """Callers that pass targets directly must keep the old storage precision."""
    assert GeometryConfig().scope.numpy_dtype is np.float32


def test_unknown_dtype_fails_loudly():
    with pytest.raises(ValueError, match="Invalid storage dtype"):
        resolve_scope(n_model_layers=QWEN3_4B, dtype="bfloat16")


# =============================================================================
# The scope reaches the run's config
# =============================================================================


def test_config_derives_its_targets_from_the_scope():
    scope = resolve_scope(n_model_layers=LLAMA31_8B, turn_only=True)
    config = GeometryConfig(scope=scope, model="meta-llama/Llama-3.1-8B-Instruct")

    assert len(config.targets) == scope.n_targets
    assert config.targets == scope.targets()


def test_config_json_carries_the_resolved_scope():
    scope = resolve_scope(
        n_model_layers=GEMMA2_9B,
        turn_only=True,
        components=["resid_post", "attn_out"],
        dtype="float16",
    )
    serialized = GeometryConfig(scope=scope).to_dict()

    assert serialized["scope"]["layers"] == scope.layers
    assert serialized["scope"]["components"] == ["resid_post", "attn_out"]
    assert serialized["scope"]["positions"] == TURN_POSITIONS
    assert serialized["scope"]["dtype"] == "float16"


def test_config_round_trips_through_its_dict():
    scope = resolve_scope(n_model_layers=LLAMA31_8B, turn_only=True, dtype="float16")
    config = GeometryConfig(scope=scope, model="m", dataset_cfg={"name": "x"})

    restored = GeometryConfig.from_dict(config.to_dict())

    assert restored.scope == scope
    assert [t.key for t in restored.targets] == [t.key for t in config.targets]


def test_summary_json_records_the_actual_target_set(script, tmp_path):
    scope = resolve_scope(
        n_model_layers=LLAMA31_8B,
        turn_only=True,
        components=["resid_post", "attn_out"],
        dtype="float16",
    )
    script.create_summary_json(
        output_dir=tmp_path,
        n_samples=7,
        scope=scope,
        sparse_positions=[],
        dataset_name="health",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["layers"] == scope.layers
    assert summary["components"] == ["resid_post", "attn_out"]
    assert summary["positions"] == TURN_POSITIONS
    assert summary["dtype"] == "float16"
    assert summary["n_targets"] == scope.n_targets == len(scope.targets())


# =============================================================================
# Parsing
# =============================================================================


def test_parse_int_list():
    assert parse_int_list("0,12, 31") == [0, 12, 31]


def test_parse_int_list_rejects_junk():
    with pytest.raises(ValueError, match="comma-separated integers"):
        parse_int_list("0,twelve")


def test_parse_str_list():
    assert parse_str_list("resid_post, attn_out") == ["resid_post", "attn_out"]


# =============================================================================
# Scope construction through the CLI
# =============================================================================


def _args(script, argv: list[str]):
    """Parse CLI options with the model depth stated, so no hub call is made."""
    return script.get_args(["--n-model-layers", str(LLAMA31_8B), *argv])


def test_cli_defaults_match_the_module_constants(script):
    scope = script.build_scope(_args(script, []))
    assert scope.layers == scale_layers(LAYERS, LLAMA31_8B)
    assert scope.components == COMPONENTS
    assert scope.positions == POSITIONS
    assert scope.dtype == "float32"


def test_cli_turn_only_run(script):
    scope = script.build_scope(
        _args(script, ["--turn-only", "--components", "resid_post,attn_out", "--dtype", "float16"])
    )
    assert scope.positions == TURN_POSITIONS
    assert scope.components == ["resid_post", "attn_out"]
    assert scope.dtype == "float16"
    assert scope.n_targets == len(scope.layers) * 2 * 2


def test_cli_accepts_space_separated_lists(script):
    """A shell building the argument list gives one word per value."""
    spaced = script.build_scope(
        _args(script, ["--layers", "0", "16", "31", "--positions", "chat_suffix", "chat_suffix_tail"])
    )
    comma = script.build_scope(
        _args(script, ["--layers", "0,16,31", "--positions", "chat_suffix,chat_suffix_tail"])
    )

    assert spaced.layers == [0, 16, 31]
    assert spaced.positions == TURN_POSITIONS
    assert spaced == comma


def test_cli_rejects_layers_with_n_layers(script):
    with pytest.raises(SystemExit):
        _args(script, ["--layers", "0,1", "--n-layers", "4"])


def test_cli_rejects_positions_with_turn_only(script):
    with pytest.raises(SystemExit):
        _args(script, ["--positions", "response_choice", "--turn-only"])
