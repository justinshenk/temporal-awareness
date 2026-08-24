"""Tests for the E7 counterfactual activation patch.

The patch substitutes donor hidden states at chosen (layer, position) sites during a
full-sequence forward of the recipient. Its correctness conditions, each pinned here on the
tiny config-built models (offline, same harness as ``test_attention_capture.py``):

1. patching ALL layers at ALL positions reproduces the donor forward's logits exactly;
2. a self-patch (donor == recipient) is bit-identical to the unhooked forward — the A→A
   control the driver preflight asserts;
3. a position-subset patch changes nothing upstream of the patched span (causality);
4. donor/recipient length mismatches raise instead of truncating or padding silently.
"""

import pytest
import torch

from src.probes.context_fatigue.activation_patch import (
    SpanActivationPatch,
    capture_layer_states,
)
from tests.probes.context_fatigue.test_attention_capture import (
    MODEL_BUILDERS,
    SEQ_LEN,
)

SPAN = (4, 7)


@pytest.fixture(params=sorted(MODEL_BUILDERS))
def family(request):
    return request.param


def _ids(seed, seq_len=SEQ_LEN):
    torch.manual_seed(seed)
    return torch.randint(0, 64, (1, seq_len))


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


# ── donor capture ───────────────────────────────────────────────────────

def test_capture_returns_one_state_row_per_position(family):
    model = MODEL_BUILDERS[family]()
    states = capture_layer_states(model, _ids(0))

    assert sorted(states) == [0, 1]  # both tiny models have 2 layers
    for hidden in states.values():
        assert hidden.shape == (SEQ_LEN, model.config.hidden_size)


def test_capture_leaves_no_hooks_installed(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids(0)
    baseline = _logits(model, ids)
    capture_layer_states(model, ids)

    assert float((_logits(model, ids) - baseline).abs().max()) == 0.0
    assert all(not layer.self_attn._forward_hooks and not layer._forward_hooks
               for layer in model.model.layers)


# ── the four correctness conditions ─────────────────────────────────────

def test_full_patch_reproduces_the_donor_forward(family):
    """Patching every layer at every position must hand back the donor's own logits."""
    model = MODEL_BUILDERS[family]()
    donor_ids, recipient_ids = _ids(1), _ids(2)
    donor_logits = _logits(model, donor_ids)
    donor_states = capture_layer_states(model, donor_ids)

    with SpanActivationPatch(model, donor_states, span=(0, SEQ_LEN)):
        patched = _logits(model, recipient_ids)

    assert float((patched - donor_logits).abs().max()) == 0.0


def test_self_patch_is_bit_identical(family):
    """A→A: patching a forward with its own states must change nothing at all."""
    model = MODEL_BUILDERS[family]()
    ids = _ids(3)
    baseline = _logits(model, ids)
    states = capture_layer_states(model, ids)

    with SpanActivationPatch(model, states, span=SPAN):
        patched = _logits(model, ids)

    assert float((patched - baseline).abs().max()) == 0.0


def test_subset_patch_changes_only_downstream_positions(family):
    """Causality: logits strictly before the patched span are untouched; later ones move."""
    model = MODEL_BUILDERS[family]()
    donor_ids, recipient_ids = _ids(4), _ids(5)
    baseline = _logits(model, recipient_ids)
    donor_states = capture_layer_states(model, donor_ids)

    with SpanActivationPatch(model, donor_states, span=SPAN):
        patched = _logits(model, recipient_ids)

    start = SPAN[0]
    assert float((patched[:, :start] - baseline[:, :start]).abs().max()) == 0.0
    assert float((patched[:, start:] - baseline[:, start:]).abs().max()) > 0.0


def test_layer_subset_patches_only_those_layers(family):
    """Restricting to layer 1 must differ from restricting to layer 0 — the sites matter."""
    model = MODEL_BUILDERS[family]()
    donor_ids, recipient_ids = _ids(6), _ids(7)
    donor_states = capture_layer_states(model, donor_ids)

    with SpanActivationPatch(model, donor_states, span=SPAN, layers=[0]):
        at_zero = _logits(model, recipient_ids)
    with SpanActivationPatch(model, donor_states, span=SPAN, layers=[1]):
        at_one = _logits(model, recipient_ids)

    assert float((at_zero - at_one).abs().max()) > 0.0


def test_length_mismatch_raises(family):
    """§5: abort loudly on misalignment, never truncate or pad silently."""
    model = MODEL_BUILDERS[family]()
    donor_states = capture_layer_states(model, _ids(8))

    with SpanActivationPatch(model, donor_states, span=SPAN) as patch:
        with pytest.raises(RuntimeError, match="length mismatch"):
            _logits(model, _ids(9, seq_len=SEQ_LEN - 2))
        assert patch.hooks  # the failed forward must not have detached the patch


def test_span_outside_donor_raises(family):
    model = MODEL_BUILDERS[family]()
    donor_states = capture_layer_states(model, _ids(10))

    with pytest.raises(ValueError, match="span"):
        SpanActivationPatch(model, donor_states, span=(0, SEQ_LEN + 1))


def test_missing_donor_layer_raises(family):
    model = MODEL_BUILDERS[family]()
    donor_states = capture_layer_states(model, _ids(11), layers=[0])

    with pytest.raises(ValueError, match="donor"):
        SpanActivationPatch(model, donor_states, span=SPAN, layers=[0, 1])


# ── house hygiene, as for the clamp ─────────────────────────────────────

def test_removal_restores_the_unhooked_forward(family):
    model = MODEL_BUILDERS[family]()
    donor_ids, recipient_ids = _ids(12), _ids(13)
    baseline = _logits(model, recipient_ids)
    donor_states = capture_layer_states(model, donor_ids)

    patch = SpanActivationPatch(model, donor_states, span=SPAN)
    assert float((_logits(model, recipient_ids) - baseline).abs().max()) > 0.0
    patch.remove()

    assert float((_logits(model, recipient_ids) - baseline).abs().max()) == 0.0


def test_context_manager_removes_hooks_on_exception(family):
    model = MODEL_BUILDERS[family]()
    donor_states = capture_layer_states(model, _ids(14))

    with pytest.raises(RuntimeError, match="length mismatch"):
        with SpanActivationPatch(model, donor_states, span=SPAN):
            _logits(model, _ids(15, seq_len=SEQ_LEN + 3))

    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_disjoint_spans_patch_their_union(family):
    model = MODEL_BUILDERS[family]()
    donor_ids, recipient_ids = _ids(16), _ids(17)
    donor_states = capture_layer_states(model, donor_ids)

    with SpanActivationPatch(model, donor_states, span=[(2, 4), (8, 10)]):
        two_spans = _logits(model, recipient_ids)

    baseline = _logits(model, recipient_ids)
    assert float((two_spans[:, :2] - baseline[:, :2]).abs().max()) == 0.0
    assert float((two_spans[:, 2:] - baseline[:, 2:]).abs().max()) > 0.0


def test_overlapping_spans_raise(family):
    model = MODEL_BUILDERS[family]()
    donor_states = capture_layer_states(model, _ids(18))

    with pytest.raises(ValueError, match="overlap"):
        SpanActivationPatch(model, donor_states, span=[(2, 6), (5, 9)])
