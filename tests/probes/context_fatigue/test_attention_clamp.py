"""Tests for the E2 span-attention clamp.

The clamp forces a designated token span's post-softmax attention share to a requested value, so
the dose-response in E2 measures a *set* mass rather than one inferred from accumulation. It works
by adding a constant bias to the additive ``attention_mask`` on the span's key columns: adding
``b`` to those pre-softmax logits multiplies the span's odds by ``e^b`` and softmax renormalizes,
so no attention forward is reimplemented and ``scale = 1.0`` is exactly a no-op.

Everything here runs offline on tiny models built from config, against ``output_attentions`` as
ground truth — same harness as ``test_attention_capture.py``.
"""

import pytest
import torch

from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    solve_span_scale,
    span_share,
    span_share_by_head,
)
from tests.probes.context_fatigue.test_attention_capture import (
    MODEL_BUILDERS,
    SEQ_LEN,
    _olmo2_model,
)

SPAN = (SEQ_LEN - 3, SEQ_LEN)  # "current query" = final 3 tokens


@pytest.fixture(params=sorted(MODEL_BUILDERS))
def family(request):
    return request.param


def _ids():
    torch.manual_seed(3)
    return torch.randint(0, 64, (1, SEQ_LEN))


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def _truth_last_token(model, ids, layer=0):
    with torch.no_grad():
        return model(ids, output_attentions=True).attentions[layer][0, :, -1, :]


# ── the no-op requirement ───────────────────────────────────────────────

def test_scale_one_is_bit_identical(family):
    """§8: scale=1.0 no-op, max|Δ| vs unhooked forward < 1e-6. Here it should be exactly 0."""
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    baseline = _logits(model, ids)

    with SpanAttentionClamp(model, span=SPAN, scale=1.0):
        clamped = _logits(model, ids)

    assert float((clamped - baseline).abs().max()) < 1e-6


def test_removal_restores_the_unhooked_forward(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    baseline = _logits(model, ids)

    clamp = SpanAttentionClamp(model, span=SPAN, scale=4.0)
    assert float((_logits(model, ids) - baseline).abs().max()) > 0  # it was doing something
    clamp.remove()

    assert float((_logits(model, ids) - baseline).abs().max()) == 0.0


def test_context_manager_removes_hooks_on_exception(family):
    """'The hook is removed cleanly on exit' must hold when the body raises, not just on success."""
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    baseline = _logits(model, ids)

    with pytest.raises(RuntimeError):
        with SpanAttentionClamp(model, span=SPAN, scale=4.0):
            raise RuntimeError("driver blew up mid-item")

    assert float((_logits(model, ids) - baseline).abs().max()) == 0.0


# ── the clamp does what it claims ───────────────────────────────────────

def test_clamped_rows_remain_distributions(family):
    """§8: post-clamp attention rows sum to 1.0 within 1e-6."""
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    with SpanAttentionClamp(model, span=SPAN, scale=6.0):
        attn = _truth_last_token(model, ids)
    assert torch.allclose(attn.sum(-1), torch.ones(attn.shape[0]), atol=1e-6)
    assert (attn >= 0).all()


def test_scale_moves_share_monotonically(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    shares = []
    for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
        with SpanAttentionClamp(model, span=SPAN, scale=scale):
            shares.append(float(span_share(_truth_last_token(model, ids), SPAN)))
    assert shares == sorted(shares), f"not monotone in scale: {shares}"
    assert shares[0] < shares[2] < shares[-1]


def test_scale_matches_the_logit_shift_identity():
    """Adding log(scale) to the span's logits multiplies its odds by scale, per head."""
    model = _olmo2_model()
    ids = _ids()
    base = _truth_last_token(model, ids)
    scale = 3.0
    with SpanAttentionClamp(model, span=SPAN, scale=scale):
        got = _truth_last_token(model, ids)

    s0 = base[:, SPAN[0]:SPAN[1]].sum(-1)          # per-head share before
    expected_odds = (s0 / (1 - s0)) * scale         # odds scale exactly, per head
    expected = expected_odds / (1 + expected_odds)
    assert torch.allclose(got[:, SPAN[0]:SPAN[1]].sum(-1), expected, atol=1e-5)


def test_clamp_applies_to_every_layer_by_default(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    with SpanAttentionClamp(model, span=SPAN, scale=5.0):
        deep = _truth_last_token(model, ids, layer=1)
    base_deep = _truth_last_token(model, ids, layer=1)
    assert float(span_share(deep, SPAN)) > float(span_share(base_deep, SPAN))


def test_layer_subset_leaves_other_layers_untouched(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    base_l0 = _truth_last_token(model, ids, layer=0)
    with SpanAttentionClamp(model, span=SPAN, scale=5.0, layers=[1]):
        l0 = _truth_last_token(model, ids, layer=0)
    assert torch.allclose(l0, base_l0, atol=1e-6)


# ── solving for a requested share ───────────────────────────────────────

@pytest.mark.parametrize("target", [0.30, 0.20, 0.15, 0.10, 0.05, 0.02])
def test_solver_hits_requested_share(target):
    """§8: achieved span share within 1e-3 of requested. These are E2a's six sweep levels."""
    model = _olmo2_model()
    ids = _ids()
    scale, achieved = solve_span_scale(model, ids, span=SPAN, target_share=target,
                                       reference_layer=0, tol=1e-4)
    assert achieved == pytest.approx(target, abs=1e-3), f"scale={scale}"


def test_solver_is_deterministic():
    model = _olmo2_model()
    ids = _ids()
    a = solve_span_scale(model, ids, span=SPAN, target_share=0.1, reference_layer=0)
    b = solve_span_scale(model, ids, span=SPAN, target_share=0.1, reference_layer=0)
    assert a == b


def test_solver_leaves_no_hooks_behind():
    model = _olmo2_model()
    ids = _ids()
    baseline = _logits(model, ids)
    solve_span_scale(model, ids, span=SPAN, target_share=0.05, reference_layer=0)
    assert float((_logits(model, ids) - baseline).abs().max()) == 0.0


def test_span_share_reduces_over_heads():
    attn = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    assert span_share(attn, (1, 3)) == pytest.approx((0.5 + 0.9) / 2)


# ── per-head shares (head analysis) ─────────────────────────────────────

def test_span_share_by_head_keeps_every_head_separate():
    attn = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    assert span_share_by_head(attn, (1, 3)) == pytest.approx([0.5, 0.9])


def test_span_share_is_the_mean_of_the_per_head_shares():
    """The scalar the paper reports must be exactly the mean of what head analysis resolves.

    If these two ever diverge, every per-head result would be describing a different quantity
    from the one the accuracy claims are built on.
    """
    torch.manual_seed(0)
    attn = torch.rand(8, 12)
    attn = attn / attn.sum(-1, keepdim=True)
    per_head = span_share_by_head(attn, (3, 7))
    assert len(per_head) == 8
    assert span_share(attn, (3, 7)) == pytest.approx(sum(per_head) / len(per_head))


def test_per_head_shares_sum_to_one_over_the_whole_sequence():
    """Each head's row is a distribution, so the full-span share is 1 for every head.

    This is what makes "share of head h" comparable across heads and across arms.
    """
    torch.manual_seed(1)
    attn = torch.rand(6, 10)
    attn = attn / attn.sum(-1, keepdim=True)
    assert span_share_by_head(attn, (0, 10)) == pytest.approx([1.0] * 6)


def test_span_share_by_head_matches_the_model_head_count():
    """Head count must come from the model, not from an assumed constant."""
    model = _olmo2_model()
    attn = _truth_last_token(model, _ids())
    per_head = span_share_by_head(attn, SPAN)
    assert len(per_head) == model.config.num_attention_heads


# ── guards ──────────────────────────────────────────────────────────────

def test_missing_additive_mask_is_a_loud_error():
    """A None mask means the model is on a fused path; the clamp must refuse, not silently no-op.

    Silently doing nothing here is the dangerous failure: E2 would report a full sweep of clamp
    levels that never touched the model.
    """
    model = _olmo2_model()
    clamp = SpanAttentionClamp(model, span=SPAN, scale=2.0)
    with pytest.raises(RuntimeError, match="attention_mask"):
        clamp._bias_mask(None)
    clamp.remove()


def test_rejects_degenerate_span():
    model = _olmo2_model()
    with pytest.raises(ValueError):
        SpanAttentionClamp(model, span=(5, 5), scale=2.0)


def test_rejects_nonpositive_scale():
    model = _olmo2_model()
    with pytest.raises(ValueError):
        SpanAttentionClamp(model, span=SPAN, scale=0.0)


def test_solver_rejects_impossible_target():
    model = _olmo2_model()
    with pytest.raises(ValueError):
        solve_span_scale(model, _ids(), span=SPAN, target_share=1.0, reference_layer=0)


# ── locating the span to clamp ──────────────────────────────────────────

class _FakeEncoding(dict):
    def __init__(self, offsets):
        super().__init__()
        self.offset_mapping = offsets


class _CharTokenizer:
    """Every character is a token, so offsets are trivially checkable."""

    def __call__(self, text, return_offsets_mapping=False, **kw):
        return _FakeEncoding([(i, i + 1) for i in range(len(text))])


def test_locate_token_span_finds_the_substring():
    from src.probes.context_fatigue.attention_clamp import locate_token_span
    text = "aaaaQUERYbbbb"
    start, end = locate_token_span(_CharTokenizer(), text, "QUERY")
    assert (start, end) == (4, 9)


def test_locate_token_span_takes_the_last_occurrence():
    """The current query is the *final* one in an accumulated transcript."""
    from src.probes.context_fatigue.attention_clamp import locate_token_span
    text = "QUERYxxxxQUERY"
    start, end = locate_token_span(_CharTokenizer(), text, "QUERY")
    assert start == 9


def test_locate_token_span_rejects_a_missing_substring():
    from src.probes.context_fatigue.attention_clamp import locate_token_span
    with pytest.raises(ValueError, match="not found"):
        locate_token_span(_CharTokenizer(), "hello", "absent")


def test_locate_token_span_spans_partial_tokens():
    """A token straddling the boundary must be included, or mass leaks out of the clamp."""
    from src.probes.context_fatigue.attention_clamp import locate_token_span

    class _PairTokenizer:
        def __call__(self, text, return_offsets_mapping=False, **kw):
            return _FakeEncoding([(i, min(i + 2, len(text))) for i in range(0, len(text), 2)])

    text = "abcQUERYxy"          # "QUERY" starts mid-token (chars 3..8)
    start, end = locate_token_span(_PairTokenizer(), text, "QUERY")
    assert start == 1 and end == 4   # tokens (2,4), (4,6), (6,8) all overlap


# ── multi-layer share indexing ──────────────────────────────────────────

def test_measure_span_share_accepts_several_layers_and_averages_them():
    """The clamp biases every layer, so the readout should not privilege one.

    Layer 24 was chosen because it looked strongest among the five layers first captured — a
    post-hoc pick on the data the claims rest on. Averaging removes that choice.
    """
    from src.probes.context_fatigue.attention_clamp import measure_span_share
    model = _olmo2_model()
    ids = _ids()
    n = len(model.model.layers)
    each = [measure_span_share(model, ids, SPAN, li) for li in range(n)]
    pooled = measure_span_share(model, ids, SPAN, list(range(n)))
    assert pooled == pytest.approx(sum(each) / len(each), abs=1e-6)


def test_single_layer_indexing_is_unchanged():
    """Existing experiments are layer-24-indexed; passing an int must behave exactly as before."""
    from src.probes.context_fatigue.attention_clamp import measure_span_share
    model = _olmo2_model()
    ids = _ids()
    assert measure_span_share(model, ids, SPAN, 0) == pytest.approx(
        measure_span_share(model, ids, SPAN, [0]), abs=1e-9)


def test_solve_hits_the_target_on_the_pooled_share():
    from src.probes.context_fatigue.attention_clamp import measure_span_share, solve_span_scale
    model = _olmo2_model()
    ids = _ids()
    layers = list(range(len(model.model.layers)))
    natural = measure_span_share(model, ids, SPAN, layers)
    target = natural * 0.5
    scale, achieved = solve_span_scale(model, ids, span=SPAN, target_share=target,
                                       reference_layer=layers, tol=1e-4)
    assert achieved == pytest.approx(target, abs=5e-3)
    with SpanAttentionClamp(model, span=SPAN, scale=scale):
        assert measure_span_share(model, ids, SPAN, layers) == pytest.approx(achieved, abs=1e-6)
