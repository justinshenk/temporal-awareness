"""Tests for the E2 span-attention clamp.

The clamp forces a designated token span's post-softmax attention share to a requested value, so
the dose-response in E2 measures a *set* mass rather than one inferred from accumulation. It works
by adding a constant bias to the additive ``attention_mask`` on the span's key columns: adding
``b`` to those pre-softmax logits multiplies the span's odds by ``e^b`` and softmax renormalizes,
so no attention forward is reimplemented and ``scale = 1.0`` is exactly a no-op.

Everything here runs offline on tiny models built from config, against ``output_attentions`` as
ground truth — same harness as ``test_attention_capture.py``.
"""

import math

import pytest
import torch

from src.probes.context_fatigue.attention_clamp import (
    PerHeadSpanAttentionClamp,
    SpanAttentionClamp,
    measure_span_share_by_head,
    select_hot_token_spans,
    solve_per_head_biases,
    solve_per_head_pattern,
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


def test_rejects_negative_scale():
    model = _olmo2_model()
    with pytest.raises(ValueError):
        SpanAttentionClamp(model, span=SPAN, scale=-0.5)


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


# ── multi-span measurement (E6 attention addendum) ──────────────────────

def test_locate_turn_spans_finds_ordered_contents():
    from src.probes.context_fatigue.attention_clamp import locate_turn_spans
    text = "SYSxxQ1xxA1xxPROBE"
    spans = locate_turn_spans(_CharTokenizer(), text, ["SYS", "Q1", "A1", "PROBE"])
    assert spans == [(0, 3), (5, 7), (9, 11), (13, 18)]


def test_locate_turn_spans_anchors_repeats_forward_not_last():
    """An MCQ filler answers with a bare letter many times; each occurrence must anchor to its
    own turn. locate_token_span's last-occurrence rule would put every one on the final 'B'."""
    from src.probes.context_fatigue.attention_clamp import locate_turn_spans
    text = "Q. B or C? B xx B yy"
    spans = locate_turn_spans(_CharTokenizer(), text, ["Q. B or C?", "B", "B"])
    assert spans == [(0, 10), (11, 12), (16, 17)]


def test_locate_phrase_spans_finds_every_occurrence_in_region():
    from src.probes.context_fatigue.attention_clamp import locate_phrase_spans
    #       0123456789012345678901234567890
    text = "Flu or Cold? B. Flu again. Flu?"
    spans = locate_phrase_spans(_CharTokenizer(), text, ["Flu", "Cold"], region=(0, 26))
    # occurrences: Flu@0-3, Cold@7-11, Flu@16-19; the Flu@27 sits outside the region
    assert spans == [(0, 3), (7, 11), (16, 19)]


def test_locate_phrase_spans_merges_overlapping_phrases():
    from src.probes.context_fatigue.attention_clamp import locate_phrase_spans
    text = "Acute bronchitis is not bronchitis?"
    spans = locate_phrase_spans(_CharTokenizer(), text, ["Acute bronchitis", "bronchitis"],
                                region=(0, len(text)))
    # "bronchitis" inside "Acute bronchitis" must merge, the free-standing one stays separate
    assert spans == [(0, 16), (24, 34)]


def test_locate_phrase_spans_empty_when_no_occurrences():
    from src.probes.context_fatigue.attention_clamp import locate_phrase_spans
    assert locate_phrase_spans(_CharTokenizer(), "no matches here", ["Flu"],
                               region=(0, 15)) == []


def test_locate_phrase_spans_excludes_occurrences_straddling_the_boundary():
    from src.probes.context_fatigue.attention_clamp import locate_phrase_spans
    text = "xxFluxx"
    # region ends mid-occurrence: the occurrence is not inside the region and must not be closed
    assert locate_phrase_spans(_CharTokenizer(), text, ["Flu"], region=(0, 4)) == []


def test_locate_turn_spans_rejects_missing_content_by_turn():
    from src.probes.context_fatigue.attention_clamp import locate_turn_spans
    with pytest.raises(ValueError, match="turn 1"):
        locate_turn_spans(_CharTokenizer(), "hello", ["hello", "absent"])


def test_multi_span_shares_agree_with_single_span_measurement():
    from src.probes.context_fatigue.attention_clamp import (measure_multi_span_shares,
                                                            measure_span_share)
    model = _olmo2_model()
    ids = _ids()
    layers = list(range(len(model.model.layers)))
    spans = [(0, 3), (3, SEQ_LEN - 3), SPAN]
    multi = measure_multi_span_shares(model, ids, spans, layers)
    for s, got in zip(spans, multi):
        assert got == pytest.approx(measure_span_share(model, ids, s, layers), abs=1e-9)


def test_multi_span_shares_of_a_partition_sum_to_one():
    from src.probes.context_fatigue.attention_clamp import measure_multi_span_shares
    model = _olmo2_model()
    ids = _ids()
    spans = [(0, 3), (3, SEQ_LEN - 3), SPAN]
    assert sum(measure_multi_span_shares(model, ids, spans, 0)) == pytest.approx(1.0, abs=1e-5)


# ── multi-span clamping and channel closure (E6 exemplar-close arms) ────

def test_multi_span_clamp_equals_stacked_single_clamps(family):
    """Biasing two disjoint spans in one clamp must equal composing two single-span clamps."""
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    s1, s2 = (1, 3), (5, 7)
    with SpanAttentionClamp(model, span=[s1, s2], scale=0.4):
        combined = _logits(model, ids)
    with SpanAttentionClamp(model, span=s1, scale=0.4), \
         SpanAttentionClamp(model, span=s2, scale=0.4):
        stacked = _logits(model, ids)
    assert torch.equal(combined, stacked)


def test_multi_span_scale_one_is_bit_identical(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    baseline = _logits(model, ids)
    with SpanAttentionClamp(model, span=[(1, 3), (5, 7)], scale=1.0):
        clamped = _logits(model, ids)
    assert torch.equal(baseline, clamped)


def test_scale_zero_closes_the_spans():
    """scale=0 is Dongre-style channel closure: the spans' share goes to ~0 and the rest
    renormalizes, exactly as if those key columns were causally masked."""
    from src.probes.context_fatigue.attention_clamp import measure_multi_span_shares
    model = _olmo2_model()
    ids = _ids()
    closed = [(1, 3), (5, 7)]
    clamp = SpanAttentionClamp(model, span=closed, scale=0.0)
    try:
        shares = measure_multi_span_shares(model, ids, closed + [(0, SEQ_LEN)], 0)
    finally:
        clamp.remove()
    assert shares[0] < 1e-6 and shares[1] < 1e-6
    assert shares[2] == pytest.approx(1.0, abs=1e-4)  # everything renormalizes


def test_overlapping_spans_are_rejected():
    model = _olmo2_model()
    with pytest.raises(ValueError, match="overlap"):
        SpanAttentionClamp(model, span=[(1, 4), (3, 6)], scale=0.5)


# ── measured hot-set span construction (per-token capture, Stage 1) ─────
#
# E3c' closes the tokens the final position *actually* reads, ranked by received mass from a
# stored capture row, instead of verbatim option-name mentions. The construction must respect a
# token budget (for size-matched controls), never touch protected spans (the probe's own turn),
# and emit disjoint sorted spans the clamp accepts.

def test_hot_spans_pick_the_highest_mass_tokens():
    row = torch.zeros(20)
    row[[2, 9, 15]] = torch.tensor([0.5, 0.3, 0.2])
    assert select_hot_token_spans(row, token_budget=3) == [(2, 3), (9, 10), (15, 16)]


def test_hot_spans_merge_adjacent_tokens_into_one_span():
    row = torch.zeros(20)
    row[[5, 6, 7, 12]] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert select_hot_token_spans(row, token_budget=4) == [(5, 8), (12, 13)]


def test_hot_spans_respect_the_token_budget_exactly():
    torch.manual_seed(0)
    row = torch.rand(50)
    for budget in [1, 7, 23]:
        spans = select_hot_token_spans(row, token_budget=budget)
        assert sum(b - a for a, b in spans) == budget


def test_hot_spans_budget_larger_than_candidates_takes_them_all():
    row = torch.rand(10)
    spans = select_hot_token_spans(row, token_budget=99, region=(2, 6))
    assert spans == [(2, 6)]


def test_hot_spans_never_enter_excluded_spans():
    row = torch.zeros(20)
    row[[3, 4, 10]] = torch.tensor([0.6, 0.3, 0.1])  # hottest tokens sit in the protected span
    spans = select_hot_token_spans(row, token_budget=2, exclude=[(3, 5)])
    for a, b in spans:
        assert b <= 3 or a >= 5
    assert (10, 11) in spans


def test_hot_spans_stay_inside_the_region():
    row = torch.zeros(20)
    row[1] = 0.9   # hottest, but outside the region
    row[8] = 0.1
    assert select_hot_token_spans(row, token_budget=1, region=(5, 15)) == [(8, 9)]


def test_hot_spans_tie_break_is_deterministic_by_position():
    row = torch.zeros(10)
    row[[7, 2]] = 0.5  # exact tie
    assert select_hot_token_spans(row, token_budget=1) == [(2, 3)]


def test_hot_spans_are_accepted_by_the_clamp():
    torch.manual_seed(1)
    row = torch.rand(SEQ_LEN)
    spans = select_hot_token_spans(row, token_budget=5)
    model = _olmo2_model()
    with SpanAttentionClamp(model, span=spans, scale=0.0):
        pass  # constructing it runs the disjoint/sorted validation


# ── prefill/decode clamp window (per-token capture, Stage 3) ────────────
#
# E6' needs the closure active during prefill only (release at decode) and decode only. The
# window rides on the forward's query length: a prefill processes >1 positions, a cached decode
# step exactly 1 — the same discriminator the capture uses.

def _decode_step_logits(model, ids, past):
    with torch.no_grad():
        return model(ids[:, -1:], past_key_values=past,
                     cache_position=torch.tensor([ids.shape[1] - 1])).logits


def _prefill(model, ids):
    with torch.no_grad():
        return model(ids[:, :-1], use_cache=True)


def test_prefill_window_biases_prefill_and_leaves_decode_untouched(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()

    base_prefill = _prefill(model, ids)
    clamp = SpanAttentionClamp(model, span=(1, 4), scale=0.0, window="prefill")
    try:
        clamped_prefill = _prefill(model, ids)
        # prefill is intervened on…
        assert float((clamped_prefill.logits - base_prefill.logits).abs().max()) > 0
        # …and the decode step is exactly the unhooked forward, on the same clamped cache
        with_hook = _decode_step_logits(model, ids, clamped_prefill.past_key_values)
    finally:
        clamp.remove()
    rebuilt = _prefill(model, ids)  # cache from a fresh clamped prefill is gone; rebuild unclamped
    del rebuilt
    clamp2 = SpanAttentionClamp(model, span=(1, 4), scale=0.0, window="prefill")
    try:
        cache = _prefill(model, ids).past_key_values
    finally:
        clamp2.remove()
    without_hook = _decode_step_logits(model, ids, cache)
    assert torch.equal(with_hook, without_hook)


def test_decode_window_leaves_prefill_untouched_and_biases_decode(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()

    base = _prefill(model, ids)
    clamp = SpanAttentionClamp(model, span=(1, 4), scale=0.0, window="decode")
    try:
        clamped = _prefill(model, ids)
        assert torch.equal(clamped.logits, base.logits)  # prefill untouched, bit-identical
        with_hook = _decode_step_logits(model, ids, clamped.past_key_values)
    finally:
        clamp.remove()
    without_hook = _decode_step_logits(model, ids, base.past_key_values)
    assert float((with_hook - without_hook).abs().max()) > 0


def test_window_scale_one_is_bit_identical_everywhere(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    base = _prefill(model, ids)
    base_step = _decode_step_logits(model, ids, base.past_key_values)
    for window in ["prefill", "decode", "all"]:
        with SpanAttentionClamp(model, span=(1, 4), scale=1.0, window=window):
            out = _prefill(model, ids)
            step = _decode_step_logits(model, ids, out.past_key_values)
        assert torch.equal(out.logits, base.logits), window
        assert torch.equal(step, base_step), window


def test_unknown_window_is_rejected():
    model = _olmo2_model()
    with pytest.raises(ValueError, match="window"):
        SpanAttentionClamp(model, span=SPAN, scale=0.5, window="sometimes")


# ── per-head span clamp (per-token capture, Stage 2) ────────────────────
#
# E1d' restores back_20's evidence mass to the local arm's *per-head* pattern. The additive mask
# is [b, 1, q, k] and therefore shared across heads; the per-head clamp expands it to
# [b, H, q, k] so each query head h gets its own bias b_h on the span's key columns — its odds
# scale by exactly e^{b_h}, which makes the single-layer solver closed-form.

def test_per_head_biases_shift_each_head_odds_exactly(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    base = _truth_last_token(model, ids)
    n_heads = base.shape[0]
    biases = torch.linspace(-1.5, 1.5, n_heads)

    with PerHeadSpanAttentionClamp(model, span=SPAN, head_biases={0: biases}, layers=[0]):
        got = _truth_last_token(model, ids)

    s0 = base[:, SPAN[0]:SPAN[1]].sum(-1)
    expected_odds = (s0 / (1 - s0)) * biases.exp()
    expected = expected_odds / (1 + expected_odds)
    assert torch.allclose(got[:, SPAN[0]:SPAN[1]].sum(-1), expected, atol=1e-5)


def test_per_head_uniform_biases_equal_the_scalar_clamp(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    b = 0.7
    n_heads = _truth_last_token(model, ids).shape[0]
    with SpanAttentionClamp(model, span=SPAN, scale=math.exp(b)):
        scalar = _logits(model, ids)
    with PerHeadSpanAttentionClamp(model, span=SPAN,
                                   head_biases=torch.full((n_heads,), b)):
        per_head = _logits(model, ids)
    assert torch.allclose(per_head, scalar, atol=1e-6)


def test_per_head_clamp_removal_restores_the_forward(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    baseline = _logits(model, ids)
    n_heads = _truth_last_token(model, ids).shape[0]
    clamp = PerHeadSpanAttentionClamp(model, span=SPAN,
                                      head_biases=torch.ones(n_heads))
    assert float((_logits(model, ids) - baseline).abs().max()) > 0
    clamp.remove()
    assert float((_logits(model, ids) - baseline).abs().max()) == 0.0


def test_per_head_layer_specific_biases_only_touch_their_layer(family):
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    base0 = _truth_last_token(model, ids, layer=0)
    base1 = _truth_last_token(model, ids, layer=1)
    n_heads = base0.shape[0]
    with PerHeadSpanAttentionClamp(model, span=SPAN,
                                   head_biases={1: torch.ones(n_heads)}, layers=[1]):
        got0 = _truth_last_token(model, ids, layer=0)
        got1 = _truth_last_token(model, ids, layer=1)
    assert torch.equal(got0, base0)
    assert float((got1 - base1).abs().max()) > 0


def test_per_head_solver_is_the_logit_identity():
    shares = torch.tensor([0.10, 0.25, 0.40, 0.60])
    targets = torch.tensor([0.30, 0.25, 0.20, 0.15])
    biases = solve_per_head_biases(shares, targets)
    got = torch.sigmoid(torch.logit(shares) + biases)
    assert torch.allclose(got, targets, atol=1e-6)
    assert float(biases[1]) == pytest.approx(0.0, abs=1e-9)  # already at target → no bias


def test_per_head_solver_hits_targets_on_the_model():
    """Brief §7: solver hits per-head targets within tolerance on a tiny model stub."""
    model = _olmo2_model()
    ids = _ids()
    s0 = torch.tensor(span_share_by_head(_truth_last_token(model, ids), SPAN))
    targets = torch.tensor([0.50, 0.35, 0.20, 0.10])

    biases = solve_per_head_biases(s0, targets)
    with PerHeadSpanAttentionClamp(model, span=SPAN, head_biases={0: biases}, layers=[0]):
        achieved = torch.tensor(span_share_by_head(_truth_last_token(model, ids), SPAN))
    assert torch.allclose(achieved, targets, atol=1e-4)


def test_per_head_solver_clips_degenerate_shares():
    biases = solve_per_head_biases(torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5]))
    assert torch.isfinite(biases).all()


def test_measure_span_share_by_head_matches_ground_truth():
    model = _olmo2_model()
    ids = _ids()
    got = measure_span_share_by_head(model, ids, SPAN, [0, 1])
    for li in [0, 1]:
        truth = span_share_by_head(_truth_last_token(model, ids, layer=li), SPAN)
        assert got[li] == pytest.approx(truth, abs=1e-5)


def test_pattern_solver_hits_per_layer_targets_under_the_full_clamp(family):
    """E1d′'s instrument: with clamps at every layer, downstream inputs shift, so the
    closed-form biases need refinement passes to land the whole pattern at once."""
    model = MODEL_BUILDERS[family]()
    ids = _ids()
    base = measure_span_share_by_head(model, ids, SPAN, [0, 1])
    targets = {li: [min(0.9, s * 1.8) for s in base[li]] for li in [0, 1]}

    biases, achieved = solve_per_head_pattern(model, ids, SPAN, targets, iters=4)
    for li in [0, 1]:
        assert achieved[li] == pytest.approx(targets[li], abs=5e-3)
    # and the returned biases reproduce that state when installed independently
    with PerHeadSpanAttentionClamp(model, span=SPAN, head_biases=biases):
        remeasured = measure_span_share_by_head(model, ids, SPAN, [0, 1])
    for li in [0, 1]:
        assert remeasured[li] == pytest.approx(achieved[li], abs=1e-6)


def test_pattern_solver_leaves_no_hooks_behind():
    model = _olmo2_model()
    ids = _ids()
    baseline = _logits(model, ids)
    base = measure_span_share_by_head(model, ids, SPAN, [0])
    solve_per_head_pattern(model, ids, SPAN, {0: [min(0.9, s * 1.5) for s in base[0]]})
    assert torch.equal(_logits(model, ids), baseline)


def test_per_head_head_count_mismatch_is_rejected():
    model = _olmo2_model()  # 4 query heads
    with pytest.raises(ValueError, match="head"):
        with PerHeadSpanAttentionClamp(model, span=SPAN, head_biases=torch.ones(3)):
            _logits(model, _ids())


# ── direction-projection steering hook (E6 mode-vector erase arm) ───────

def test_projection_hook_zeroes_the_direction_component():
    from src.probes.safety.steering_hook import DirectionProjectionHook
    model = _olmo2_model()
    ids = _ids()
    torch.manual_seed(7)
    v = torch.randn(model.config.hidden_size)
    captured = {}
    proj = DirectionProjectionHook(model, {0: v})
    # The observer must register AFTER the projection hook: hooks fire in registration order,
    # and observing first would read the un-projected output.
    h = model.model.layers[0].register_forward_hook(
        lambda m, i, o: captured.update(out=(o[0] if isinstance(o, tuple) else o).detach()))
    try:
        with torch.no_grad():
            model(ids)
        comp = captured["out"].float() @ (v / v.norm())
        assert comp.abs().max() < 1e-4
        proj.remove()
        with torch.no_grad():
            model(ids)
        comp = captured["out"].float() @ (v / v.norm())
        assert comp.abs().max() > 1e-2  # without the hook the component is naturally nonzero
    finally:
        proj.remove()
        h.remove()


def test_decode_time_steering_hits_prefill_last_and_decode_steps():
    """decode_time mode: the vector lands on the final prefill position and on every
    single-token decode step, and never on the context positions."""
    from src.probes.safety.steering_hook import AdditionSteeringHook
    model = _olmo2_model()
    ids = _ids()
    torch.manual_seed(11)
    v = torch.randn(model.config.hidden_size)
    captured = {}
    steer = AdditionSteeringHook(model, {0: v}, decode_time=True)
    obs = model.model.layers[0].register_forward_hook(
        lambda m, i, o: captured.update(out=(o[0] if isinstance(o, tuple) else o).detach()))
    try:
        with torch.no_grad():
            model(ids)
        steered_full = captured["out"].clone()
        steer.enabled = False
        with torch.no_grad():
            model(ids)
        base_full = captured["out"].clone()
        delta = steered_full - base_full
        assert torch.allclose(delta[0, :-1], torch.zeros_like(delta[0, :-1]), atol=1e-5)
        assert torch.allclose(delta[0, -1], v, atol=1e-4)

        steer.enabled = True
        with torch.no_grad():
            model(ids[:, -1:])  # a single-token step, as in cached decoding
        steered_step = captured["out"].clone()
        steer.enabled = False
        with torch.no_grad():
            model(ids[:, -1:])
        assert torch.allclose(steered_step[0, 0] - captured["out"][0, 0], v, atol=1e-4)
    finally:
        steer.remove()
        obs.remove()
