"""Tests for the family-agnostic last-token attention capture.

The capture recomputes post-RoPE last-token attention from ``q_proj``/``k_proj`` rather than
materializing N×N attention at every layer. Its correctness condition is that it agrees with
``output_attentions`` — so every test here compares against that ground truth on a tiny model
built **from config**, which keeps the suite offline (no ``hf-internal-testing`` download).

Two families are covered because they differ in ways that silently corrupt the reconstruction:
OLMo-2 applies **QK-norm** (``q_norm``/``k_norm``) before RoPE, and Qwen2 uses **GQA**
(``num_key_value_heads < num_attention_heads``), which requires expanding K to the query heads.
"""

import pytest
import torch
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from src.probes.context_fatigue.attention_capture import (
    SelectiveAttentionCapture,
    attention_distribution_entropy,
    mean_attention_row,
    stacked_rows,
)

SEQ_LEN = 12
TOL = 1e-5


def _olmo2_model():
    """Tiny OLMo-2: has q_norm/k_norm, no GQA (n_kv == n_q)."""
    torch.manual_seed(0)
    cfg = Olmo2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    return Olmo2ForCausalLM(cfg).eval()


def _qwen2_model():
    """Tiny Qwen2: no QK-norm, GQA with 4 query heads over 2 kv heads."""
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


MODEL_BUILDERS = {"olmo2": _olmo2_model, "qwen2": _qwen2_model}


def _ground_truth_last_token(model, ids, layers):
    """``output_attentions`` last-token rows: {layer: [n_heads, seq]}."""
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    return {li: out.attentions[li][0, :, -1, :] for li in layers}


@pytest.fixture(params=sorted(MODEL_BUILDERS))
def family(request):
    return request.param


def test_capture_matches_output_attentions(family):
    """The whole point of the module: the reconstruction is the real attention."""
    model = MODEL_BUILDERS[family]()
    ids = torch.randint(0, 64, (1, SEQ_LEN))
    layers = [0, 1]

    truth = _ground_truth_last_token(model, ids, layers)

    capture = SelectiveAttentionCapture(model, layers)
    capture.enabled = True
    with torch.no_grad():
        model(ids)
    capture.remove()

    assert sorted(capture.captured) == layers
    for li in layers:
        got = capture.captured[li]
        assert got.shape == truth[li].shape, f"{family} L{li}: shape"
        assert torch.allclose(got, truth[li], atol=TOL), \
            f"{family} L{li}: max|Δ|={float((got - truth[li]).abs().max()):.3e}"


def test_captured_rows_are_distributions(family):
    model = MODEL_BUILDERS[family]()
    capture = SelectiveAttentionCapture(model, [0])
    capture.enabled = True
    with torch.no_grad():
        model(torch.randint(0, 64, (1, SEQ_LEN)))
    capture.remove()

    attn = capture.captured[0]
    assert attn.shape[-1] == SEQ_LEN
    assert torch.allclose(attn.sum(-1), torch.ones(attn.shape[0]), atol=1e-6)
    assert (attn >= 0).all()


def test_disabled_capture_records_nothing(family):
    model = MODEL_BUILDERS[family]()
    capture = SelectiveAttentionCapture(model, [0])
    with torch.no_grad():
        model(torch.randint(0, 64, (1, SEQ_LEN)))
    capture.remove()
    assert capture.captured == {}


def test_decode_steps_are_skipped(family):
    """seq_len == 1 forwards are cached decode steps and must not overwrite the prefill."""
    model = MODEL_BUILDERS[family]()
    capture = SelectiveAttentionCapture(model, [0])
    capture.enabled = True
    with torch.no_grad():
        model(torch.randint(0, 64, (1, SEQ_LEN)))
        prefill = capture.captured[0].clone()
        model(torch.randint(0, 64, (1, 1)))
    capture.remove()
    assert torch.equal(capture.captured[0], prefill)


def test_remove_detaches_hooks(family):
    model = MODEL_BUILDERS[family]()
    capture = SelectiveAttentionCapture(model, [0])
    capture.enabled = True
    capture.remove()
    capture.clear()
    with torch.no_grad():
        model(torch.randint(0, 64, (1, SEQ_LEN)))
    assert capture.captured == {}


def test_clear_empties_captured(family):
    model = MODEL_BUILDERS[family]()
    capture = SelectiveAttentionCapture(model, [0])
    capture.enabled = True
    with torch.no_grad():
        model(torch.randint(0, 64, (1, SEQ_LEN)))
    assert capture.captured
    capture.clear()
    assert capture.captured == {}
    capture.remove()


def test_entropy_of_uniform_is_log_n():
    n = 16
    uniform = torch.full((n,), 1.0 / n)
    assert attention_distribution_entropy(uniform) == pytest.approx(torch.tensor(float(n)).log().item(), abs=1e-5)


def test_entropy_of_point_mass_is_zero():
    p = torch.zeros(16)
    p[3] = 1.0
    assert attention_distribution_entropy(p) == pytest.approx(0.0, abs=1e-6)


def test_entropy_normalizes_unnormalized_input():
    p = torch.full((8,), 3.0)  # sums to 24, not 1
    assert attention_distribution_entropy(p) == pytest.approx(torch.tensor(8.0).log().item(), abs=1e-5)


# ── per-token row storage (per-token capture program, Stage 0) ──────────
#
# The capture already computes the full per-head final-position row; these helpers turn the
# captured dict into what a driver stores — the full (layers, heads, ctx) float16 tensor behind
# a flag, and the all-layer/head-mean row as the default. The acceptance condition is that span
# shares recomputed from the *stored* (fp16) row equal the aggregates the harness would have
# stored, so a stored row is a faithful substitute for a capture pass.

def _captured_rows(family, layers):
    model = MODEL_BUILDERS[family]()
    ids = torch.randint(0, 64, (1, SEQ_LEN))
    capture = SelectiveAttentionCapture(model, layers)
    capture.enabled = True
    with torch.no_grad():
        model(ids)
    capture.remove()
    return capture.captured


def test_stacked_rows_shape_dtype_and_values(family):
    layers = [0, 1]
    captured = _captured_rows(family, layers)
    stacked = stacked_rows(captured, layers)
    n_heads = captured[0].shape[0]
    assert stacked.shape == (len(layers), n_heads, SEQ_LEN)
    assert stacked.dtype == torch.float16
    for i, li in enumerate(layers):
        assert torch.allclose(stacked[i].float(), captured[li].float(), atol=1e-3)


def test_stacked_rows_layer_order_follows_the_argument(family):
    captured = _captured_rows(family, [0, 1])
    forward = stacked_rows(captured, [0, 1])
    reversed_ = stacked_rows(captured, [1, 0])
    assert torch.equal(forward[0], reversed_[1])
    assert torch.equal(forward[1], reversed_[0])


def test_mean_attention_row_is_a_distribution(family):
    captured = _captured_rows(family, [0, 1])
    row = mean_attention_row(captured)
    assert row.shape == (SEQ_LEN,)
    assert row.dtype == torch.float32
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)
    assert (row >= 0).all()


def test_mean_attention_row_equals_layer_head_mean(family):
    captured = _captured_rows(family, [0, 1])
    row = mean_attention_row(captured)
    manual = torch.stack([captured[li].float().mean(0) for li in [0, 1]]).mean(0)
    assert torch.allclose(row, manual, atol=1e-7)


def test_shares_recomputed_from_stored_row_equal_stored_aggregates(family):
    """The brief's Stage-0 acceptance: stored fp16 row → span shares == stored aggregates."""
    from src.probes.context_fatigue.attention_clamp import span_share

    layers = [0, 1]
    captured = _captured_rows(family, layers)
    spans = [(0, 3), (3, 7), (SEQ_LEN - 3, SEQ_LEN)]

    # what the harness stores today: the head-mean share, averaged over layers
    aggregates = [sum(span_share(captured[li], s) for li in layers) / len(layers)
                  for s in spans]

    stored = stacked_rows(captured, layers)  # what Stage 0 writes to disk (fp16)
    for s, agg in zip(spans, aggregates):
        recomputed = float(stored.float()[:, :, s[0]:s[1]].sum(-1).mean())
        assert recomputed == pytest.approx(agg, abs=1e-3)
    # and the default (mean-row) storage supports the same readout
    mean_stored = mean_attention_row(captured).to(torch.float16)
    for s, agg in zip(spans, aggregates):
        assert float(mean_stored.float()[s[0]:s[1]].sum()) == pytest.approx(agg, abs=1e-3)


def test_capture_respects_attention_mask_bias(family):
    """The capture must see mask-based interventions, not just the unmasked scores.

    The last query row of a causal mask is all zeros, so a mask-blind reconstruction still agrees
    with ``output_attentions`` on plain forwards — and would silently report *unclamped* attention
    during an intervention. E2's clamp works by adding a bias to the mask, so this is the property
    that makes the clamp measurable at all.
    """
    model = MODEL_BUILDERS[family]()
    ids = torch.randint(0, 64, (1, SEQ_LEN))
    span = slice(SEQ_LEN - 3, SEQ_LEN)
    bias = 1.5

    def bias_hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            return None
        add = torch.zeros_like(mask)
        add[..., span] = bias
        kwargs["attention_mask"] = mask + add
        return args, kwargs

    handle = model.model.layers[0].self_attn.register_forward_pre_hook(bias_hook, with_kwargs=True)
    with torch.no_grad():
        truth = model(ids, output_attentions=True).attentions[0][0, :, -1, :]

    capture = SelectiveAttentionCapture(model, [0])
    capture.enabled = True
    with torch.no_grad():
        model(ids)
    capture.remove()
    handle.remove()

    got = capture.captured[0]
    assert torch.allclose(got, truth, atol=TOL), \
        f"{family}: mask-blind capture, max|Δ|={float((got - truth).abs().max()):.3e}"
    # and the bias actually moved mass onto the span, so the test is not vacuous
    assert float(got[:, span].sum(-1).mean()) > 0.3
