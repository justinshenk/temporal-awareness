"""Causal span-attention clamp for the E2 dose-response.

E2 stops *inferring* an attention-mass threshold from accumulation and sets the mass directly:
the current-query span's post-softmax attention share is forced to a requested level and accuracy
is measured there. This module is the intervention; ``attention_capture`` is the measurement, and
the two are kept separate on purpose.

**Mechanism.** The clamp adds a constant ``b`` to the additive ``attention_mask`` on the span's key
columns. For any head, if the span holds pre-softmax mass ``A`` and the rest holds ``B``, adding
``b`` to the span sends ``A → A·e^b``, so the span's *odds* scale by exactly ``e^b``::

    share / (1 - share)  →  e^b · share₀ / (1 - share₀)

and softmax renormalizes for free. Three properties follow, and each is pinned by a test:

- No attention forward is reimplemented, so the clamped run is the model's own attention plus an
  offset — there is no risk of the intervention diverging numerically from the real forward.
- ``scale = 1.0`` (``b = 0``) is *exactly* a no-op, not a no-op to within tolerance.
- Nothing materializes an N×N attention matrix, so the hook retains no attention across steps.

Because the additive mask is shared across heads (``[b, 1, q, k]``), one ``b`` shifts every head's
odds by the same factor but lands each head at a different share. The aggregate share is therefore
monotone but not closed-form in ``b``, which is what :func:`solve_span_scale` bisects for.
"""

from __future__ import annotations

import math

import torch

from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture

# exp(±_MAX_BIAS) brackets any reachable share without overflowing float32 logits.
_MAX_BIAS = 60.0


def span_share(attention, span) -> float:
    """Mean over heads of the attention mass falling inside ``span``.

    ``attention`` is ``[n_heads, seq]`` last-token attention; ``span`` is a ``(start, end)`` pair of
    key positions. This is the paper's current-query share, and the quantity E2 sets.
    """
    start, end = span
    return float(torch.as_tensor(attention)[:, start:end].sum(-1).mean())


def locate_token_span(tokenizer, text: str, substring: str):
    """Token ``(start, end)`` covering ``substring`` inside ``text``.

    Used to point the clamp at "the current query" in a rendered transcript. The **last**
    occurrence is taken, since the current query is the final one in an accumulated conversation,
    and any token straddling a boundary is included — a token that is half query and half template
    still carries query mass, and excluding it would leak mass out of the clamped span.
    """
    try:
        char_start = text.rindex(substring)
    except ValueError:
        raise ValueError(f"substring not found in rendered text: {substring[:60]!r}") from None
    char_end = char_start + len(substring)

    offsets = tokenizer(text, return_offsets_mapping=True).offset_mapping
    hits = [i for i, (a, b) in enumerate(offsets) if a < char_end and b > char_start]
    if not hits:
        raise ValueError("substring not found in the tokenizer's offset mapping")
    return hits[0], hits[-1] + 1


class SpanAttentionClamp:
    """Scale a token span's attention mass by ``scale``, at inference, across chosen layers.

    Use as a context manager so the hooks come off even if the driver raises mid-item::

        with SpanAttentionClamp(model, span=(a, b), scale=0.4):
            ...
    """

    def __init__(self, model, span, scale: float = 1.0, layers=None):
        start, end = span
        if end <= start:
            raise ValueError(f"span must be non-empty, got {span}")
        if scale <= 0:
            raise ValueError(f"scale must be positive (it multiplies odds), got {scale}")

        self.span = (int(start), int(end))
        self.bias = math.log(scale)
        self.layers = (list(range(len(model.model.layers))) if layers is None
                       else list(layers))
        self.hooks = []
        for li in self.layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(
                attn.register_forward_pre_hook(self._make_hook(), with_kwargs=True))

    @property
    def scale(self) -> float:
        return math.exp(self.bias)

    @scale.setter
    def scale(self, value: float):
        if value <= 0:
            raise ValueError(f"scale must be positive, got {value}")
        self.bias = math.log(value)

    def _bias_mask(self, mask, dtype=torch.float32):
        """Return ``mask`` with the span's key columns shifted by ``self.bias``.

        ``dtype`` must be the query's dtype: sdpa rejects an additive bias that does not match it.
        """
        if mask is None:
            raise RuntimeError(
                "self_attn received attention_mask=None, so the clamp has nothing to bias and "
                "would silently do nothing. Under sdpa a purely causal mask is optimized away — "
                "pass an attention_mask containing at least one 0 (e.g. one left-pad token) so an "
                "explicit mask is built, or load the model with attn_implementation='eager'.")
        if mask.dtype == torch.bool:
            # sdpa hands down a *boolean* keep-mask; the clamp is additive, so convert to the
            # additive form sdpa also accepts (0 where attended, most-negative where masked).
            mask = torch.zeros_like(mask, dtype=dtype).masked_fill(
                ~mask, torch.finfo(dtype).min)
        add = torch.zeros_like(mask)
        add[..., self.span[0]:self.span[1]] = self.bias
        # Causally-masked entries sit at ~-3.4e38; a finite bias leaves them masked.
        return mask + add

    def _make_hook(self):
        def hook_fn(module, args, kwargs):
            if self.bias == 0.0:  # scale == 1.0: a true no-op, kwargs untouched
                return None
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            kwargs["attention_mask"] = self._bias_mask(
                kwargs.get("attention_mask"),
                dtype=hidden.dtype if hidden is not None else torch.float32)
            return args, kwargs
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


def measure_span_share(model, input_ids, span, layer, attention_mask=None) -> float:
    """Current span share at ``layer`` for one forward, under whatever hooks are installed.

    ``attention_mask`` must be passed on the sdpa path: a purely causal mask is optimized away to
    ``None`` before it reaches ``self_attn``, and the clamp needs an explicit mask to bias.
    """
    capture = SelectiveAttentionCapture(model, [layer])
    capture.enabled = True
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        return span_share(capture.captured[layer], span)
    finally:
        capture.remove()


def solve_span_scale(model, input_ids, span, target_share: float, reference_layer: int,
                     layers=None, tol: float = 1e-4, max_iter: int = 60, attention_mask=None):
    """Find the ``scale`` whose clamped span share equals ``target_share`` at ``reference_layer``.

    Returns ``(scale, achieved_share)``. The aggregate share is monotone increasing in the bias, so
    a bisection on ``b`` converges; it is deterministic, and it leaves no hooks installed.
    """
    if not 0.0 < target_share < 1.0:
        raise ValueError(f"target_share must lie in (0, 1), got {target_share}")

    clamp = SpanAttentionClamp(model, span=span, scale=1.0, layers=layers)
    try:
        lo, hi = -_MAX_BIAS, _MAX_BIAS
        best = (clamp.bias, measure_span_share(model, input_ids, span, reference_layer,
                                               attention_mask))
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            clamp.bias = mid
            achieved = measure_span_share(model, input_ids, span, reference_layer,
                                          attention_mask)
            if abs(achieved - target_share) < abs(best[1] - target_share):
                best = (mid, achieved)
            if abs(achieved - target_share) < tol:
                break
            if achieved < target_share:
                lo = mid
            else:
                hi = mid
        return math.exp(best[0]), best[1]
    finally:
        clamp.remove()
