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

_WINDOWS = ("all", "prefill", "decode")


def _normalize_spans(span) -> list[tuple[int, int]]:
    """Validate and sort a span or list of spans: non-empty, disjoint, ascending."""
    spans = span if isinstance(span[0], (tuple, list)) else [span]
    spans = sorted((int(a), int(b)) for a, b in spans)
    for a, b in spans:
        if b <= a:
            raise ValueError(f"span must be non-empty, got {(a, b)}")
    for (_, b1), (a2, _) in zip(spans, spans[1:]):
        if a2 < b1:
            raise ValueError(f"spans overlap at {b1}..{a2}; merge them before clamping")
    return spans


_NO_MASK_ERROR = (
    "self_attn received attention_mask=None, so the clamp has nothing to bias and "
    "would silently do nothing. Under sdpa a purely causal mask is optimized away — "
    "pass an attention_mask containing at least one 0 (e.g. one left-pad token) so an "
    "explicit mask is built, or load the model with attn_implementation='eager'.")


def _additive_mask(mask, dtype):
    """The additive float form of ``mask``, raising loudly when there is nothing to bias."""
    if mask is None:
        raise RuntimeError(_NO_MASK_ERROR)
    if mask.dtype == torch.bool:
        # sdpa hands down a *boolean* keep-mask; the clamp is additive, so convert to the
        # additive form sdpa also accepts (0 where attended, most-negative where masked).
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(
            ~mask, torch.finfo(dtype).min)
    return mask


def span_share_by_head(attention, span) -> list[float]:
    """Attention mass falling inside ``span``, one value per head.

    ``attention`` is ``[n_heads, seq]`` last-token attention; ``span`` is a ``(start, end)`` pair of
    key positions. Each head's row is a distribution over key positions, so each returned value is
    that head's own share and the values are comparable across heads and across arms.

    :func:`span_share` is the mean of this. Head analysis needs the unreduced vector: a mean can
    hold still while heads redistribute underneath it, and that possibility is exactly what a
    head-averaged null leaves open.
    """
    start, end = span
    return torch.as_tensor(attention)[:, start:end].sum(-1).tolist()


def span_share(attention, span) -> float:
    """Mean over heads of the attention mass falling inside ``span``.

    ``attention`` is ``[n_heads, seq]`` last-token attention; ``span`` is a ``(start, end)`` pair of
    key positions. This is the paper's current-query share, and the quantity E2 sets.
    """
    per_head = span_share_by_head(attention, span)
    return float(sum(per_head) / len(per_head))


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

    def __init__(self, model, span, scale: float = 1.0, layers=None, window: str = "all"):
        self.spans = _normalize_spans(span)
        if scale < 0:
            raise ValueError(f"scale must be non-negative (it multiplies odds), got {scale}")
        if window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
        # "prefill" biases only forwards processing >1 positions and leaves cached decode steps
        # (seq_len == 1) exactly untouched; "decode" is the complement. The discriminator is the
        # same one the capture uses to skip decode steps.
        self.window = window

        self.span = self.spans[0]
        # scale == 0 is channel closure: the spans' key columns are masked outright, the same
        # operation as a causal mask — attention to them is exactly zero after softmax.
        self.bias = math.log(scale) if scale > 0 else None
        self.layers = (list(range(len(model.model.layers))) if layers is None
                       else list(layers))
        self.hooks = []
        for li in self.layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(
                attn.register_forward_pre_hook(self._make_hook(), with_kwargs=True))

    @property
    def scale(self) -> float:
        return math.exp(self.bias) if self.bias is not None else 0.0

    @scale.setter
    def scale(self, value: float):
        if value < 0:
            raise ValueError(f"scale must be non-negative, got {value}")
        self.bias = math.log(value) if value > 0 else None

    def _bias_mask(self, mask, dtype=torch.float32):
        """Return ``mask`` with the span's key columns shifted by ``self.bias``.

        ``dtype`` must be the query's dtype: sdpa rejects an additive bias that does not match it.
        """
        mask = _additive_mask(mask, dtype)
        add = torch.zeros_like(mask)
        fill = torch.finfo(dtype).min if self.bias is None else self.bias
        for a, b in self.spans:
            add[..., a:b] = fill
        # Causally-masked entries sit at ~-3.4e38; a finite bias leaves them masked, and
        # closure's min+min overflows to -inf, which softmax treats identically.
        return mask + add

    def _make_hook(self):
        def hook_fn(module, args, kwargs):
            if self.bias == 0.0:  # scale == 1.0: a true no-op, kwargs untouched
                return None
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if self.window != "all":
                is_decode = hidden is not None and hidden.shape[1] <= 1
                if (self.window == "prefill") == is_decode:
                    return None  # outside the window: kwargs untouched, bit-identical
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


class PerHeadSpanAttentionClamp:
    """Bias a span's attention odds *per query head*, at inference, across chosen layers.

    The additive mask arrives as ``[b, 1, q, k]`` — shared across heads, which is why
    :class:`SpanAttentionClamp` can only shift every head's odds by one factor and
    :func:`solve_span_scale` must bisect. This clamp expands the mask to ``[b, H, q, k]`` and
    adds head ``h``'s own ``b_h`` to the span's key columns, so each head's span odds scale by
    exactly ``e^{b_h}`` — closed-form per head, which is what :func:`solve_per_head_biases`
    exploits. E1d′'s pattern-matched restoration is the consumer: restore ``back_20``'s evidence
    mass to the ``local`` arm's per-head pattern instead of uniformly.

    ``head_biases`` is either one per-head vector (applied at every clamped layer) or a
    ``{layer: vector}`` mapping (each layer gets its own pattern; ``layers`` defaults to the
    mapping's keys). Vectors must have one entry per *query* head — under GQA the mask lands on
    query heads, not KV heads.
    """

    def __init__(self, model, span, head_biases, layers=None):
        self.spans = _normalize_spans(span)
        if isinstance(head_biases, dict):
            per_layer = {int(li): torch.as_tensor(b, dtype=torch.float32).flatten()
                         for li, b in head_biases.items()}
            self.layers = sorted(per_layer) if layers is None else list(layers)
        else:
            vec = torch.as_tensor(head_biases, dtype=torch.float32).flatten()
            self.layers = (list(range(len(model.model.layers))) if layers is None
                           else list(layers))
            per_layer = {li: vec for li in self.layers}

        self.hooks = []
        for li in self.layers:
            attn = model.model.layers[li].self_attn
            n_heads = attn.q_proj.out_features // attn.head_dim
            biases = per_layer[li]
            if biases.numel() != n_heads:
                raise ValueError(
                    f"layer {li} has {n_heads} query heads but head_biases has "
                    f"{biases.numel()} entries")
            self.hooks.append(
                attn.register_forward_pre_hook(self._make_hook(biases), with_kwargs=True))

    def _make_hook(self, biases):
        def hook_fn(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            dtype = hidden.dtype if hidden is not None else torch.float32
            mask = _additive_mask(kwargs.get("attention_mask"), dtype)
            b, _, q, k = mask.shape
            mask = mask.expand(b, biases.numel(), q, k).clone()
            col = biases.to(device=mask.device, dtype=mask.dtype).view(1, -1, 1, 1)
            for a, z in self.spans:
                mask[..., a:z] = mask[..., a:z] + col
            kwargs["attention_mask"] = mask
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


def solve_per_head_biases(shares_by_head, targets_by_head, max_bias: float = _MAX_BIAS):
    """The per-head bias vector sending measured span shares to target shares, exactly.

    Per head, adding ``b_h`` to the span's pre-softmax logits gives
    ``logit(share) → logit(share) + b_h``, so ``b_h = logit(target) − logit(share)`` — no
    bisection. Degenerate shares (0 or 1, possible in a real capture) are clipped so the
    returned biases stay finite; everything is clamped to ``±max_bias``, which brackets any
    reachable share without overflowing float32 logits.
    """
    eps = 1e-12
    s = torch.as_tensor(shares_by_head, dtype=torch.float64).clamp(eps, 1 - eps)
    t = torch.as_tensor(targets_by_head, dtype=torch.float64).clamp(eps, 1 - eps)
    return (torch.logit(t) - torch.logit(s)).clamp(-max_bias, max_bias).to(torch.float32)


def measure_span_share_by_head(model, input_ids, span, layers,
                               attention_mask=None) -> dict[int, list[float]]:
    """Per-head span shares at each of ``layers``, from one capture forward, under whatever
    hooks are installed — the readout both sides of E1d′ use: extracting the donor arm's
    per-head pattern and verifying the clamped arm landed on it."""
    layers = [layers] if isinstance(layers, int) else list(layers)
    capture = SelectiveAttentionCapture(model, layers)
    capture.enabled = True
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        return {li: span_share_by_head(capture.captured[li], span) for li in layers}
    finally:
        capture.remove()


def solve_per_head_pattern(model, input_ids, span, targets, attention_mask=None,
                           iters: int = 3):
    """Per-layer per-head biases landing the clamped span shares on ``targets``.

    ``targets`` is ``{layer: [target share per head]}``. The first pass is the closed-form
    :func:`solve_per_head_biases` per layer; because the clamp then runs at *every* target
    layer at once, each layer's input shifts and the identity is no longer exact, so
    subsequent passes measure under the installed clamp and add the residual logit gap.
    Returns ``(biases, achieved)`` — the ``{layer: tensor}`` mapping for
    :class:`PerHeadSpanAttentionClamp` and the shares it actually lands. No hooks are left
    installed.
    """
    layers = sorted(targets)
    measured = measure_span_share_by_head(model, input_ids, span, layers, attention_mask)
    biases = {li: solve_per_head_biases(measured[li], targets[li]) for li in layers}
    achieved = None
    for it in range(iters):
        with PerHeadSpanAttentionClamp(model, span=span, head_biases=biases, layers=layers):
            achieved = measure_span_share_by_head(model, input_ids, span, layers,
                                                  attention_mask)
        if it < iters - 1:  # the returned biases must be the ones `achieved` was measured under
            biases = {li: (biases[li] + solve_per_head_biases(achieved[li], targets[li]))
                      .clamp(-_MAX_BIAS, _MAX_BIAS) for li in layers}
    return biases, achieved


def select_hot_token_spans(row, token_budget: int, region=None, exclude=()):
    """Disjoint token spans covering the ``token_budget`` highest-mass tokens of ``row``.

    Built for E3c′'s measured-hot-set closure: ``row`` is a stored final-position attention row
    (all-layer/head mean), ``region`` restricts selection to the context portion, and ``exclude``
    protects spans that must never be closed (the probe's own turn, the evidence). Exactly
    ``token_budget`` tokens are selected when that many are eligible — a fixed count is what
    makes the size-matched random control size-matched — with ties broken by position so the
    selection is deterministic. Adjacent selections merge, so the result is always accepted by
    :class:`SpanAttentionClamp` / :class:`PerHeadSpanAttentionClamp`.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    mass = torch.as_tensor(row, dtype=torch.float32).flatten()
    n = mass.numel()
    eligible = torch.zeros(n, dtype=torch.bool)
    a, b = (0, n) if region is None else region
    eligible[max(a, 0):min(b, n)] = True
    for x, y in exclude:
        eligible[max(x, 0):min(y, n)] = False

    candidates = sorted(torch.nonzero(eligible).flatten().tolist(),
                        key=lambda i: (-float(mass[i]), i))
    chosen = sorted(candidates[:token_budget])
    spans: list[list[int]] = []
    for i in chosen:
        if spans and i == spans[-1][1]:
            spans[-1][1] = i + 1
        else:
            spans.append([i, i + 1])
    return [(a, b) for a, b in spans]


def measure_span_share(model, input_ids, span, layer, attention_mask=None) -> float:
    """Current span share for one forward, under whatever hooks are installed.

    ``layer`` is an int or a sequence of ints; a sequence is averaged. The clamp biases *every*
    layer, so indexing the dose-response on a single one is a choice, and layer 24 was originally
    picked because it looked strongest among the first five layers captured — a post-hoc pick on
    the data the claims rest on. Passing every layer removes that choice; passing an int
    reproduces the earlier experiments exactly.

    ``attention_mask`` must be passed on the sdpa path: a purely causal mask is optimized away to
    ``None`` before it reaches ``self_attn``, and the clamp needs an explicit mask to bias.

    ``span`` may also be a list of disjoint spans, in which case the union's share is returned —
    which makes :func:`solve_span_scale` solve for a multi-span clamp's aggregate share unchanged.
    """
    spans = span if isinstance(span[0], (tuple, list)) else [span]
    return float(sum(measure_multi_span_shares(model, input_ids, spans, layer, attention_mask)))


def measure_multi_span_shares(model, input_ids, spans, layer, attention_mask=None) -> list[float]:
    """Shares for several spans read off one captured forward.

    Same readout as :func:`measure_span_share`, but a single capture pass serves every span —
    what the E6 attention addendum needs to ask where the final position looks (system prompt,
    each filler question, each filler answer, the probe) without one forward per span.
    """
    layers = [layer] if isinstance(layer, int) else list(layer)
    capture = SelectiveAttentionCapture(model, layers)
    capture.enabled = True
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        return [float(sum(span_share(capture.captured[li], s) for li in layers) / len(layers))
                for s in spans]
    finally:
        capture.remove()


def locate_turn_spans(tokenizer, text: str, contents) -> list[tuple[int, int]]:
    """Token spans for an ordered sequence of turn contents inside a rendered transcript.

    Forward search from a moving cursor — each content is looked up *after* the previous turn's
    end — so a short repeated content (an MCQ filler's bare-letter answer) anchors to its own
    turn, where :func:`locate_token_span`'s last-occurrence rule would put every copy on the
    final one. Boundary-straddling tokens are included, as in :func:`locate_token_span`.
    """
    offsets = tokenizer(text, return_offsets_mapping=True).offset_mapping
    spans, cursor = [], 0
    for ti, content in enumerate(contents):
        try:
            char_start = text.index(content, cursor)
        except ValueError:
            raise ValueError(f"turn {ti} content not found after char {cursor}: "
                             f"{content[:60]!r}") from None
        char_end = char_start + len(content)
        cursor = char_end
        hits = [i for i, (a, b) in enumerate(offsets) if a < char_end and b > char_start]
        if not hits:
            raise ValueError(f"turn {ti} not found in the tokenizer's offset mapping")
        spans.append((hits[0], hits[-1] + 1))
    return spans


def locate_phrase_spans(tokenizer, text: str, phrases, region: tuple[int, int]):
    """Disjoint token spans covering every occurrence of any phrase inside a char region.

    Built for E3c's competitor closure: ``phrases`` are the probe's option names and ``region``
    is the context portion of the rendered transcript (everything before the evidence turn), so
    the probe's own option list is never touched. Occurrences must lie wholly inside the region
    — one straddling the boundary is not context and is excluded. Overlapping token spans (one
    phrase containing another, or adjacent occurrences sharing a boundary token) are merged, so
    the result is always accepted by :class:`SpanAttentionClamp`.
    """
    region_start, region_end = region
    char_spans = []
    for phrase in phrases:
        pos = text.find(phrase, region_start)
        while pos != -1 and pos + len(phrase) <= region_end:
            char_spans.append((pos, pos + len(phrase)))
            pos = text.find(phrase, pos + 1)
    if not char_spans:
        return []

    offsets = tokenizer(text, return_offsets_mapping=True).offset_mapping
    tok_spans = []
    for a, b in char_spans:
        hits = [i for i, (x, y) in enumerate(offsets) if x < b and y > a]
        if hits:
            tok_spans.append((hits[0], hits[-1] + 1))

    tok_spans.sort()
    merged = [tok_spans[0]]
    for a, b in tok_spans[1:]:
        if a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def solve_span_scale(model, input_ids, span, target_share: float, reference_layer: int,
                     layers=None, tol: float = 1e-4, max_iter: int = 60, attention_mask=None):
    """Find the ``scale`` whose clamped span share equals ``target_share`` at ``reference_layer``.

    ``reference_layer`` may be a sequence, in which case the bisection targets the mean share over
    those layers -- the readout the clamp's own all-layer intervention implies.

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
