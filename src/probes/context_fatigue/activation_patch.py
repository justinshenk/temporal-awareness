"""Counterfactual activation patching for E7.

E7 asks whether intervening-context hidden states *causally carry* which format instruction
was given: states are harvested from a donor forward (system prompt A) and substituted into a
recipient forward (system prompt B) at chosen (layer, position) sites, with the readout taken
downstream. This module is the instrument; the driver owns condition construction, span
choice, and scoring.

**Mechanism.** A forward hook on each patched decoder layer replaces that layer's output
hidden states at the patched positions with the donor's states for the same sites. Nothing is
recomputed or reimplemented, so:

- patching every layer at every position IS the donor forward from the last layer onward —
  the final norm and lm_head see only donor states, and the logits match the donor's exactly;
- a self-patch writes back bit-identical values, so the A→A control is exact, not
  approximate — the driver preflight asserts ``max|Δ| == 0.0``;
- positions strictly before the first patched position are causally upstream of every
  patched site and come out untouched.

Donor and recipient must be position-aligned: the hook refuses to run when the recipient's
sequence length differs from the donor's (§5 of the brief — abort loudly, never truncate or
pad silently). Alignment of token ids outside the counterfactual span is the driver's
assertion, since only the driver knows which span is counterfactual.
"""

from __future__ import annotations

import torch


def _normalise_spans(span):
    """Sorted, validated ``[(start, end), ...]`` from one span or a list of spans."""
    spans = span if isinstance(span[0], (tuple, list)) else [span]
    spans = sorted((int(a), int(b)) for a, b in spans)
    for a, b in spans:
        if b <= a:
            raise ValueError(f"span must be non-empty, got {(a, b)}")
    for (_, b1), (a2, _) in zip(spans, spans[1:]):
        if a2 < b1:
            raise ValueError(f"spans overlap at {b1}..{a2}; merge them before patching")
    return spans


def capture_layer_states(model, input_ids, layers=None, attention_mask=None) -> dict:
    """Donor harvest: ``{layer: [seq, hidden]}`` of each decoder layer's output states.

    One batch-1 forward with temporary hooks; the hooks are removed even if the forward
    raises. States are detached clones, so they survive the donor context being freed and
    cannot backprop into the donor graph.
    """
    if input_ids.shape[0] != 1:
        raise ValueError(f"patching is batch-1 by design, got batch {input_ids.shape[0]}")
    layers = list(range(len(model.model.layers))) if layers is None else list(layers)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[li] = hidden[0].detach().clone()
        return hook

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        for h in hooks:
            h.remove()
    return captured


class SpanActivationPatch:
    """Substitute donor hidden states at ``layers`` × ``span`` during recipient forwards.

    ``donor_states`` is :func:`capture_layer_states` output; ``span`` is a ``(start, end)``
    pair or a list of disjoint pairs (token positions, end-exclusive); ``layers`` defaults to
    every layer the donor provides. Use as a context manager so the hooks come off even if
    the driver raises mid-item::

        with SpanActivationPatch(model, donor_states, span=(a, b), layers=[8, 9]):
            ...
    """

    def __init__(self, model, donor_states, span, layers=None):
        self.spans = _normalise_spans(span)
        self.layers = sorted(donor_states) if layers is None else list(layers)
        missing = [li for li in self.layers if li not in donor_states]
        if missing:
            raise ValueError(f"donor states missing for layers {missing}")
        self.donor = {li: donor_states[li] for li in self.layers}

        seq_lens = {tuple(v.shape) for v in self.donor.values()}
        if len(seq_lens) > 1:
            raise ValueError(f"donor layers disagree on shape: {sorted(seq_lens)}")
        self.donor_len = next(iter(self.donor.values())).shape[0]
        if self.spans[-1][1] > self.donor_len:
            raise ValueError(f"span {self.spans[-1]} exceeds donor length {self.donor_len}")
        self.index = torch.cat([torch.arange(a, b) for a, b in self.spans])

        self.hooks = [model.model.layers[li].register_forward_hook(self._make_hook(li))
                      for li in self.layers]

    def _make_hook(self, li):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[0] != 1:
                raise RuntimeError(f"patching is batch-1 by design, got batch "
                                   f"{hidden.shape[0]} at layer {li}")
            if hidden.shape[1] != self.donor_len:
                raise RuntimeError(
                    f"donor/recipient length mismatch at layer {li}: donor {self.donor_len} "
                    f"vs recipient {hidden.shape[1]} — refusing to patch misaligned states")
            donor = self.donor[li]
            idx = self.index.to(hidden.device)
            patched = hidden.clone()
            patched[:, idx, :] = donor[idx].to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            return patched
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False
