"""LoReFT: an input-conditional affine edit in a learned orthonormal subspace.

``LoReFT(h) = h + (Wh + b − hR)Rᵀ`` (Wu et al. 2024) — the same conditional-linear-map family as
the ridge map ``W·a`` (refusal frontier) and the same subspace machinery as DAS, but the injected
*value* is learned by task loss instead of taken from an expert's δ. ``R`` reuses
:class:`OrthoSubspace` (orthonormal every step via QR), so the commonsense reproduction shares its
intervention code path with the DAS null on GSM8K.

The paper's commonsense recipe intervenes on the first/last 7 *prompt* tokens of every layer with
weights shared across positions: one :class:`LoReFTIntervention` per layer, applied by
:class:`PositionEditHook` at the :func:`intervention_locations` of each sample (offset by left
padding). Only geometry/edit glue lives here (unit-testable on a tiny Llama); model loading,
tokenization and the train/eval loops live in the commonsense scripts.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from src.probes.attribution.das_subspace import OrthoSubspace


class LoReFTIntervention(nn.Module):
    """``h + (source(h) − hR)Rᵀ`` with trainable orthonormal ``R`` (d×r) and ``source``: d→r.

    When ``source(h) = hR`` the edit is the identity; at ``r=d`` it rewrites ``h`` entirely to
    ``source(h)Rᵀ``. The shift always lies in span(R). Trainable budget is ``2dr + r`` (the paper's
    commonsense config, d=4096 r=8, is 65 544 params per layer).
    """

    def __init__(self, d: int, r: int, seed: int = 0):
        super().__init__()
        self.subspace = OrthoSubspace(d, r, seed=seed)
        self.source = nn.Linear(d, r)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        R = self.subspace()
        return h + (self.source(h) - h @ R) @ R.T


def intervention_locations(plen: int, n_prefix: int, n_suffix: int, offset: int = 0) -> list[int]:
    """Sorted unique f{n_prefix}+l{n_suffix} positions of a ``plen``-token prompt, plus ``offset``.

    ``offset`` is the sample's left-padding length, so the returned indices address the padded
    batch row directly. Short prompts collapse the overlap (plen ≤ prefix+suffix ⇒ every prompt
    position, once).
    """
    if plen < 1:
        raise ValueError(f"need plen >= 1, got {plen}")
    head = range(min(plen, n_prefix))
    tail = range(max(0, plen - n_suffix), plen)
    return [p + offset for p in sorted(set(head) | set(tail))]


def response_labels(input_ids: torch.Tensor, response_starts: torch.Tensor) -> torch.Tensor:
    """LM labels supervising only the response: ``−100`` strictly before each sample's start.

    With left padding the pads precede the prompt, so masking ``[:start)`` covers both; nothing
    after ``start`` is touched (the response's eos keeps its label even when pad_id == eos_id).
    """
    labels = input_ids.clone()
    mask = torch.arange(input_ids.shape[1]) < response_starts.unsqueeze(1)
    labels[mask] = -100
    return labels


class PositionEditHook:
    """Applies a per-layer intervention module to chosen positions of the residual stream.

    Registers on ``model.model.layers[li]`` (Llama layout) for each layer in ``interventions``.
    The caller sets ``locations`` to a ``(batch, n_pos)`` index tensor (ragged location lists
    padded by repeating an index — duplicates rewrite the same edited value, computed once from
    the pre-edit row). Edits run in f32 and cast back, with gradients flowing to the intervention.

    Decode steps pass through untouched: when the running forward is shorter than the largest
    location (a 1-token forward on a KV cache), the prompt positions aren't in this forward, so
    the hook skips — the edit lands exactly once, at prefill, like LoReFT.
    """

    def __init__(self, model, interventions: dict[int, nn.Module]):
        self.interventions = dict(interventions)
        self.locations: torch.Tensor | None = None
        self.enabled: bool = False
        self._hooks = [model.model.layers[li].register_forward_hook(self._make_hook(li))
                       for li in self.interventions]

    def set_locations(self, locations: torch.Tensor) -> None:
        self.locations = locations

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled or self.locations is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            loc = self.locations.to(hs.device)
            if hs.shape[1] <= int(loc.max()):
                return output
            bidx = torch.arange(hs.shape[0], device=hs.device).unsqueeze(1).expand_as(loc)
            edited = self.interventions[li](hs[bidx, loc].float()).to(hs.dtype)
            hs = hs.clone()
            hs[bidx, loc] = edited
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        return hook_fn

    @contextmanager
    def injecting(self):
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
