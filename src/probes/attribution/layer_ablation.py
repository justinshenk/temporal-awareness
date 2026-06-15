"""Identity-ablation of decoder layers.

Makes chosen decoder blocks no-ops: the layer returns its *input* hidden state, so its attention+MLP
contribution to the residual stream is removed while all other layers run normally. This measures how
much downstream computation is load-bearing — combined with an L20 activation patch, it asks whether
base's layers 21–31 transform the patched state into the answer (recovery collapses as they are
ablated) or merely transcribe an answer already present at L20 (recovery survives ablation).

Registers a forward hook on ``model.model.layers[li]`` (Llama layout), reading the block input from the
hook's ``inputs`` so no separate pre-hook is needed. Batch=1, matching
:class:`~src.probes.attribution.lockstep_oracle.OverwriteResidualHook`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


class IdentityAblationHook:
    """Turns chosen decoder layers into identity maps while :meth:`ablating` is active.

    The hook replaces each selected layer's output hidden state with the hidden state that entered the
    layer (``inputs[0]``), so the block contributes nothing to the residual stream. Non-selected layers
    and inactive periods pass through untouched. The active set is mutable via :meth:`set_layers`, so a
    cumulative sweep ({31}, {30,31}, …) reuses one hook registration.
    """

    def __init__(self, model, layers: Iterable[int]):
        self.active: set[int] = set()
        self.enabled: bool = False
        self._hooks: list = []
        for li in layers:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def set_layers(self, layers: Iterable[int]) -> None:
        self.active = set(layers)

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled or li not in self.active:
                return output
            hs_in = inputs[0]
            assert hs_in.shape[0] == 1, f"IdentityAblationHook assumes batch=1, got {hs_in.shape[0]}"
            return (hs_in,) + tuple(output[1:]) if isinstance(output, tuple) else hs_in

        return hook_fn

    @contextmanager
    def ablating(self):
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
