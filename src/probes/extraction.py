"""Per-token residual-stream extraction hooks.

Captures the residual stream output at specified layers for every token in a
forward pass. Produces `dict[int, Tensor]` keyed by layer index, each tensor
shape `(seq_len, hidden_dim)` on CPU in float32.
"""

from __future__ import annotations

from typing import Sequence

import torch


class PerTokenResidualCapture:
    """Captures per-token residual streams at a set of layers.

    Usage:
        capture = PerTokenResidualCapture(model, layers=[0, 10, 20, 30, 41])
        capture.enabled = True
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        capture.enabled = False
        acts = capture.captured  # dict[int, Tensor(seq_len, hidden_dim)]
        capture.clear()
        ...
        capture.remove()  # release hooks when done
    """

    def __init__(self, model, layers: Sequence[int]):
        self.layers = list(layers)
        self.captured: dict[int, torch.Tensor] = {}
        self.enabled: bool = False
        self._hooks: list = []

        for li in self.layers:
            hook = model.model.layers[li].register_forward_hook(self._make_hook(li))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return
            hs = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = hs[0].detach().float().cpu()

        return hook_fn

    def clear(self) -> None:
        self.captured = {}

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
