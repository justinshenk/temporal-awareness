"""Additive activation steering: add a fixed per-layer vector to the residual stream.

Used to test the *sufficiency* of a shift component — steer the base model by the
parallel-only or orthogonal-only part of the LoRA shift and observe behavior.
Register on the underlying base model (hooks fire under a PEFT forward too).
"""

from __future__ import annotations

import torch


class AdditionSteeringHook:
    """Adds ``vectors[layer]`` to the output of each named decoder layer."""

    def __init__(self, model, vectors: dict[int, torch.Tensor]):
        self.vectors = {li: v.detach().float() for li, v in vectors.items()}
        self.enabled = True
        self._hooks = []
        for li in self.vectors:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            v = self.vectors[layer_idx].to(device=hs.device, dtype=hs.dtype)
            hs = hs + v
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
