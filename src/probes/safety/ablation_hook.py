"""Directional ablation of the residual stream (Arditi et al. 2024 style).

Removes a fixed unit direction from every decoder layer's output at all token
positions: ``h' = h - (h·d̂) d̂``. Used to test whether ablating the finetune-only
"harm" direction restores refusal while preserving the (orthogonal) task direction.

Register on the underlying base model; the hooks fire during a LoRA/PEFT forward
because the PEFT wrapper reuses the same decoder-layer modules.
"""

from __future__ import annotations

import torch


class DirectionalAblationHook:
    """Projects a unit direction out of every decoder layer's residual output."""

    def __init__(self, model, direction: torch.Tensor):
        d = direction.detach().float().flatten()
        self.direction = d / d.norm()
        self.enabled = True
        self._hooks = []
        for layer in model.model.layers:
            self._hooks.append(layer.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            d = self.direction.to(device=hs.device, dtype=hs.dtype)
            proj = (hs * d).sum(dim=-1, keepdim=True)
            hs = hs - proj * d
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
