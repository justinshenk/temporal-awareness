"""Forward hooks for activation patching experiments (v6)."""

from __future__ import annotations

from contextlib import contextmanager

import torch


class DirectionPatchHook:
    """Adds a position-specific delta along a learned direction at selected token positions.

    Unlike ProbeSteeringHook (v2), this hook:
      - Applies at only a given set of token positions, not every token
      - Uses per-position delta vectors (precomputed), not a uniform target
    """

    def __init__(self, model, layer: int):
        self.layer = layer
        self.patch_map: dict[int, torch.Tensor] = {}
        self.enabled = False
        self._device = None
        self._dtype = None
        self._hook = model.model.layers[layer].register_forward_hook(self._hook_fn)

    def set_patches(self, patches: dict[int, torch.Tensor]) -> None:
        self.patch_map = patches

    def _hook_fn(self, module, inputs, output):
        if not self.enabled or not self.patch_map:
            return output
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        assert hs.shape[0] == 1, f"batch=1 required, got {hs.shape[0]}"
        for pos, delta in self.patch_map.items():
            if delta.device != hs.device or delta.dtype != hs.dtype:
                delta = delta.to(device=hs.device, dtype=hs.dtype)
                self.patch_map[pos] = delta
            hs[0, pos] = hs[0, pos] + delta
        if is_tuple:
            return (hs,) + output[1:]
        return hs

    @contextmanager
    def patching(self):
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def remove(self) -> None:
        self._hook.remove()


class FullResidualPatchHook:
    """Replaces the residual at selected token positions with stored values."""

    def __init__(self, model, layer: int):
        self.layer = layer
        self.patch_map: dict[int, torch.Tensor] = {}
        self.enabled = False
        self._hook = model.model.layers[layer].register_forward_hook(self._hook_fn)

    def set_patches(self, patches: dict[int, torch.Tensor]) -> None:
        self.patch_map = patches

    def _hook_fn(self, module, inputs, output):
        if not self.enabled or not self.patch_map:
            return output
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        assert hs.shape[0] == 1
        for pos, value in self.patch_map.items():
            if value.device != hs.device or value.dtype != hs.dtype:
                value = value.to(device=hs.device, dtype=hs.dtype)
                self.patch_map[pos] = value
            hs[0, pos] = value
        if is_tuple:
            return (hs,) + output[1:]
        return hs

    @contextmanager
    def patching(self):
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def remove(self) -> None:
        self._hook.remove()
