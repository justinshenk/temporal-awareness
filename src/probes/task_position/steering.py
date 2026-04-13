"""Activation steering using trained ridge probe directions.

Provides a forward hook that pushes the ridge-probe readout of every token in
the residual stream to a target value, by adding a per-token multiple of the
probe direction.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from src.probes.task_position.probes import RidgeProbe


class ProbeSteeringHook:
    """Forward hook that steers a single transformer layer's residual stream
    so that a ridge probe's readout at every token equals a target value.

    Steering math:
        r(x) = x · w + b  (probe readout)
        Δx   = (target - r(x)) · w / ||w||²
        x'   = x + Δx
        => r(x') = target

    Usage:
        hook = ProbeSteeringHook(model, layer=10, probe=probe, target=0.1)
        with hook.steering(), torch.no_grad():
            _ = model(input_ids, use_cache=False)
        hook.remove()
    """

    def __init__(self, model, layer: int, probe: RidgeProbe, target: float):
        self.layer = layer
        self.target = float(target)
        w_np = probe.direction()
        b_np = float(probe._model.intercept_)
        self._w = torch.tensor(w_np, dtype=torch.float32)
        self._b = b_np
        self._w_norm_sq = float((self._w * self._w).sum())
        self.enabled = False
        self._device = None
        self._dtype = None

        self._hook = model.model.layers[layer].register_forward_hook(self._hook_fn)

    def _ensure_on(self, hs: torch.Tensor) -> None:
        if self._device != hs.device or self._dtype != hs.dtype:
            self._device = hs.device
            self._dtype = hs.dtype
            self._w_dev = self._w.to(device=hs.device, dtype=hs.dtype)

    def _hook_fn(self, module, inputs, output):
        if not self.enabled:
            return output
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        # hs shape: (batch, seq, hidden)
        assert hs.shape[0] == 1, f"steering assumes batch=1, got {hs.shape[0]}"
        self._ensure_on(hs)
        # readout[t] = hs[0, t] @ w + b
        readout = (hs[0] @ self._w_dev) + self._b  # (seq,)
        alpha = (self.target - readout) / self._w_norm_sq  # (seq,)
        delta = alpha.unsqueeze(-1) * self._w_dev  # (seq, hidden)
        new_hs = hs.clone()
        new_hs[0] = hs[0] + delta
        if is_tuple:
            return (new_hs,) + output[1:]
        return new_hs

    @contextmanager
    def steering(self):
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def remove(self) -> None:
        self._hook.remove()
