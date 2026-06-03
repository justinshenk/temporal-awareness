"""Additive activation steering: add a fixed per-layer vector to the residual stream.

Used to test the *sufficiency* of a shift component — steer the base model by the
parallel-only or orthogonal-only part of the LoRA shift and observe behavior.
Register on the underlying base model (hooks fire under a PEFT forward too).
"""

from __future__ import annotations

import torch


class AdditionSteeringHook:
    """Adds ``vectors[layer]`` to the output of each named decoder layer.

    ``last_token=True`` steers only the final prefill position (the generation site) and
    skips cached decode steps — the function-vector-style application that leaves the
    per-token content of earlier positions untouched.
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor], last_token: bool = False):
        self.vectors = {li: v.detach().float() for li, v in vectors.items()}
        self.last_token = last_token
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
            if self.last_token:
                if hs.shape[1] <= 1:  # skip cached decode steps
                    return output
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + v
            else:
                hs = hs + v
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class NormPreservingSteeringHook:
    """Add ``vectors[layer]`` then rescale each position back to its original norm.

    Diagnoses/fixes the high-magnitude 'overwrite' failure: ``a' = ‖a‖ · (a+v)/‖a+v‖``.
    If the high-α accuracy cliff vanishes under this, the cliff was a norm effect.
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor]):
        self.vectors = {li: v.detach().float() for li, v in vectors.items()}
        self.enabled = True
        self._hooks = [model.model.layers[li].register_forward_hook(self._make_hook(li))
                       for li in self.vectors]

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            v = self.vectors[li].to(device=hs.device, dtype=hs.dtype)
            orig = hs.norm(dim=-1, keepdim=True)
            steered = hs + v
            hs = steered / (steered.norm(dim=-1, keepdim=True) + 1e-6) * orig
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class ProjectionSteeringHook:
    """Add ``vectors[layer]`` then project back onto the natural activation subspace.

    ``bases[layer] = (mean (d,), V (d,k))`` is the top-k PCA of natural activations.
    ``a' = mean + (a+v-mean) V Vᵀ`` removes the off-manifold component of the steer. If the
    high-α cliff vanishes under this, the cliff was an off-manifold effect.
    """

    def __init__(self, model, vectors, bases):
        self.vectors = {li: v.detach().float() for li, v in vectors.items()}
        self.bases = {li: (m.detach().float(), V.detach().float()) for li, (m, V) in bases.items()}
        self.enabled = True
        self._hooks = [model.model.layers[li].register_forward_hook(self._make_hook(li))
                       for li in self.vectors]

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            v = self.vectors[li].to(hs.device, hs.dtype)
            mean, V = (t.to(hs.device, hs.dtype) for t in self.bases[li])
            centered = hs + v - mean
            hs = mean + (centered @ V) @ V.t()
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
