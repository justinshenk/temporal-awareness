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


class LinearConditionalSteerHook:
    """Add ``α·(hs @ Aᵀ @ C)`` per position — an input-conditional task-shift prediction.

    The map ``W = Aᵀ C`` is a per-layer closed-form ridge fit (dual form) of each case's
    ICL task-shift from its own activation, stored factored as ``Aᵀ`` ``(d, n)`` and
    ``C`` ``(n, d)`` so the hook is two small matmuls instead of a dense ``d×d``.
    Unlike a fixed steering vector, the added shift depends on the input ``hs``.
    """

    def __init__(self, model, maps, alpha, last_token: bool = False):
        self.maps = {li: (At.to(torch.float32), C.to(torch.float32)) for li, (At, C) in maps.items()}
        self.alpha, self.last_token, self.enabled, self._hooks = alpha, last_token, True, []
        for li in self.maps:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            At, C = (t.to(hs.device, hs.dtype) for t in self.maps[li])
            if self.last_token:
                if hs.shape[1] <= 1:  # skip cached decode steps
                    return output
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + self.alpha * ((hs[:, -1, :] @ At) @ C)
            else:
                hs = hs + self.alpha * ((hs @ At) @ C)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class LinearPrimalSteerHook:
    """Add ``α·(hs @ Wᵀ)`` per position — a dense primal-ridge task-shift prediction.

    Unlike :class:`LinearConditionalSteerHook` (dual form, stored factored as ``Aᵀ``, ``C``),
    this carries a dense ``d×d`` map ``W`` fit by streamed primal ridge over CoT tokens
    (``W·a ≈ δ``). The shift is applied at every position by default (``last_token=False``),
    matching how ``W`` was fit — including single-position decode steps under a KV cache.
    """

    def __init__(self, model, maps: dict[int, torch.Tensor], alpha: float, last_token: bool = False):
        self.maps = {li: W.to(torch.float32) for li, W in maps.items()}
        self.alpha, self.last_token, self.enabled, self._hooks = alpha, last_token, True, []
        for li in self.maps:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            W = self.maps[li].to(hs.device, hs.dtype)
            if self.last_token:
                if hs.shape[1] <= 1:  # skip cached decode steps in function-vector mode
                    return output
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + self.alpha * (hs[:, -1, :] @ W.T)
            else:
                hs = hs + self.alpha * (hs @ W.T)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

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
