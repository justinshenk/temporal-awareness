"""Per-attention-head capture and patching for function-vector (FV) extraction.

Implements the activation hooks Todd et al. (ICLR 2024, "Function Vectors in LLMs") need:
the per-head contribution to the residual stream is ``x_h @ W_O[:, h]^T`` where ``x_h`` is the
slice of the o_proj input belonging to head ``h``. We therefore (a) capture the o_proj input at
the last token (so it can be split into heads), and (b) patch a single head's o_proj input slice
with a fixed mean vector to measure that head's causal indirect effect.

All hooks register on ``model.model.layers[L].self_attn.o_proj`` and work under a batched,
left-padded forward (last token at position -1 for every row).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class PerHeadOprojCapture:
    """Capture each layer's o_proj input at the last token, reshaped to (batch, n_heads, head_dim)."""

    def __init__(self, model, layers, n_heads: int, head_dim: int):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.captured: dict[int, torch.Tensor] = {}
        self._active = False
        self._hooks = []
        for li in layers:
            op = model.model.layers[li].self_attn.o_proj
            self._hooks.append(op.register_forward_pre_hook(self._make(li)))

    def _make(self, li: int):
        def pre(module, args):
            if self._active:
                x = args[0]
                self.captured[li] = (
                    x[:, -1, :].detach().reshape(x.shape[0], self.n_heads, self.head_dim).float()
                )
            return None

        return pre

    @contextmanager
    def capturing(self):
        self._active = True
        try:
            yield
        finally:
            self._active = False

    def clear(self) -> None:
        self.captured = {}

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class HeadMeanPatch:
    """Overwrite one head's o_proj input slice at the last token with a fixed vector (broadcast over batch)."""

    def __init__(self, model, layer: int, head: int, vec: torch.Tensor, head_dim: int):
        self.head = head
        self.head_dim = head_dim
        self.vec = vec.detach()
        op = model.model.layers[layer].self_attn.o_proj
        self._h = op.register_forward_pre_hook(self._pre)

    def _pre(self, module, args):
        x = args[0].clone()
        s = self.head * self.head_dim
        x[:, -1, s : s + self.head_dim] = self.vec.to(dtype=x.dtype, device=x.device)
        return (x,) + tuple(args[1:])

    def remove(self) -> None:
        self._h.remove()


def head_output_vector(o_proj_weight: torch.Tensor, head_input: torch.Tensor, head: int,
                       head_dim: int) -> torch.Tensor:
    """Residual-stream contribution of one head: W_O[:, h_slice] @ head_input (hidden,)."""
    s = head * head_dim
    w = o_proj_weight[:, s : s + head_dim].float()
    return w @ head_input.float()
