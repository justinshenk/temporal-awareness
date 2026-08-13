"""Lockstep dual-forward residual oracle.

The *oracle* version of all-step residual steering. At each greedy decode step over a context
that the **base** model itself generated, a **LoRA** forward supplies the true residual at a set
of layers and we **overwrite** the base forward's residual there before reading the next token.

Both forwards run on the same base-generated tokens (positions stay aligned) and the context is
base's own (no in-context-correctness leak), so this is the exact oracle of injecting ``α·Wᵀa``
every decode step — with the LoRA's real residual instead of a fitted map. A null here (no layer
recovers capability) proves the steering null was *fundamental*, not estimator-driven.

Degeneracy, by design: overwriting **every** layer forces the base forward's final residual to be
the LoRA's, so its logits become the LoRA's and lockstep decoding reproduces LoRA's greedy
trajectory exactly. That is the *positive control* (validates wiring/position-alignment). The
signal is the **single-layer** sweep, where base layers above the patched one still run with base
weights.

``lockstep_generate`` is deliberately pure orchestration over two callables so it is testable
without PEFT; the GSM8K script wires the real shared base/adapter into them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch


class OverwriteResidualHook:
    """Overwrites the residual-stream output of chosen decoder layers with supplied values.

    Registers on ``model.model.layers[li]`` (Llama layout). Per step the caller sets
    ``values[li]`` to a ``(num_positions, hidden)`` tensor (the LoRA residual captured on the same
    context); the hook replaces the layer output at all current positions while ``injecting()`` is
    active. Layers absent from ``values`` pass through untouched, so the capture forward is never
    disturbed by a stale value.
    """

    def __init__(self, model, layers: Iterable[int]):
        self.layers = list(layers)
        self.values: dict[int, torch.Tensor] = {}
        self.enabled: bool = False
        self._hooks: list = []
        for li in self.layers:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def set_values(self, values: dict[int, torch.Tensor]) -> None:
        self.values = values

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            if not self.enabled or li not in self.values:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            assert hs.shape[0] == 1, f"OverwriteResidualHook assumes batch=1, got {hs.shape[0]}"
            v = self.values[li].to(device=hs.device, dtype=hs.dtype)
            hs = hs.clone()
            hs[0, :, :] = v
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


def blended_injection(a: torch.Tensor, lora_resid: torch.Tensor, W: torch.Tensor,
                      t: float) -> torch.Tensor:
    """Interpolate the injected residual between steering (``t=0``) and the oracle (``t=1``).

    Returns ``a + W·a + t·(δ_true − W·a)`` where ``δ_true = lora_resid − a``. At ``t=0`` this is the
    α=1 steering value ``a + W·a``; at ``t=1`` it collapses to ``lora_resid`` (the true oracle). The
    fidelity sweep over ``t`` shows how exact the injected shift must be to recover capability.
    """
    wa = a @ W.T
    d_true = lora_resid - a
    return a + wa + t * (d_true - wa)


def projected_injection(a: torch.Tensor, lora_resid: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Inject only the part of the true shift inside ``span(V)``: ``a + Π_V(δ_true)``.

    ``V`` is ``(d, k)`` with orthonormal columns (a PCA band of δ). At full rank this returns
    ``lora_resid`` (the full oracle); restricting ``V`` to a band tests whether that band of the
    shift carries the recovery — locating the capability-critical directions in variance-space.
    """
    delta = lora_resid - a
    return a + (delta @ V) @ V.T


def control_injection(a: torch.Tensor, lora_resid: torch.Tensor, mode: str,
                      generator: torch.Generator | None = None,
                      alpha: float = 1.0, prompt_len: int = 0) -> torch.Tensor:
    """Inject a *content-destroyed* shift of the same magnitude: the oracle's empirical floor.

    On a procedure a broken injection scores ~0 — it cannot emit the right integer by accident — so
    the oracle needs no floor. On a k-way register task (commonsense: 2–4 choices) a merely garbled
    intervention still scores at chance, and a partial recovery cannot be distinguished from a
    destroyed one without measuring what "destroyed" looks like. Both modes keep δ's magnitude and
    direction statistics and destroy only its *content*:

    - ``mean_delta`` — replace every position's shift with the trajectory mean, i.e. the best
      possible *fixed vector*. This is the pointwise/register hypothesis stated as an intervention:
      if a register is what installs, averaging over time should cost little; if a temporally dense
      procedure is what installs, it should collapse.
    - ``shuffle_positions`` — apply the true per-token shifts in a permuted order, destroying
      time alignment while preserving the exact multiset of shifts.
    - ``random_matched`` — random directions with the **per-token norms of the true δ**. This is
      the floor an oracle claim actually needs: it answers "would any perturbation of this size do
      as well?", which on a k-way task is a live worry and on GSM8K is not.

    ``prompt_len`` marks where the generated tokens start, and it is **load-bearing**. Statistics
    taken over *all* positions are dominated by the prompt: measured at L20 on commonsense, prompt
    positions outnumber generated ones ~15:1 and their diverse directions cancel, so ‖mean δ‖ over
    all positions is 11 where over generated positions it is 29 — a 2.7× dilution that turned an
    intended control into a no-op. Shuffling has the same defect: with ~100 prompt rows almost every
    swap is prompt↔prompt.

    ``alpha`` scales the injected shift, separating "this direction cannot install the behaviour"
    from "this magnitude breaks the model". Measured on commonsense at L20: α ≤ 3 leaves greedy
    decoding **byte-identical to base**, α ≥ 10 produces gibberish. ``alpha=0`` is exactly the
    unpatched base.
    """
    delta = lora_resid - a
    gen = delta[prompt_len:] if delta.shape[0] > prompt_len else delta
    if mode == "mean_delta":
        return a + alpha * gen.mean(dim=0, keepdim=True).expand_as(delta)
    if mode == "shuffle_positions":
        out = delta.clone()
        perm = torch.randperm(gen.shape[0], generator=generator).to(delta.device)
        out[delta.shape[0] - gen.shape[0]:] = gen[perm]
        return a + alpha * out
    if mode == "random_matched":
        noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta.device)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return a + alpha * noise.to(delta.dtype) * delta.norm(dim=1, keepdim=True)
    raise ValueError(f"unknown control mode: {mode!r} "
                     f"(want 'mean_delta', 'shuffle_positions' or 'random_matched')")


@torch.no_grad()
def lockstep_generate(
    input_ids: torch.Tensor,
    capture_residuals: Callable[[torch.Tensor], dict[int, torch.Tensor]],
    base_logits: Callable[[torch.Tensor], torch.Tensor],
    inject: OverwriteResidualHook,
    inject_layers: Iterable[int],
    max_new: int,
    eos_id: int | None,
    device,
) -> torch.Tensor:
    """Greedy lockstep dual-forward decode; returns the full ``(1, prompt+gen)`` token ids.

    ``capture_residuals(S)`` runs the LoRA forward on ``S`` and returns its residuals
    ``{layer: (positions, hidden)}``. ``base_logits(S)`` runs the base forward on ``S`` with this
    ``inject`` hook active and returns ``(1, seq, vocab)`` logits; the loop sets the hook's values
    (restricted to ``inject_layers``) from the captured residuals before each call.
    """
    inject_layers = list(inject_layers)
    S = input_ids
    for _ in range(max_new):
        caps = capture_residuals(S)
        inject.set_values({li: caps[li] for li in inject_layers})
        nxt = int(base_logits(S)[0, -1].argmax())
        S = torch.cat([S, torch.tensor([[nxt]], device=device)], dim=1)
        if eos_id is not None and nxt == eos_id:
            break
    return S
