"""Assemble per-token ``(a_t, δ_t)`` blocks from two teacher-forced residual captures.

Model-free glue between :class:`PerTokenResidualCapture` and :class:`GramAccumulator`
(mirrors the script/library split of ``shift_extraction.py``): the model forwards live
in the collection script, the alignment logic lives here so it is unit-testable.

``a_t`` = base residual at a CoT token; ``δ_t`` = LoRA residual − base residual at the
same position. Both passes run on the identical ``prompt+CoT`` token sequence, so the
two captures are aligned position-for-position and the CoT slice is ``[prompt_len:]``.
"""

from __future__ import annotations

import torch


def cot_token_slice(prompt_len: int, full_len: int, positions: str = "cot") -> slice:
    """The window of positions the ridge map is fit on, within the teacher-forced sequence.

    ``"cot"`` (default) keeps only the generated tokens — the original GSM8K choice, where the CoT
    is both the thing being installed and the bulk of the sequence (~250 of ~400 positions).

    ``"all"`` keeps the prompt too, so the fit distribution matches where the map is **applied**:
    ``LinearPrimalSteerHook`` steers every position, prompt included. That mismatch is mild on
    GSM8K and severe on a task whose supervised target is ~6 tokens against a ~97-token prompt,
    where ~94% of the steered positions are off the fit distribution. It also multiplies the fit
    set by ~17× there. The prompt is not a free ride: measured per-token ‖δ‖ at L20 on commonsense
    is 27.6–30.0 on prompt positions against 41.5–43.1 on generated ones, so those positions carry
    real shift — the earlier "dilution" result (‖mean δ‖ 29 → 11) is about prompt directions
    cancelling one another, not about their being small.
    """
    if not 0 <= prompt_len <= full_len:
        raise ValueError(f"need 0 <= prompt_len <= full_len, got {prompt_len}, {full_len}")
    if positions == "cot":
        return slice(prompt_len, full_len)
    if positions == "all":
        return slice(0, full_len)
    raise ValueError(f"unknown fit window: {positions!r} (want 'cot' or 'all')")


def assemble_blocks(base_capture: dict[int, torch.Tensor],
                    lora_capture: dict[int, torch.Tensor],
                    layer: int, tok_slice: slice) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(a_block, d_block)`` float64 ``(n_cot, d)`` for one layer over the CoT slice."""
    if layer not in base_capture or layer not in lora_capture:
        raise ValueError(f"layer {layer} missing from a capture")
    base = base_capture[layer]
    lora = lora_capture[layer]
    if base.shape != lora.shape:
        raise ValueError(f"capture shape mismatch at layer {layer}: {tuple(base.shape)} vs {tuple(lora.shape)}")
    a = base[tok_slice].to(torch.float64)
    d = lora[tok_slice].to(torch.float64) - a
    return a, d
