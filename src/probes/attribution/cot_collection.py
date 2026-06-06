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


def cot_token_slice(prompt_len: int, full_len: int) -> slice:
    """Positions of the generated CoT tokens within the teacher-forced sequence."""
    if not 0 <= prompt_len <= full_len:
        raise ValueError(f"need 0 <= prompt_len <= full_len, got {prompt_len}, {full_len}")
    return slice(prompt_len, full_len)


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
