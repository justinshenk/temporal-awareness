"""Assemble LoRA / ICL activation-shift sets from captured residuals.

Both the ICL shift and the LoRA shift are measured against the *same* baseline
(base model, clean prompt) at the prediction site (the final token before the
model emits its answer letter), per layer:

    icl_shift[i]  = resid(base, case i WITH context)  - resid(base, case i clean)
    lora_shift[i] = resid(lora, case i clean)         - resid(base, case i clean)

This module holds the model-free assembly logic so it is unit-testable on CPU;
the actual model forward passes live in ``scripts/lora_icl/extract_shifts.py``.
"""

from __future__ import annotations

import numpy as np


def _to_2d_array(hidden_states) -> np.ndarray:
    """Coerce a ``(seq, d)`` tensor/array to a float64 NumPy array."""
    if hasattr(hidden_states, "detach"):  # torch.Tensor
        hidden_states = hidden_states.detach().float().cpu().numpy()
    arr = np.asarray(hidden_states, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (seq, d) hidden states, got shape {arr.shape}")
    return arr


def last_token_residual(capture: dict[int, object]) -> dict[int, np.ndarray]:
    """Reduce a ``{layer: (seq, d)}`` capture to the final-token vector per layer."""
    if not capture:
        raise ValueError("capture is empty")
    return {layer: _to_2d_array(hs)[-1] for layer, hs in capture.items()}


def stack_shift_set(
    reference: list[dict[int, np.ndarray]],
    variant: list[dict[int, np.ndarray]],
    layer: int,
) -> np.ndarray:
    """Build a ``(n, d)`` shift set ``variant[i] - reference[i]`` at one layer.

    Args:
        reference: per-example ``{layer: vector}`` baseline residuals.
        variant: per-example ``{layer: vector}`` residuals to subtract the
            baseline from. Must be index-aligned and the same length as
            ``reference``.
        layer: which layer's residual to difference.
    """
    if len(reference) != len(variant):
        raise ValueError(
            f"reference ({len(reference)}) and variant ({len(variant)}) must align"
        )
    if not reference:
        raise ValueError("shift set is empty")
    rows = []
    for i, (ref, var) in enumerate(zip(reference, variant)):
        if layer not in ref or layer not in var:
            raise ValueError(f"example {i} missing layer {layer}")
        r = np.asarray(ref[layer], dtype=np.float64)
        v = np.asarray(var[layer], dtype=np.float64)
        if r.shape != v.shape:
            raise ValueError(
                f"example {i}: shape mismatch ref {r.shape} vs variant {v.shape}"
            )
        rows.append(v - r)
    return np.stack(rows, axis=0)
