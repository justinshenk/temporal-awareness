"""N+P-position ablation: the causal complement to the staircase.

The staircase is observational: it shows that target-position probes
do not beat earlier-position probes in domains like code. The natural
causal claim — "information was *flowing* from N+P positions to the
target" — needs an intervention. This module provides it.

Procedure:
  1. Run a normal forward pass and extract target-position activations.
  2. Run a second forward pass with the activations at the EARLIER
     positions (the N+P-analog region) ablated:
       - "zero" ablation: replace with zeros
       - "mean" ablation: replace with the per-channel mean computed
         across all examples (less off-distribution than zero)
  3. Re-extract target-position activations under ablation.
  4. Train probes on both, compare accuracies. A collapse under
     ablation = info was flowing from N+P; persistence = something
     fresh is computed at the target.

Implementation note:
  The hook intervenes on the *INPUT* of each subsequent layer (via the
  pre-hook), so the modification propagates through the residual stream
  to the target position the same way the original information would
  have. Intervening at a single layer L means: from layer L onward, the
  earlier positions are zeroed. Caller picks L.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils.types import ActivationCache, PlanningExample
from .hf_activation_extraction import find_transformer_blocks

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Hook implementation
# ──────────────────────────────────────────────────────────────────────

class _PositionAblator:
    """Forward pre-hook that ablates activations at specified positions.

    Use as a context manager. Hooks are registered on transformer
    blocks starting at `intervention_layer` and propagate through the
    residual stream.

    Modes:
        "zero":     replace activations at given positions with zeros
        "mean":     replace with `mean_replacement` tensor (caller supplies)
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        intervention_layer: int,
        ablate_positions: list[int],
        mode: Literal["zero", "mean"] = "zero",
        mean_replacement: Optional[torch.Tensor] = None,
    ):
        self.blocks = blocks
        self.intervention_layer = intervention_layer
        self.ablate_positions = list(ablate_positions)
        self.mode = mode
        self.mean_replacement = mean_replacement
        self.handles: list = []

        if mode == "mean" and mean_replacement is None:
            raise ValueError("mode='mean' requires mean_replacement tensor (d_model,)")

    def __enter__(self):
        # Register a forward-pre-hook on every block from intervention
        # layer onward. The pre-hook modifies the residual stream INPUT
        # to the block; downstream blocks see the modified residual.
        for li in range(self.intervention_layer, len(self.blocks)):
            handle = self.blocks[li].register_forward_pre_hook(self._make_pre_hook())
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_pre_hook(self):
        positions = self.ablate_positions
        mode = self.mode
        mean = self.mean_replacement

        def pre_hook(module, inputs):
            # inputs is a tuple; the first element is the residual stream
            if not inputs:
                return inputs
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return inputs
            # Some HF blocks pass (hidden, attention_mask, ...) — only
            # mutate the hidden state.
            x = x.clone()  # avoid in-place writes on the original tensor
            seq_len = x.shape[1]
            valid = [p for p in positions if 0 <= p < seq_len]
            if not valid:
                return inputs
            if mode == "zero":
                x[:, valid, :] = 0.0
            elif mode == "mean":
                # Broadcast mean (d_model,) across the selected positions
                x[:, valid, :] = mean.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            return (x,) + inputs[1:]
        return pre_hook


# ──────────────────────────────────────────────────────────────────────
# Activation collection alongside ablation
# ──────────────────────────────────────────────────────────────────────

class _TargetCollector:
    """Captures activations at the OUTPUT of a specific layer.

    Used to grab the target-position activation after ablation has
    propagated through the network.
    """
    def __init__(self, blocks: nn.ModuleList, layer_idx: int):
        self.blocks = blocks
        self.layer_idx = layer_idx
        self.handle = None
        self.captured: Optional[torch.Tensor] = None

    def __enter__(self):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured = hidden.detach().to("cpu")
        self.handle = self.blocks[self.layer_idx].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_with_ablation(
    model: nn.Module,
    tokenizer,
    examples: list[PlanningExample],
    earlier_positions_per_example: list[list[int]],
    intervention_layer: int,
    record_layer: int,
    mode: Literal["zero", "mean"] = "zero",
    mean_replacement: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> list[dict]:
    """Run forward pass with N+P ablation; return target-layer activations.

    Args:
        model: HF causal LM.
        tokenizer: matching tokenizer.
        examples: PlanningExamples to process.
        earlier_positions_per_example: list of token-index lists. Each
            sub-list specifies which positions to ablate for that
            example. Caller supplies these (usually from
            `get_earlier_positions(...)` after target resolution).
        intervention_layer: ablate from this block onward.
        record_layer: capture activations at the OUTPUT of this block.
        mode: "zero" | "mean".
        mean_replacement: required for mode='mean'; shape (d_model,).
        show_progress: tqdm.

    Returns:
        List of dicts: {example_id, token_ids, token_strings,
                       layer_activation_after_ablation: ndarray (seq_len, d_model)}
    """
    if mode == "mean" and mean_replacement is None:
        raise ValueError("mode='mean' requires mean_replacement (np.ndarray of shape (d_model,))")

    device = next(model.parameters()).device
    _, blocks = find_transformer_blocks(model)
    n_layers = len(blocks)
    if not (0 <= intervention_layer < n_layers):
        raise ValueError(f"intervention_layer {intervention_layer} out of range")
    if not (0 <= record_layer < n_layers):
        raise ValueError(f"record_layer {record_layer} out of range")
    if record_layer < intervention_layer:
        logger.warning(
            f"record_layer ({record_layer}) < intervention_layer ({intervention_layer}); "
            f"recorded activations will be unaffected by ablation."
        )

    mean_tensor = None
    if mode == "mean":
        mean_tensor = torch.from_numpy(np.asarray(mean_replacement, dtype=np.float32))

    iterator = examples
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(examples, desc=f"Ablation ({mode}) L{intervention_layer}")
        except ImportError:
            pass

    results: list[dict] = []
    for ex, ablate_positions in zip(iterator, earlier_positions_per_example):
        encoded = tokenizer(ex.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        token_ids = input_ids[0].tolist()
        token_strings = [
            tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            for tid in token_ids
        ]

        with _PositionAblator(
            blocks=blocks,
            intervention_layer=intervention_layer,
            ablate_positions=ablate_positions,
            mode=mode,
            mean_replacement=mean_tensor,
        ), _TargetCollector(blocks=blocks, layer_idx=record_layer) as tc:
            _ = model(input_ids=input_ids)

        if tc.captured is None:
            raise RuntimeError(f"Failed to capture record_layer={record_layer} output")

        # (1, seq_len, d_model) → (seq_len, d_model)
        arr = tc.captured[0].to(torch.float32).numpy()
        results.append({
            "example_id": ex.example_id,
            "token_ids": token_ids,
            "token_strings": token_strings,
            "layer_activation_after_ablation": arr,
            "ablated_positions": list(ablate_positions),
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Compute the per-channel mean (for mean-ablation)
# ──────────────────────────────────────────────────────────────────────

def compute_position_mean(
    caches: list[ActivationCache],
    layer: int,
    positions_per_example: list[list[int]],
) -> np.ndarray:
    """Per-channel mean activation across (example, position) pairs.

    Used as the replacement value for mean-ablation. Computing this
    over the SAME positions we will ablate keeps the replacement
    on-distribution for the ablated region.
    """
    parts: list[np.ndarray] = []
    for cache, positions in zip(caches, positions_per_example):
        for p in positions:
            if 0 <= p < len(cache.token_ids):
                parts.append(cache.activations[layer][p])
    if not parts:
        raise ValueError("No valid positions across any example — cannot compute mean.")
    stack = np.stack(parts, axis=0)  # (N, d_model)
    return stack.mean(axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Top-level orchestration helper used by the main runner
# ──────────────────────────────────────────────────────────────────────

def run_np_ablation_experiment(
    model,
    tokenizer,
    examples: list[PlanningExample],
    baseline_caches: list[ActivationCache],
    earlier_positions_per_example: list[list[int]],
    intervention_layer: int,
    record_layer: int,
    modes: tuple[str, ...] = ("zero", "mean"),
) -> dict:
    """Run zero and/or mean ablation, return target-layer activations.

    Returns a dict:
        {
          "zero": list[dict] (per-example ablation results),
          "mean": list[dict] (...),
        }

    The runner then trains probes on these and compares to baseline.
    """
    out: dict = {}
    for mode in modes:
        mean_repl = None
        if mode == "mean":
            mean_repl = compute_position_mean(
                caches=baseline_caches,
                layer=intervention_layer,
                positions_per_example=earlier_positions_per_example,
            )
        out[mode] = extract_with_ablation(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            earlier_positions_per_example=earlier_positions_per_example,
            intervention_layer=intervention_layer,
            record_layer=record_layer,
            mode=mode,
            mean_replacement=mean_repl,
        )
    return out


__all__ = [
    "extract_with_ablation",
    "compute_position_mean",
    "run_np_ablation_experiment",
]
