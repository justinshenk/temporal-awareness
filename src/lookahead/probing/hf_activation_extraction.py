"""HuggingFace-based activation extractor (replaces TransformerLens for new models).

Why a second extractor?
    TransformerLens (used by the workshop's `activation_extraction.py`) does
    not yet support Gemma3, Qwen3, Llama-3.3-70B and several other newer
    architectures Maar et al. test. To reproduce their model coverage we
    need a raw-HuggingFace path that works on any AutoModelForCausalLM.

    This module produces the same ActivationCache interface as the TL
    extractor, so downstream probing / baseline / patching code is
    unchanged. Pick the backend per model.

What is extracted:
    Residual-stream activations at the *output of each transformer block*
    (post-MLP, post-residual-add). This is what `hook_resid_post`
    captures in TransformerLens. For raw HF models, we hook each block's
    forward and capture its output tensor.

Layer-name auto-detection:
    Walks the model tree and finds the `nn.ModuleList` of transformer
    blocks. Known good for:
      - GPT-2 family            (transformer.h)
      - GPT-NeoX / Pythia       (gpt_neox.layers)
      - Llama 2/3 + variants    (model.layers)
      - Gemma 1/2/3             (model.layers)
      - Qwen 1/2/3              (model.layers)
      - Mistral / Mixtral       (model.layers)
      - SantaCoder / StarCoder  (transformer.h)
      - CodeLlama               (model.layers)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils.types import ActivationCache, PlanningExample

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Layer discovery — works across all current decoder LLMs
# ──────────────────────────────────────────────────────────────────────

def find_transformer_blocks(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Find the ModuleList of transformer blocks inside the model.

    Tries known dotted paths first (fast), falls back to a generic
    search if none match.

    Returns:
        (dotted_path, module_list)
    """
    known_paths = (
        "transformer.h",          # GPT-2, GPT-J, StarCoder, SantaCoder, BLOOM
        "gpt_neox.layers",        # Pythia, GPT-NeoX
        "model.layers",           # Llama, Gemma, Qwen, Mistral, CodeLlama, Falcon-3
        "model.decoder.layers",   # OPT
        "transformer.blocks",     # MPT
    )
    for path in known_paths:
        node = model
        ok = True
        for part in path.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok and isinstance(node, nn.ModuleList) and len(node) > 0:
            return path, node

    # Generic fallback: walk the tree and find the longest ModuleList of
    # things that look like transformer blocks.
    best_path = None
    best_modlist = None
    best_len = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            # Heuristic: the block list is the longest one in the model
            best_path = name
            best_modlist = mod
            best_len = len(mod)
    if best_modlist is not None:
        logger.warning(
            f"Using fallback layer discovery: {best_path} "
            f"({best_len} blocks). Verify this is correct."
        )
        return best_path, best_modlist

    raise RuntimeError(
        "Could not find transformer blocks in model. "
        "Please specify them manually."
    )


# ──────────────────────────────────────────────────────────────────────
# Hook-based activation capture
# ──────────────────────────────────────────────────────────────────────

class _ActivationCollector:
    """Registers forward hooks on requested transformer blocks.

    Use as a context manager so hooks are always removed on exit even
    when exceptions are raised inside the forward pass.
    """

    def __init__(self, blocks: nn.ModuleList, layer_indices: list[int]):
        self.blocks = blocks
        self.layer_indices = list(layer_indices)
        self.handles: list = []
        self.captured: dict[int, torch.Tensor] = {}

    def __enter__(self):
        for li in self.layer_indices:
            handle = self.blocks[li].register_forward_hook(self._make_hook(li))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Transformer blocks return either a Tensor or a tuple
            # (hidden, ...). The first element is the residual stream.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Detach, move to CPU once, drop grads / batch dim handling later
            self.captured[layer_idx] = hidden.detach().to("cpu")
        return hook


# ──────────────────────────────────────────────────────────────────────
# Public API — matches src/lookahead/probing/activation_extraction.py
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_activations_all_positions(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layers: Optional[list[int]] = None,
    include_logits: bool = False,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> ActivationCache:
    """Extract residual stream activations at every position and layer.

    Args:
        model: Any AutoModelForCausalLM. No special architecture support
            needed — uses forward hooks on the transformer block list.
        tokenizer: The matching tokenizer.
        prompt: Input text.
        layers: Layer indices to extract (None = all blocks).
        include_logits: If True, also returns the full logits tensor.
        device: Where to run the forward pass. Defaults to the model's
            device.
        dtype: Override for the forward dtype (mostly for fp16 → fp32
            casting; activations are stored as float32 numpy regardless).

    Returns:
        ActivationCache with the same schema as the TransformerLens
        extractor, so downstream code is interchangeable.
    """
    if device is None:
        device = next(model.parameters()).device

    blocks_path, blocks = find_transformer_blocks(model)
    n_layers = len(blocks)
    if layers is None:
        layers = list(range(n_layers))
    else:
        # Validate
        for li in layers:
            if li < 0 or li >= n_layers:
                raise ValueError(
                    f"Layer index {li} out of range [0, {n_layers})."
                )

    # Tokenize (no padding, single sequence)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    token_ids = input_ids[0].tolist()
    token_strings = [
        tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        for tid in token_ids
    ]

    # Forward pass with hooks
    with _ActivationCollector(blocks, layers) as collector:
        outputs = model(input_ids=input_ids)

    # Convert captured tensors to numpy (drop batch dim)
    activations: dict[int, np.ndarray] = {}
    for li in layers:
        if li not in collector.captured:
            raise RuntimeError(
                f"Layer {li} was not captured. "
                f"Block list path used: {blocks_path}"
            )
        # Captured shape: (1, seq_len, d_model). Drop batch dim, to float32.
        arr = collector.captured[li][0].to(torch.float32).numpy()
        activations[li] = arr

    logits_np = None
    if include_logits:
        logits_np = outputs.logits[0].to(torch.float32).cpu().numpy()

    return ActivationCache(
        example_id="",  # caller sets this
        token_ids=token_ids,
        token_strings=token_strings,
        activations=activations,
        logits=logits_np,
    )


def extract_activations_batch(
    model: nn.Module,
    tokenizer,
    examples: list[PlanningExample],
    layers: Optional[list[int]] = None,
    include_logits: bool = False,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> list[ActivationCache]:
    """Extract activations for a list of examples (one at a time).

    Variable-length prompts make naive batching contaminate activations
    via padding; we process serially, which matches the TL extractor and
    keeps the result format identical.
    """
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(examples, desc="Extracting activations (HF)")
        except ImportError:
            iterator = examples
    else:
        iterator = examples

    caches: list[ActivationCache] = []
    for ex in iterator:
        cache = extract_activations_all_positions(
            model=model,
            tokenizer=tokenizer,
            prompt=ex.prompt,
            layers=layers,
            include_logits=include_logits,
            device=device,
        )
        cache.example_id = ex.example_id
        caches.append(cache)
    return caches


# ──────────────────────────────────────────────────────────────────────
# Helper: configure a sensible layer sample for a given model size
# ──────────────────────────────────────────────────────────────────────

def default_layer_sample(n_layers: int, n_samples: int = 6) -> list[int]:
    """Workshop's 6-layer sampling pattern: [0, n/6, n/3, n/2, 2n/3, n-1].

    For very small models (< 12 layers), uses every other layer instead.
    """
    if n_layers < n_samples * 2:
        return list(range(0, n_layers, max(1, n_layers // n_samples)))
    return [
        0,
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        (2 * n_layers) // 3,
        n_layers - 1,
    ]


def maar_layer_sample(n_layers: int) -> list[int]:
    """Maar et al.'s layer range: roughly 10%-85% depth, dense sampling.

    Used for rhyme/QA where steering effects are concentrated in the
    middle band of the network.
    """
    lo = max(1, int(0.10 * n_layers))
    hi = max(lo + 1, int(0.85 * n_layers))
    return list(range(lo, hi + 1))


@contextmanager
def model_eval_mode(model: nn.Module):
    """Ensure the model is in eval mode; restore previous state on exit."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


__all__ = [
    "find_transformer_blocks",
    "extract_activations_all_positions",
    "extract_activations_batch",
    "default_layer_sample",
    "maar_layer_sample",
    "model_eval_mode",
]
