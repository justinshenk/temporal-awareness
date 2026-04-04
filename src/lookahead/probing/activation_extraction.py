"""Activation extraction at every token position and layer.

Unlike the temporal probes (which extract only the last-token activation),
lookahead planning detection requires activations at EVERY position.

The key question: at position t, do activations already encode information
about structure at position t+k?

This module handles:
1. Running a forward pass and caching all residual stream activations
2. Organizing activations by (layer, position) for efficient probing
3. Computing logit-lens predictions at each position (what the model
   "would predict" at each intermediate layer)
4. Token-level metadata (which tokens correspond to targets, anchors, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch

from ..utils.types import ActivationCache, PlanningExample

logger = logging.getLogger(__name__)


def extract_activations_all_positions(
    model,
    tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    include_logits: bool = False,
    device: str = "cpu",
) -> ActivationCache:
    """Extract residual stream activations at every position and layer.
    
    Args:
        model: A TransformerLens HookedTransformer model
        tokenizer: The model's tokenizer
        prompt: Input text
        layers: Which layers to extract (None = all layers)
        include_logits: Whether to also cache per-position logits
        device: Device for computation
        
    Returns:
        ActivationCache with activations[layer] of shape (seq_len, d_model)
    """
    # Tokenize
    tokens = model.to_tokens(prompt, prepend_bos=True)
    token_ids = tokens[0].tolist()
    token_strings = [model.to_string(torch.tensor([tid])) for tid in token_ids]
    
    n_layers = model.cfg.n_layers
    if layers is None:
        layers = list(range(n_layers))
    
    # Build names filter for residual stream at requested layers
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    
    def names_filter(name: str) -> bool:
        return name in hook_names
    
    # Run forward pass with cache
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
    
    # Extract activations: (seq_len, d_model) per layer
    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        act = cache[hook_name][0]  # remove batch dim → (seq_len, d_model)
        activations[layer] = act.cpu().numpy()
    
    # Optionally extract per-position logits
    logits_np = None
    if include_logits:
        logits_np = logits[0].cpu().numpy()  # (seq_len, vocab_size)
    
    return ActivationCache(
        example_id="",  # caller sets this
        token_ids=token_ids,
        token_strings=token_strings,
        activations=activations,
        logits=logits_np,
    )


def extract_activations_batch(
    model,
    tokenizer,
    examples: list[PlanningExample],
    layers: list[int] | None = None,
    include_logits: bool = False,
    device: str = "cpu",
    show_progress: bool = True,
) -> list[ActivationCache]:
    """Extract activations for a batch of examples.
    
    Processes examples one at a time (variable-length prompts can't be
    trivially batched without padding, which would contaminate activations).
    
    Args:
        model: TransformerLens HookedTransformer
        tokenizer: Model tokenizer
        examples: Planning examples to process
        layers: Layers to extract (None = all)
        include_logits: Cache logits too
        device: Device
        show_progress: Show tqdm progress bar
        
    Returns:
        List of ActivationCache, one per example
    """
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(examples, desc="Extracting activations")
        except ImportError:
            iterator = examples
    else:
        iterator = examples
    
    caches = []
    for example in iterator:
        cache = extract_activations_all_positions(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            layers=layers,
            include_logits=include_logits,
            device=device,
        )
        cache.example_id = example.example_id
        
        # Find target token positions in the tokenized sequence
        if example.target_value:
            target_positions = _find_target_positions(
                token_strings=cache.token_strings,
                token_ids=cache.token_ids,
                target_value=example.target_value,
                tokenizer=tokenizer,
            )
            example.target_token_positions = target_positions
        
        caches.append(cache)
    
    return caches


def _find_target_positions(
    token_strings: list[str],
    token_ids: list[int],
    target_value: str,
    tokenizer,
) -> list[int]:
    """Find which token positions correspond to the target value.
    
    For rhyme: find the position of the rhyme word in the completion.
    For acrostic: find the position of the next letter.
    For code: find the position of the return type annotation.
    
    Returns list of positions (0-indexed in the full tokenized sequence).
    """
    # Simple approach: tokenize the target and search for it
    target_token_ids = tokenizer.encode(target_value, add_special_tokens=False)
    
    if not target_token_ids:
        return []
    
    positions = []
    # Sliding window search
    for i in range(len(token_ids) - len(target_token_ids) + 1):
        if token_ids[i:i + len(target_token_ids)] == target_token_ids:
            positions.extend(range(i, i + len(target_token_ids)))
    
    # Also try with leading space variants (common in GPT-2 tokenizer)
    if not positions:
        target_with_space = " " + target_value
        space_token_ids = tokenizer.encode(target_with_space, add_special_tokens=False)
        for i in range(len(token_ids) - len(space_token_ids) + 1):
            if token_ids[i:i + len(space_token_ids)] == space_token_ids:
                positions.extend(range(i, i + len(space_token_ids)))
    
    return positions


def compute_logit_lens(
    model,
    activations: dict[int, np.ndarray],
    target_token_ids: list[int],
) -> dict[int, np.ndarray]:
    """Compute logit-lens predictions: what does each layer predict at each position?
    
    The logit lens applies the unembedding matrix to intermediate layer
    activations to see what token the model "would predict" at that layer.
    
    For planning detection, this tells us:
    - At which (layer, position) does the target token first appear in top-k?
    - Does the target probability increase monotonically across layers?
    
    Args:
        model: TransformerLens model (needed for W_U unembedding matrix)
        activations: dict[layer -> (seq_len, d_model)]
        target_token_ids: Token IDs we're looking for in predictions
        
    Returns:
        dict[layer -> (seq_len, n_targets)] probability of each target at each position
    """
    W_U = model.W_U.detach()  # (d_model, vocab_size)
    b_U = getattr(model, 'b_U', None)
    
    # Layer norm before unembedding
    ln_final = model.ln_final
    
    target_probs = {}
    
    for layer, acts in activations.items():
        acts_tensor = torch.tensor(acts, dtype=W_U.dtype, device=W_U.device)
        
        # Apply final layer norm
        normed = ln_final(acts_tensor)
        
        # Project to vocab space
        logits = normed @ W_U  # (seq_len, vocab_size)
        if b_U is not None:
            logits = logits + b_U
        
        probs = torch.softmax(logits, dim=-1)
        
        # Extract target token probabilities
        target_probs_layer = []
        for tid in target_token_ids:
            target_probs_layer.append(probs[:, tid].cpu().numpy())
        
        target_probs[layer] = np.stack(target_probs_layer, axis=-1)  # (seq_len, n_targets)
    
    return target_probs


def compute_logit_lens_rank(
    model,
    activations: dict[int, np.ndarray],
    target_token_ids: list[int],
    k: int = 10,
) -> dict[int, np.ndarray]:
    """Compute rank of target tokens in logit-lens predictions.
    
    Returns dict[layer -> (seq_len, n_targets)] with the rank (0=top-1)
    of each target token at each position.
    
    A target entering top-k at an early position/layer is evidence of planning.
    """
    W_U = model.W_U.detach()
    b_U = getattr(model, 'b_U', None)
    ln_final = model.ln_final
    
    target_ranks = {}
    
    for layer, acts in activations.items():
        acts_tensor = torch.tensor(acts, dtype=W_U.dtype, device=W_U.device)
        normed = ln_final(acts_tensor)
        logits = normed @ W_U
        if b_U is not None:
            logits = logits + b_U
        
        # Compute ranks
        # sorted_indices: (seq_len, vocab_size), descending
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        
        ranks_layer = []
        for tid in target_token_ids:
            # For each position, find where tid appears in the sorted order
            rank = (sorted_indices == tid).nonzero(as_tuple=True)[1]
            ranks_layer.append(rank.cpu().numpy())
        
        target_ranks[layer] = np.stack(ranks_layer, axis=-1)
    
    return target_ranks
