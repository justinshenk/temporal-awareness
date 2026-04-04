"""Causal patching for commitment verification.

Once we've identified commitment points via probing, we need to verify
them causally:

1. NECESSITY: If we patch (replace) activations at the commitment point,
   does the model's planned output change?
   → If yes: the commitment point is necessary for the plan.

2. SUFFICIENCY: If we inject a "commitment vector" (the activation
   difference between committing to target A vs target B), does this
   steer the model's output toward the injected target?
   → If yes: the commitment signal is sufficient to control the plan.

3. TIMING: Is intervention more effective at the commitment point
   than at other positions? Compare pre-commitment, at-commitment,
   and post-commitment patching effectiveness.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ..utils.types import PatchingResult, PlanningExample, ActivationCache

logger = logging.getLogger(__name__)


def patch_necessity(
    model,
    example: PlanningExample,
    cache: ActivationCache,
    patch_layer: int,
    patch_position: int,
    source_cache: ActivationCache,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
) -> PatchingResult:
    """Test necessity: does patching at the commitment point change the output?
    
    We replace the activation at (layer, position) in the original example
    with the activation from a different example (source_cache) that has
    a different target. If the output changes to match the source, this
    position is necessary for the plan.
    
    Args:
        model: TransformerLens HookedTransformer
        example: The original example
        cache: Original example's activation cache
        patch_layer: Layer to patch
        patch_position: Position to patch
        source_cache: Activation cache from a contrastive example (different target)
        max_new_tokens: Tokens to generate
        temperature: Generation temperature (0 = greedy)
        
    Returns:
        PatchingResult with before/after comparison
    """
    # Get original output (greedy, no intervention)
    tokens_original = model.to_tokens(example.prompt, prepend_bos=True)
    
    with torch.no_grad():
        original_output_ids = model.generate(
            tokens_original,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-10),
            stop_at_eos=True,
            verbose=False,
        )
    original_output = model.to_string(original_output_ids[0, tokens_original.shape[1]:])
    
    # Get original target probability
    with torch.no_grad():
        original_logits = model(tokens_original)
    original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
    
    # Get patched output
    source_activation = torch.tensor(
        source_cache.activations[patch_layer][patch_position],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    
    hook_name = f"blocks.{patch_layer}.hook_resid_post"
    
    def patch_hook(act, hook):
        act[0, patch_position, :] = source_activation
        return act
    
    with torch.no_grad():
        # Run with the patch active
        patched_output_ids = model.generate(
            tokens_original,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-10),
            stop_at_eos=True,
            verbose=False,
            fwd_hooks=[(hook_name, patch_hook)],
            use_past_kv_cache=False,  # Must disable KV cache for patches to work
        )
    patched_output = model.to_string(patched_output_ids[0, tokens_original.shape[1]:])
    
    # Get patched target probability  
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            tokens_original,
            fwd_hooks=[(hook_name, patch_hook)],
        )
    patched_probs = torch.softmax(patched_logits[0, -1, :], dim=-1)
    
    # Compute target token probability
    tokenizer = model.tokenizer
    target_token_ids = tokenizer.encode(example.target_value, add_special_tokens=False)
    
    if target_token_ids:
        orig_target_prob = original_probs[target_token_ids[0]].item()
        patched_target_prob = patched_probs[target_token_ids[0]].item()
    else:
        orig_target_prob = 0.0
        patched_target_prob = 0.0
    
    return PatchingResult(
        example_id=example.example_id,
        patch_layer=patch_layer,
        patch_position=patch_position,
        mode="necessity",
        original_target=example.target_value,
        original_output=original_output.strip(),
        original_target_prob=orig_target_prob,
        intervened_output=patched_output.strip(),
        intervened_target_prob=patched_target_prob,
        prob_delta=patched_target_prob - orig_target_prob,
        output_changed=original_output.strip() != patched_output.strip(),
        target_flipped=False,  # computed below
    )


def patch_sufficiency(
    model,
    example: PlanningExample,
    cache: ActivationCache,
    contrastive_cache: ActivationCache,
    patch_layer: int,
    patch_position: int,
    contrastive_target: str,
    alpha: float = 1.0,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
) -> PatchingResult:
    """Test sufficiency: does injecting a commitment vector steer the output?
    
    Compute the "commitment vector" as the activation difference between
    the contrastive example and the original example at the same position.
    Then add this vector (scaled by alpha) to the original activations.
    
    If the output shifts toward the contrastive target, the commitment
    signal is sufficient to control the plan.
    
    Args:
        model: TransformerLens model
        example: Original example
        cache: Original activations
        contrastive_cache: Activations from contrastive example (different target)
        patch_layer: Layer to steer
        patch_position: Position to steer
        contrastive_target: What the contrastive example was targeting
        alpha: Steering strength (1.0 = full, 0.0 = none)
        max_new_tokens: Tokens to generate
        temperature: Generation temperature
        
    Returns:
        PatchingResult
    """
    # Compute steering vector: contrastive - original at the patch position
    original_act = cache.activations[patch_layer][patch_position]
    contrastive_act = contrastive_cache.activations[patch_layer][patch_position]
    steering_vector = torch.tensor(
        (contrastive_act - original_act) * alpha,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    
    tokens = model.to_tokens(example.prompt, prepend_bos=True)
    hook_name = f"blocks.{patch_layer}.hook_resid_post"
    
    def steer_hook(act, hook):
        act[0, patch_position, :] += steering_vector
        return act
    
    # Get steered output
    with torch.no_grad():
        steered_output_ids = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-10),
            stop_at_eos=True,
            verbose=False,
            fwd_hooks=[(hook_name, steer_hook)],
            use_past_kv_cache=False,
        )
    steered_output = model.to_string(steered_output_ids[0, tokens.shape[1]:])
    
    # Get steered target probability
    with torch.no_grad():
        steered_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, steer_hook)],
        )
    steered_probs = torch.softmax(steered_logits[0, -1, :], dim=-1)
    
    # Original output for comparison
    with torch.no_grad():
        original_logits = model(tokens)
    original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
    original_output_ids = model.generate(
        tokens, max_new_tokens=max_new_tokens, do_sample=False,
        stop_at_eos=True, verbose=False,
    )
    original_output = model.to_string(original_output_ids[0, tokens.shape[1]:])
    
    # Target probabilities
    tokenizer = model.tokenizer
    target_ids = tokenizer.encode(contrastive_target, add_special_tokens=False)
    
    if target_ids:
        orig_prob = original_probs[target_ids[0]].item()
        steered_prob = steered_probs[target_ids[0]].item()
    else:
        orig_prob = 0.0
        steered_prob = 0.0
    
    return PatchingResult(
        example_id=example.example_id,
        patch_layer=patch_layer,
        patch_position=patch_position,
        mode="sufficiency",
        original_target=example.target_value,
        original_output=original_output.strip(),
        original_target_prob=orig_prob,
        intervened_output=steered_output.strip(),
        intervened_target_prob=steered_prob,
        prob_delta=steered_prob - orig_prob,
        output_changed=original_output.strip() != steered_output.strip(),
        target_flipped=False,
    )


def timing_comparison(
    model,
    example: PlanningExample,
    cache: ActivationCache,
    source_cache: ActivationCache,
    patch_layer: int,
    commitment_position: int,
    positions_before: list[int] | None = None,
    positions_after: list[int] | None = None,
    max_new_tokens: int = 30,
) -> dict:
    """Compare patching effectiveness at pre-, at-, and post-commitment positions.
    
    This tests H4: intervention is easier before the commitment point.
    
    Args:
        model: TransformerLens model
        example: Original example
        cache: Original activations
        source_cache: Contrastive activations
        patch_layer: Layer to patch
        commitment_position: The identified commitment position
        positions_before: Positions before commitment to test
        positions_after: Positions after commitment to test
        max_new_tokens: Tokens to generate
        
    Returns:
        dict with patching results at each position
    """
    if positions_before is None:
        # Test 1, 2, 3 positions before commitment
        positions_before = [
            max(0, commitment_position - k)
            for k in [1, 2, 3]
            if commitment_position - k >= 0
        ]
    
    if positions_after is None:
        seq_len = len(cache.token_ids)
        positions_after = [
            min(seq_len - 1, commitment_position + k)
            for k in [1, 2, 3]
            if commitment_position + k < seq_len
        ]
    
    all_positions = {
        "before": positions_before,
        "at": [commitment_position],
        "after": positions_after,
    }
    
    results = {}
    
    for timing, positions in all_positions.items():
        for pos in positions:
            result = patch_necessity(
                model=model,
                example=example,
                cache=cache,
                patch_layer=patch_layer,
                patch_position=pos,
                source_cache=source_cache,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            results[f"{timing}_pos{pos}"] = {
                "timing": timing,
                "position": pos,
                "relative_to_commitment": pos - commitment_position,
                "prob_delta": result.prob_delta,
                "output_changed": result.output_changed,
                "original_output": result.original_output,
                "patched_output": result.intervened_output,
            }
    
    return results
