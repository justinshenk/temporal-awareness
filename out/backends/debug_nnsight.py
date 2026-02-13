#!/usr/bin/env python3
"""Debug NNsight backend by comparing with HuggingFace."""

import sys
sys.path.insert(0, "/Users/unrulyabstractions/work/temporal-awareness")

import torch
import numpy as np

# Test prompts
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
]


def compare_logits(logits1, logits2, name1="A", name2="B"):
    """Compare two logit tensors."""
    diff = (logits1 - logits2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Compare probabilities
    probs1 = torch.softmax(logits1[0, -1, :], dim=-1)
    probs2 = torch.softmax(logits2[0, -1, :], dim=-1)
    prob_diff = (probs1 - probs2).abs()
    max_prob_diff = prob_diff.max().item()

    # Top-k agreement
    top1_1 = logits1[0, -1].argmax().item()
    top1_2 = logits2[0, -1].argmax().item()

    print(f"  Logits max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
    print(f"  Probs max diff: {max_prob_diff:.6f}")
    print(f"  Top-1: {name1}={top1_1}, {name2}={top1_2}, match={top1_1 == top1_2}")

    # Top-5 overlap
    top5_1 = set(logits1[0, -1].topk(5).indices.tolist())
    top5_2 = set(logits2[0, -1].topk(5).indices.tolist())
    overlap = len(top5_1 & top5_2)
    print(f"  Top-5 overlap: {overlap}/5")

    return max_diff, top1_1 == top1_2


def compare_activations(cache1, cache2, name1="A", name2="B"):
    """Compare activation caches."""
    print(f"\n  Activation comparison ({len(cache1)} keys in {name1}, {len(cache2)} in {name2}):")

    # Find common keys
    common_keys = set(cache1.keys()) & set(cache2.keys())
    print(f"  Common keys: {len(common_keys)}")

    if not common_keys:
        print(f"  Keys in {name1}: {list(cache1.keys())[:5]}...")
        print(f"  Keys in {name2}: {list(cache2.keys())[:5]}...")
        return

    # Compare values for common keys
    total_diff = 0
    for key in sorted(common_keys)[:5]:  # Compare first 5 common keys
        v1 = cache1[key]
        v2 = cache2[key]
        print(f"    {key}:")
        print(f"      Shape: {name1}={v1.shape}, {name2}={v2.shape}")

        if v1.shape == v2.shape:
            diff = (v1 - v2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"      Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            total_diff += max_diff
        else:
            print(f"      SHAPE MISMATCH!")

    return total_diff


def main():
    print("=" * 60)
    print("NNsight vs HuggingFace Debug Comparison")
    print("=" * 60)

    # Import after path setup
    from src.inference.model_runner import ModelRunner
    from src.inference.backends import ModelBackend

    model_name = "gpt2"
    device = "cpu"  # Use CPU for reproducibility
    dtype = torch.float32  # Use float32 for better numerical comparison

    print(f"\n[1] Loading HuggingFace backend...")
    hf_runner = ModelRunner(model_name, device=device, dtype=dtype, backend=ModelBackend.HUGGINGFACE)

    print(f"\n[2] Loading NNsight backend...")
    nn_runner = ModelRunner(model_name, device=device, dtype=dtype, backend=ModelBackend.NNSIGHT)

    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    for prompt in PROMPTS:
        print(f"\nPrompt: '{prompt}'")

        # Tokenize with both
        hf_tokens = hf_runner.tokenize(prompt)
        nn_tokens = nn_runner.tokenize(prompt)

        print(f"  HF tokens shape: {hf_tokens.shape}, values: {hf_tokens[0].tolist()}")
        print(f"  NN tokens shape: {nn_tokens.shape}, values: {nn_tokens[0].tolist()}")

        if not torch.equal(hf_tokens, nn_tokens):
            print("  WARNING: Tokens differ!")

        # Forward pass
        hf_logits = hf_runner._backend.forward(hf_tokens)
        nn_logits = nn_runner._backend.forward(nn_tokens)

        print(f"  HF logits shape: {hf_logits.shape}")
        print(f"  NN logits shape: {nn_logits.shape}")

        compare_logits(hf_logits, nn_logits, "HF", "NN")

    print("\n" + "=" * 60)
    print("Testing Run With Cache")
    print("=" * 60)

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")

    # Run with cache
    hf_logits, hf_cache = hf_runner.run_with_cache(prompt)
    nn_logits, nn_cache = nn_runner.run_with_cache(prompt)

    print(f"\nLogits comparison:")
    compare_logits(hf_logits, nn_logits, "HF", "NN")
    compare_activations(hf_cache, nn_cache, "HF", "NN")

    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60)

    for prompt in PROMPTS:
        print(f"\nPrompt: '{prompt}'")

        hf_gen = hf_runner.generate(prompt, max_new_tokens=10, temperature=0.0)
        nn_gen = nn_runner.generate(prompt, max_new_tokens=10, temperature=0.0)

        print(f"  HF: '{hf_gen}'")
        print(f"  NN: '{nn_gen}'")
        print(f"  Match: {hf_gen == nn_gen}")

    print("\n" + "=" * 60)
    print("Low-level NNsight Investigation")
    print("=" * 60)

    # Direct investigation of NNsight model structure
    nn_model = nn_runner._model
    print(f"\nNNsight model type: {type(nn_model)}")
    print(f"Has transformer: {hasattr(nn_model, 'transformer')}")
    print(f"Has lm_head: {hasattr(nn_model, 'lm_head')}")

    if hasattr(nn_model, 'transformer'):
        print(f"Has transformer.h: {hasattr(nn_model.transformer, 'h')}")
        if hasattr(nn_model.transformer, 'h'):
            print(f"Number of layers: {len(nn_model.transformer.h)}")
            layer0 = nn_model.transformer.h[0]
            print(f"Layer 0 type: {type(layer0)}")
            print(f"Layer 0 has attn: {hasattr(layer0, 'attn')}")
            print(f"Layer 0 has mlp: {hasattr(layer0, 'mlp')}")

            if hasattr(layer0, 'attn'):
                print(f"Layer 0 attn type: {type(layer0.attn)}")

    # Test direct trace access - MUST follow execution order
    print("\n--- Testing direct trace (following execution order) ---")
    input_ids = nn_runner.tokenize("Hello")
    print(f"Input IDs: {input_ids}")

    # In GPT-2, execution order within each layer is:
    # ln_1 -> attn -> residual add -> ln_2 -> mlp -> residual add -> output
    # So we must access in order: attn -> mlp -> layer_output

    saved_values = {}

    with nn_model.trace(input_ids):
        for layer_idx in range(12):
            layer = nn_model.transformer.h[layer_idx]

            # Attention output first (comes first in forward)
            saved_values[f"layer{layer_idx}_attn"] = layer.attn.output[0].save()

            # MLP output second
            saved_values[f"layer{layer_idx}_mlp"] = layer.mlp.output.save()

            # Layer residual output last (this is the final output after residual connections)
            saved_values[f"layer{layer_idx}_out"] = layer.output[0].save()

        # lm_head at the very end
        saved_values["logits"] = nn_model.lm_head.output.save()

    print("\n--- Post-trace values ---")
    print(f"Layer 0 attn shape: {saved_values['layer0_attn'].shape}")
    print(f"Layer 0 mlp shape: {saved_values['layer0_mlp'].shape}")
    print(f"Layer 0 output shape: {saved_values['layer0_out'].shape}")
    print(f"Logits shape: {saved_values['logits'].shape}")

    # Compare with HF forward
    print("\n--- Compare with HF layer outputs ---")
    hf_model = hf_runner._model
    hf_input_ids = hf_runner.tokenize("Hello")

    # Get HF hidden states
    with torch.no_grad():
        hf_outputs = hf_model(hf_input_ids, output_hidden_states=True)
        hf_hidden = hf_outputs.hidden_states

    print(f"HF has {len(hf_hidden)} hidden states (including embedding)")
    print(f"HF hidden[1] shape (after layer 0): {hf_hidden[1].shape}")

    # Compare NNsight layer outputs with HF hidden states
    # NOTE: HF hidden_states[12] (final) is AFTER transformer.ln_f, not comparable to layer output!
    print("\n--- Layer-by-layer comparison ---")
    print("  (Note: HF hidden_states[-1] is AFTER ln_f, layer.output is BEFORE ln_f)")
    for i in range(12):
        nn_layer_out = saved_values[f"layer{i}_out"]
        hf_layer_out = hf_hidden[i + 1]  # +1 because hidden[0] is embedding

        # For the last layer, HF hidden_states[12] is AFTER ln_f
        # So we can't directly compare - let's check if ln_f explains the diff
        if i == 11:
            # Apply ln_f to NN output to match HF
            ln_f = hf_model.transformer.ln_f
            with torch.no_grad():
                nn_after_ln_f = ln_f(nn_layer_out)
            diff = (nn_after_ln_f - hf_layer_out).abs()
            max_diff = diff.max().item()
            print(f"  Layer {i:2d} resid_post (AFTER ln_f): max_diff={max_diff:.6f}")
        else:
            diff = (nn_layer_out - hf_layer_out).abs()
            max_diff = diff.max().item()
            print(f"  Layer {i:2d} resid_post: max_diff={max_diff:.6f}")

    print("\n" + "=" * 60)
    print("Extended Activation Tests")
    print("=" * 60)

    # Test with longer prompt
    prompt = "The quick brown fox jumps over the lazy dog"
    print(f"\nPrompt: '{prompt}'")

    # Get activations from both
    hf_logits2, hf_cache2 = hf_runner.run_with_cache(prompt)
    nn_logits2, nn_cache2 = nn_runner.run_with_cache(prompt)

    print(f"\nLogits comparison:")
    compare_logits(hf_logits2, nn_logits2, "HF", "NN")

    # Compare resid_post at all layers
    print("\n--- resid_post comparison at all layers ---")
    for i in range(12):
        key = f"blocks.{i}.hook_resid_post"
        if key in hf_cache2 and key in nn_cache2:
            diff = (hf_cache2[key] - nn_cache2[key]).abs()
            print(f"  Layer {i:2d}: max_diff={diff.max().item():.6f}")

    # Compare specific positions
    print("\n--- Position-specific comparison (layer 5, all positions) ---")
    key = "blocks.5.hook_resid_post"
    hf_act = hf_cache2[key]
    nn_act = nn_cache2[key]
    seq_len = hf_act.shape[1]
    for pos in range(seq_len):
        diff = (hf_act[0, pos, :] - nn_act[0, pos, :]).abs()
        print(f"  Position {pos:2d}: max_diff={diff.max().item():.6f}")

    # Test interventions if supported
    print("\n" + "=" * 60)
    print("Testing Interventions")
    print("=" * 60)

    from src.inference.interventions import Intervention, Target

    # Create a simple steering intervention
    steering_vector = torch.randn(768, dtype=dtype, device=device) * 0.1
    intervention = Intervention(
        layer=5,
        component="resid_post",
        target=Target(axis="all"),
        mode="add",
        values=steering_vector.numpy(),
    )

    # Apply intervention with NNsight
    print("\nApplying add intervention at layer 5...")
    try:
        nn_logits_interv = nn_runner.run_with_intervention(prompt, intervention)
        print(f"  NNsight intervention logits shape: {nn_logits_interv.shape}")

        # Compare with non-intervened
        diff = (nn_logits_interv - nn_logits2).abs()
        print(f"  Logits changed by intervention: max_diff={diff.max().item():.2f}")

        # Check top prediction changed
        top1_before = nn_logits2[0, -1].argmax().item()
        top1_after = nn_logits_interv[0, -1].argmax().item()
        print(f"  Top-1 prediction: before={top1_before}, after={top1_after}")
    except Exception as e:
        print(f"  Intervention failed: {e}")


if __name__ == "__main__":
    main()
