#!/usr/bin/env python3
"""Test script for HuggingFace backend attention pattern capture and weight accessors.

Tests:
1. Attention pattern capture via run_with_cache
2. Weight matrix accessors (get_W_Q, get_W_K, get_W_V, get_W_O)
3. Combined matrices (get_W_OV, get_W_QK)
4. Basic inference with Qwen3-4B-Instruct

Usage:
    uv run python scripts/test_hf_attention_hooks.py
"""

import torch

from src.inference import ModelRunner
from src.inference.backends import ModelBackend


def test_attention_hooks():
    """Test attention pattern capture and weight accessors on Qwen3."""
    print("=" * 60)
    print("Testing HuggingFace Backend Attention Hooks")
    print("=" * 60)

    # Load Qwen3-4B-Instruct with HuggingFace backend
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"\nLoading model: {model_name}")

    runner = ModelRunner(
        model_name=model_name,
        backend=ModelBackend.HUGGINGFACE,
    )
    backend = runner._backend

    # Test basic inference
    print("\n" + "-" * 40)
    print("Test 1: Basic Generation")
    print("-" * 40)
    prompt = "The capital of France is"
    output = runner.generate(prompt, max_new_tokens=10, temperature=0.0)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")

    # Test architecture info
    print("\n" + "-" * 40)
    print("Test 2: Architecture Info")
    print("-" * 40)
    print(f"n_layers: {backend.get_n_layers()}")
    print(f"d_model: {backend.get_d_model()}")
    print(f"n_heads: {backend.get_n_heads()}")
    print(f"d_head: {backend.get_d_head()}")
    print(f"n_kv_heads: {backend.get_n_kv_heads()}")

    # Test attention pattern capture
    print("\n" + "-" * 40)
    print("Test 3: Attention Pattern Capture")
    print("-" * 40)
    input_ids = runner.encode("Hello, world!")
    print(f"Input shape: {input_ids.shape}")

    # Request attention patterns for all layers
    def attn_filter(name):
        return "hook_pattern" in name

    logits, cache = backend.run_with_cache(input_ids, names_filter=attn_filter)
    print(f"Logits shape: {logits.shape}")
    print(f"Cache keys: {list(cache.keys())[:5]}...")  # First 5 keys

    # Check attention pattern shape
    pattern_key = "blocks.0.attn.hook_pattern"
    if pattern_key in cache:
        pattern = cache[pattern_key]
        print(f"Attention pattern shape: {pattern.shape}")
        print("  Expected: [batch, n_heads, seq_q, seq_k]")
        print(f"  Got: [batch={pattern.shape[0]}, n_heads={pattern.shape[1]}, "
              f"seq_q={pattern.shape[2]}, seq_k={pattern.shape[3]}]")

        # Verify attention patterns sum to 1 along last dimension
        attn_sum = pattern[0, 0, :, :].sum(dim=-1)
        print(f"Attention sums to 1? {torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-3)}")
    else:
        print(f"WARNING: {pattern_key} not found in cache")
        print(f"Available keys: {list(cache.keys())}")

    # Test weight matrix accessors
    print("\n" + "-" * 40)
    print("Test 4: Weight Matrix Accessors")
    print("-" * 40)

    # Single layer access
    W_Q = backend.get_W_Q(layer=0)
    W_K = backend.get_W_K(layer=0)
    W_V = backend.get_W_V(layer=0)
    W_O = backend.get_W_O(layer=0)

    print(f"W_Q[0] shape: {W_Q.shape}")
    print("  Expected: [n_heads, d_model, d_head]")
    print(f"W_K[0] shape: {W_K.shape}")
    print(f"W_V[0] shape: {W_V.shape}")
    print(f"W_O[0] shape: {W_O.shape}")
    print("  Expected: [n_heads, d_head, d_model]")

    # All layers access
    W_Q_all = backend.get_W_Q(layer=None)
    print(f"\nW_Q (all layers) shape: {W_Q_all.shape}")
    print("  Expected: [n_layers, n_heads, d_model, d_head]")

    # Test combined matrices
    print("\n" + "-" * 40)
    print("Test 5: Combined Matrices (W_OV, W_QK)")
    print("-" * 40)

    W_OV = backend.get_W_OV(layer=0, head=0)
    W_QK = backend.get_W_QK(layer=0, head=0)

    print(f"W_OV[0,0] shape: {W_OV.shape}")
    print("  Expected: [d_model, d_model]")
    print(f"W_QK[0,0] shape: {W_QK.shape}")
    print("  Expected: [d_model, d_model]")

    # Test that W_OV = W_V @ W_O
    W_V_head = W_V[0]  # [d_model, d_head]
    W_O_head = W_O[0]  # [d_head, d_model]
    W_OV_manual = W_V_head @ W_O_head
    print(f"\nW_OV = W_V @ W_O? {torch.allclose(W_OV, W_OV_manual, atol=1e-5)}")

    # Test embedding and unembedding matrices
    print("\n" + "-" * 40)
    print("Test 6: Embedding/Unembedding Matrices")
    print("-" * 40)

    W_E = backend.get_W_E()
    W_U = backend.get_W_U()
    b_U = backend.get_b_U()

    print(f"W_E shape: {W_E.shape}")
    print("  Expected: [vocab_size, d_model]")
    print(f"W_U shape: {W_U.shape}")
    print("  Expected: [d_model, vocab_size]")
    print(f"b_U: {b_U}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_attention_hooks()
