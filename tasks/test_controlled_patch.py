"""
CONTROLLED EXPERIMENT: resid_pre[L+1] vs resid_post[L] patching differences

This script investigates why patching resid_post[L] vs resid_pre[L+1] produces
different effects, even when the tensors should be mathematically identical.
"""

import torch
import transformer_lens as tl
from transformer_lens import HookedTransformer
import numpy as np


def run_experiment():
    print("=" * 80)
    print("CONTROLLED EXPERIMENT: resid_pre[L+1] vs resid_post[L] Patching")
    print("=" * 80)

    # Load GPT-2 small
    print("\n[1] Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    # Simple test input
    test_input = "The quick brown fox"
    print(f"\n[2] Test input: '{test_input}'")

    # First, verify that resid_post[5] and resid_pre[6] are identical in a clean run
    print("\n[3] Verifying resid_post[5] == resid_pre[6] in clean forward pass...")

    cache = {}
    def cache_hook_post_5(activation, hook):
        cache['resid_post_5'] = activation.clone()
        return activation

    def cache_hook_pre_6(activation, hook):
        cache['resid_pre_6'] = activation.clone()
        return activation

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", cache_hook_post_5),
                ("blocks.6.hook_resid_pre", cache_hook_pre_6),
            ]
        )

    resid_post_5 = cache['resid_post_5']
    resid_pre_6 = cache['resid_pre_6']

    print(f"   resid_post[5] shape: {resid_post_5.shape}")
    print(f"   resid_pre[6] shape: {resid_pre_6.shape}")
    print(f"   Are they identical? {torch.allclose(resid_post_5, resid_pre_6)}")
    print(f"   Max difference: {(resid_post_5 - resid_pre_6).abs().max().item()}")

    # Create a specific patch value (random but fixed)
    torch.manual_seed(42)
    patch_value = torch.randn_like(resid_post_5)
    print(f"\n[4] Created patch value with shape {patch_value.shape}")
    print(f"   Patch value mean: {patch_value.mean().item():.6f}")
    print(f"   Patch value std: {patch_value.std().item():.6f}")

    # EXPERIMENT A: Patch resid_post[5]
    print("\n[5] EXPERIMENT A: Patching resid_post[5]...")

    def patch_resid_post_5(activation, hook):
        return patch_value.clone()

    with torch.no_grad():
        logits_A = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_resid_post_5)]
        )

    print(f"   Output logits shape: {logits_A.shape}")
    print(f"   Logits mean: {logits_A.mean().item():.6f}")
    print(f"   Logits[0, -1, :10]: {logits_A[0, -1, :10].tolist()}")

    # EXPERIMENT B: Patch resid_pre[6]
    print("\n[6] EXPERIMENT B: Patching resid_pre[6]...")

    def patch_resid_pre_6(activation, hook):
        return patch_value.clone()

    with torch.no_grad():
        logits_B = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_resid_pre_6)]
        )

    print(f"   Output logits shape: {logits_B.shape}")
    print(f"   Logits mean: {logits_B.mean().item():.6f}")
    print(f"   Logits[0, -1, :10]: {logits_B[0, -1, :10].tolist()}")

    # COMPARE RESULTS
    print("\n" + "=" * 80)
    print("COMPARISON OF RESULTS")
    print("=" * 80)

    are_identical = torch.allclose(logits_A, logits_B, atol=1e-6)
    max_diff = (logits_A - logits_B).abs().max().item()
    mean_diff = (logits_A - logits_B).abs().mean().item()

    print(f"\n   Are logits identical (atol=1e-6)? {are_identical}")
    print(f"   Max absolute difference: {max_diff:.10e}")
    print(f"   Mean absolute difference: {mean_diff:.10e}")

    if not are_identical:
        print("\n   !!! LOGITS DIFFER - INVESTIGATING FURTHER !!!")

        # Find where they differ most
        diff = (logits_A - logits_B).abs()
        max_idx = diff.argmax()
        max_pos = np.unravel_index(max_idx.item(), diff.shape)
        print(f"   Max diff at position: {max_pos}")
        print(f"   Logits_A value: {logits_A[max_pos].item():.10e}")
        print(f"   Logits_B value: {logits_B[max_pos].item():.10e}")

        # DEEPER INVESTIGATION
        print("\n" + "=" * 80)
        print("DEEPER INVESTIGATION")
        print("=" * 80)

        # Check if the issue is in what happens AFTER the patch point
        print("\n[7] Checking activations AFTER patch point...")

        cache_A = {}
        cache_B = {}

        def capture_post_A(layer):
            def hook(activation, hook):
                cache_A[f'resid_post_{layer}'] = activation.clone()
                return activation
            return hook

        def capture_post_B(layer):
            def hook(activation, hook):
                cache_B[f'resid_post_{layer}'] = activation.clone()
                return activation
            return hook

        # Run A with capturing
        hooks_A = [("blocks.5.hook_resid_post", patch_resid_post_5)]
        for layer in [6, 7, 8, 9, 10, 11]:
            hooks_A.append((f"blocks.{layer}.hook_resid_post", capture_post_A(layer)))

        with torch.no_grad():
            _ = model.run_with_hooks(test_input, fwd_hooks=hooks_A)

        # Run B with capturing
        hooks_B = [("blocks.6.hook_resid_pre", patch_resid_pre_6)]
        for layer in [6, 7, 8, 9, 10, 11]:
            hooks_B.append((f"blocks.{layer}.hook_resid_post", capture_post_B(layer)))

        with torch.no_grad():
            _ = model.run_with_hooks(test_input, fwd_hooks=hooks_B)

        print("\n   Comparing activations at each layer after patch:")
        for layer in [6, 7, 8, 9, 10, 11]:
            key = f'resid_post_{layer}'
            if key in cache_A and key in cache_B:
                diff = (cache_A[key] - cache_B[key]).abs().max().item()
                print(f"   Layer {layer} resid_post max diff: {diff:.10e}")

        # EXPERIMENT C: Check if both patches even get applied
        print("\n[8] Verifying patches are actually applied...")

        cache_verify = {}

        def verify_patch_applied_post(activation, hook):
            cache_verify['post_5_before'] = activation.clone()
            result = patch_value.clone()
            cache_verify['post_5_after'] = result.clone()
            return result

        def verify_patch_applied_pre(activation, hook):
            cache_verify['pre_6_before'] = activation.clone()
            result = patch_value.clone()
            cache_verify['pre_6_after'] = result.clone()
            return result

        # Verify experiment A
        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[("blocks.5.hook_resid_post", verify_patch_applied_post)]
            )

        print(f"\n   Patch A (resid_post[5]):")
        print(f"   - Input to hook differs from patch? {not torch.allclose(cache_verify['post_5_before'], patch_value)}")
        print(f"   - Output from hook equals patch? {torch.allclose(cache_verify['post_5_after'], patch_value)}")

        cache_verify.clear()

        # Verify experiment B
        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[("blocks.6.hook_resid_pre", verify_patch_applied_pre)]
            )

        print(f"\n   Patch B (resid_pre[6]):")
        print(f"   - Input to hook differs from patch? {not torch.allclose(cache_verify['pre_6_before'], patch_value)}")
        print(f"   - Output from hook equals patch? {torch.allclose(cache_verify['pre_6_after'], patch_value)}")

        # EXPERIMENT D: Check if there's something BETWEEN resid_post[5] and resid_pre[6]
        print("\n[9] Checking what happens BETWEEN resid_post[5] and resid_pre[6]...")

        # Get all hook points available
        print(f"\n   Model hook points that contain 'resid':")
        for name in model.hook_dict.keys():
            if 'resid' in name and ('5' in name or '6' in name):
                print(f"   - {name}")

        # EXPERIMENT E: What if we patch BOTH at the same time?
        print("\n[10] EXPERIMENT: Patching BOTH resid_post[5] AND resid_pre[6]...")

        def patch_both_post(activation, hook):
            return patch_value.clone()

        def patch_both_pre(activation, hook):
            return patch_value.clone()

        with torch.no_grad():
            logits_both = model.run_with_hooks(
                test_input,
                fwd_hooks=[
                    ("blocks.5.hook_resid_post", patch_both_post),
                    ("blocks.6.hook_resid_pre", patch_both_pre),
                ]
            )

        print(f"   Logits_both == Logits_A? {torch.allclose(logits_both, logits_A, atol=1e-6)}")
        print(f"   Logits_both == Logits_B? {torch.allclose(logits_both, logits_B, atol=1e-6)}")
        print(f"   Max diff (both vs A): {(logits_both - logits_A).abs().max().item():.10e}")
        print(f"   Max diff (both vs B): {(logits_both - logits_B).abs().max().item():.10e}")

        # EXPERIMENT F: Check hook execution order
        print("\n[11] Checking hook execution order...")

        execution_order = []

        def order_tracker(name):
            def hook(activation, hook):
                execution_order.append(name)
                return activation
            return hook

        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[
                    ("blocks.5.hook_resid_post", order_tracker("resid_post_5")),
                    ("blocks.6.hook_resid_pre", order_tracker("resid_pre_6")),
                    ("blocks.6.hook_attn_out", order_tracker("attn_out_6")),
                    ("blocks.6.hook_mlp_out", order_tracker("mlp_out_6")),
                    ("blocks.6.hook_resid_mid", order_tracker("resid_mid_6")),
                    ("blocks.6.hook_resid_post", order_tracker("resid_post_6")),
                ]
            )

        print(f"   Execution order: {execution_order}")

        # EXPERIMENT G: Test if the issue is that resid_pre affects MORE than resid_post
        print("\n[12] CRITICAL TEST: Does resid_pre[6] also affect layer 6's attention input?")

        cache_attn = {}

        def capture_attn_input(activation, hook):
            cache_attn['attn_input'] = activation.clone()
            return activation

        # Without any patching
        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[("blocks.6.hook_attn_in", capture_attn_input)]
            )
        clean_attn_input = cache_attn['attn_input'].clone()

        # With resid_post[5] patching
        cache_attn.clear()
        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[
                    ("blocks.5.hook_resid_post", patch_resid_post_5),
                    ("blocks.6.hook_attn_in", capture_attn_input),
                ]
            )
        patched_A_attn_input = cache_attn['attn_input'].clone()

        # With resid_pre[6] patching
        cache_attn.clear()
        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[
                    ("blocks.6.hook_resid_pre", patch_resid_pre_6),
                    ("blocks.6.hook_attn_in", capture_attn_input),
                ]
            )
        patched_B_attn_input = cache_attn['attn_input'].clone()

        print(f"\n   Clean attn_input[6] shape: {clean_attn_input.shape}")
        print(f"   Patched A (resid_post[5]) attn_input[6] equals patch? {torch.allclose(patched_A_attn_input, patch_value)}")
        print(f"   Patched B (resid_pre[6]) attn_input[6] equals patch? {torch.allclose(patched_B_attn_input, patch_value)}")
        print(f"   Patched A attn_input[6] == Patched B attn_input[6]? {torch.allclose(patched_A_attn_input, patched_B_attn_input)}")
        print(f"   Max diff A vs B attn_input: {(patched_A_attn_input - patched_B_attn_input).abs().max().item():.10e}")

        # FINAL ANALYSIS
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS")
        print("=" * 80)

        # Check if there's layer norm between resid_post and resid_pre
        print("\n[13] Checking for layer norm or other ops between hooks...")

        # Look at model architecture
        print(f"\n   Model architecture for block 5 and 6:")
        print(f"   Block 5: {model.blocks[5]}")
        print(f"\n   Checking if ln1 is applied between resid_post and resid_pre...")

        # The key insight: TransformerLens applies layer norm AFTER hook_resid_pre
        # but hook_attn_in is the LN'd version. Let's verify.

        print("\n[14] CHECKING: Is there processing between resid_post[L] and resid_pre[L+1]?")

        # Cache everything in detail
        detail_cache = {}

        def detail_hook(name):
            def hook(activation, hook):
                detail_cache[name] = activation.clone()
                return activation
            return hook

        with torch.no_grad():
            _ = model.run_with_hooks(
                test_input,
                fwd_hooks=[
                    ("blocks.5.hook_resid_post", detail_hook("block5_resid_post")),
                    ("blocks.5.hook_resid_pre", detail_hook("block5_resid_pre")),
                    ("blocks.6.hook_resid_pre", detail_hook("block6_resid_pre")),
                    ("blocks.6.hook_resid_post", detail_hook("block6_resid_post")),
                ]
            )

        print(f"\n   block5_resid_pre == block5_resid_post? {torch.allclose(detail_cache.get('block5_resid_pre', torch.zeros(1)), detail_cache.get('block5_resid_post', torch.ones(1)))}")
        print(f"   block5_resid_post == block6_resid_pre? {torch.allclose(detail_cache['block5_resid_post'], detail_cache['block6_resid_pre'])}")

    else:
        print("\n   Logits are IDENTICAL - no investigation needed.")
        print("   This suggests the hooks work correctly and the tensors flow as expected.")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return {
        'logits_A': logits_A,
        'logits_B': logits_B,
        'are_identical': are_identical,
        'max_diff': max_diff,
    }


def run_advanced_experiment():
    """
    More advanced tests to find edge cases where resid_pre and resid_post patching diverge.
    """
    print("\n" + "=" * 80)
    print("ADVANCED EXPERIMENT: Edge Cases")
    print("=" * 80)

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    test_input = "The quick brown fox"

    # Get original activations
    with torch.no_grad():
        _, cache = model.run_with_cache(test_input)

    original_resid_post_5 = cache['blocks.5.hook_resid_post'].clone()
    original_resid_pre_6 = cache['blocks.6.hook_resid_pre'].clone()

    print(f"\n[1] Original activations identical? {torch.allclose(original_resid_post_5, original_resid_pre_6)}")

    # TEST 1: Partial position patching (only position 2)
    print("\n" + "-" * 40)
    print("TEST 1: Partial position patching (only position 2)")
    print("-" * 40)

    torch.manual_seed(42)
    patch_single_pos = torch.randn(1, 1, 768)

    def patch_post_5_pos2(activation, hook):
        result = activation.clone()
        result[:, 2:3, :] = patch_single_pos
        return result

    def patch_pre_6_pos2(activation, hook):
        result = activation.clone()
        result[:, 2:3, :] = patch_single_pos
        return result

    with torch.no_grad():
        logits_A = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_5_pos2)]
        )
        logits_B = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6_pos2)]
        )

    print(f"   Logits identical? {torch.allclose(logits_A, logits_B, atol=1e-6)}")
    print(f"   Max diff: {(logits_A - logits_B).abs().max().item():.10e}")

    # TEST 2: Patching with a DIFFERENT input's activation (cross-input patching)
    print("\n" + "-" * 40)
    print("TEST 2: Cross-input patching (patch with activation from different input)")
    print("-" * 40)

    other_input = "Hello world test"
    with torch.no_grad():
        _, other_cache = model.run_with_cache(other_input)

    # Shapes might differ, so let's handle that
    other_resid = other_cache['blocks.5.hook_resid_post']
    print(f"   Original input length: 5, Other input length: {other_resid.shape[1]}")

    # Use min length for comparison
    min_len = min(5, other_resid.shape[1])

    def patch_post_5_cross(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = other_resid[:, :min_len, :]
        return result

    def patch_pre_6_cross(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = other_resid[:, :min_len, :]
        return result

    with torch.no_grad():
        logits_A = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_5_cross)]
        )
        logits_B = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6_cross)]
        )

    print(f"   Logits identical? {torch.allclose(logits_A, logits_B, atol=1e-6)}")
    print(f"   Max diff: {(logits_A - logits_B).abs().max().item():.10e}")

    # TEST 3: Ablation to zero
    print("\n" + "-" * 40)
    print("TEST 3: Zero ablation")
    print("-" * 40)

    def patch_post_5_zero(activation, hook):
        return torch.zeros_like(activation)

    def patch_pre_6_zero(activation, hook):
        return torch.zeros_like(activation)

    with torch.no_grad():
        logits_A = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_5_zero)]
        )
        logits_B = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6_zero)]
        )

    print(f"   Logits identical? {torch.allclose(logits_A, logits_B, atol=1e-6)}")
    print(f"   Max diff: {(logits_A - logits_B).abs().max().item():.10e}")

    # TEST 4: Mean ablation
    print("\n" + "-" * 40)
    print("TEST 4: Mean ablation (replace with mean activation)")
    print("-" * 40)

    mean_activation = original_resid_post_5.mean(dim=(0, 1), keepdim=True).expand_as(original_resid_post_5)

    def patch_post_5_mean(activation, hook):
        return mean_activation.clone()

    def patch_pre_6_mean(activation, hook):
        return mean_activation.clone()

    with torch.no_grad():
        logits_A = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_5_mean)]
        )
        logits_B = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6_mean)]
        )

    print(f"   Logits identical? {torch.allclose(logits_A, logits_B, atol=1e-6)}")
    print(f"   Max diff: {(logits_A - logits_B).abs().max().item():.10e}")

    # TEST 5: Check hook_resid_mid behavior
    print("\n" + "-" * 40)
    print("TEST 5: Compare patching resid_mid vs resid_pre for layer 6")
    print("-" * 40)

    torch.manual_seed(42)
    patch_value = torch.randn_like(original_resid_post_5)

    def patch_pre_6(activation, hook):
        return patch_value.clone()

    def patch_mid_6(activation, hook):
        return patch_value.clone()

    with torch.no_grad():
        logits_pre = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6)]
        )
        logits_mid = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_mid", patch_mid_6)]
        )

    print(f"   Patching resid_pre[6] vs resid_mid[6]:")
    print(f"   Logits identical? {torch.allclose(logits_pre, logits_mid, atol=1e-6)}")
    print(f"   Max diff: {(logits_pre - logits_mid).abs().max().item():.10e}")

    # TEST 6: Multiple layers simultaneously
    print("\n" + "-" * 40)
    print("TEST 6: Patching multiple layers (5, 6, 7) with same value")
    print("-" * 40)

    def patch_fn(activation, hook):
        return patch_value.clone()

    # Using resid_post for layers 5,6,7
    with torch.no_grad():
        logits_post = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", patch_fn),
                ("blocks.6.hook_resid_post", patch_fn),
                ("blocks.7.hook_resid_post", patch_fn),
            ]
        )

    # Using resid_pre for layers 6,7,8
    with torch.no_grad():
        logits_pre = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.6.hook_resid_pre", patch_fn),
                ("blocks.7.hook_resid_pre", patch_fn),
                ("blocks.8.hook_resid_pre", patch_fn),
            ]
        )

    print(f"   Patching resid_post[5,6,7] vs resid_pre[6,7,8]:")
    print(f"   Logits identical? {torch.allclose(logits_post, logits_pre, atol=1e-6)}")
    print(f"   Max diff: {(logits_post - logits_pre).abs().max().item():.10e}")

    # TEST 7: Check if the order of hooks matters
    print("\n" + "-" * 40)
    print("TEST 7: Hook ordering test")
    print("-" * 40)

    captured = {'order': []}

    def capture_order(name):
        def hook(activation, hook):
            captured['order'].append(name)
            return activation
        return hook

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.6.hook_resid_pre", capture_order("resid_pre_6")),
                ("blocks.5.hook_resid_post", capture_order("resid_post_5")),
            ]
        )

    print(f"   Hook execution order when registered out of order: {captured['order']}")

    # TEST 8: Using run_with_cache vs run_with_hooks
    print("\n" + "-" * 40)
    print("TEST 8: Comparing run_with_hooks vs ActivationCache + patching")
    print("-" * 40)

    # Direct hook patching
    def patch_for_test8(activation, hook):
        return patch_value.clone()

    with torch.no_grad():
        logits_hooks = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_for_test8)]
        )

    # Using model.run_with_cache - can we patch via cache?
    # Actually, run_with_cache doesn't allow patching, so this is just to show the API
    print("   (Note: run_with_cache doesn't support patching - hooks are the way)")

    # TEST 9: Check if there's any dropout or stochastic behavior
    print("\n" + "-" * 40)
    print("TEST 9: Determinism test - run same patch twice")
    print("-" * 40)

    def patch_for_test9(activation, hook):
        return patch_value.clone()

    with torch.no_grad():
        logits_1 = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_for_test9)]
        )
        logits_2 = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_for_test9)]
        )

    print(f"   Two identical patches produce identical results? {torch.allclose(logits_1, logits_2)}")

    # TEST 10: The REAL culprit check - is there a difference in how the actual experiment code works?
    print("\n" + "-" * 40)
    print("TEST 10: Check if batch processing causes issues")
    print("-" * 40)

    # Batch of 2 identical inputs
    batch_input = ["The quick brown fox", "The quick brown fox"]

    torch.manual_seed(42)
    batch_patch = torch.randn(2, 5, 768)

    def patch_post_batch(activation, hook):
        return batch_patch.clone()

    def patch_pre_batch(activation, hook):
        return batch_patch.clone()

    with torch.no_grad():
        logits_A_batch = model.run_with_hooks(
            batch_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_batch)]
        )
        logits_B_batch = model.run_with_hooks(
            batch_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_batch)]
        )

    print(f"   Batch patching - logits identical? {torch.allclose(logits_A_batch, logits_B_batch, atol=1e-6)}")
    print(f"   Max diff: {(logits_A_batch - logits_B_batch).abs().max().item():.10e}")

    print("\n" + "=" * 80)
    print("ADVANCED EXPERIMENT COMPLETE")
    print("=" * 80)

    return True


def check_actual_experiment_code():
    """
    Check how the actual experiment code does patching to find the source of divergence.
    """
    print("\n" + "=" * 80)
    print("CHECKING ACTUAL EXPERIMENT CODE PATTERNS")
    print("=" * 80)

    # We need to look at what the actual patching code does differently
    # This is a simulation of potential issues

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    test_input = "The quick brown fox"

    # Get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(test_input)

    # Pattern 1: What if the hook receives a tensor that's a view?
    print("\n[1] Testing if tensor view vs clone matters...")

    original = cache['blocks.5.hook_resid_post']

    def patch_with_view(activation, hook):
        # Return a view
        return original  # This is the original tensor, not a clone

    def patch_with_clone(activation, hook):
        # Return a clone
        return original.clone()

    with torch.no_grad():
        logits_view = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_with_view)]
        )
        logits_clone = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_with_clone)]
        )

    print(f"   View vs clone identical? {torch.allclose(logits_view, logits_clone)}")

    # Pattern 2: What if there's in-place modification?
    print("\n[2] Testing in-place modification behavior...")

    def patch_inplace(activation, hook):
        activation[:] = original
        return activation

    def patch_return_new(activation, hook):
        return original.clone()

    with torch.no_grad():
        logits_inplace = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_inplace)]
        )
        logits_new = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_return_new)]
        )

    print(f"   In-place vs new tensor identical? {torch.allclose(logits_inplace, logits_new)}")

    # Pattern 3: Does the activation get modified between hooks?
    print("\n[3] Check if activation is modified between resid_post[L] and resid_pre[L+1]...")

    modifications = {}

    def check_post_5(activation, hook):
        modifications['post_5_id'] = id(activation)
        modifications['post_5_data'] = activation.clone()
        return activation

    def check_pre_6(activation, hook):
        modifications['pre_6_id'] = id(activation)
        modifications['pre_6_data'] = activation.clone()
        return activation

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", check_post_5),
                ("blocks.6.hook_resid_pre", check_pre_6),
            ]
        )

    print(f"   Same tensor ID? {modifications['post_5_id'] == modifications['pre_6_id']}")
    print(f"   Same tensor values? {torch.allclose(modifications['post_5_data'], modifications['pre_6_data'])}")

    # CRUCIAL INSIGHT: If they're not the same tensor ID, the hook mechanism creates copies
    # which could explain differences if one gets the copy and one gets the original

    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS COMPLETE")
    print("=" * 80)


def investigate_hook_mechanism():
    """
    Deep investigation into the TransformerLens hook mechanism itself.
    Investigates all possible sources of divergence.
    """
    print("\n" + "=" * 80)
    print("DEEP INVESTIGATION: TransformerLens Hook Mechanism")
    print("=" * 80)

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    test_input = "The quick brown fox"

    # Test 1: Verify the transformer block architecture
    print("\n[1] TransformerBlock Structure:")
    block = model.blocks[5]
    print(f"   Block attributes: {[a for a in dir(block) if not a.startswith('_')]}")

    # Test 2: Trace the exact data flow within a block
    print("\n[2] Tracing EXACT data flow in block 5 -> block 6 transition:")

    trace = {}

    def trace_hook(name):
        def hook(activation, **kwargs):
            trace[name] = {
                'id': id(activation),
                'data_ptr': activation.data_ptr(),
                'shape': activation.shape,
                'mean': activation.mean().item(),
                'snapshot': activation[0, 0, :5].clone().tolist()
            }
            return activation
        return hook

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                # Block 5 hooks
                ("blocks.5.hook_resid_pre", trace_hook("B5_resid_pre")),
                ("blocks.5.hook_attn_out", trace_hook("B5_attn_out")),
                ("blocks.5.hook_resid_mid", trace_hook("B5_resid_mid")),
                ("blocks.5.hook_mlp_out", trace_hook("B5_mlp_out")),
                ("blocks.5.hook_resid_post", trace_hook("B5_resid_post")),
                # Block 6 hooks
                ("blocks.6.hook_resid_pre", trace_hook("B6_resid_pre")),
                ("blocks.6.hook_attn_out", trace_hook("B6_attn_out")),
                ("blocks.6.hook_resid_mid", trace_hook("B6_resid_mid")),
                ("blocks.6.hook_mlp_out", trace_hook("B6_mlp_out")),
                ("blocks.6.hook_resid_post", trace_hook("B6_resid_post")),
            ]
        )

    print("\n   Data flow trace:")
    for name, info in trace.items():
        print(f"   {name}:")
        print(f"      tensor id: {info['id']}")
        print(f"      data_ptr: {info['data_ptr']}")
        print(f"      mean: {info['mean']:.6f}")
        print(f"      sample: {info['snapshot']}")

    # Check if B5_resid_post and B6_resid_pre are the SAME tensor
    print("\n   CRITICAL CHECK:")
    print(f"   B5_resid_post tensor id == B6_resid_pre tensor id? {trace['B5_resid_post']['id'] == trace['B6_resid_pre']['id']}")
    print(f"   B5_resid_post data_ptr == B6_resid_pre data_ptr? {trace['B5_resid_post']['data_ptr'] == trace['B6_resid_pre']['data_ptr']}")

    # Test 3: What if we patch and capture simultaneously?
    print("\n[3] Patching resid_post[5] and capturing resid_pre[6]:")

    torch.manual_seed(42)
    patch_value = torch.randn(1, 5, 768)

    capture = {}

    def patch_post_5(activation, **kwargs):
        capture['pre_patch_post_5'] = activation.clone()
        return patch_value.clone()

    def capture_pre_6(activation, **kwargs):
        capture['post_patch_pre_6'] = activation.clone()
        return activation

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", patch_post_5),
                ("blocks.6.hook_resid_pre", capture_pre_6),
            ]
        )

    print(f"   resid_pre[6] after patching resid_post[5] equals patch_value? {torch.allclose(capture['post_patch_pre_6'], patch_value)}")
    print(f"   Max diff: {(capture['post_patch_pre_6'] - patch_value).abs().max().item():.10e}")

    # Test 4: Reverse - patch resid_pre[6] and capture resid_post[5]
    print("\n[4] Patching resid_pre[6] and capturing resid_post[5]:")

    capture.clear()

    def capture_post_5(activation, **kwargs):
        capture['pre_patch_post_5'] = activation.clone()
        return activation

    def patch_pre_6(activation, **kwargs):
        capture['pre_patch_pre_6'] = activation.clone()
        return patch_value.clone()

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", capture_post_5),
                ("blocks.6.hook_resid_pre", patch_pre_6),
            ]
        )

    print(f"   resid_post[5] is UNAFFECTED by patching resid_pre[6]? {not torch.allclose(capture['pre_patch_post_5'], patch_value)}")
    print(f"   (This is expected - resid_post[5] runs BEFORE resid_pre[6])")

    # Test 5: THE KEY INSIGHT - Hook execution order
    print("\n[5] HOOK EXECUTION ORDER TEST:")

    order = []

    def order_hook(name):
        def hook(activation, **kwargs):
            order.append(name)
            return activation
        return hook

    with torch.no_grad():
        _ = model.run_with_hooks(
            test_input,
            fwd_hooks=[
                ("blocks.5.hook_resid_post", order_hook("5_resid_post")),
                ("blocks.6.hook_resid_pre", order_hook("6_resid_pre")),
            ]
        )

    print(f"   Execution order: {order}")
    print(f"   5_resid_post runs BEFORE 6_resid_pre? {'5_resid_post' == order[0] if order else 'EMPTY'}")

    # Test 6: What about when we use run_with_cache?
    print("\n[6] Testing run_with_cache behavior:")

    with torch.no_grad():
        logits, cache = model.run_with_cache(test_input)

    print(f"   cache['blocks.5.hook_resid_post'] == cache['blocks.6.hook_resid_pre']?")
    same_tensor = torch.allclose(
        cache['blocks.5.hook_resid_post'],
        cache['blocks.6.hook_resid_pre']
    )
    print(f"   {same_tensor}")

    # Test 7: Check if the issue might be in our codebase's specific usage
    print("\n[7] Investigating POSITION-SPECIFIC patching (potential source of bugs):")

    # Sometimes the issue is in how positions are handled
    # Let's test patching at a specific position

    def patch_post_5_pos2_only(activation, **kwargs):
        result = activation.clone()
        result[:, 2, :] = patch_value[:, 2, :]
        return result

    def patch_pre_6_pos2_only(activation, **kwargs):
        result = activation.clone()
        result[:, 2, :] = patch_value[:, 2, :]
        return result

    with torch.no_grad():
        logits_post = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", patch_post_5_pos2_only)]
        )
        logits_pre = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", patch_pre_6_pos2_only)]
        )

    print(f"   Position-specific patching: logits identical? {torch.allclose(logits_post, logits_pre, atol=1e-6)}")
    print(f"   Max diff: {(logits_post - logits_pre).abs().max().item():.10e}")

    # Test 8: Test the BOUNDARY case - layer 0 and layer n_layers-1
    print("\n[8] Boundary layer testing:")

    n_layers = model.cfg.n_layers
    print(f"   Model has {n_layers} layers")

    # resid_pre[0] is the embedding, resid_post[n_layers-1] is before unembed
    with torch.no_grad():
        _, cache = model.run_with_cache(test_input)

    # Check embedding boundary
    embed = cache['hook_embed'] if 'hook_embed' in cache else None
    resid_pre_0 = cache['blocks.0.hook_resid_pre']
    print(f"   hook_embed exists? {embed is not None}")
    if embed is not None:
        print(f"   hook_embed == resid_pre[0]? {torch.allclose(embed, resid_pre_0)}")

    # Check final layer
    resid_post_last = cache[f'blocks.{n_layers-1}.hook_resid_post']
    ln_final = cache['ln_final.hook_normalized'] if 'ln_final.hook_normalized' in cache else None
    print(f"   ln_final.hook_normalized exists? {ln_final is not None}")

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


def find_real_discrepancy():
    """
    Systematically search for ANY scenario where resid_post[L] != resid_pre[L+1].
    """
    print("\n" + "=" * 80)
    print("SYSTEMATIC SEARCH FOR DISCREPANCY")
    print("=" * 80)

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    test_inputs = [
        "The quick brown fox",
        "Hello world",
        "1 2 3 4 5",
        "",  # Empty string
        "A",  # Single character
    ]

    for test_input in test_inputs:
        if not test_input:
            continue  # Skip empty

        print(f"\n[Testing input: '{test_input[:30]}...']")

        with torch.no_grad():
            _, cache = model.run_with_cache(test_input)

        # Check ALL layer boundaries
        discrepancies = []
        for layer in range(model.cfg.n_layers - 1):
            post = cache[f'blocks.{layer}.hook_resid_post']
            pre = cache[f'blocks.{layer+1}.hook_resid_pre']

            if not torch.allclose(post, pre):
                max_diff = (post - pre).abs().max().item()
                discrepancies.append((layer, max_diff))

        if discrepancies:
            print(f"   DISCREPANCIES FOUND!")
            for layer, diff in discrepancies:
                print(f"   Layer {layer}: max_diff = {diff}")
        else:
            print(f"   All layer boundaries identical (as expected)")

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)


def test_intervention_code_patterns():
    """
    Test specific patterns that might be used in the experiment code.
    """
    print("\n" + "=" * 80)
    print("TESTING INTERVENTION CODE PATTERNS")
    print("=" * 80)

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()

    test_input = "The quick brown fox"

    # Get clean activations
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(test_input)

    clean_post_5 = clean_cache['blocks.5.hook_resid_post'].clone()
    clean_pre_6 = clean_cache['blocks.6.hook_resid_pre'].clone()

    # Get activations from a different input (corrupted)
    corrupted_input = "A different sentence here"
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_input)

    # Shapes might differ - need to handle
    corrupted_post_5 = corrupted_cache['blocks.5.hook_resid_post']
    corrupted_pre_6 = corrupted_cache['blocks.6.hook_resid_pre']

    print(f"\n[1] Clean input shape: {clean_post_5.shape}")
    print(f"   Corrupted input shape: {corrupted_post_5.shape}")

    # Pattern: Denoising patching (patch clean into corrupted run)
    print("\n[2] DENOISING PATTERN: Patch clean activations into corrupted run")

    # Make shapes match by padding/truncating
    min_len = min(clean_post_5.shape[1], corrupted_post_5.shape[1])
    patch_value = clean_post_5[:, :min_len, :].clone()

    def denoise_post_5(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = patch_value
        return result

    def denoise_pre_6(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = patch_value
        return result

    with torch.no_grad():
        logits_post = model.run_with_hooks(
            corrupted_input,
            fwd_hooks=[("blocks.5.hook_resid_post", denoise_post_5)]
        )
        logits_pre = model.run_with_hooks(
            corrupted_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", denoise_pre_6)]
        )

    print(f"   Denoising patching identical? {torch.allclose(logits_post, logits_pre, atol=1e-6)}")
    print(f"   Max diff: {(logits_post - logits_pre).abs().max().item():.10e}")

    # Pattern: Noising patching (patch corrupted into clean run)
    print("\n[3] NOISING PATTERN: Patch corrupted activations into clean run")

    patch_value_corrupted = corrupted_post_5[:, :min_len, :].clone()

    def noise_post_5(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = patch_value_corrupted
        return result

    def noise_pre_6(activation, hook):
        result = activation.clone()
        result[:, :min_len, :] = patch_value_corrupted
        return result

    with torch.no_grad():
        logits_post = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.5.hook_resid_post", noise_post_5)]
        )
        logits_pre = model.run_with_hooks(
            test_input,
            fwd_hooks=[("blocks.6.hook_resid_pre", noise_pre_6)]
        )

    print(f"   Noising patching identical? {torch.allclose(logits_post, logits_pre, atol=1e-6)}")
    print(f"   Max diff: {(logits_post - logits_pre).abs().max().item():.10e}")

    # Check if there's maybe a gradient-related issue
    print("\n[4] GRADIENT TEST: Does grad mode affect results?")

    # Note: run_with_hooks defaults to no grad context
    # But what if grad is enabled?

    def patch_fn(activation, hook):
        return patch_value.clone().requires_grad_(False)

    logits_post = model.run_with_hooks(
        test_input,
        fwd_hooks=[("blocks.5.hook_resid_post", patch_fn)]
    )
    logits_pre = model.run_with_hooks(
        test_input,
        fwd_hooks=[("blocks.6.hook_resid_pre", patch_fn)]
    )

    print(f"   No-grad patching identical? {torch.allclose(logits_post, logits_pre, atol=1e-6)}")

    print("\n" + "=" * 80)
    print("PATTERN TESTING COMPLETE")
    print("=" * 80)

    return True


if __name__ == "__main__":
    results = run_experiment()

    # Run advanced experiments
    run_advanced_experiment()

    # Check actual patterns
    check_actual_experiment_code()

    # Deep investigation
    investigate_hook_mechanism()

    # Systematic search
    find_real_discrepancy()

    # Test specific patterns
    test_intervention_code_patterns()
