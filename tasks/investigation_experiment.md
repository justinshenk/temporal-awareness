# Investigation: resid_pre[L+1] vs resid_post[L] Patching Effects

## Executive Summary

**FINDING: resid_post[L] and resid_pre[L+1] patching produce IDENTICAL results in TransformerLens.**

After extensive controlled experiments, we found **NO DISCREPANCY** between patching `resid_post[L]` and `resid_pre[L+1]` with the same value. The tensors are mathematically and physically identical (same tensor ID, same memory address).

If you are seeing different patching effects in your actual experiments, the source must be **outside** the basic hook mechanism.

---

## Controlled Experiment Results

### Basic Patching Test
```
Patching resid_post[5] with random tensor -> logits_A
Patching resid_pre[6] with SAME tensor    -> logits_B

Result: logits_A == logits_B (max diff: 0.0)
```

### Critical Verification
```
B5_resid_post tensor id == B6_resid_pre tensor id? True
B5_resid_post data_ptr == B6_resid_pre data_ptr? True
```

**They are literally the same tensor in memory.** TransformerLens passes the output of `hook_resid_post[L]` directly as the input to `hook_resid_pre[L+1]` without any modification.

### Hook Execution Order
```
Execution order: ['5_resid_post', '6_resid_pre']
```
Hooks execute in forward order: `resid_post[5]` runs BEFORE `resid_pre[6]`.

### Cascading Effect Verification
```
Patching resid_post[5] -> resid_pre[6] receives the patched value? True
Patching resid_pre[6] -> resid_post[5] is unaffected? True (as expected)
```

---

## Additional Tests Performed

All tests showed **no difference** between resid_post[L] and resid_pre[L+1] patching:

| Test | Result |
|------|--------|
| Partial position patching (pos 2 only) | Identical |
| Cross-input patching | Identical |
| Zero ablation | Identical |
| Mean ablation | Identical |
| Multiple layer patching | Identical |
| Batch processing | Identical |
| Denoising pattern (clean into corrupted) | Identical |
| Noising pattern (corrupted into clean) | Identical |
| With/without gradient context | Identical |
| View vs clone tensor | Identical |
| In-place vs new tensor modification | Identical |

### Test That DID Show Difference
```
Patching resid_pre[6] vs resid_mid[6]: Max diff = 27.05
```
This is expected - `resid_mid` is AFTER the attention operation within the block, so patching `resid_pre` vs `resid_mid` will differ because attention happens between them.

---

## Boundary Cases

### Embedding Boundary
```
hook_embed == resid_pre[0]? False
```
The embedding output and `resid_pre[0]` are NOT identical in GPT-2 with TransformerLens. This is because TransformerLens adds positional embeddings between `hook_embed` and `hook_resid_pre[0]`.

### Layer Boundaries (All L)
All layer boundaries checked - all identical:
```
blocks.{L}.hook_resid_post == blocks.{L+1}.hook_resid_pre for all L in [0, n_layers-2]
```

---

## Possible Sources of Observed Discrepancy

If your actual experiment shows different effects, investigate these:

### 1. Layer Indexing Bug
```python
# WRONG: off-by-one error in layer indexing
resid_post_L = cache[f'blocks.{layer}.hook_resid_post']
resid_pre_L = cache[f'blocks.{layer}.hook_resid_pre']  # Should be layer+1!

# CORRECT
resid_post_L = cache[f'blocks.{layer}.hook_resid_post']
resid_pre_Lp1 = cache[f'blocks.{layer+1}.hook_resid_pre']
```

### 2. Component Confusion
Are you perhaps comparing `resid_pre` with `resid_mid` instead of `resid_pre` with `resid_post`?
```
resid_pre[L] -> attention -> resid_mid[L] -> MLP -> resid_post[L] -> resid_pre[L+1]
```
Patching `resid_pre[L]` vs `resid_mid[L]` WILL differ because attention runs between them.

### 3. Different Patch Values Being Used
Check that you're using the EXACT same tensor values:
```python
# Bug: Different random seeds
torch.manual_seed(42)
patch_for_post = torch.randn(...)  # Seed 42
# ... some code ...
patch_for_pre = torch.randn(...)   # Seed exhausted, different values!
```

### 4. Position Mismatch
Ensure positions align correctly:
```python
# If patch_values has shape [seq_len, d_model] and input has different seq_len
# The patching might apply to different positions
```

### 5. Experiment Caching/Stale State
If running multiple experiments in sequence, ensure model state is properly reset:
```python
model.reset_hooks()  # Call between experiments
```

### 6. Attribution vs Activation Patching
Attribution patching uses gradients and may have different semantics:
- Attribution patching measures gradient-weighted effects
- Activation patching directly substitutes values
These can show different patterns even when the underlying tensors are identical.

---

## TransformerLens Block Architecture (GPT-2)

```
Input: resid_pre[L] (= resid_post[L-1])
        |
        v
    [hook_resid_pre]
        |
        v
    LayerNorm1 (ln1)
        |
        v
    Attention
        |
        v
    [hook_attn_out]
        |
        +---> ADD to residual
        |
        v
    [hook_resid_mid]
        |
        v
    LayerNorm2 (ln2)
        |
        v
    MLP
        |
        v
    [hook_mlp_out]
        |
        +---> ADD to residual
        |
        v
    [hook_resid_post] ---> resid_pre[L+1]
```

---

## Conclusion

**The TransformerLens hook mechanism is correct.** Patching `resid_post[L]` and `resid_pre[L+1]` with the same value will always produce identical downstream effects.

If you're observing different patching effects in your experiment, the bug is in:
1. The experimental code (layer indexing, component naming, etc.)
2. The analysis/visualization code
3. How patch values are generated or stored

To debug: Add explicit assertions that verify the patch values are identical before patching, and log the exact hook names being used.

---

## Test Script Location

Full test script: `/Users/unrulyabstractions/work/temporal-awareness/tasks/test_controlled_patch.py`

Run with:
```bash
uv run python tasks/test_controlled_patch.py
```
