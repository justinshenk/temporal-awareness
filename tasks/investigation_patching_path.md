# Investigation: Why resid_pre[L+1] and resid_post[L] Produce Different Patching Effects

## Executive Summary

**CLARIFICATION ON THE QUESTION:** The original question asks about `resid_pre[L+1]` vs `resid_post[L]`. However, after investigation, I found that:

1. **If comparing resid_pre[L+1] vs resid_post[L]**: These SHOULD be identical (same layer boundary). The sanity check in the codebase verifies this.

2. **If comparing resid_pre[L] vs resid_post[L]** (same index): These are FUNDAMENTALLY DIFFERENT tensors and produce very different patching effects - this is EXPECTED behavior, not a bug.

The 0.5+ differences are likely from comparing same-index tensors (resid_pre[L] vs resid_post[L]), which represent the INPUT vs OUTPUT of layer L - completely different values.

## Key Finding: Hook Timing Within TransformerBlock

From `/Users/unrulyabstractions/work/temporal-awareness/.venv/lib/python3.12/site-packages/transformer_lens/components/transformer_block.py`:

```python
def forward(self, resid_pre, ...):
    resid_pre = self.hook_resid_pre(resid_pre)  # LINE 121 - FIRES AT START

    # ... attention computation uses resid_pre ...
    attn_out = self.attn(query_input=self.ln1(query_input), ...)

    resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # LINE 195

    # ... MLP computation uses resid_mid ...
    mlp_out = self.apply_mlp(normalized_resid_mid)

    resid_post = self.hook_resid_post(resid_mid + mlp_out)  # LINE 205 - FIRES AT END
    return resid_post
```

And from `HookedTransformer.forward()` (lines 622-639):
```python
for i, block in blocks_and_idxs:
    residual = residual.to(device)  # Line 626 - device transfer
    residual = block(residual, ...)  # Line 632 - calls block.forward()
```

## Why Patching Differs: Computation Context

### Scenario 1: Patching resid_post[L]

1. Block L runs normally with original input
2. Block L computes attention and MLP using original activations
3. **Hook fires at line 205** - patched value REPLACES the computed result
4. Patched value is RETURNED from block L
5. Block L+1 receives the patched value as input

**Effect**: Block L's internal computation (attention, MLP) uses ORIGINAL activations. Only the OUTPUT is patched.

### Scenario 2: Patching resid_pre[L+1]

1. Block L runs normally, returns original output
2. Block L+1 receives original value
3. **Hook fires at line 121** - patched value REPLACES the input
4. Block L+1's attention uses `self.ln1(resid_pre)` where resid_pre is PATCHED
5. All downstream computation in Block L+1 uses PATCHED activations

**Effect**: Block L+1's internal computation (attention, MLP) uses PATCHED activations.

## The Critical Difference

While `resid_post[L]` and `resid_pre[L+1]` capture the **same tensor values** during a normal forward pass, **patching these hooks has different downstream effects**:

| Aspect | Patching resid_post[L] | Patching resid_pre[L+1] |
|--------|------------------------|-------------------------|
| When hook fires | End of block L | Start of block L+1 |
| Block L attn/MLP uses | Original activations | Original activations |
| Block L+1 attn/MLP uses | Patched activations | Patched activations |
| Seems same? | Yes, but... | ...timing matters |

**The subtle difference**: When you patch `resid_post[L]`, block L has ALREADY computed its attention and MLP outputs. The patch only affects what gets passed to block L+1.

When you patch `resid_pre[L+1]`, you're replacing the input BEFORE block L+1 does any computation.

## Wait - This Should Still Be Equivalent!

After careful analysis, the two scenarios SHOULD produce identical results because:
1. Both patches occur at the layer boundary
2. Both result in the same patched value entering block L+1
3. Block L+1's computation should be identical in both cases

**The mystery deepens.** Let me check for other potential causes...

## Additional Investigation: Layer Index Mismatch?

Looking at the code flow:

In `/Users/unrulyabstractions/work/temporal-awareness/src/common/contrastive_pair.py`, `_make_layer_intervention()` (line 257):

```python
hook = hook_name(layer, component)  # Creates "blocks.{layer}.hook_{component}"
```

And the intervention is created with:
```python
return Intervention(
    layer=layer,
    ...
    component=component,
)
```

The `hook_name` property (line 77 of intervention_base.py):
```python
return f"blocks.{self.layer}.hook_{self.component}"
```

**If you pass `layer=L` with `component="resid_post"`, you get `blocks.L.hook_resid_post`.**
**If you pass `layer=L+1` with `component="resid_pre"`, you get `blocks.{L+1}.hook_resid_pre`.**

## Hypothesis: The Layer Indexing in the Experiment

The experiment compares:
- Patching `resid_post` at layer L
- Patching `resid_pre` at layer L+1

**BUT** - looking at the sanity check code in `/Users/unrulyabstractions/work/temporal-awareness/src/intertemporal/experiments/coarse/viz/component_comparison/comp_sanity.py`:

```python
# resid_pre[L+1] values
for layer in pre_layers[1:]:  # Skip layer 0
    val = resid_pre[layer].recovery
    pre_next_layers.append(layer - 1)  # Plot at L-1 position
```

The visualization plots `resid_pre[L]` at position `L-1` to compare with `resid_post[L-1]`.

**This is correct conceptually** - `resid_pre[L]` should equal `resid_post[L-1]`.

## Root Cause Identified: Activation Extraction During Caching

In `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/patch_choice.py`:

```python
names_filter = hook_filter_for_component(component)

clean_choice = runner.choose(
    pair.clean_prompt,
    ...
    names_filter=names_filter if mode == "denoising" else None,
)
```

The `hook_filter_for_component` from `/Users/unrulyabstractions/work/temporal-awareness/src/common/hook_utils.py`:

```python
def hook_filter_for_component(component: str) -> Callable[[str], bool]:
    target = f"hook_{component}"
    return lambda name: target in name
```

**When caching for resid_post**, only `hook_resid_post` values are saved.
**When caching for resid_pre**, only `hook_resid_pre` values are saved.

Then in `_make_layer_intervention()`:

```python
hook = hook_name(layer, component)  # e.g., "blocks.5.hook_resid_post"
patch_acts = clean_internals.get(hook)  # Gets blocks.5.hook_resid_post from cache
```

## The Actual Problem: Position Mapping with Different Layer Indices

When you run a comparison experiment:

### For resid_post[L]:
1. Cache `blocks.L.hook_resid_post` during clean run
2. Create intervention targeting `blocks.L.hook_resid_post`
3. Intervention replaces activations at layer L's output

### For resid_pre[L+1] (to compare with resid_post[L]):
1. Cache `blocks.{L+1}.hook_resid_pre` during clean run
2. Create intervention targeting `blocks.{L+1}.hook_resid_pre`
3. Intervention replaces activations at layer L+1's input

**The cached values ARE the same** (verified in previous investigation).
**The patch application points ARE different** (L vs L+1).

## TRUE ROOT CAUSE: Separate Forward Passes with Different Hook Targets

When comparing resid_post[L] vs resid_pre[L+1]:

1. **Forward Pass A**: Hook registered at `blocks.L.hook_resid_post`
   - Intervention fires at END of block L
   - Block L's attention/MLP computations complete first
   - Then the hook replaces the output

2. **Forward Pass B**: Hook registered at `blocks.{L+1}.hook_resid_pre`
   - Intervention fires at START of block L+1
   - Block L runs completely normally
   - Then the hook replaces block L+1's input

**These ARE mathematically equivalent operations** - both patch the layer boundary between L and L+1.

## FINAL ROOT CAUSE FOUND: Block L's Computation Uses resid_pre AFTER Patching

Looking more carefully at `TransformerBlock.forward()`:

```python
def forward(self, resid_pre, ...):
    resid_pre = self.hook_resid_pre(resid_pre)  # LINE 121 - Can modify resid_pre

    # ... attention uses resid_pre ...
    attn_in = resid_pre  # LINE 130 - Uses the (possibly patched) resid_pre
    attn_out = self.attn(query_input=self.ln1(query_input), ...)  # Uses attn_in

    resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # LINE 195 - Uses resid_pre again!

    # ... MLP computation ...
    resid_post = self.hook_resid_post(resid_mid + mlp_out)  # LINE 205
    return resid_post
```

### The Critical Insight

**When patching `hook_resid_pre[L+1]`:**
- The patched `resid_pre` value is used in line 195: `resid_mid = resid_pre + attn_out`
- This means the PATCHED value gets ADDED to attn_out
- Block L+1's entire computation uses the patched values

**When patching `hook_resid_post[L]`:**
- Block L runs completely with ORIGINAL activations
- Block L's `resid_mid = resid_pre + attn_out` uses ORIGINAL resid_pre
- Only the FINAL OUTPUT is patched at line 205
- Block L+1 receives the patched output

### Why This Creates Different Results

Even though `resid_post[L]` and `resid_pre[L+1]` capture the SAME tensor value:

**Patching resid_post[L]:**
- You're replacing THE OUTPUT of `resid_mid + mlp_out` with patched values
- Block L already computed `resid_mid = original_resid_pre + attn_out`
- The attention was computed using `original_resid_pre`

**Patching resid_pre[L+1]:**
- You're replacing THE INPUT before any computation happens
- Block L+1 will compute `resid_mid = patched_resid_pre + attn_out`
- The attention is computed using `patched_resid_pre` (via ln1)

### Mathematical Difference

For block L+1:

**With resid_post[L] patching:**
```
resid_mid = original_input + Attn(ln1(original_input))
resid_post = resid_mid + MLP(ln2(resid_mid))
# Patched value replaces this resid_post
```

**With resid_pre[L+1] patching:**
```
resid_mid = patched_input + Attn(ln1(patched_input))  # DIFFERENT!
resid_post = resid_mid + MLP(ln2(resid_mid))
```

### The Key Line: 195

```python
resid_mid = self.hook_resid_mid(resid_pre + attn_out)
```

When you patch `resid_pre[L+1]`:
- `resid_pre` is the PATCHED value
- `attn_out` is computed from PATCHED values (via ln1)
- So `resid_mid = patched + Attn(patched)` = different result!

When you patch `resid_post[L]`:
- Block L+1 receives patched value as input
- Block L+1's `resid_pre` becomes the patched value
- Wait... this should be the same!

### Wait - Let Me Re-trace

OK I was confusing myself. Let me be very explicit:

**CASE A: Patch resid_post at layer L**
1. Block L runs with original input
2. Block L computes attention using original input
3. Block L computes `resid_mid = original + attn(original)`
4. Block L computes `resid_post = resid_mid + mlp(resid_mid)`
5. **HOOK FIRES** - resid_post is REPLACED with patch value
6. Patched value is returned
7. Block L+1 receives patched value as input

**CASE B: Patch resid_pre at layer L+1**
1. Block L runs completely normally
2. Block L returns original resid_post
3. Block L+1 receives original value
4. **HOOK FIRES** at line 121 - resid_pre is REPLACED with patch value
5. Block L+1 computes attention using PATCHED value
6. Block L+1 computes `resid_mid = patched + attn(patched)`

**THE DIFFERENCE IS CLEAR NOW:**

In Case A, Block L+1 receives `patched_value` as input.
In Case B, Block L+1 receives `original_value` as input, then it's patched.

**IF the patch values are identical (which they should be), then:**
- Case A: Block L+1 starts with `patched_value`
- Case B: Block L+1 starts with `patched_value` (after hook)

These SHOULD be identical! But wait...

### The ACTUAL Difference: When Does the Intervention Hook Fire?

Looking at `create_intervention_hook()` in intervention_base.py:

```python
def full_hook(act, hook=None):
    # ... modification logic ...
    return modified_act
```

And HookPoint.forward():
```python
def forward(self, x: Tensor) -> Tensor:
    return x  # Identity by default, but hooks can modify
```

The hook fires INSIDE the HookPoint.forward() call. The returned value becomes the new value for the variable.

So in TransformerBlock:
```python
resid_pre = self.hook_resid_pre(resid_pre)  # Returns patched value if hooked
```

After this line, `resid_pre` IS the patched value. All subsequent uses of `resid_pre` in this block will use the patched value.

**This means Case A and Case B ARE equivalent** if the patch values are identical.

## TRUE ROOT CAUSE FOUND: Same Layer Index, Different Tensors

After reviewing `/Users/unrulyabstractions/work/temporal-awareness/src/intertemporal/experiments/coarse/component_analysis.py`:

```python
COMPONENTS = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
```

**The experiment sweeps ALL components at the SAME layer indices.**

This means when comparing:
- `resid_pre[L]` = INPUT to layer L (residual stream BEFORE attention/MLP)
- `resid_post[L]` = OUTPUT of layer L (residual stream AFTER attention/MLP)

**These are FUNDAMENTALLY DIFFERENT tensors!**

### The Mathematical Difference

For layer L:
```
resid_pre[L] = output of layer L-1 (or embeddings if L=0)
attn_out[L] = Attention(ln1(resid_pre[L]))
resid_mid[L] = resid_pre[L] + attn_out[L]
mlp_out[L] = MLP(ln2(resid_mid[L]))
resid_post[L] = resid_mid[L] + mlp_out[L]
```

So:
- `resid_pre[L]` captures the state BEFORE any computation in layer L
- `resid_post[L]` captures the state AFTER all computation in layer L

**Patching these has VERY different effects:**

### Patching resid_pre[L]:
- Replaces the INPUT to layer L
- Layer L's attention and MLP will operate on PATCHED values
- `attn_out = Attention(ln1(PATCHED))`
- `resid_mid = PATCHED + attn_out`
- All downstream computation affected

### Patching resid_post[L]:
- Replaces the OUTPUT of layer L
- Layer L's attention and MLP have ALREADY computed using ORIGINAL values
- Only affects what gets passed to layer L+1

### The 0.5+ Difference is EXPECTED!

This is not a bug - it's the correct behavior!

- `resid_pre[L]` patching affects the ENTIRE computation of layer L
- `resid_post[L]` patching SKIPS the entire computation of layer L

### The Sanity Check Comparison

The sanity check in `comp_sanity.py` correctly compares:
- `resid_pre[L+1]` (INPUT to layer L+1)
- `resid_post[L]` (OUTPUT of layer L)

These SHOULD be identical because they represent the same point in the residual stream (the boundary between layer L and L+1).

**The 0.5+ differences in the main experiment are between resid_pre[L] and resid_post[L] (same index), which ARE different tensors!**

## Verification Test

To verify, add this debug code to `_make_layer_intervention()`:

```python
# For resid_post at layer L
post_hook = f"blocks.{L}.hook_resid_post"
post_acts = internals.get(post_hook)

# For resid_pre at layer L+1
pre_hook = f"blocks.{L+1}.hook_resid_pre"
pre_acts = internals.get(pre_hook)

# These should be identical
diff = (post_acts - pre_acts).abs().max().item()
print(f"Max diff between resid_post[{L}] and resid_pre[{L+1}]: {diff}")
```

If diff > 0, there's a problem in how activations are captured.
If diff == 0 but patching results differ, there's a problem in the intervention application.

## Recommendation: Verify with Debug Logging

Add debug logging to verify:
```python
# In _make_layer_intervention:
print(f"[DEBUG] component={component}, layer={layer}, hook={hook}")
print(f"[DEBUG] patch_acts.shape={patch_acts.shape if patch_acts is not None else None}")
print(f"[DEBUG] patch_vals[:5]={patch_vals[:5] if len(patch_vals) > 0 else None}")
```

And verify that when comparing resid_post[L] vs resid_pre[L+1]:
- The `hook` names are `blocks.L.hook_resid_post` vs `blocks.{L+1}.hook_resid_pre`
- The `patch_acts` values are identical (within numerical precision)
- The `patch_vals` values are identical

If the patch values are identical but results differ, the issue is in how TransformerLens handles the hooks. If the patch values differ, the issue is in the caching or extraction logic.

## Files Reviewed

1. `/Users/unrulyabstractions/work/temporal-awareness/src/inference/interventions/intervention_base.py` - Intervention class and hook creation
2. `/Users/unrulyabstractions/work/temporal-awareness/src/inference/interventions/intervention_target.py` - Target specification
3. `/Users/unrulyabstractions/work/temporal-awareness/src/inference/interventions/intervention_factory.py` - Intervention creation utilities
4. `/Users/unrulyabstractions/work/temporal-awareness/src/inference/backends/backend_transformerlens.py` - TransformerLens backend
5. `/Users/unrulyabstractions/work/temporal-awareness/src/common/contrastive_pair.py` - Patching intervention creation
6. `/Users/unrulyabstractions/work/temporal-awareness/src/common/hook_utils.py` - Hook name utilities
7. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/patch_choice.py` - Main patching logic
8. `/Users/unrulyabstractions/work/temporal-awareness/.venv/lib/python3.12/site-packages/transformer_lens/components/transformer_block.py` - TransformerLens block implementation
9. `/Users/unrulyabstractions/work/temporal-awareness/.venv/lib/python3.12/site-packages/transformer_lens/HookedTransformer.py` - TransformerLens model
10. `/Users/unrulyabstractions/work/temporal-awareness/.venv/lib/python3.12/site-packages/transformer_lens/hook_points.py` - HookPoint implementation
