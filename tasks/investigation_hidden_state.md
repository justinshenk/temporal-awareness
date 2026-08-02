# Investigation: Why resid_pre[L+1] and resid_post[L] Produce Different Patching Effects

## Executive Summary

After thorough investigation of the TransformerLens source code and this project's backend code, I found that **resid_pre[L+1] and resid_post[L] are mathematically identical tensors** - the same tensor passes through both hooks without any transformation. However, **patching on these hooks can produce different effects due to the timing of when hooks fire and how the modified value propagates**.

## Key Finding: Direct Tensor Pass-Through

Looking at `HookedTransformer.forward()` (lines 621-639):

```python
for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
    residual = residual.to(get_device_for_block_index(i, self.cfg))
    residual = block(
        residual,
        past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
        shortformer_pos_embed=shortformer_pos_embed,
        attention_mask=attention_mask,
    )
```

And in `TransformerBlock.forward()` (lines 103-218):

```python
def forward(self, resid_pre, ...):
    resid_pre = self.hook_resid_pre(resid_pre)  # HOOK FIRES HERE

    # ... attention computation ...

    resid_post = self.hook_resid_post(resid_mid + mlp_out)  # HOOK FIRES HERE
    return resid_post  # This becomes next layer's input
```

**The flow is:**
1. Block L's `resid_post` value is returned
2. HookedTransformer passes it directly to Block L+1 as `resid_pre` argument
3. Block L+1's `hook_resid_pre` fires on this same tensor

**There is NO computation between `resid_post[L]` and `resid_pre[L+1]`.**

## Why Patching Can Still Produce Different Effects

### 1. Hook Execution Order and Return Value Semantics

When you patch `resid_post[L]`:
- The hook fires at the END of block L
- The **returned value** from the hook replaces the tensor
- This modified value is what gets passed to block L+1

When you patch `resid_pre[L+1]`:
- The hook fires at the START of block L+1
- The **returned value** from the hook replaces the tensor
- This modified value is used for block L+1's computation

**If the hook returns the modified value correctly, both should have identical effects.**

### 2. Potential Divergence: Hook Return Value Handling

Looking at `HookPoint.forward()` in `hook_points.py` (line 360):

```python
def forward(self, x: Tensor) -> Tensor:
    return x
```

The HookPoint is an identity function. The hooks registered via `add_hook()` modify the behavior through PyTorch's hook mechanism.

In `hook_points.py` (lines 192-232), the full_hook wrapper:

```python
def full_hook(module, module_input, module_output):
    # ... conversion logic ...
    hook_result = hook(module_output, hook=self)

    if hook_result is not None and self.hook_conversion is not None:
        hook_result = self.hook_conversion.revert(hook_result)

    return hook_result  # This replaces the output!
```

**Critical: If the hook function returns `None` instead of the modified tensor, the modification is lost!**

### 3. Potential Issue: In-Place Modification vs. Return

Looking at `intervention_base.py`'s `create_intervention_hook()`:

```python
def hook(act, hook=None):
    for i, pos in enumerate(positions):
        if pos < act.shape[1]:
            act[:, pos] = _apply_position(act[:, pos], v, mode, tv, alpha)
    return act  # IMPORTANT: Returns the modified tensor
```

The code correctly returns the modified tensor. However, note that:
- For position-specific patches: `act[:, pos] = ...` modifies in-place
- The function still returns `act`

**If there's any code path that forgets to return, the patch would be lost.**

## Investigation: No Hidden State Found

I searched for:
1. **KV Cache effects**: The KV cache (`TransformerLensKeyValueCache`) is passed to each block but does not affect the residual stream directly - it only caches keys/values for attention
2. **LayerNorm between layers**: There is NO LayerNorm applied between `resid_post[L]` and `resid_pre[L+1]`. LayerNorm is applied:
   - Before attention: `self.ln1(query_input)`
   - Before MLP: `self.ln2(mlp_in)`
   - At the very end: `self.ln_final(residual)` (before unembedding)
3. **Device transfers**: The only operation between blocks is `residual.to(device)` for multi-GPU, which preserves values
4. **Any hook side effects**: Hooks don't have persistent side effects - they only modify the value they receive

## Most Likely Explanation for Observed Differences

If verified identical tensors produce different patching effects, the most likely causes are:

### A. Cache Key Mismatch
When caching activations, the cache key is the hook name. If you're:
- Caching with `resid_post` filter
- But creating interventions targeting `resid_pre`

The interventions won't find the cached values because the keys don't match!

### B. Layer Index Off-by-One
If you're patching `resid_pre[L+1]` with values from `resid_post[L]`, you need to:
- Get activations from `blocks.L.hook_resid_post`
- Apply them at `blocks.(L+1).hook_resid_pre`

If the layer indices aren't adjusted correctly, you might be patching the wrong layer.

### C. Intervention Target Resolution
In `contrastive_pair.py` (line 191):
```python
layers = target.resolve_layers(available)
```

If `available` layers don't include the requested layer (because the wrong hook names were cached), the intervention would be skipped.

## Verification Steps

1. **Verify tensor identity**: Add debug logging to confirm `resid_post[L]` tensor id equals `resid_pre[L+1]` tensor id at runtime
2. **Check cache keys**: Print all keys in the activation cache to verify correct hooks are being captured
3. **Verify intervention creation**: Print the hook names in the created interventions to ensure they match what was intended
4. **Check intervention application**: Add debug prints in `create_intervention_hook` to confirm the hook is actually being called

## Conclusion

**There is no hidden computation between resid_post[L] and resid_pre[L+1].** They are the same tensor. If patching produces different effects, it's most likely due to:

1. **Cache key mismatch** - caching one hook type but patching another
2. **Layer index confusion** - not adjusting indices when translating between post and pre
3. **Bug in intervention creation or application** - hook not being registered or not returning the modified value

The TransformerLens architecture guarantees that `resid_post[L]` and `resid_pre[L+1]` are mathematically identical. Any observed differences must come from the patching infrastructure, not the model itself.

---

## Appendix: Project-Specific Code Paths

### This Project's Hook Name Generation

From `src/common/hook_utils.py`:
```python
def hook_name(layer: int, component: str) -> str:
    """Generate hook name: blocks.{layer}.hook_{component}"""
    if component == "attn_z":
        return f"blocks.{layer}.attn.hook_z"
    return f"blocks.{layer}.hook_{component}"
```

### Components Defined in `src/common/patching_types.py`:
```python
COMPONENTS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z")
PATCHING_COMPONENTS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z")
```

### Intervention Flow in This Project

1. **Activation Caching** (`patch_choice.py`):
   ```python
   names_filter = hook_filter_for_component(component)
   clean_choice = runner.choose(..., names_filter=names_filter)
   ```
   - Caches activations for a specific component (e.g., `resid_post`)
   - Cache keys are full hook names like `blocks.5.hook_resid_post`

2. **Intervention Creation** (`contrastive_pair.py`):
   ```python
   hook = hook_name(layer, component)  # e.g., "blocks.5.hook_resid_post"
   patch_acts = clean_internals.get(hook)  # Look up in cache
   ```
   - Creates intervention for each layer in target
   - **Must use same component for cache lookup and intervention!**

3. **Intervention Application** (`backend_transformerlens.py`):
   ```python
   for intervention in interventions:
       hook_fn, _ = create_intervention_hook(...)
       fwd_hooks.append((intervention.hook_name, hook_fn))
   ```
   - Registers hook at `intervention.hook_name`
   - **Hook name must match the patching target!**

### Key Insight: The Patching Component Determines Everything

If you:
1. Run with `component="resid_post"` (caches `blocks.L.hook_resid_post`)
2. Create intervention with `component="resid_pre"` (hooks `blocks.L.hook_resid_pre`)

You would be patching layer L's `resid_pre` with values from layer L's `resid_post` - which are **different tensors within the same layer**!

To patch `resid_pre[L+1]` with `resid_post[L]` values:
1. Cache `resid_post` at layer L
2. Create intervention targeting `resid_pre` at layer L+1
3. **Manually adjust the layer index when creating the intervention**

This is likely NOT what the current code does - it probably uses the same layer for both cache and intervention.

### CONFIRMED: Same Component Used Throughout

Looking at `_make_layer_intervention()` in `contrastive_pair.py`:

```python
def _make_layer_intervention(self, layer, component, target, ...):
    hook = hook_name(layer, component)  # Same component for lookup
    patch_acts = clean_internals.get(hook)  # Get from cache
    # ...
    return Intervention(
        layer=layer,
        component=component,  # Same component for target
        ...
    )
```

**The code uses the SAME component for both:**
1. Looking up cached activations (`clean_internals.get(hook)`)
2. Creating the intervention (`component=component`)

This means if you request `component="resid_post"`:
- It looks for `blocks.L.hook_resid_post` in cache
- It creates intervention targeting `blocks.L.hook_resid_post`

And if you request `component="resid_pre"`:
- It looks for `blocks.L.hook_resid_pre` in cache
- It creates intervention targeting `blocks.L.hook_resid_pre`

**There is no built-in support for "patch resid_pre[L+1] with values from resid_post[L]".**

### Root Cause Hypothesis

If someone is comparing:
- Patching `resid_post` at layer L
- Patching `resid_pre` at layer L

**These target different tensors!** In the data flow:
```
resid_pre[L] -> attention -> resid_mid[L] -> MLP -> resid_post[L]
                                                        |
                                                        v
                                              resid_pre[L+1] (same tensor)
```

To get equivalent effects, you would need to:
- Patch `resid_post[L]` OR
- Patch `resid_pre[L+1]` (not `resid_pre[L]`)

If someone patches `resid_pre[L]` expecting the same effect as `resid_post[L]`, they will see different results because `resid_pre[L]` is the input to layer L, while `resid_post[L]` is the output.
