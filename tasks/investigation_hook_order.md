# Investigation: Why resid_pre[L+1] and resid_post[L] Produce Different Patching Effects

## Executive Summary

**SURPRISING FINDING**: After extensive testing with actual TransformerLens, patching `hook_resid_pre[L+1]` and `hook_resid_post[L]` produces **IDENTICAL** downstream effects when the cached values are identical.

**If you're seeing different effects, the cause must be elsewhere.**

## Verified Findings

### 1. The Tensors ARE Identical
```python
# From actual TransformerLens testing:
resid_post[5] shape: torch.Size([1, 4, 768])
resid_pre[6] shape: torch.Size([1, 4, 768])
Are they exactly equal? True
Max diff: 0.0
```

### 2. Patching Either Hook Produces Identical Results
```python
# Tested with gpt2-small in TransformerLens:
Are patched logits exactly equal? True
Max logit diff: 0.0
Last position logit diff: 0.0

Top 5 tokens from resid_post[5] patch: [' on', ' in', ' down', ' at', ' next']
Top 5 tokens from resid_pre[6] patch: [' on', ' in', ' down', ' at', ' next']
```

### 3. Observation of Hook Flow During Patching

When patching `resid_post[5]`:
```
resid_pre[5]: -3.147998
resid_post[5] [PATCH]: -2.558491 -> -0.223222  (patched)
resid_pre[6]: -0.223222  (sees patched value)
resid_post[6]: 0.759105
```

When patching `resid_pre[6]`:
```
resid_pre[5]: -3.147998
resid_post[5]: -2.558491  (unchanged)
resid_pre[6] [PATCH]: -2.558491 -> -0.223222  (patched)
resid_post[6]: 0.759105  (same result!)
```

Both produce the same `resid_post[6]` value.

## Potential Causes of Observed Differences

If you're seeing different patching effects, investigate these:

### Cause 1: Different Activation Being Patched (MOST COMMON)
Check if you're actually comparing the same layer indices. Off-by-one errors are common.

```python
# WRONG: These are NOT equivalent!
cache["blocks.5.hook_resid_post"]  # Output of layer 5
cache["blocks.5.hook_resid_pre"]   # Input to layer 5 (= output of layer 4!)

# CORRECT: These ARE equivalent
cache["blocks.5.hook_resid_post"]  # Output of layer 5
cache["blocks.6.hook_resid_pre"]   # Input to layer 6 (= output of layer 5)
```

### Cause 2: Using `get_act_name` Incorrectly
```python
from transformer_lens import utils

# These map to the same activation:
utils.get_act_name("resid_post", 5)  # -> "blocks.5.hook_resid_post"
utils.get_act_name("resid_pre", 6)   # -> "blocks.6.hook_resid_pre"

# But NOT these:
utils.get_act_name("resid_post", 5)  # -> "blocks.5.hook_resid_post"
utils.get_act_name("resid_pre", 5)   # -> "blocks.5.hook_resid_pre" (DIFFERENT!)
```

### Cause 3: Different Cached Values Due to Hook Registration Order
If you registered caching hooks in a different order for `resid_post` vs `resid_pre`, and another hook modified the tensor, you might have cached different values.

### Cause 4: Patching at Different Positions
If you're patching specific positions, make sure the position indices match:
```python
# These should be identical if done correctly
corrupted[:, pos, :] = clean_cache["blocks.5.hook_resid_post"][:, pos, :]
corrupted[:, pos, :] = clean_cache["blocks.6.hook_resid_pre"][:, pos, :]
```

### Cause 5: Numerical Precision Issues
In rare cases, `.to(device)` operations between blocks can introduce tiny numerical differences. Check:
```python
diff = (resid_post_L - resid_pre_L_plus_1).abs().max()
print(f"Max difference: {diff.item()}")  # Should be 0.0 or very close
```

### Cause 6: Multi-GPU Setup
TransformerLens does `residual = residual.to(get_device_for_block_index(i, self.cfg))` between blocks. This could potentially cause issues if devices differ.

## Technical Details: Hook Mechanics

### How TransformerLens Hooks Work

From `transformer_lens/hook_points.py`:

```python
class HookPoint(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x  # Just identity - hooks do the work

    def add_hook(self, hook, ...):
        def full_hook(module, module_input, module_output):
            hook_result = hook(module_output, hook=self)
            return hook_result  # Return value replaces module output

        self.register_forward_hook(full_hook, prepend=prepend)
```

**Key insight**: When a hook returns a value, it REPLACES the module's output. The next module in the chain receives this modified value.

### Block Structure

From `transformer_lens/components/transformer_block.py`:

```python
def forward(self, resid_pre, ...):
    resid_pre = self.hook_resid_pre(resid_pre)  # <-- FIRST hook

    # ... attention and MLP computation ...

    resid_post = self.hook_resid_post(...)      # <-- LAST hook
    return resid_post
```

### Forward Pass Flow

From `transformer_lens/HookedTransformer.py`:

```python
def forward(self, input, ...):
    residual = self.input_to_embed(input, ...)

    for i, block in enumerate(self.blocks):
        residual = residual.to(device)  # Potential tensor copy
        residual = block(residual, ...)  # block returns hook_resid_post output

    residual = self.ln_final(residual)
    return self.unembed(residual)
```

The key observation: `residual` after `block(residual)` IS the output of `hook_resid_post`, which becomes the input to `hook_resid_pre` of the next block.

### Computation Flow Diagram

```
embed -> hook_resid_pre[0] -> Block 0 internals -> hook_resid_post[0]
                                                          |
                                                          v
         hook_resid_pre[1] <- (same tensor object) <------+
                |
                v
         Block 1 internals -> hook_resid_post[1]
                                    |
                                    v
         hook_resid_pre[2] <- ------+
                |
                v
         Block 2 internals -> hook_resid_post[2] -> ln_final -> unembed
```

## Verification Code

To verify the hooks produce identical effects:

```python
import torch
import transformer_lens

model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

clean_tokens = model.to_tokens("The cat sat")
corrupted_tokens = model.to_tokens("The dog ran")

# Get clean cache
_, cache = model.run_with_cache(clean_tokens)

# Verify tensors are identical
L = 5
resid_post_L = cache[f"blocks.{L}.hook_resid_post"]
resid_pre_L1 = cache[f"blocks.{L+1}.hook_resid_pre"]
print(f"Tensors equal: {torch.equal(resid_post_L, resid_pre_L1)}")

# Patch at specific position
pos = -1

def patch_hook(clean_val, pos):
    def hook(tensor, hook):
        tensor[:, pos, :] = clean_val[:, pos, :]
        return tensor
    return hook

# Test 1: Patch resid_post[L]
logits_1 = model.run_with_hooks(
    corrupted_tokens,
    fwd_hooks=[(f"blocks.{L}.hook_resid_post", patch_hook(resid_post_L, pos))]
)

# Test 2: Patch resid_pre[L+1]
logits_2 = model.run_with_hooks(
    corrupted_tokens,
    fwd_hooks=[(f"blocks.{L+1}.hook_resid_pre", patch_hook(resid_pre_L1, pos))]
)

print(f"Logits equal: {torch.equal(logits_1, logits_2)}")
print(f"Max diff: {(logits_1 - logits_2).abs().max().item()}")
```

## Debugging Checklist

If you see different effects:

1. **Verify layer indices**: `resid_post[L]` should match `resid_pre[L+1]`, NOT `resid_pre[L]`
2. **Print cached values**: `torch.equal(cache[resid_post_L], cache[resid_pre_L1])`
3. **Check position indices**: Are you patching the same positions?
4. **Add logging hooks**: Print tensor values before and after patching
5. **Check for other hooks**: Are there other hooks that might interfere?
6. **Check device consistency**: Are tensors on the same device?

## Conclusion

**The hooks ARE interchangeable for patching when used correctly.** The timing difference between `hook_resid_post[L]` (end of block L) and `hook_resid_pre[L+1]` (start of block L+1) does NOT cause different downstream effects because they pass through the same value to subsequent computation.

If you're seeing different effects:
1. Double-check layer indices (most common error)
2. Verify the cached tensors are actually identical
3. Ensure patch positions match
4. Look for other hooks that might interfere
5. Check for device/precision issues
