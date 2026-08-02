# Investigation: Metric Calculation for resid_pre[L+1] vs resid_post[L]

## Executive Summary

**FINDING: The metric calculation code treats ALL components identically.** There is NO component-specific logic in how recovery/disruption are computed. The metrics are purely a function of the logit differences from the model output, not the component being patched.

This means if resid_pre[L+1] and resid_post[L] produce DIFFERENT patching effects despite being VERIFIED IDENTICAL tensors, the difference MUST come from one of:
1. How the hook names are generated/matched
2. How activations are stored/retrieved from the cache
3. How the intervention is applied by the backend

## Detailed Analysis

### 1. Metric Calculation Flow

The recovery/disruption metrics are computed in `src/common/math/faithfulness_scores.py`:

```python
def compute_recovery(y_intervened: float, y_clean: float, y_corrupted: float) -> float:
    """Raw recovery toward clean: R = (y_intervened - y_corrupted) / (y_clean - y_corrupted)."""
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_intervened - y_corrupted) / delta

def compute_disruption(y_intervened: float, y_clean: float, y_corrupted: float) -> float:
    """Raw disruption toward corrupt: D = (y_clean - y_intervened) / (y_clean - y_corrupted)."""
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_clean - y_intervened) / delta
```

**Key observation:** These functions take ONLY the y values (logit differences) - they have NO knowledge of which component was patched.

### 2. How y Values Are Computed

In `src/activation_patching/intervened_choice.py`, the IntervenedChoice class computes recovery/disruption:

```python
@property
def recovery(self) -> float:
    y_clean = self._get_logit_diff(self.baseline_clean)
    y_corrupted = self._get_logit_diff(self.baseline_corrupted)
    y_intervened = self._get_logit_diff(self.intervened)
    return compute_recovery(y_intervened, y_clean, y_corrupted)
```

The `_get_logit_diff` method extracts logit differences from the choice objects:

```python
def _get_logit_diff(self, choice: ChoiceType) -> float:
    logits = choice.divergent_logits
    if logits:
        if self.switched:
            return logits[1] - logits[0]
        return logits[0] - logits[1]
    lps = choice.divergent_logprobs
    if self.switched:
        return lps[1] - lps[0]
    return lps[0] - lps[1]
```

**Key observation:** This also has NO component-specific logic. It simply extracts logit/logprob differences from the model's final output.

### 3. Component Name Generation

In `src/common/hook_utils.py`:

```python
def hook_name(layer: int, component: str) -> str:
    """Generate hook name: blocks.{layer}.hook_{component}"""
    if component == "attn_z":
        return f"blocks.{layer}.attn.hook_z"
    return f"blocks.{layer}.hook_{component}"
```

So for resid_pre at layer L+1 vs resid_post at layer L:
- `resid_pre` at L+1: `blocks.{L+1}.hook_resid_pre`
- `resid_post` at L: `blocks.{L}.hook_resid_post`

These are DIFFERENT hook names even though the tensors may be identical!

### 4. Where Activations Are Retrieved

In `src/common/contrastive_pair.py`, the `_make_layer_intervention` method:

```python
def _make_layer_intervention(self, layer, component, ...):
    if component == "attn_z":
        hook = f"blocks.{layer}.attn.hook_z"
    else:
        hook = hook_name(layer, component)  # e.g., "blocks.{layer}.hook_{component}"

    # Get source activations
    if mode == "denoising":
        patch_acts = clean_internals.get(hook)  # <-- USES HOOK NAME AS KEY
    else:
        patch_acts = corrupted_internals.get(hook)
```

**CRITICAL FINDING:** The activations are retrieved using the hook name as the dictionary key!

### 5. Potential Root Cause

If you're patching:
- `resid_pre[L+1]` -> looks for key `blocks.{L+1}.hook_resid_pre` in the cache
- `resid_post[L]` -> looks for key `blocks.{L}.hook_resid_post` in the cache

Even if these tensors are mathematically identical in the model architecture, they are stored under DIFFERENT keys in the activation cache.

**The question becomes:** Are BOTH hooks being captured when running the forward pass with caching?

### 6. Hook Filter Logic

In `src/common/hook_utils.py`:

```python
def hook_filter_for_component(component: str) -> Callable[[str], bool]:
    """Filter for a specific component."""
    if component == "attn_z":
        return lambda name: "attn.hook_z" in name
    target = f"hook_{component}"
    return lambda name: target in name
```

When capturing activations for `resid_post`:
- Filter matches: `hook_resid_post`
- This would NOT capture `hook_resid_pre` hooks!

When capturing activations for `resid_pre`:
- Filter matches: `hook_resid_pre`
- This would NOT capture `hook_resid_post` hooks!

### 7. Key Insight About Cache Behavior

Looking at `src/activation_patching/patch_choice.py`:

```python
clean_choice = runner.choose(
    pair.clean_prompt,
    pair.choice_prefix,
    labels,
    with_cache=(mode == "denoising"),
    names_filter=names_filter if mode == "denoising" else None,  # <-- Component-specific filter!
)
```

The `names_filter` is derived from `hook_filter_for_component(component)`, which only captures hooks for the SPECIFIED component.

**ROOT CAUSE IDENTIFIED:**

If you patch `resid_pre[L+1]`, the system:
1. Creates filter for `hook_resid_pre`
2. Captures ONLY `resid_pre` activations
3. Creates intervention looking for `blocks.{L+1}.hook_resid_pre`

If you patch `resid_post[L]`, the system:
1. Creates filter for `hook_resid_post`
2. Captures ONLY `resid_post` activations
3. Creates intervention looking for `blocks.{L}.hook_resid_post`

Even though resid_pre[L+1] == resid_post[L] mathematically, they go through DIFFERENT code paths because:
1. Different hook names
2. Different cache keys
3. Potentially different capture filters

## Verification Steps

To verify this is the issue:

1. **Check if both hooks are captured:** Add logging to see what keys are in the activation cache after a forward pass with `resid_pre` vs `resid_post` filter.

2. **Check hook order in backend:** The TransformerLens model may execute hooks in a specific order. If resid_post[L] is captured BEFORE resid_pre[L+1], they might capture different states.

3. **Check tensor identity at runtime:** In the backend, verify that the tensors returned by both hooks are actually the same object or at least numerically identical.

## Conclusion

The metric calculation itself is component-agnostic. The difference in patching effects MUST come from how:
1. Hook names are generated
2. Activations are filtered/captured
3. The intervention is applied

The most likely culprit is that the hook capturing filter (`names_filter`) only captures one type of hook at a time, and there may be subtle timing/ordering differences in when resid_pre vs resid_post hooks fire during the forward pass.

## Files Examined

1. `/Users/unrulyabstractions/work/temporal-awareness/src/common/math/faithfulness_scores.py` - Recovery/disruption formulas (no component logic)
2. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/intervened_choice.py` - IntervenedChoice recovery/disruption properties
3. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/act_patch_metrics.py` - Metric extraction (no component-specific normalization)
4. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/act_patch_results.py` - Result containers
5. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/patch_choice.py` - Patching orchestration
6. `/Users/unrulyabstractions/work/temporal-awareness/src/common/contrastive_pair.py` - Intervention creation
7. `/Users/unrulyabstractions/work/temporal-awareness/src/common/hook_utils.py` - Hook name generation and filtering
8. `/Users/unrulyabstractions/work/temporal-awareness/src/common/patching_types.py` - Component type definitions
9. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/coarse/sweep_runners.py` - Layer sweep logic
10. `/Users/unrulyabstractions/work/temporal-awareness/src/activation_patching/coarse/coarse_results.py` - Coarse patching results
11. `/Users/unrulyabstractions/work/temporal-awareness/src/common/choice/simple_binary_choice.py` - divergent_logprobs extraction
12. `/Users/unrulyabstractions/work/temporal-awareness/src/common/choice/grouped_binary_choice.py` - Multi-fork aggregation

## Next Steps

The investigation should now move to the BACKEND layer:
1. Examine how TransformerLens registers/fires hooks for resid_pre vs resid_post
2. Check if there's any caching or memoization at the tensor level
3. Verify the exact timing when each hook fires during the forward pass
