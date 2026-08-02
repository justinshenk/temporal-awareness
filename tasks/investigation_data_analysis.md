# Investigation: resid_pre[L+1] vs resid_post[L] Patching Differences

## Executive Summary

**ROOT CAUSE IDENTIFIED**: Bug in `backend_huggingface.py` where intervention hooks always modify `output[0]` regardless of whether the component is `resid_pre` or `resid_post`. This causes patching `resid_pre[L]` to actually patch `resid_post[L]` (effectively `resid_pre[L+1]`), leading to off-by-one layer effects.

## Data Analysis

### 1. Sanity Check (Patch ALL layers, ALL positions)

| Component   | Recovery | Disruption |
|-------------|----------|------------|
| resid_pre   | 0.4901   | 0.8632     |
| resid_post  | 1.0000   | 0.0000     |

**Key Finding**: Patching ALL positions with `resid_pre` gives only 49% recovery, while `resid_post` gives 100% recovery. This is impossible if both are hooking the same tensors.

### 2. Per-Layer Comparison

For single-layer patching (all positions), comparing `resid_post[L]` vs `resid_pre[L+1]`:

| Layer L | resid_post[L] | resid_pre[L+1] | Difference |
|---------|---------------|----------------|------------|
| 16      | 0.0107        | 0.0499         | -0.0393    |
| 17      | 0.0180        | -0.0439        | +0.0619    |
| 19      | 0.2943        | 0.3242         | -0.0299    |
| 20      | 0.5852        | 0.4880         | +0.0972    |
| 23      | 0.8156        | 0.6172         | +0.1984    |
| 34      | 0.9241        | 0.4901         | +0.4340    |

**These should be identical if both patch the same tensor!**

### 3. Cross-Sample Consistency

The pattern is consistent across samples:

| Layer | Sample 0 Diff | Sample 1 Diff |
|-------|---------------|---------------|
| 20    | +0.0972       | +0.0437       |
| 25    | +0.0646       | +0.0626       |
| 30    | +0.0386       | +0.0294       |
| 34    | +0.4340       | +0.3345       |

The differences are not noise - they are systematic and reproducible.

## Root Cause Analysis

### Code Path Investigation

#### 1. Caching (backend_huggingface.py:644-658)

```python
# CORRECT: Different handling for resid_pre vs others
def make_hook(hook_name, use_input=False):
    def hook_fn(mod, inp, out):
        if use_input:  # resid_pre uses input
            val = inp[0] if isinstance(inp, tuple) else inp
        else:          # resid_post uses output
            val = out[0] if isinstance(out, tuple) else out
        cache[hook_name] = val.detach()
    return hook_fn

use_input = component == "resid_pre"  # <-- Correctly distinguishes
hooks.append(module.register_forward_hook(make_hook(name, use_input)))
```

This is **CORRECT**: `resid_pre` captures `inp[0]` (layer input), `resid_post` captures `out[0]` (layer output).

#### 2. Intervention (backend_huggingface.py:885-954)

```python
# BUG: ALWAYS uses output, ignoring component type
def make_hook(values, target, mode, target_values, alpha):
    def intervention_hook(mod, input, output):
        if isinstance(output, tuple):
            hidden = output[0]  # <-- ALWAYS uses output!
        else:
            hidden = output
        # ... modifies hidden ...
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return intervention_hook
```

This is **WRONG**: For `resid_pre`, we should be modifying `input[0]`, not `output[0]`.

### Impact

When we think we're patching `resid_pre[L]`:
1. We capture activations from `layer[L].input[0]` = the actual `resid_pre[L]`
2. We patch `layer[L].output[0]` = `resid_post[L]` = `resid_pre[L+1]`

This means:
- **Patching `resid_pre[L]` actually patches `resid_pre[L+1]`** (off by one layer!)
- The source activations are from `resid_pre[L]` but applied at `resid_pre[L+1]`
- This explains why results differ between `resid_pre[L]` and `resid_post[L-1]`

### Verification

The data supports this:
- At layer 35 (last layer): `resid_pre` recovery = 0.490, `resid_post` recovery = 1.000
- Patching `resid_pre[35]` with the intervention bug actually patches `resid_post[35]`
- But the cached source values are from `resid_pre[35]` (one layer earlier in the residual stream)
- This mismatch causes incomplete recovery

## Why This Matters

1. **All `resid_pre` patching experiments are off by one layer**
2. **The tensor values being patched are correct, but they're applied at the wrong point**
3. **Results will show `resid_post[L]` effects when intending `resid_pre[L]` effects**

## Recommended Fix

In `backend_huggingface.py`, modify `run_with_intervention` to handle `resid_pre` using a pre-hook that modifies input:

```python
def run_with_intervention(...):
    hooks = []
    for intervention in interventions:
        component = intervention.component

        if component == "resid_pre":
            # Use pre-hook to modify input BEFORE the layer
            def make_pre_hook(values, target, mode, target_values, alpha):
                def pre_hook(mod, input):
                    hidden = input[0]
                    # Apply intervention to hidden...
                    return (hidden,) + input[1:] if len(input) > 1 else (hidden,)
                return pre_hook

            hook = module.register_forward_pre_hook(
                make_pre_hook(values, target, mode, target_values, alpha)
            )
        else:
            # Use post-hook to modify output AFTER the layer
            hook = module.register_forward_hook(
                make_hook(values, target, mode, target_values, alpha)
            )

        hooks.append(hook)
```

## Files Affected

- `/Users/unrulyabstractions/work/temporal-awareness/src/inference/backends/backend_huggingface.py`
  - `run_with_intervention()` - needs pre-hook for resid_pre
  - `run_with_intervention_and_cache()` - same fix needed

- `/Users/unrulyabstractions/work/temporal-awareness/src/inference/backends/backend_nnsight.py`
  - Same bug exists: always uses `module.output[0]`

## Data Files Analyzed

- `/Users/unrulyabstractions/work/temporal-awareness/out/experiments/nano/aggregated/coarse/resid_pre.json`
- `/Users/unrulyabstractions/work/temporal-awareness/out/experiments/nano/aggregated/coarse/resid_post.json`
- `/Users/unrulyabstractions/work/temporal-awareness/out/experiments/nano/pairs/pair_0/coarse/sweep_resid_pre/coarse_results.json`
- `/Users/unrulyabstractions/work/temporal-awareness/out/experiments/nano/pairs/pair_0/coarse/sweep_resid_post/coarse_results.json`
