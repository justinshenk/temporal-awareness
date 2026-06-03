# Beating the fixed-vector ceiling: input-conditional & last-token steering (Qwen, DDXPlus)

The constant mean task vector caps task accuracy and, at α≈1, erodes refusal (a blanket
refusal-axis leak on every prompt). Two independent fixes — and both also fix safety.
Eval split: base task 0.324, refusal 1.00; n_eval=40, n_harmful=25, n_fit=120, single seed.

## Results (α=1.0)

| method | task | refusal |
|---|--:|--:|
| base | 0.324 | 1.00 |
| mean vector, all-position | 0.475 | **0.24** |
| mean vector, **last-token** | 0.550 | 1.00 |
| conditional (λ=1), all-position | 0.575 | 1.00 |
| **conditional (λ=3), all-position** | **0.600** | 1.00 |
| conditional, last-token | 0.45–0.50 | 1.00 |

## Two fixes for the "direction problem"
The all-position mean vector does two bad things at once: it **overwrites per-case content**
(caps task) and **blanket-leaks onto the refusal axis** (erodes safety). Both fixes reduce
the "constant blanket application", so both repair task *and* safety:

- **A — last-token steering** (`AdditionSteeringHook(..., last_token=True)`): steer only the
  generation site. Mean vector 0.475/0.24 → **0.55/1.00**. Higher task (stops clobbering
  content tokens) and no erosion (the leak lands on one position of harmful prompts, not all).
- **B — input-conditional map** (`run_conditional_steer.py`): a closed-form ridge map W
  (per layer, dual form) predicts each case's shift from its own activation,
  `steer = α·(W·a)`. All-position: **0.60/1.00** (λ=3) — the best task, fully safe. It's
  safe because `W·a` is input-aware: on a harmful prompt it predicts a *different, smaller,
  off-refusal-axis* shift than the constant medical-task vector.

## They do NOT stack
Conditional + last-token (0.45–0.50) is **worse** than conditional + all-position
(0.575–0.60). The conditional map's whole benefit is supplying an *appropriate per-position*
shift; restricting it to the last token discards that. So: use **last-token** for the simple
constant vector, **all-position** for the conditional map.

## Takeaways
- The original all-position mean vector (0.475 / 0.24) was the worst option on **both** axes;
  either fix dominates it.
- Best task while fully safe: **all-position input-conditional map** (≈0.60). Simplest fully
  safe option: **last-token mean vector** (0.55).
- Both stay activation-space and **training-free** (the conditional map is closed-form). The
  conditional map is on the spectrum toward LoRA (input-gated) but without gradient training
  and — unlike the finetune — it does **not** erode refusal.

## Caveats
Single eval split (absolute numbers are split-sensitive — another split had the mean vector
at 0.65; the *within-split rankings* are the signal, not the absolutes), n_eval=40,
n_harmful=25, single seed. Needs multi-seed / multi-split replication with CIs, and a check
that the conditional map doesn't distort behavior on unrelated (non-medical, non-harmful)
prompts. The constant-vector erosion fix here is *positional*; the orthogonalized-vector fix
(`steering_safety`) is *directional* — they're complementary safety levers.

## Reproduce
`scripts/safety/run_conditional_steer.py` (`--lambdas`, `--alphas`; runs all-position and
last-token); hooks in `src/probes/safety/steering_hook.py`
(`AdditionSteeringHook(last_token=)`, `LinearConditionalSteerHook`). JSON:
`conditional_steer.json`.
