# On-policy DAgger refit: closing the off-policy loop doesn't rescue steering

**Date:** 2026-06-07 · **Model:** Llama-2-7B + MetaMath LoRA · **Probe:**
`scripts/attribution/dagger_refit_gsm8k.py`

## Question

Every map so far was fit **off-policy** — on sequences the LoRA (expert) generated, with clean-base
activations — but deployed on the **steered base's own rollout**, which is off-distribution. The
standard fix is DAgger: roll out with the current policy, label the visited states with the expert,
aggregate into the dataset, refit, repeat. If the wall is distribution shift, accuracy should climb
over rounds.

**Loop (per round k):** steer base with the current map → greedy on-policy CoT → teacher-force each
sequence through clean base + LoRA → `a` = base residual, `δ` = LoRA − base over the CoT slice →
**aggregate** into GramAccumulators seeded from the saved off-policy `train_L*.pt` → refit `W` at
each layer's global λ* → steer + eval. n_fit=48 rollouts/round, n_eval=30, α=1.0, 256-token budgets.

## Results

base **0.000** · LoRA **0.500** · budget **+0.500**

| round | source | agg tokens/layer | steer acc | recovery |
|---|---|---|---|---|
| 0 | off-policy global map | 750 | 0.000 | +0.00 |
| 1 | +48 on-policy seqs (12,288 tok) | 13,038 | 0.000 | +0.00 |
| 2 | +48 on-policy seqs (12,288 tok) | 25,326 | 0.000 | +0.00 |

By round 2 the aggregated dataset is ~97% on-policy (25,326 vs 750 seed tokens/layer) — effectively
a pure on-policy refit — and still **0.00**.

## The loop can't bootstrap out of degeneracy

Every rollout hit **exactly** the 256-token cap (48 × 256 = 12,288 tokens each round): the steered
policy **never emits a terminus** ("The answer is:" / EOS). So the on-policy states DAgger collects
are entirely degenerate, non-terminating continuation. Refitting `W` on the states the steered
model visits just re-teaches that distribution — there is no expert-corrected gradient back toward
terminating, arithmetic-bearing CoT, because the linear map applied to those degenerate activations
can't produce one. DAgger normally repairs compounding error by showing the learner the expert's
action in the states it drifts into; here the "action" is a residual shift that is itself the source
of the drift, so iterating doesn't converge.

## Verdict — the off-policy explanation is ruled out

Distribution shift is **not** why static steering fails. On-policy aggregation over the steered
model's own visited states, ~97% on-policy by round 2, recovers 0.00 — the same as the off-policy
global map. This closes the static side of the investigation:

| condition | recovery |
|---|---|
| global map (off-policy, CoT-token fit) | 0.00 |
| per-context local map (prompt-prefix fit) | 0.00 |
| short-output / single-step | 0.00 |
| **on-policy DAgger refit (≤97% on-policy)** | **0.00** |
| ICL (re-derives state each step via attention) | ~0.27 |

Globally or locally, off-policy or on-policy, however regularized — a fixed per-token linear shift
added to the residual stream installs a reasoning-shaped *register* but never the *computation*.
The MetaMath LoRA's competence is a procedure executed over the trajectory; only a mechanism that
re-derives the reasoning state each decode step (attention over in-context demonstrations) recovers
any of it. The linear-injection mechanism is the wall.
