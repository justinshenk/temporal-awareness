# Per-context local refit: a problem-specific map doesn't transfer either

**Date:** 2026-06-07 · **Model:** Llama-2-7B + MetaMath LoRA · **Probe:**
`scripts/attribution/local_refit_gsm8k.py`

## Question

Every static result so far used ONE global map `W` (fit across all problems on LoRA-generated
CoT tokens, off-policy) → 0.00 recovery. Maybe the wall is the map's *generality*: a single
4096×4096 linear map can't cover every problem's activation geometry. Test it by **refitting per
context** — for each problem, fit a fresh per-layer ridge map on that problem's *own prompt-prefix*
activations, then steer that problem's CoT with its own map.

- **Fit:** teacher-force only the question prompt through base (adapter off) and LoRA; take
  `(a, δ)` over the prompt positions `[0:prompt_len]`; solve the per-layer ridge. No CoT/answer
  leakage — the map sees only the question.
- **Steer:** all-32-layer joint injection of that problem's maps; generate; score.
- **λ sweep** `{1, 100, global-λ*}`: the global λ* (median 1000) is calibrated for the global
  fit's ~10k tokens and over-regularizes a ~90-token local fit, so λ is swept down to keep the
  local map expressive. α ∈ {0.5, 1.0}. n=30 GSM8K test.

## Results

base **0.000** · LoRA **0.533** · budget **+0.533**

| λ | α=0.5 | α=1.0 |
|---|---|---|
| 1 | 0.000 | 0.000 |
| 100 | 0.000 | 0.000 |
| global-λ* | 0.000 | 0.000 |

**0.00 recovery in every cell** — identical to the global map.

## The maps are expressive, not no-ops

Per-layer fit norms are on par with the global maps (global `|W|_fro` ranged 2.85–8.61):

| λ | mean `\|W_L\|_fro` | generation (problem 0: "Janet's ducks lay 16 eggs/day…") |
|---|---|---|
| 1 | 7.02 | "The fear here is that there may be too many irrelevant details that need to be checked…" |
| 100 | 5.24 | "The amount she makes is limited to certain jurisuries where the number of validatable units…" |
| global | 3.19 | "…the amount she makes at the fathers's arrogance is not limited to a certain amount…" |

Grammatical, clearly steered away from base, but **off-topic reasoning prose** — no arithmetic,
no contact with the problem (no eggs, breakfast, baking). The same decode-but-can't-steer
signature as the global joint map at low α: fluent register shift, zero computation.

## Verdict

**The global map's generality is not the wall.** A map fit on *this problem's own* activations,
expressive (‖W‖ on par with global) and swept across regularization, still recovers **0.00**.
Combined with the prior strands:

- global map (CoT-token fit, off-policy) → 0.00
- per-context local map (prompt-prefix fit) → 0.00
- short-output / single-step → 0.00; injection destructive even at 1–2 tokens
- only ICL (re-derives the state each step via attention) → ~27% of budget

The failure converges on the **linear-injection mechanism itself**: adding a fitted direction to
the residual stream — globally or locally, however regularized — installs a reasoning-shaped
*register* but not the *computation*. The MetaMath LoRA's competence is a procedure executed over
the trajectory, and no fixed per-token linear shift reproduces a procedure.

*Caveat:* the local map is fit on prompt-prefix tokens (boilerplate + question), whose δ may differ
from CoT-token δ. But the global map *was* fit on CoT tokens and also failed — the two fitting
regimes converge on the same null, which is the point.
