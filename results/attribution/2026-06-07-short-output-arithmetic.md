# Short-output arithmetic: trajectory-length is not the wall — compute doesn't transfer

**Date:** 2026-06-07 · **Model:** Llama-2-7B + MetaMath LoRA (rank-16/α-32) · **Maps:** all-32-layer
primal-ridge `W_L` (λ*), all-step joint injection · **Probes:** `scripts/attribution/short_arithmetic.py`,
`src/probes/attribution/arithmetic_problems.py`

## Question

GSM8K (multi-step CoT, `max_new=512`): all-step joint steering recovers **0.00** of the LoRA's
budget (base 0.00 → LoRA 0.60). Two readings were still open:

- **(H1) Long-trajectory drift** — the injected δ is a sound shift, but compounds/drifts over a
  long self-maintained chain, so coherence collapses before the answer.
- **(H2) Compute doesn't transfer** — the linear map simply doesn't install the LoRA's arithmetic,
  at any length.

The clean separator: single-**operation** arithmetic (one reasoning step), run under two prompt
modes that shrink the self-maintained trajectory while holding the prompt/steering machinery
identical to GSM8K. 30 problems, 10 each over `mul2x2` (two 2-digit factors), `add3`, `sub3`
(two 3-digit terms).

- **`direct`** (`max_new=12`): response forced to begin `### Response: The answer is: ` → the
  number must come out immediately, **zero** self-maintained CoT.
- **`cot`** (`max_new=256`): natural `Let's think step by step` → **one short** reasoning step,
  budget large enough that the LoRA finishes.

## Results

| mode | base | LoRA | budget | steer α=0.5 | steer α=1.0 |
|---|---|---|---|---|---|
| `direct` (0-step) | **0.767** | 0.600 | **−0.167** | 0.000 | 0.000 |
| `cot` (1-step) | 0.000 | 0.733 | +0.733 | 0.000 (rec +0.00) | 0.000 (rec +0.00) |
| GSM8K (multi-step) | 0.000 | 0.600 | +0.600 | 0.000 | 0.000 |

Per-tier (`mul2x2` / `add3` / `sub3`):
- `direct`: base `0.5 / 0.8 / 1.0`, LoRA `0.3 / 0.8 / 0.7`
- `cot`: base `0 / 0 / 0`, LoRA `0.2 / 1.0 / 1.0`

## Verdict — H1 rejected, H2 confirmed

**Shortening the trajectory does not rescue steering.** Single-operation, single-step `cot`
steering recovers **0.00** — the same as multi-step GSM8K. Reducing the chain the model must
self-maintain from "many operations over 512 tokens" to "one operation over ~80 tokens" changes
nothing. The wall is not trajectory drift; **the linear map does not install the compute.**

Two sharper findings fall out:

1. **The LoRA's arithmetic is procedural, not a readable value.** Forced to answer *directly*
   (`direct` mode), the LoRA (0.600) is **worse than base** (0.767): −0.167 budget. The MetaMath
   LoRA doesn't store "23×47 → 1081"; it executes a step-by-step procedure, and skipping the
   procedure degrades it below the untuned base. There is no static target for a single δ to hit —
   which is *why* a fixed activation shift can't reproduce it.

2. **Injection corrupts even a 1–2 token output.** In `direct` mode the base model alone answers
   0.767 correctly, but base+steer drops to **0.000** — the injected direction pushes the model off
   its functional manifold even when only one number need be emitted. This rules out the last
   form of H1 ("the direction is fine, it just accumulates"): it is destructive *immediately*, not
   cumulatively.

## Where this lands the arc

- Static injection (single-layer, all-layer joint, norm-preserve, manifold-projected, α-annealed,
  prefill-only, **and now short/single-step**) → **0.00** everywhere.
- ICL (dynamic, demonstrations re-attended every step) → restores coherent terminating CoT and
  ~27% of budget (peak 4-shot 0.16).
- The capability is latent and reachable, but only by a mechanism that **re-derives** the reasoning
  state each step (attention over context), not by **re-adding** a fixed direction. The MetaMath
  LoRA's competence is a *procedure over the trajectory*, and a per-token linear map — however well
  it decodes δ in held-out R² — cannot install a procedure.
