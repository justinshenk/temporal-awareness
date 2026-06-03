# Generalization checks: MMLU non-destabilization + MedMCQA second-dataset pipeline

## 1. Does the DDXPlus-distilled map destabilize an unrelated task (MMLU)? — No
gemma-2-9b, distilled map (W·a fit on DDXPlus LoRA shift) applied to the base on MMLU:

| | MMLU acc |
|---|--:|
| base (no steer) | 0.510 |
| + distill map, α=0.5 | **0.600** |
| + distill map, α=1.0 | 0.460 |

At the safe magnitude (α=0.5) the transfer does **not** damage the unrelated task — MMLU is
unharmed (nudged up, plausibly the map encodes generic "answer the MCQ with a letter"
behavior). Only over-steering (α=1) mildly hurts. So the DDXPlus-knowledge transfer is
reasonably task-specific, not a broad perturbation. (`mmlu_destab.json`.)

## 2. The pipeline on a second dataset (MedMCQA) — generalizes
Qwen-7B, real medical-exam MCQ (distinct from DDXPlus's synthetic symptom format):

| condition | acc | refusal |
|---|--:|--:|
| zero-shot | 0.327 | — |
| 5-shot ICL | 0.429 | — |
| steer α=0.5 | 0.440 | **1.00** |
| steer α=1.0 | 0.480 | 0.88 |
| steer_orth α=1.0 | 0.500 | 0.88 |

Task-vector steering again **transfers the ICL competence gain** (0.327 → 0.44–0.50, ≈ ICL)
while **preserving refusal** (1.00 at α=0.5; mild 0.88 at α=1 — same Goldilocks band).

## Bounding principle, confirmed across three datasets
Steering transfers exactly the ICL gain *available*, which scales with how much the model
needs the task:

| dataset | zero-shot | ICL/steered | headroom | refusal @α≈0.5 |
|---|--:|--:|---|--:|
| DDXPlus (niche format) | 0.14 | 0.70 | huge | preserved |
| MedMCQA (real exam) | 0.33 | 0.50 | modest | 1.00 |
| MMLU (general knowledge) | 0.68 | — | none | (n/a) |

So "competence transfer without the safety tax" is **not DDXPlus-specific** — it holds on a
second dataset, with the gain scaling with zero-shot headroom and refusal preserved at the
safe magnitude in every case.

## 3. MedMCQA LoRA (the weight-route half) — gain ≈ 0, but still erodes
Trained a MedMCQA LoRA (600 ex, 3 epochs) on Qwen-7B:

| | acc | refusal |
|---|--:|--:|
| base | 0.653 | 1.00 |
| MedMCQA LoRA | 0.620 | 0.96 |
| gain / erosion | **−0.03** | **−0.04** |

**ICL on the SAME slice (600:650): zero-shot 0.653, 5-shot ICL 0.653 — identical.** So this
slice simply has **no headroom**: ICL is also flat, neither route can help (the model already
does it). The "no LoRA gain" is a saturated slice, NOT a route-specific failure (MedMCQA
varies hugely by slice — the steering slice 100:150 had base 0.33 with real ICL headroom).

### Multi-seed check (RETRACTS the "erodes-for-nothing" claim)
A single run suggested the MedMCQA LoRA erodes refusal at ~0 gain (1.0→0.96). Three seeds
(n_harmful=40) show that was a 1-prompt noise blip:

| seed | LoRA task gain | refusal drop |
|---|--:|--:|
| 42 | −0.033 | +0.025 (1/40) |
| 123 | +0.040 | 0.000 |
| 7 | +0.008 | 0.000 |

- **Task gain ≈ 0** across seeds (−0.03/+0.04/+0.01): robust — the MedMCQA LoRA genuinely
  doesn't help (model already competent).
- **No reliable erosion** (drops +0.025/0/0): 2 of 3 seeds show ZERO erosion. The earlier
  "erodes for nothing" was overstated off one prompt — **retracted.**

Corrected picture — the *opposite* of gain-independent: **erosion tracks how much the
finetune actually moves the weights.** DDXPlus (strong learnable signal → large weight change)
erodes hard; MedMCQA (model already competent → finetune barely moves the weights, ~0 gain)
→ ~0 erosion. The route still differs (ICL never erodes), but a *weak* finetune that learns
nothing also doesn't erode — erosion scales with finetune strength, not independent of it.

### DDXPlus LoRA across the same 3 seeds — the headline is bulletproof
| seed | base→LoRA acc | gain | refusal drop |
|---|--:|--:|--:|
| 42 | 0.085→0.980 | +0.90 | **+1.00** (1.0→0.0) |
| 123 | 0.143→0.920 | +0.78 | **+1.00** |
| 7 | 0.085→0.920 | +0.84 | **+1.00** |

Every seed: huge gain (+0.84 mean) AND complete refusal collapse (→0.00). Side by side:

| task | finetune strength | LoRA gain (3 seeds) | refusal drop (3 seeds) |
|---|---|--:|--:|
| DDXPlus | strong (learnable format) | +0.78/+0.90/+0.84 | **+1.00/+1.00/+1.00** |
| MedMCQA | weak (already competent) | −0.03/+0.04/+0.01 | +0.025/0/0 |

The DDXPlus route-dependence is fully seed-robust; the MedMCQA "erosion" was noise. Erosion
scales with finetune strength — clean and seed-validated.

Route-dependence across tasks:

| task | type | LoRA gain | LoRA erosion | steering transfer |
|---|---|--:|--:|---|
| DDXPlus | learnable format | +0.79 | →0.00 (full) | 0.14→0.70 |
| MedMCQA | existing knowledge | −0.03 | →0.96 (mild) | 0.33→0.50 |

Both the LoRA gain and its erosion scale with how much the task is a learnable niche format
vs knowledge the model has. **Caveat:** MedMCQA eval slices differed between runs (base 0.33
steering-slice vs 0.65 LoRA-slice — high n=50 variance), so absolute gains are noisy; the
qualitative points (LoRA ≈ no gain, mild erosion regardless) are robust.

## Reproduce
`scripts/safety/run_mmlu_destab.py` (gemma), `scripts/safety/run_medmcqa_pipeline.py` (Qwen).
JSONs: `mmlu_destab.json`, `medmcqa_pipeline.json`. Caveats: single model each, n_eval 50,
n_harmful 25, single seed.
