# Antonym function vector vs LoRA — same task vector, two routes?

`google/gemma-2-9b-it` | bare `word: antonym` | 10-shot | 30 held-out queries | FV via zero-shot-corrupted AIE over 42x16 heads, top-10.

## Accuracies (does each route install the task?)

| condition | antonym acc |
|-----------|------------:|
| zero-shot (base) | 0.00 |
| 10-shot ICL (base) | 0.70 |
| zero-shot + LoRA | 0.67 |
| zero-shot + FV @L9 | 0.00 |

## Top FV heads (zero-shot-corrupted AIE)

| rank | layer | head | AIE |
|-----:|------:|-----:|----:|
| 1 | 28 | 6 | +0.0001 |
| 2 | 16 | 13 | +0.0001 |
| 3 | 15 | 14 | +0.0001 |
| 4 | 24 | 13 | +0.0000 |
| 5 | 17 | 5 | +0.0000 |
| 6 | 9 | 4 | +0.0000 |
| 7 | 7 | 7 | +0.0000 |
| 8 | 11 | 9 | +0.0000 |
| 9 | 19 | 5 | +0.0000 |
| 10 | 28 | 14 | +0.0000 |

## FV insert-layer sweep (zero-shot acc + FV)

| insert layer | acc |
|-------------:|----:|
| 9 | 0.00 |
| 14 | 0.00 |
| 20 | 0.00 |

## Direction comparison across depth (cosine; random-null std ≈ 0.017)

| layer | ICL-taskvec · LoRA | FV · LoRA | FV · taskvec |
|------:|-------------------:|----------:|-------------:|
| 4 | +0.502 | +0.065 | +0.026 |
| 9 | +0.498 | +0.043 | +0.151 |
| 14 | +0.388 | +0.042 | +0.126 |
| 20 | +0.458 | +0.131 | +0.179 |
| 28 | +0.750 | +0.116 | +0.163 |
| 35 | +0.766 | +0.100 | +0.112 |

## Reading

- **Signal is real:** zero-shot 0.00 vs 10-shot 0.70, and the LoRA generalizes to held-out words (0.67) — unlike DDXPlus, both routes genuinely install the antonym function (not memorization, not prior-knowledge leakage).
- **Same task vector, two routes — YES (coarse).** cos(ICL-task-vector, LoRA-shift) peaks at **+0.766 @L35** (~46× the random-null std). The in-context demos and the in-weights LoRA install a substantially shared residual-space direction — mechanism-level support for the subspace-convergence result, now on a genuine ICL task.
- **But the head-localized FV did NOT extract.** Zero-shot+FV stays 0.00 (no lift) and single-head AIE ≈ 0 for every head; cos(FV, LoRA) ≈ 0. The antonym task is **distributed across heads** on this model — single-head causal mediation (Todd-style) finds no sparse FV here, even though the coarse task vector (Hendel-style) cleanly does. A real limit of the sparse-head account on a 9B instruct model.
- **Scope:** one model, 30 held-out queries; FV = top-10 heads, zero-shot-corrupted AIE, first-token readout; task vector = mean ICL−zeroshot last-token residual; LoRA shift = mean LoRA−base zero-shot residual.
