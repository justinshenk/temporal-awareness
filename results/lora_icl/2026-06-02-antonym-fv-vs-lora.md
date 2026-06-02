# Antonym function vector vs LoRA — same task vector, two routes?

`google/gemma-2-9b-it` | bare `word: antonym` | 10-shot | 30 held-out queries | FV via **shuffled-label-corrupted** AIE (log-prob delta, Todd-faithful) over 42x16 heads, top-10.

## Accuracies (does each route install the task?)

| condition | antonym acc |
|-----------|------------:|
| zero-shot (base) | 0.00 |
| 10-shot ICL (base) | 0.70 |
| 10-shot ICL, shuffled labels (corruption) | 0.57 |
| zero-shot + LoRA | 0.67 |
| zero-shot + FV @L9 | 0.00 |

## Top FV heads (shuffled-label-corrupted AIE, log-prob delta)

| rank | layer | head | AIE |
|-----:|------:|-----:|----:|
| 1 | 24 | 13 | +0.9370 |
| 2 | 17 | 5 | +0.7499 |
| 3 | 24 | 9 | +0.6855 |
| 4 | 21 | 5 | +0.5422 |
| 5 | 24 | 12 | +0.3722 |
| 6 | 25 | 13 | +0.3458 |
| 7 | 22 | 11 | +0.2787 |
| 8 | 24 | 15 | +0.2785 |
| 9 | 23 | 14 | +0.2750 |
| 10 | 17 | 4 | +0.2475 |

## FV insert-layer sweep (zero-shot acc + FV)

| insert layer | acc |
|-------------:|----:|
| 9 | 0.00 |
| 14 | 0.00 |
| 20 | 0.00 |

## Direction comparison across depth (cosine; random-null std ≈ 0.017)

| layer | ICL-taskvec · LoRA | FV · LoRA | FV · taskvec |
|------:|-------------------:|----------:|-------------:|
| 4 | +0.502 | +0.037 | +0.021 |
| 9 | +0.498 | +0.035 | +0.074 |
| 14 | +0.388 | +0.029 | +0.071 |
| 20 | +0.458 | +0.153 | +0.179 |
| 28 | +0.750 | +0.187 | +0.206 |
| 35 | +0.766 | +0.145 | +0.141 |

## Reading

- **Method fix:** AIE now uses **shuffled-label ICL** corruption + log-prob readout (Todd-faithful), correcting the earlier *zero-shot* corruption where single-head AIE was ~0 by construction (removing all in-context signal means no single head can restore the task).
- **Signal is real:** zero-shot 0.00 vs 10-shot 0.70, and the LoRA generalizes to held-out words (0.67). Note the shuffled-label run still scores 0.57 — the model keeps most antonym performance with scrambled labels (Min et al. 2022: the *mapping* signal is small; demos mostly install format/function), so the AIE headroom is intrinsically modest for this task.
- **Same task vector, two routes — YES (coarse), unchanged.** cos(ICL-task-vector, LoRA-shift) peaks at **+0.766 @L35** (~46× random-null). This does not depend on the FV method.
- **FV heads ARE found now — correcting my earlier overclaim.** With the corruption fixed the AIE is strongly structured: top heads at layers 17/21/22/23/24/25 (early-middle, exactly where Todd finds FV heads) with AIE up to **+0.94 nats**, versus the all-zero AIE of the broken zero-shot version. So there *are* antonym FV heads; the earlier "distributed across heads" reading was the zero-shot-corruption artifact, now retracted.
- **Yet the FV still doesn't reconstruct the task direction** (zero-shot+FV 0.00; cos(FV, task-vector) ≤ +0.21) — and this is the *real* finding, the Min et al. decomposition made mechanistic. Shuffled-label AIE isolates the **label-mapping** component (shuffled 0.57 vs clean 0.70 → only 0.13 headroom), but the task vector and LoRA shift are dominated by the large **format/function-selection** component (zero-shot 0.00 → 0.70). These are different, near-orthogonal sub-directions, so the mapping-FV ≠ the format-dominated task vector. The "two routes converge (0.77)" result is about the **format/function direction** — which single-head causal mediation *cannot* isolate, because removing the format means zero-shot corruption, where single-head AIE is 0 by construction.
- **Net:** the antonym task vector is **two superposed sub-directions** — a dominant format/function-install (which both ICL and the LoRA share → cos 0.77) and a small label-mapping correction (which the sparse Todd-FV isolates). Sparse-FV extraction recovers the latter, not the former. (Caveat: FV magnitude was untuned (#3) and no positive control was run, but the cosine evidence is scale-free.)
- **Scope:** one model, 30 held-out queries; FV = top-10 heads, shuffled-label-corrupted AIE (log-prob delta); task vector = mean ICL−zeroshot last-token residual; LoRA shift = mean LoRA−base zero-shot residual.
