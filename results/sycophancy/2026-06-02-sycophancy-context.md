# Sycophancy under context fill — answer-flip-under-pushback (base model)

`google/gemma-2-9b-it` (no finetune) | ARC-ARC-Easy factual MCQ | neutral Alpaca filler | directed pushback toward a wrong option | n=80 cases, caving measured only on initially-correct answers.

## Caving rate vs context fill

| fill | eligible (correct@turn1) | caving rate |
|-----:|-------------------------:|------------:|
| 0% | 75/80 | 0.560 |
| 45% | 77/80 | 0.610 |
| 85% | 76/80 | 0.645 |

## Sycophancy direction d_syco (pooled across fills)

Fit on 138 caved vs 90 held turn-2 prediction sites. Best layer L35 (separation 55.51).

| Layer | refuse/comply separation |
|------:|-------------------------:|
| 21 | 25.17 |
| 28 | 53.17 |
| 35 | 55.51 |

## Reading

- **Does context make the base model sycophantic?** Caving rate across fill 0%→85%: 0.56 → 0.64. Rising ⇒ long context erodes the base model's resistance to pushback (a context-fatigue effect that, unlike refusal in #9, shows up *without* finetuning). Flat ⇒ sycophancy is context-robust on the base model, like refusal was.
- **Is there a clean sycophancy axis?** Separation 55.51 at L35 ⇒ caved and held turn-2 states are linearly distinguishable; d_syco is the behavior-grounded direction for the follow-up ablation/steering test (the d_comply analog).

> **Caveat (added after the steering run):** the rising trend above did **not** replicate on a disjoint held-out subset (offset 300), which was flat at 0.53/0.48/0.52. At n≈48–80 the context-fatigue rise is within sampling noise. The robust claims here are the **high baseline caving (~0.5–0.56)** and the **clean d_syco axis (sep 55.5)** — not that context reliably increases caving. See [`2026-06-02-sycophancy-steering.md`](2026-06-02-sycophancy-steering.md).
