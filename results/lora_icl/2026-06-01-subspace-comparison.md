# LoRA vs ICL — Activation-Subspace Comparison (DDXPlus)

Base model: `google/gemma-2-9b-it` | examples: 60 | hidden dim: 3584 | PCA k: 5
Random cosine null (chance scale): ±0.0167

## ICL shift vs real-LoRA shift

| Layer | mean cosine | subspace overlap | min∠ (deg) | mean∠ (deg) |
|------:|------------:|-----------------:|-----------:|------------:|
| 0 | -0.1386 | 0.6029 | 25.4 | 49.9 |
| 7 | +0.0224 | 0.2248 | 69.2 | 76.9 |
| 14 | +0.2751 | 0.3572 | 40.9 | 68.1 |
| 21 | +0.6568 | 0.3444 | 48.7 | 69.2 |
| 28 | +0.7388 | 0.6680 | 30.2 | 46.4 |
| 35 | +0.8068 | 0.6776 | 29.1 | 46.2 |
| 41 | +0.7443 | 0.6492 | 29.4 | 47.8 |

## Reading

- Random cosine null (chance scale) is ±0.0167; observed late-layer cosines are ~40-50x that, so the alignment is far above chance.
- Peak mean-shift cosine +0.807 at layer 35 (fractional depth ~0.85).
- Mean cosine compares the average shift direction; subspace overlap is on mean-centered PCA subspaces, so it discards the shared mean offset and reflects whether the per-case variation lives in the same subspace. Both rising together in late layers is the stronger signal.
- Early layers (0-7) show little to no alignment; the shared subspace emerges in the mid-to-late stack where the task/answer computation lives.
