# LoRA vs ICL — geometry (DDXPlus, per layer)

From `results/lora_icl/shifts` (60 examples, hidden dim 3584, PCA k=5).

| Layer | cos(ΔICL, ΔLoRA) | LoRA energy in ICL k-subspace | LoRA energy on mean-ICL line | ‖LoRA‖ kept by parallel |
|------:|-----------------:|------------------------------:|-----------------------------:|------------------------:|
| 0 | -0.139 | 0.118 | 0.019 | 0.139 |
| 7 | +0.022 | 0.016 | 0.001 | 0.022 |
| 14 | +0.275 | 0.050 | 0.073 | 0.275 |
| 21 | +0.657 | 0.186 | 0.384 | 0.657 |
| 28 | +0.739 | 0.205 | 0.464 | 0.739 |
| 35 | +0.807 | 0.325 | 0.537 | 0.807 |
| 41 | +0.744 | 0.437 | 0.486 | 0.744 |

## Reading

- **Q1** cos(ΔICL, ΔLoRA): near 0 early, rising to ~0.8 late — the mean shifts align in the back half of the stack.
- **Q2** the ICL 5-dim PCA subspace is fit to ICL's per-example *variation*, so it captures only a modest share of LoRA energy; the 1-D mean-ICL line captures the directional overlap (≈ cos² of the means). Late layers show the largest shared share.
- **Q3/Q4 (geometry)** `‖LoRA‖ kept by parallel` is the fraction of the LoRA shift retained if you keep only the ICL-parallel component; `1 −` that is the orthogonal remainder. The *functional* effect of keeping each (task accuracy / refusal) is measured by steering — see `2026-06-01-keep-component-steering.md`.
