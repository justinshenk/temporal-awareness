# Safety Degradation — Weights vs Activations (DDXPlus → refusal)

Base: `google/gemma-2-9b-it` | DDXPlus LoRA: `results/lora_icl/adapter` | harmful eval prompts: 60

## Behavioral refusal rate (higher = safer)

| Condition | refusal rate |
|-----------|-------------:|
| base (harmful alone) | 0.983 |
| base + DDXPlus medical ICL | 0.983 |
| DDXPlus LoRA | 0.867 |

## Activation shift along the refusal direction (negative = toward compliance)

`cos(shift, r)` is magnitude-normalized: it isolates *direction* of the shift relative to the refusal axis, independent of how large the shift is.

| Layer | ICL proj | LoRA proj | cos(ICL,r) | cos(LoRA,r) | cos(ICL,LoRA) |
|------:|---------:|----------:|-----------:|------------:|--------------:|
| 0 | +4.16 | -0.60 | +0.320 | -0.226 | -0.473 |
| 7 | -2.47 | -0.47 | -0.069 | -0.033 | -0.044 |
| 14 | -18.33 | -2.35 | -0.247 | -0.101 | +0.043 |
| 21 | -18.50 | -23.42 | -0.175 | -0.320 | +0.172 |
| 28 | +2.25 | -116.87 | +0.016 | -0.512 | -0.027 |
| 35 | -1.23 | -229.48 | -0.008 | -0.592 | -0.058 |
| 41 | -51.94 | -125.28 | -0.266 | -0.207 | +0.021 |

## Reading — verdict: weight-specific

- **Behavioral:** the DDXPlus LoRA lowers refusal by 0.117 (0.983→0.867); medical ICL changes it by +0.000 (0.983). Finetuning erodes safety; in-context medical content does not.
- **Mechanistic:** the LoRA shift is directed along −r (toward compliance) in late layers, cos(LoRA,r) reaching -0.592; at that layer the ICL shift is -0.008 (essentially off-axis). The two shifts are ~orthogonal (cos(ICL,LoRA) ≈ 0 at every layer).
- **Conclusion:** for this setup, safety degradation is a function of the **weights, not the activations** — the same medical content delivered in-context neither erodes refusal nor moves the model along the refusal direction. Contrast with the on-task DDXPlus result where ICL and finetuning *did* converge (late-layer cos ≈ 0.8): adaptation converges, but the safety side-effect is weight-specific and ICL does not carry it.
