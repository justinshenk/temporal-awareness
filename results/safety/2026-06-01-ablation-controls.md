# Ablation capstone — controls

Base `google/gemma-2-9b-it` | layer 35 | 50 harmful, 50 medical | reference: base 0.98/0.10, LoRA 0.84/1.00, LoRA+ablate_harm 0.98/0.98

| Condition | refusal rate | DDXPlus acc |
|-----------|-------------:|------------:|
| base + ablate harm dir | 0.980 | 0.000 |
| LoRA + ablate random dir | 0.820 | 1.000 |

## Reading

- **Accuracy is weight-provided.** base+ablate_harm task acc = 0.00 (vs LoRA+ablate_harm 0.98). Without the LoRA weights the preserved residual direction carries ~no task ability — so the capstone's retained accuracy is LoRA weight robustness, NOT surgical sparing of a task direction. (Directional residual ablation never disables the weights.)
- **Safety recovery is harm-specific.** LoRA+ablate_random refusal = 0.82 ≈ un-ablated LoRA (0.84), nowhere near the harm-ablation 0.98. A random direction does not restore refusal, so the recovery is specific to the harm direction (also ≠ task direction, which gave 0.68).
- **Net:** the harm direction is a specific, low-rank, safety-relevant direction whose ablation restores refusal; the task survives ablation because it is weight-resident and redundant, not because it was spared. The practical upshot (ablate harm → 0.98 refusal / 0.98 acc) holds; the mechanism for the retained accuracy is LoRA robustness.
