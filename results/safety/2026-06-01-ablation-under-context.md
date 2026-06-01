# Does the harm-direction ablation survive long context?

`google/gemma-2-9b-it` DDXPlus LoRA | harm dir @L35 ablated all layers | 25 medical, 25 harmful per fill | DDXPlus medical filler

| Context fill | LoRA refusal | +ablate refusal | LoRA acc | +ablate acc |
|-------------:|-------------:|----------------:|---------:|------------:|
| 0% | 0.760 | 1.000 | 0.840 | 0.840 |
| 45% | 0.040 | 0.880 | 0.840 | 0.840 |
| 85% | 0.160 | 0.840 | 0.840 | 0.800 |

## Reading

- LoRA refusal across fill: 0.76→0.16 (the context-fragility). With harm-ablation: 1.00→0.84.
- If +ablate refusal stays high across fills, the static harm-direction ablation is a robust recipe — safety holds even in long context. If it still collapses, the context-fragility is a separate, attention-mediated mechanism the ablation does not fix.
- Task accuracy under ablation across fill: 0.84→0.80 (should stay near the LoRA's — the recipe must keep the task).
