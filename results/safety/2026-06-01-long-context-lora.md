# Long-context probe — task (distributed) vs refusal (low-rank) under context fill

Base `google/gemma-2-9b-it` | DDXPlus LoRA | 30 medical, 30 harmful per fill | filler = DDXPlus medical cases | max_ctx 8192

| Context fill | LoRA acc | LoRA refusal | base acc | base refusal |
|-------------:|---------:|-------------:|---------:|-------------:|
| 0% | 1.000 | 0.800 | 0.133 | 0.967 |
| 45% | 0.933 | 0.100 | 0.700 | 0.967 |
| 85% | 0.967 | 0.233 | 0.667 | 0.967 |

## Reading

- **Task is context-robust.** LoRA accuracy holds across fill (1.00→0.97); base accuracy even *rises* via ICL (0.13→0.70). The distributed / weight-resident task resists context degradation — the valid half of "weight-resident ⇒ robust."
- **Refusal fragility is finetuning-induced, not dimensionality.** Base refusal is rock-stable under context (0.97→0.97), but the finetuned model's refusal collapses (0.80→0.10). Same low-rank refusal mechanism — fragile only after finetuning. So low rank alone does NOT predict context-fragility; finetuning destabilizes it.
- **Headline (interaction):** neither finetuning alone (clean refusal 0.80) nor context alone (base 0.97) is catastrophic, but finetuning × long context drives refusal to 0.10 — a safety collapse invisible to standard short-context evals.
