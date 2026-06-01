# Capstone — ablate the finetune-only harm direction

Base `google/gemma-2-9b-it` | DDXPlus LoRA `results/lora_icl/adapter` | direction layer 35 | 50 harmful prompts, 50 medical cases

| Condition | refusal rate (safer↑) | DDXPlus acc (task↑) |
|-----------|----------------------:|--------------------:|
| base | 0.980 | 0.100 |
| LoRA | 0.840 | 1.000 |
| LoRA + ablate harm dir | 0.980 | 0.980 |
| LoRA + ablate task dir (control) | 0.680 | 1.000 |

## Reading

- **Ablating the harm direction recovers safety at ~no task cost:** refusal 0.84→0.98 (back to base 0.98) while DDXPlus accuracy stays 0.98 (LoRA 1.00). The compliance drift is causally carried by the finetune-only, ICL-orthogonal direction.
- **Specificity control:** ablating the *task* direction does NOT restore safety (refusal 0.68) — so the recovery is specific to the harm direction, not a generic side effect of ablating something.
- **Caveat:** ablating the task direction did not degrade accuracy (1.00); the LoRA's task solution is redundant and not bottlenecked by that single ICL-derived direction, so the task half of the dissociation is not demonstrated. The safety half is clean and is the load-bearing result.
- **Upshot:** you can keep a finetune's task gain (accuracy 1.00→0.98) and remove its safety cost (refusal 0.84→0.98) by projecting out one ICL-orthogonal direction — a direct causal demonstration that the harm is separable from the beneficial task subspace.
