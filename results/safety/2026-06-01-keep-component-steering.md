# Keep-only-component steering — what each LoRA component does (base model)

Base `google/gemma-2-9b-it` steered by the mean DDXPlus LoRA shift at layers [14, 21, 28, 35, 41] | 30 medical, 30 harmful | LoRA reference: acc 1.00, refusal 0.84.

| Steering | DDXPlus acc (task↑) | refusal rate |
|----------|--------------------:|-------------:|
| base (no steer) | 0.133 | 0.967 |
| base + steer full | 0.433 | 0.000 |
| base + steer parallel | 0.000 | 0.000 |
| base + steer orthogonal | 0.733 | 0.000 |

## Reading

- **CONFOUND — refusal is uninformative here.** Every steering condition (full, parallel, orthogonal) drove refusal to 0.00. That is a non-specific *magnitude* artifact: the mean late-layer LoRA shifts are large, and adding them at 5 layers (coeff 1.0, all positions) over-drives the model so refusal breaks regardless of direction. The safety half of Q4 needs a coefficient sweep to read; do not interpret the refusal column.
- **Accuracy (provisional, surprising):** orthogonal-only RAISES DDXPlus acc 0.13→0.73, while parallel-only DROPS it 0.13→0.00 (full intermediate, 0.43). So the task-answering capability rides on the LoRA-**orthogonal** component, and the ICL-**parallel** direction alone does not carry it (it is actively disruptive). This is consistent with the ICL shift being dominated by a "context-present / positional" mode rather than the answer — steering base toward "you're deep in a context" with no actual examples degrades it.
- **Caveat:** still an over-driven regime (refusal artifact) and only 5/42 layers, so treat the accuracy ordering as suggestive, not final. If it survives a magnitude sweep it would REFINE the earlier "shared subspace = the beneficial task part" reading: the shared direction is safety-neutral and ICL-aligned, but the *answer* capability is largely in the orthogonal LoRA-specific part.
