# Keep-only-component steering — coefficient sweep (base model)

Base `google/gemma-2-9b-it` | layers [14, 21, 28, 35, 41] | 25 medical, 25 harmful | base (no steer): acc 0.16, refusal 1.00 | LoRA ref: acc 1.00, refusal 0.84.

## DDXPlus accuracy (task↑)

| component | c=0.1 | c=0.25 | c=0.5 | c=1.0 |
|---|---|---|---|---|
| full | 0.72 | 0.72 | 0.80 | 0.52 |
| parallel | 0.32 | 0.68 | 0.56 | 0.00 |
| orthogonal | 0.60 | 0.72 | 0.72 | 0.80 |

## Refusal rate (safer↑)

| component | c=0.1 | c=0.25 | c=0.5 | c=1.0 |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 0.00 | 0.00 |
| parallel | 1.00 | 1.00 | 0.16 | 0.00 |
| orthogonal | 1.00 | 1.00 | 1.00 | 0.00 |

## Reading (corrects the over-driven c=1.0 single-point run)

- **Both components carry task signal.** At the safe magnitude c=0.25 (refusal intact at 1.00 for all), every steering type lifts accuracy comparably — parallel 0.68, orthogonal 0.72, full 0.72, vs base 0.16. The earlier "parallel destroys the task" was purely a c=1.0 over-drive artifact (parallel→0.00 only at c=1.0).
- **The refusal damage is carried by the ICL-shared (parallel) direction, not the orthogonal one.** As coeff rises, orthogonal preserves refusal far longer: at c=0.5 orthogonal keeps refusal 1.00 *and* accuracy 0.72, while parallel collapses to 0.16 and full to 0.00. Steering toward the ICL-aligned "context-mode" direction is what erodes refusal; the LoRA-specific (orthogonal) direction lifts accuracy without that cost over a wider magnitude range.
- **This ties back to the long-context result.** The parallel component is the ICL/context-mode signal; steering base toward it erodes refusal — the same axis real long context drives the model along (which collapsed the finetuned model's refusal 0.80→0.10). Refusal erosion rides on the context-mode axis.
- Caveat: n=25, so accuracy values are noisy (±~0.1); the refusal transitions are sharper and the load-bearing signal. Steering at 5/42 layers, so magnitudes are approximate.
