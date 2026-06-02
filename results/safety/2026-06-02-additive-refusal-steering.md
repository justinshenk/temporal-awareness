# Additive context-aware fix — steer the LoRA toward refusal across context fills

`google/gemma-2-9b-it` DDXPlus LoRA | neutral filler | additive steer toward refusal at layers [21, 28, 35] (each by `coeff x (mean resid|refuse - mean resid|comply)`, natural scale; ‖vec‖@L35=143 vs resid norm 694) | held-out refusal n=24, DDXPlus n=24. coeff=1 adds the full refuse-comply separation at each layer.

## Refusal vs (steer coeff x context fill)  —  >0 fill is where the un-steered LoRA collapses

| coeff | fill 0% | fill 45% | fill 85% | DDXPlus acc (clean) |
|------:|------:|------:|------:|--------------------:|
| 0 | 0.75 | 0.54 | 0.62 | 0.83  ← un-steered |
| 0.5 | 0.88 | 0.71 | 0.83 | 0.83 |
| 1 | 0.96 | 0.92 | 0.92 | 0.92  ← sweet spot |
| 2 | 0.75 | 0.29 | 0.12 | 0.96 |
| 4 | 0.00 | 0.00 | 0.00 | 0.96 |

## Reading

- **The additive fix works — and it is the clean mirror of the failed ablation.** Un-steered, the LoRA's refusal collapses 0.75→0.54 under context fill. Steering *toward* refusal (the `-d_comply` direction, additive) at coeff 1 holds refusal **0.92–0.96 across all fills** while DDXPlus accuracy is *unharmed* (0.83→0.92). The probe's ablation of the same axis drove refusal *down* (0.54→0.08) — same direction, wrong sign. So the context erosion really is carried by the behavior-grounded compliance axis, and intervening on it in the correct (additive) sense restores safety under long context.
- **It is a Goldilocks curve — magnitude-sensitivity made visible.** Refusal is non-monotone in the steer coefficient: 0.5 under-corrects (partial rescue), 1.0 is optimal, and 2.0–4.0 *over-drive* and paradoxically destroy refusal (→0.12, then →0.00) as the model is pushed off-distribution and emits degenerate text the classifier no longer scores as a refusal. This is the same magnitude artifact documented in the keep-component sweep — the fix exists only inside a calibrated operating band, not as "more steering = safer."
- **Why a single late layer failed first.** An earlier single-layer steer at L35 (even at coeff 64 on the unit axis) barely moved refusal: a fixed additive offset at one layer is renormalized away by the LayerNorms above it. Distributing the natural-scale push across all three fit layers (21/28/35) is what makes it bite — consistent with Arditi-style interventions needing to act through the stack, not at one point.
- **Contrast with the static-r recipe.** The capstone's harm-direction *ablation* (a different geometry — a finetune-added vector whose removal reverts to base) is robust to long context (#10). The static `r` itself does not transfer to the context regime. This additive steer along the *context-fit* `d_comply` is the intervention that matches the context-fatigue geometry: it is built from the LoRA's actual refuse/comply behavior under fill, and it is the one that holds under fill.
- **Scope:** additive steer at layers [21, 28, 35] (each layer's own natural-scale toward-refusal vector), one model/task, held-out n=24, near-ceiling base refusal. coeff=1 = the full refuse-comply separation per layer. Task acc is the over-drive guard, measured at clean context (the over-drive rows show task survives even where refusal breaks — confirming the high-coeff refusal collapse is a steering artifact, not a task collapse).
