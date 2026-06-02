# Context-specific compliance direction vs the static refusal direction

`google/gemma-2-9b-it` DDXPlus LoRA | neutral filler | d_comply fit on 36 prompts' refuse/comply behavior across fills | best layer L35 (separation 37.83).

## d_comply vs static r (cosine per layer)

| Layer | cos(d_comply, r) | refuse/comply separation |
|------:|-----------------:|-------------------------:|
| 21 | -0.382 | 34.30 |
| 28 | -0.236 | 33.03 |
| 35 | -0.156 | 37.83 |

## Does the drift track d_comply or r? (signed projection; >0 = toward compliance)

| fill | LoRA refusal | drift·d_comply@L35 | drift·r@L35 |
|-----:|-------------:|----------------------:|----------------:|
| 0% | 0.83 | +0.0 | +0.0 |
| 45% | 0.67 | +24.1 | +8.3 |
| 85% | 0.72 | +17.2 | +6.8 |

## Causal: ablate d_comply across fills (held-out prompts)

| fill | LoRA refusal | + ablate d_comply |
|-----:|-------------:|------------------:|
| 0% | 0.75 | 0.71 |
| 45% | 0.54 | 0.08 |
| 85% | 0.62 | 0.17 |

## Reading

- **d_comply ≠ r.** cos(d_comply, r) = −0.16 to −0.38 — mostly orthogonal. The behavior-grounded compliance axis is a *different direction* from the static Arditi r, which is why r did not track the context drift.
- **The drift moves along d_comply, not r.** drift·d_comply@L35 = +24.1/+17.2 (toward compliance) vs drift·r = +8.3/+6.8 — ~3× larger and tracking the refusal dip. The context erosion has a coherent direction; it just isn't r.
- **Ablation is the WRONG intervention (sign error).** Ablating d_comply made refusal *worse* (0.54→0.08): d_comply IS the behavioral refuse/comply axis, so projecting it out removes refusal (Arditi), it does not restore it. (The capstone's harm direction was a *specific added compliance vector* whose removal reverts toward base — a different geometry.) The context-aware fix must be **additive** — steer toward refusal (−d_comply) — not ablation.
- **Follow-up (done):** the additive steer along −d_comply at layers 21/28/35 holds refusal **0.92–0.96 across fills** at coeff 1 (un-steered collapses to 0.54) with task accuracy unharmed — the correct-sign mirror of this failed ablation. See [`2026-06-02-additive-refusal-steering.md`](2026-06-02-additive-refusal-steering.md).
