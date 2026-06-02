# Additive fix for sycophancy — steer the base model toward holding its ground

`google/gemma-2-9b-it` (no finetune) | ARC-ARC-Easy | additive steer toward holding ground (`-d_syco`) at layers [21, 28, 35] (natural scale; ‖vec‖@L35=157) | fit n=60, held-out test n=48 | caving measured on initially-correct answers.

## Caving rate vs (steer coeff x context fill)  —  lower is less sycophantic

| coeff | fill 0% | fill 45% | fill 85% | turn-1 acc (clean) |
|------:|------:|------:|------:|-------------------:|
| 0 | 0.53 | 0.48 | 0.52 | 0.98  ← un-steered (mean 0.51) |
| 0.5 | 0.48 | 0.33 | 0.33 | 0.96  ← sweet spot (mean 0.38) |
| 1 | 0.72 | 0.39 | 0.41 | 0.96 |
| 2 | 1.00 | 0.97 | 1.00 | 0.65 |
| 4 | nan | nan | nan | 0.00 |

## Reading

- **The sycophancy axis is causal, but the additive fix is modest.** Steering toward holding ground at coeff 0.5 cuts mean caving 0.51→0.38 (strongest under fill: 0.48→0.33 @45%, 0.52→0.33 @85%) while turn-1 accuracy stays 0.96 (un-steered 0.98). So pushing along d_syco does make the model hold its correct answer — but the reduction is *partial*, not the near-total rescue the refusal fix gave (#13: 0.54→0.92).
- **Over-drive is catastrophic and narrow.** By coeff 2 turn-1 accuracy collapses (0.98→0.65); the caving=1.00 rows there are an artifact of the model emitting non-answers, not real agreement. The usable band is small (low coeff only) and the dose-response is non-monotone (coeff 1 is *worse* than 0.5 at fill 0%) — the magnitude-sensitivity seen throughout, sharper here.
- **Why weaker than refusal (#13)?** There the harm was a *low-rank, finetune-added* direction whose additive counter cleanly restored refusal. Here sycophancy is a deeply ingrained *base*-model disposition (no finetune): a single-direction additive steer only partially counteracts it before over-driving — consistent with a more distributed mechanism than one added vector.
- **Caveat on the context effect.** Un-steered caving on this held-out subset is flat across fill (0.53/0.48/0.52), *not* rising as in the probe subset (0.56→0.65). The context-fatigue rise does not replicate across subsets at this n — treat it as within noise; the robust claim is that base sycophancy is *high* (~0.5), not that context reliably worsens it.
- **Scope:** additive steer at layers [21, 28, 35], base model, held-out n=48; turn-1 accuracy is the over-drive guard. coeff=1 = the full held−caved separation per layer.
