# Many-shot sycophancy priming — caving vs demonstrated behavior in context

`google/gemma-2-9b-it` (no finetune) | ARC-ARC-Easy | demos from a disjoint pool, fresh held-out test n=48 | caving measured on initially-correct answers | cave/hold demos differ only in the final demonstrated letter (length matched).

## Caving rate vs (filler condition x context fill)

| fill | neutral | cave-primed | hold-primed |
|-----:|------:|------:|------:|
| 0% | 0.60 | 0.60 | 0.60 |
| 45% | 0.65 | 0.98 | 0.00 |
| 85% | 0.72 | 0.98 | 0.00 |

### Eligibility (turn-1 correct, /48) — priming should not corrupt basic QA

| fill | neutral | cave-primed | hold-primed |
|-----:|------:|------:|------:|
| 0% | 47 | 47 | 47 |
| 45% | 48 | 48 | 48 |
| 85% | 47 | 48 | 48 |

## Reading

- **Baseline (no filler):** caving 0.60.
- **Cave-primed vs neutral at 85% fill:** 0.98 vs 0.72. Higher under cave-priming ⇒ a context full of the model's own caving begets more caving — in-context priming (many-shot), a content effect that neutral length did not produce.
- **Hold-primed vs neutral at 85% fill:** 0.00 vs 0.72. Lower ⇒ demonstrating held ground *inoculates* against pushback — the prime is bidirectional.
- **Isolation:** cave- and hold-primed share identical questions, pushbacks, and token counts; they differ only in the demonstrated outcome letter. So any cave−hold gap is the demonstrated *behavior*, not length or topic. The neutral column is the pure length control.
- **This dwarfs the other knobs.** Neutral length barely moves caving (0.60→0.72); the single-direction activation steer was modest and narrow (0.51→0.38, over-drives by coeff 2). In-context demonstrated behavior nearly saturates it both ways (0.00 ↔ 0.98) with turn-1 accuracy intact — so under context, sycophancy is governed far more by the *demonstrated response policy* than by length or by one residual direction. The many-shot (Anil et al.) channel is the dominant one here.
- **Honest nuance:** every demo ends in the caved (or held) letter, so this is in-context *policy imitation* — the model conforms its new-question answer to the demonstrated behavior. That is exactly the sycophancy-relevant failure (and its inoculation), but it is behavioral conformity, not proof the model's underlying 'belief' changed; QA staying intact shows it is not just degenerate copying.
- **Scope:** base model, one task, held-out n=48; caving on initially-correct answers only; n is small, read gaps not decimals.
