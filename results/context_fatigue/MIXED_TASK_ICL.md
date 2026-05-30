# Does context rot lower accuracy once in-context learning is removed?

In the homogeneous DDXPlus / single-subject streams, per-case accuracy stays flat
or rises with context because the model in-context-learns the one repeating task
(`CONTEXT_ROT_ATTENTION.md` §4). This experiment removes that crutch to test
whether the rot reaches the *score*: each accumulated question is drawn **uniformly
at random from all 56 MMLU subjects**, so the prior context carries **no predictive
information** about the next question — only the constant MCQ answer *format* is
learnable, not the task or domain. Contrast: a **coherent** single-subject stream
where ICL can help. Same model (OLMo-2-Instruct), matched lengths, pooled over 12
sessions each.

Script: `scripts/context_fatigue/run_random_context.py`. Data:
`results/random_context/`.

## Result — accuracy is flat even with ICL deterred

| context fill | random (no predictive info) | coherent (ICL works) |
|---|---:|---:|
| 0–20%   | 0.57 (n=80) | 0.83 (n=107) |
| 20–40%  | 0.64 (n=84) | 0.86 (n=111) |
| 40–60%  | 0.65 (n=75) | 0.84 (n=111) |
| 60–80%  | 0.69 (n=74) | 0.86 (n=111) |
| 80–100% | 0.45 (n=31) | 0.77 (n=43)  |
| **overall** | **0.62** (n=344) | **0.84** (n=483) |
| **corr(correct, fill)** | **−0.00** | **−0.01** |

- **Random (ICL deterred): essentially flat** across the first ~80% of context
  (0.57 → 0.69), `corr(correct, fill) = −0.00`. The lower absolute level (0.62 vs
  0.84) just reflects that random questions span hard subjects; the *slope* — the
  context-rot question — is zero.
- A **mild dip appears only in the last 80–100% bin** (random 0.45, coherent
  0.77), present in *both* conditions on small n (31 / 43). If anything this is a
  near-context-limit effect, not an ICL-specific collapse.

## Interpretation

The hypothesis — "remove ICL and accuracy will decay as context fills" — is **not
supported**. Even when the accumulated context is non-predictive noise, OLMo-2's
per-case accuracy does not systematically degrade with context length. This
sharpens the overall picture:

> Context rot in OLMo-2 is real and strongly measurable in **attention allocation**
> (system-prompt erosion, recency bias, diffusion) and in **confidence/calibration**
> (entropy collapses ~5×), but it does **not** manifest as falling task accuracy —
> and this holds even after removing the in-context-learning benefit that could
> have masked it. The deployment hazard is the growing *confidently-wrong* gap, not
> a decaying score.

Caveat: only the final ~20% of the 4096-token window shows any dip, on small n;
a longer-context model would be needed to test whether accuracy eventually erodes
deep into a much larger context.
