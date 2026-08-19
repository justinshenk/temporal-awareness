# E3 — competition at fixed distance

**Verdict: CONFIRMED. Confusable context costs accuracy with the evidence local and fill held
fixed — near-duplicate context is worth −0.085 [−0.140, −0.030] against the natural stream. In a
joint fit within E3, option overlap carries the coefficient and fill carries none, the same shape
as E1's distance result. The decline is *not* monotone in confusability: mild overlap is
indistinguishable from none, and the whole cost sits at the near-duplicate end.**

Run 2026-08-19 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · n = **365 paired probes per arm** ·
artifacts `results/context_fatigue/e3_competition/` (`turns.csv`, `summary.json`) · driver
`scripts/context_fatigue/run_competition_sweep.py` · brief `tasks/e3_competition_brief.md`.

## What was varied

The evidence stays at the current query (distance 0) and every arm accumulates **8 prior DDXPlus
cases**, each answered in the transcript with its gold letter. The only thing that moves is how
much the accumulated cases' 5-option differentials **overlap** the current probe's:

| arm | construction | mean options shared with the probe (of 5) |
|---|---|---|
| `disjoint` | 0 shared options, by construction | 0.00 |
| `random` | sampled uniformly (the natural stream) | 0.80 |
| `near_dup` | ≥3 shared, **different gold pathology** | 3.75 |

No context case anywhere has the probe's gold as its *answer* (**0 leaks over 1,095 rows**,
asserted in-driver). The probe's gold does appear as a *distractor* inside `near_dup` context
cases — that is the manipulation.

## Deviation from the parent brief, and why

The brief specified MMLU arms (`unrelated` / `same_subject` / `near_dup`). Measured on the real
pool, that instrument does not work: near_dup and same_subject differ by nothing that matters
(question-stem Jaccard 0.106 vs 0.102; only 16% of picks share even one option). A null there
would have meant "MMLU contains no near-duplicates", not "competition has no effect". DDXPlus
cases draw from a shared 46-pathology universe and separate the arms **5×** at ≤1.1% difference in
context tokens. All three arms are DDXPlus, so format and in-context-learning affordance are
constant; the brief's MMLU `unrelated` arm would have confounded competition with the ICL this
paper credits for holding accuracy up.

## Results

Chance = 0.200. Guard skipped **4 of 384** probes; **15** were dropped because one arm could not
be filled (`near_dup` needs 8 candidates at overlap ≥3; 97% of probes have them). Both drops
apply to *all three arms at once*, so the panel stays paired.

| arm | shared options | n | accuracy | mean fill | ctx tokens | parsed |
|---|---|---|---|---|---|---|
| `random` | 0.80 | 365 | **0.512** | 0.751 | 3077 | 93.2% |
| `disjoint` | 0.00 | 365 | 0.485 | 0.755 | 3093 | 97.5% |
| `near_dup` | 3.75 | 365 | **0.427** | 0.738 | 3024 | 93.4% |

**Paired contrasts** (same 365 probes in every arm; case-resampled bootstrap, 10,000 draws):

| contrast | Δ accuracy | 95% CI | |
|---|---|---|---|
| `random` − `near_dup` | **+0.0849** | [+0.0301, +0.1397] | **significant** |
| `disjoint` − `near_dup` | **+0.0575** | [+0.0055, +0.1123] | **significant** |
| `random` − `disjoint` | +0.0274 | [−0.0192, +0.0740] | not significant |

**Joint fit within E3, accuracy ~ fill + shared options:**

| predictor | β | 95% CI | significant |
|---|---|---|---|
| shared options | **−0.0208** | [−0.0392, −0.0029] | **yes** |
| fill | −0.2844 | [−0.6396, +0.0751] | no |

## Against §6 of the brief

- *Confirms:* accuracy falls with confusability at fixed distance and fill, `near_dup`'s CI
  excluding zero ✔. Competition is isolated from distance — the contrast needle setups cannot make.
- *Falsifies:* would have required all three arms flat. They are not.
- *Control-arm agreement:* the brief requires the low-competition arms to land on E1's `local`
  (0.464) or the run is void. `random` 0.512 (+0.049 [−0.037, +0.134]) and `disjoint` 0.485
  (+0.021 [−0.065, +0.107]) both agree. The harness did not drift.

## The non-monotonicity is real and worth stating

Confusability does not order the arms. `random` (0.80 shared) is the *best* arm, above `disjoint`
(0 shared), though not significantly. Two forces plausibly oppose each other: mild topical overlap
supplies useful priming over the probe's option space, while near-duplication supplies competitors
that actively mislead. Only the second is large enough to measure here. The honest claim is
therefore **near-duplicate context is costly**, not "cost rises with overlap"; we have one
significant step, not a gradient.

## Robustness

- **Not a parsing artifact.** Unparsed responses score as wrong and the arms differ in parse rate
  (97.5% `disjoint` vs 93.4% `near_dup`). Restricted to parsed responses only (paired n = 318),
  `random` − `near_dup` = **+0.0818 [+0.0252, +0.1384]**, still significant; `disjoint` −
  `near_dup` = +0.0472 [−0.0094, +0.1038] loses significance.
- **Not a length artifact.** Context tokens do not order the arms: `disjoint` is the *longest*
  (3093) and mid-accuracy, `near_dup` the shortest (3024). If anything the shorter `near_dup`
  context should have helped it, so length biases against the result.
- **Not a fill artifact.** Fill spans 0.738–0.755 and carries no coefficient in the joint fit.

## Attention addendum — competition is a SECOND channel, not more dilution

Re-run attention-only over the same 384 probes (`run_competition_sweep.py --attention-only`,
artifacts `e3_attention/`, paired n = 365, identical selection and seeds), measuring the evidence
span's share at L24 under each arm:

| arm | evidence share @L24 | question share @L24 | accuracy |
|---|---|---|---|
| `disjoint` | 0.03409 | 0.11512 | 0.485 |
| `random` | 0.03230 | 0.11378 | 0.512 |
| `near_dup` | 0.03257 | 0.10782 | 0.427 |

| contrast | Δ evidence share | Δ accuracy |
|---|---|---|
| `random` − `near_dup` | **−0.00027 [−0.00088, +0.00035]** n.s. | **+0.0849** sig |
| `disjoint` − `near_dup` | +0.00152 [+0.00089, +0.00216] sig | +0.0575 sig |
| `random` − `disjoint` | −0.00179 [−0.00220, −0.00137] sig | +0.0274 n.s. |

**The headline contrast moves accuracy 8.5 points with no detectable change in the evidence's
attention mass.** Quantitatively: E1f's dose-response is 6.29 accuracy per unit share (R² = 0.966
over the balanced panel), so the observed share difference predicts an accuracy gap of **0.0017 —
2.0% of the 0.0849 actually measured**, and only 6.5% even at the upper end of its CI. Producing
this accuracy drop through mass would take a share change of 0.0135, **50× larger** than what is
there. Note also that the two contrasts run *opposite* ways: `random` − `disjoint` has a
significant share difference and no accuracy difference, while `random` − `near_dup` has a
significant accuracy difference and no share difference.

**Conclusion: distance and competition are two different mechanisms.** Displacement costs accuracy
by draining the evidence's attention mass (E1b/E1c/E1f: sufficient, graded, dose-dependent).
Competition costs accuracy while leaving that mass intact. "Attention dilution" is the right
account of the first and the wrong account of the second, and a paper that folds them together
under one word will mispredict every setting where the two come apart — which is every
needle-in-a-haystack benchmark, because burying a needle does both at once.

The one measurable attention correlate of competition is on the **question** span, not the
evidence: `near_dup` depresses question share by 0.0060 [0.0046, 0.0073] against `random`. That is
the direction the per-case error signature in §7 associates with *correct* answers, so it does not
support a simple "competition steals attention from the query" reading either, and we do not push
it further here.
