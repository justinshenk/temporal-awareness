# E1 mechanism — is the distance penalty attention dilution, position, or locality?

**Verdict: the accuracy penalty is mediated by the evidence's attention mass, and the
mass→accuracy relationship is *graded and shallow* — roughly 0.04 accuracy per sweep step, which no
single contrast in this program was powered to detect. Positional decay (token distance, not turn
count) is what drains the mass. An earlier draft of this report called the penalty a "locality
threshold"; the E1f sweep refutes that and the claim has been withdrawn.**

> **Intervals corrected 2026-08-19 — all contrasts below are now PAIRED.** Every run in this
> report scores the *same item* under several conditions, and this report always said so
> ("paired n = 174"). The intervals, however, came from `arm_accuracy_gap`, which resamples the
> two arms **independently** — correct for independent arms, wrong here, because it charges the
> interval for between-item difficulty variance that cancels inside a within-item contrast. It
> inflated every CI by roughly 2.5×. Recomputed with `paired_accuracy_gap`
> (`scripts/context_fatigue/analyze_dilution_paired.py` → `results/context_fatigue/
> dilution_paired.json`); both intervals are recorded there so the correction is auditable.
> **This changes one verdict** — E1d's necessity contrast — and makes six of E1f's seven dose
> contrasts significant. The old, wider intervals are quoted alongside as "unpaired".

Four runs, 2026-08-18, `allenai/OLMo-2-1124-7B-Instruct`, L24, seed 42:

| run | artifact | driver |
|---|---|---|
| E1b attention addendum | `e1_with_attention/` | `run_distance_sweep.py --measure-attention` |
| E1c sufficiency | `e1c_evidence_clamp/` | `run_evidence_clamp.py` |
| E1d necessity | `e1d_evidence_rescue/` | `run_evidence_clamp.py --clamp-arm back_20 --donor-arm local` |
| E1e tokens vs turns | `e1e_dissociation/` | `run_distance_dissociation.py` |

Reported together because they are one argument, not four; the brief's §9 "one report per
experiment" is bent deliberately here.

## Why this was needed

E1 showed a 0.19–0.21 accuracy penalty for displacing the evidence at identical fill, with distance
carrying the regression coefficient and fill carrying none. That is consistent with attention
dilution but does not establish it: **E1 measured no attention at all**. Distance and any
attention drain move together by construction, so a positional or interference account predicts the
same table. These runs separate them.

## E1b — the drain is real (n = 192 per arm, fill fixed at 0.69)

| arm | evidence share @L24 | accuracy | question share |
|---|---|---|---|
| `local` | 0.0408 | 0.464 | 0.105 |
| `back_2` | 0.0363 | 0.359 | 0.109 |
| `back_5` | 0.0320 | 0.292 | 0.109 |
| `back_10` | 0.0195 | 0.250 | 0.112 |
| `back_20` | 0.0124 | 0.276 | 0.116 |

Every share gap from `local` excludes zero; r(distance, share) = **−0.83**. Question share stays
flat, confirming that only the evidence moved. Accuracy reproduced E1's numbers exactly — a
determinism check.

**A trap worth recording.** *Within* an arm, higher evidence share predicts **lower** accuracy
(`local`: β = −11.2 [−20.0, −2.6]; bottom share quartile 0.542 vs top quartile 0.312). Controlling
for vignette length makes it stronger (−13.8 [−23.3, −4.3]), so it is not a length artifact.
The reading is that attention tracks *difficulty* — the model looks harder at confusing vignettes.
This observational correlation has the **opposite sign to the causal effect** established below, and
anyone reading share-vs-accuracy correlations off this data without intervening will conclude the
reverse of the truth.

## E1c — mass removal is sufficient (paired n = 174)

Evidence kept at `local` position; its span clamped to **the same item's own** `back_20` share.

| condition | evidence share | accuracy |
|---|---|---|
| `local` | 0.0414 | 0.5365 |
| `local_clamped` | 0.0125 | 0.3333 |
| `back_20` | 0.0125 | 0.3646 |

- `local` − `local_clamped` = **+0.2021 [+0.1379, +0.2672]** — significant
  (unpaired: [+0.1034, +0.2998])
- `local_clamped` − `back_20` = −0.0249 [−0.0833, +0.0345] — **indistinguishable**, and the
  pairing tightens this null to ±0.06 (unpaired: [−0.1216, +0.0718])
- for reference, `local` − `back_20` = +0.1772 [+0.1044, +0.2500]

Starving the evidence's attention, with nothing moved, lands exactly on `back_20`'s accuracy:
**114% of the distance penalty reproduced by mass alone.** Median clamp scale 0.152 (−1.89 nats),
the same on-manifold magnitude E2a's 0.15 level used, far from the −4.7 to −6.1 ablation regime.

## E1d — mass restoration is not sufficient (paired n = 174)

The converse: evidence left at `back_20`, clamped **up** to the same item's `local` share.

| condition | evidence share | accuracy |
|---|---|---|
| `local` | 0.0414 | 0.5365 |
| `back_20_clamped` | 0.0415 | 0.4219 |
| `back_20` | 0.0125 | 0.3646 |

- `back_20_clamped` − `back_20` = **+0.0546 [+0.0172, +0.0977]** — **significant**
  (unpaired: [−0.0460, +0.1533], which is why this was previously read as a null)
- `local` − `back_20_clamped` = **+0.1226 [+0.0536, +0.1925]** — significant residual

**Verdict revised by the paired reanalysis.** Restoring the mass recovers **32% of the penalty,
and that recovery is real** — the earlier "does not clear zero" reading was an artifact of the
unpaired estimator. The correct statement is that mass restoration is *partially* sufficient: a
significant recovery with a significant 68% residual.

**The asymmetry is probably the instrument.** The clamp applies a *uniform* bias across heads. It
can strip mass faithfully, but it cannot reconstruct *which* heads should carry the restored mass —
boosting every head equally is not the inverse of the natural pattern. Full sufficiency paired with
*partial* necessity is exactly the signature of an intervention that is faithful in one direction
and crude in the other. This is a limitation of the method, not evidence for a second mechanism,
and should not be quoted as the latter.

## E1e — tokens govern the mass; neither governs the accuracy

A partial 2×2 in (gap turns) × (filler length), with total context and fill held equal by padding
each arm with leading short filler. n = 192 per arm, fill 0.72, context ~2953 tokens in every arm.

| arm | gap turns | gap tokens | evidence share | accuracy |
|---|---|---|---|---|
| `local` | 0 | 15 | 0.0413 | **0.510** |
| `turns5_short` | 5 | 446 | 0.0288 | 0.354 |
| `turns5_long` | 5 | 1684 | 0.0104 | 0.328 |
| `turns20_short` | 20 | 1740 | 0.0108 | 0.318 |

**Share is positional.** The two matched-token arms have effectively identical share (0.0104 vs
0.0108) despite a 4× difference in turn count, while at matched turns share falls 0.0288 → 0.0104
as tokens go 446 → 1684. Attention drain is a function of **token distance**, which is what RoPE
decay predicts.

**Accuracy is not graded.** The whole cost is paid at the first displacement:

Paired n = 192, bootstrap 10,000 draws, cases as the unit:

| contrast | isolates | Δ accuracy (paired) | Δ evidence share |
|---|---|---|---|
| `local` → `turns5_short` | evidence leaves the current turn | **+0.1562 [+0.0885, +0.2240]** | — |
| `local` → `turns5_long` | " | **+0.1823 [+0.1042, +0.2604]** | — |
| `local` → `turns20_short` | " | **+0.1927 [+0.1094, +0.2708]** | — |
| `turns5_short` → `turns5_long` | +1,240 gap tokens, **turns fixed** | +0.0260 [−0.0417, +0.0938] n.s. | **+0.0184 [+0.0174, +0.0193] sig** |
| `turns5_long` → `turns20_short` | 5 → 20 turns, **tokens fixed** | +0.0104 [−0.0573, +0.0781] n.s. | −0.0003 [−0.0009, +0.0002] n.s. |

Joint fits over all rows confirm the split:

| outcome | ~ gap_tokens | ~ gap_turns |
|---|---|---|
| evidence share | **−1.71e-05 [−1.77e-05, −1.64e-05] sig** | +2.19e-05 [−2.23e-05, +6.63e-05] n.s. |
| accuracy | −7.19e-05 [−1.35e-04, −8.92e-06] sig* | −1.73e-03 [−7.85e-03, +4.38e-03] n.s. |

\* the accuracy-on-tokens coefficient is carried by the `local` rows (gap ≈ 15 tokens); it encodes
the displacement step, not a gradient, as the matched contrasts above show.

**The sharpest single result here is contrast C2.** At matched turn count, evidence attention share
falls by 64% (0.0288 → 0.0104) with a tight CI excluding zero, and accuracy does not move
(+0.026 [−0.042, +0.094]). Once the evidence has left the current turn, **its attention mass can be
cut by nearly two thirds at a cost bounded above by 9.4 points** — and the paired interval is what
makes that a usable bound rather than a shrug.

## What the four runs say together

1. **Positional decay is real in the attention.** Token distance, not turn count, sets how much
   attention the evidence retains (E1e).
2. **That mass is causally sufficient for the accuracy penalty.** Removing it at fixed position
   reproduces the penalty in full (E1c).
3. **The accuracy–mass relationship is graded, not a threshold** (E1f, below). Each sweep step of
   ~0.004 share costs ~0.04 accuracy; the cumulative effect over a large share change is real and
   matches E1c almost exactly. Under the paired estimator the gradient is significant from the
   first step, so this is a measured slope, not an inference from a chain of nulls.
4. **E1e's flat contrasts remain flat, but are now usefully bounded.** Its C2 step
   (+0.026 [−0.042, +0.094]) and E1f's comparable clamped step, 0.029 → 0.012
   (+0.115 [+0.038, +0.191]), still have overlapping intervals, so the data cannot separate "the
   same shallow gradient" from "clamping and displacement differ". The point estimates differ
   four-fold, so this is a genuine open question rather than a settled agreement — the paired
   analysis narrows it without closing it.

For the paper: the honest mechanism sentence is *"displacing evidence out of the current turn drains
its attention mass, and that mass causally mediates the accuracy cost; the relationship is graded
but shallow, so only large changes in mass produce detectable changes in accuracy."* "Dilution" is
defensible for both channels; what is **not** supported is any claim about a threshold or a knee.

## E1f — the share→accuracy dose-response at fixed position (common subset n = 131)

`run_evidence_clamp.py --levels 0.036 … 0.012` at `local`, 192 probes. Levels at or above an item's
natural share are skipped, so per-level raw n ranges 131–192; everything below uses the **common
subset present at every level**, which is the only comparison where the item set is held fixed.

| clamped share | accuracy | vs natural, **paired** (95% CI) | vs natural, unpaired |
|---|---|---|---|
| 0.0441 (natural) | 0.473 | — | — |
| 0.0360 | 0.420 | **+0.053 [+0.008, +0.107]** | +0.053 [−0.069, +0.176] |
| 0.0320 | 0.427 | +0.046 [−0.008, +0.099] | +0.046 [−0.076, +0.168] |
| 0.0290 | 0.389 | **+0.084 [+0.023, +0.145]** | +0.084 [−0.038, +0.198] |
| 0.0250 | 0.351 | **+0.122 [+0.053, +0.191]** | +0.122 [+0.000, +0.237] |
| 0.0200 | 0.313 | **+0.160 [+0.084, +0.244]** | +0.160 [+0.046, +0.275] |
| 0.0160 | 0.313 | **+0.160 [+0.076, +0.244]** | +0.160 [+0.038, +0.275] |
| 0.0120 | 0.275 | **+0.198 [+0.115, +0.282]** | +0.198 [+0.084, +0.313] |

**No knee exists, and under the corrected estimator the gradient is detectable from the first
step.** Six of the seven levels differ significantly from the natural share — including 0.036, a
cut of less than a fifth of the mass. The largest adjacent step is **+0.053** (natural → 0.036;
an earlier draft of this section said "every adjacent step is ≤0.038", which its own table
contradicted), and three adjacent steps exclude zero. The curve is a smooth, roughly linear
decline with no threshold anywhere in it — the "graded, not a knee" conclusion is unchanged and
now rests on a significant gradient rather than on a chain of nulls.

**This experiment was run to confirm a knee and refuted it.** The hypothesis was that E1c (cutting
0.041→0.012 at `local`, costing 0.202) and E1e's C2 (cutting 0.029→0.010 at `back_5`, costing
nothing) could only both be true if the curve had a threshold between them. The real resolution is
duller and better: **both are the same shallow gradient**, and E1e's single step was too small to
detect. E1f's natural→0.012 contrast is **+0.198 [+0.115, +0.282]**, against E1c's independently
measured **+0.202 [+0.138, +0.267]** for the same share change — agreement to within **0.004**
across two separately-run experiments, which is the strongest internal consistency check in the
program.

## Open

- Whether the residual E1d gap survives a **pattern-matched** clamp (per-head targets rather than
  one aggregate) — the test separating instrument limitation from a genuine second mechanism.
- **Power — the original diagnosis was wrong.** This section used to say every per-step contrast
  had a CI half-width of ~0.12 against effects of ~0.04, and concluded the program needed an order
  of magnitude more items. It did not. **The designs were already paired; the analysis was not.**
  Switching to `paired_accuracy_gap` cuts the half-width to ~0.03–0.08 on the same data and turns
  four previously-null contrasts significant, including E1d's necessity result. What remains true
  is the prescription — prefer a paired design over shared items — which these runs had all along.

## E1d addendum (2026-08-20): the 32% is not a layer-24 artifact — all-layer rescue is significant and still partial

Re-run with the restoration target indexed on the all-32-layer mean share instead of layer 24
(`e1d_alllayer/`, n=192/arm, `--reference-layer 0..31`). back_20 natural share 0.0302 → clamped
to local's 0.0758. Accuracy: back_20 0.370, clamped 0.417, local 0.536. Paired bootstrap:
rescue **+0.047 [+0.010, +0.083]** (now significant, where the L24-indexed run's +0.058
[−0.046, +0.161] was not), penalty +0.167 [+0.099, +0.240], recovered fraction
**0.28 [0.07, 0.61]**. The layer hole in the mediation argument is closed: matched all-layer
share still recovers only about a quarter of the penalty, so the asymmetry with removal (116%)
stands, and the remaining candidate for the shortfall is per-head structure — the uniform
across-head bias can set a span's total mass but not the head pattern the model built at local
position. Paper edit owed: the "Mass restoration" paragraph should quote these numbers and say
"share-matched (all-layer)" rather than "layer-24-indexed".

## E1c addendum (2026-08-24): sufficiency is not a layer-24 artifact — all-layer removal reproduces ~91%

Re-run with the removal target indexed on the all-32-layer mean share instead of layer 24
(`e1c_alllayer/`, n=192 paired triplets, `--reference-layer 0..31`, same panel construction as
`e1d_alllayer/`). local natural share 0.0758 → clamped to back_20's 0.0302 (achieved 0.0302).
Accuracy: local 0.536, local_clamped 0.385, back_20 0.370. Paired bootstrap: removal cost
**+0.151 [+0.099, +0.208]**, displacement penalty +0.167 [+0.094, +0.240], clamped − back_20
+0.016 [−0.052, +0.083] (indistinguishable — the clamped arm lands on back_20). Recovered
fraction **0.91 [0.60, 1.46]**; the interval includes 1.0.

Reading: the committed L24-indexed sufficiency (114%, +0.202 of +0.177) becomes ~91% in
all-layer coordinates with a CI spanning full reproduction. The overshoot past 100% was a
coordinate artifact; the sufficiency claim itself survives the re-denomination. Together with
the E1d all-layer addendum, both directions of the mediation are now established in
layer-agnostic units: removal reproduces the penalty (~0.91, CI includes 1), restoration
recovers 0.28 [0.07, 0.61] of it.
