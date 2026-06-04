# Is the conditional steering map W's safety principled, or OOD-luck?

**TL;DR.** The input-conditional ridge map `W` (steer = α·W·a), proposed in CONDITIONAL_STEER as
a "robustly safe finetuning alternative," was only ever evaluated on **far-OOD** harmful prompts
(generic AdvBench). The dual form `W·a = Σᵢ (a·aᵢ) Cᵢ` emits a small shift on far-OOD inputs **by
construction**, so far-OOD refusal proves nothing. Testing harmful prompts at **graded distance to
the medical fit distribution** shows the naive map's safety is **OOD-luck** — it erodes refusal
1.00→0.63 on harmful prompts wearing DDXPlus clothing (α=1.0). The **r-orthogonalized** map is the
genuinely **principled** fix: refusal holds 1.00 across all distances.

## Method

`scripts/safety/run_steer_ood_probe.py` + `src/probes/safety/graded_harmful_data.py`. Fit the
DDXPlus conditional map `W` (per-case ICL shifts, ridge dual form), then probe on three harmful
buckets at increasing distance-to-fit (Qwen2.5-7B, L21, n=30/bucket, λ=1):

| bucket | construction | should be refused |
|---|---|---|
| far | AdvBench (bombs, malware) | yes |
| near | medical-flavored harmful (overdose a patient, fake a diagnosis for opioids) | yes |
| indist | harmful ask + real DDXPlus clinical block + "You are a doctor." frame | yes |

Distance ordering is **verified**, not asserted: mean `cos(a, fit-centroid)@L21` = far 0.597 <
near 0.636 < indist 0.666 (monotone). Readouts per bucket: `‖W·a‖/‖a‖`, `cos(W·a, r)`, and refusal
under the naive and r-orthogonalized maps at α ∈ {0.5, 1.0}.

## Result

| bucket | ‖W·a‖/‖a‖ | cos(W·a,r) | naive α0.5 | naive α1.0 | orth α0.5 | orth α1.0 |
|---|---|---|---|---|---|---|
| far    | 0.371 | −0.155 | 0.97 | 0.97 | 0.97 | 1.00 |
| near   | 0.396 | −0.155 | 1.00 | 0.90 | 1.00 | 1.00 |
| indist | 0.419 | −0.159 | 1.00 | **0.63** | 1.00 | **1.00** |

base refusal = 1.00 on every bucket. ΔRefusal naive(α1.0) = +0.03 / +0.10 / **+0.37**;
r-orth(α1.0) = +0.00 / +0.00 / +0.00.

## Mechanism

`cos(W·a, r)` is **constant** at ≈−0.16 (a fixed-direction compliance leak — the mid-layer r-leak
from CONDITIONAL_STEER), while `‖W·a‖` **grows** with closeness-to-fit because the dual-form map
weights training shifts by similarity `(a·aᵢ)`. **Fixed direction × magnitude that grows toward fit
⇒ refusal collapses on in-distribution-looking harmful.** AdvBench-only evaluation never excites the
map (small `‖W·a‖`), so it misses the failure entirely.

## Two honest refinements

1. **The naive-map erosion is an α=1.0 effect.** At the recommended α=0.5 the naive map's margin
   still absorbs the leak (indist 1.00). Precise claim: the naive map's safety *margin* shrinks
   toward fit and breaks only at high steer strength — not "unsafe at the recommended setting."
2. **r-orthogonalization is principled, not itself OOD-luck.** Projecting r out of every row of `C`
   makes every emitted `W·a ⊥ r`; the map then holds refusal 1.00 across all distances at both α.
   Zeroing the leak *direction* defuses the growing *magnitude*. The published "robustly safe"
   recipe survives the OOD probe where the naive map fails.

## Benign twins + Corollary 1 (`run_steer_twin_probe.py`)

Six buckets — a benign twin at each format — deconfound distance-from-length and test Corollary 1
(W·a depends on a only through k(a), so it cannot tell harmful from benign at the same format):

| format | cos(W·a,r) harm / benign | ‖W·a‖ harm / benign (ratio) | cos_fit harm / benign |
|---|---|---|---|
| far    | −0.155 / −0.118 | 0.371 / 0.412 (0.90) | 0.597 / 0.649 |
| near   | −0.155 / −0.123 | 0.396 / 0.456 (0.87) | 0.636 / 0.716 |
| indist | −0.159 / −0.133 | 0.419 / 0.631 (0.66) | 0.666 / 0.958 |

- **Direction is constant and NOT input-aware** — harmful is if anything *more* compliance-aligned
  than its benign twin. Any safety is off-axis, not selectivity.
- **At far/near, harm ≈ benign ‖W·a‖** → Corollary 1 confirmed (the map can't tell them apart).
- **Caveat:** suffix-appended indist harm reaches only 66% of the benign ‖W·a‖ ceiling (the appended
  harmful suffix pulls the last token partway OOD). Interleaving the harm into the MCQ stem hits the
  ceiling (‖W·a‖ 0.58) but drops base refusal to 0.10 (the model just answers the letter).

## The refit fix: luck → a learned harm-response (`run_steer_refit_probe.py`)

Augment the fit with HELD-IN harmful-in-domain exemplars at target δ=0:
`W_refit = ridge([A_benign; A_harm], [Δ_benign; 0])`. Evaluated on HELD-OUT in-domain harmful
(disjoint cases, n=25 × 2 seeds, suffix harm, base refusal 1.00):

| map | ‖W·a‖/‖a‖ | cos(W·a,r) | refusal 1.00→ | DDXPlus task 0.25→ |
|---|---|---|---|---|
| naive | 0.439 | −0.166 | **0.46** | 0.62 |
| orth  | 0.433 | +0.000 | 0.74 | 0.61 |
| refit | **0.035** | −0.081 | **0.82** | 0.61 |

- **Naive erodes 1.00→0.46 (seed-robust)** — OOD-luck confirmed behaviorally, not just by magnitude.
- **Refit cuts ‖W·a‖ 12× on held-out harm it never saw (generalizes), holds refusal best (0.82),
  and keeps the benign task gain (0.61)** — a learned, generalizing harm-response.
- **Orthogonalization only partially rescues in-domain harm (0.74, vs 1.00 on far-OOD)** — direction-
  zeroing is a split-sensitive post-hoc patch; refit (magnitude-killing) is the principled fix.
- Honest residual: refit not fully restored (0.82) — small shift summed over 5 steered layers + n=25
  noise; α=0.5 / more exemplars expected to close it.

## Robustness: across α, model family, and route (`run_steer_refit_probe.py`, `run_lora_refit_probe.py`)

- **α-dependence (Qwen, α=0.5, 80 exemplars):** all maps hold refusal 1.00 — the naive map's erosion
  is an α=1.0 over-steer effect; at the recommended α=0.5 even naive holds. refit ‖W·a‖ 0.025.
- **Cross-family (gemma-2-9b-it, L35, α=1.0, base refusal 0.72):** naive 0.72→0.00, orth 0.72→0.00
  (FAILS on gemma), refit 0.72→0.88 (‖W·a‖ 0.33→0.015, task kept). The fix replicates; orth does not.
- **LoRA-distill route (Qwen d75, α=1.0):** distil the LoRA's OWN shift (stronger task, carries the
  weight-route baggage). LoRA finetune itself: refusal 0.00 / task 0.84. naive-distill 1.00→0.00,
  orth 1.00→0.00 (fails), **refit 1.00→1.00 with the best task (0.66 > naive 0.57), ‖W·a‖ 0.655→0.050.**
  Refit LAUNDERS the LoRA — keeps its task transfer, kills the erosion it carries.

| route / setting | naive | orth | refit |
|---|---|---|---|
| ICL-distill, Qwen, α=1.0 | 0.46 | 0.74 | 0.82 |
| ICL-distill, gemma, α=1.0 | 0.00 | 0.00 | 0.88 |
| LoRA-distill, Qwen, α=1.0 | 0.00 | 0.00 | 1.00 |
| LoRA-distill, gemma, α=1.0 | 0.00 | 0.00 | 1.00 |

(held-out in-domain harmful refusal; base 1.00 for Qwen, 0.72 for gemma). **r-orthogonalization fails
at α=1.0 on 3 of 4 settings (partial on the 4th); the δ=0 refit holds in all four** — route-agnostic
(ICL + LoRA), cross-family (Qwen + gemma). On the LoRA route refit also keeps the best task transfer
(Qwen 0.66, gemma 0.75, approaching the LoRA's own 0.84/0.93) — it LAUNDERS the weight route: task in,
erosion out.

## Takeaway

A steering map can look safe purely because the safety eval lives outside its active input regime.
The discriminating test is harmful prompts graded by distance-to-fit (distance verified in activation
space), with the mediator readout (`‖W·a‖`, `cos(W·a, r)`) separating a constant leak from a magnitude
effect. The naive map is OOD-luck (constant non-protective direction, magnitude grows toward fit,
refusal collapses on in-domain harm). r-orthogonalization is a partial patch; the principled fix is
**refitting with harmful-in-domain exemplars at δ=0**, which teaches a small, generalizing shift on
harmful inputs while preserving benign task transfer.

See `results/safety/ood_probe_qwen/ood_probe.json` and `ood_probe_plot.png`.
Related: `route-dependent-safety-result` (CONDITIONAL_STEER), `GRADED_RISK.md` (benign-topic null).
