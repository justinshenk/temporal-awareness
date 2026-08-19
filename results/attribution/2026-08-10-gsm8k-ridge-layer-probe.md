# P5 — the GSM8K ridge layer probe (2026-08-10)

**Brief:** `tasks/p5_execution_brief.md`. **Question:** the paper's ladder axis claimed a divergence
— multihop ridge steering leaks ~¼ of the recovery budget at L20 and ~½ at L24, "where every GSM8K
ridge measurement is 0.00" — but GSM8K had never been probed per-layer at L20/L24, and its committed
nulls were too underpowered (n≤50) to support the comparison. This run rebuilds the GSM8K maps and
measures the same per-layer protocol multihop got, matched layer-for-layer.

**Setup.** `NousResearch/Llama-2-7b-hf` + `LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32`, seed 42,
config `configs/attribution/metamath_llama2_gsm8k.yaml` throughout. Collection n_fit 200 / n_te 60 at
max_new 512 (34,893 / 10,663 CoT tokens); the generation phase reproduced the aborted 2026-08-06 run
**token-for-token at every 10-problem checkpoint**, confirming determinism. Steering: n_eval 200,
α=1.0, max_new 512, base/LoRA references measured under the same protocol.

## §6 branch adjudication (the point of the run)

**The result lands between the brief's two branches and must be read as both:**

1. **The divergence survives as a difference of *degree*.** At the matched layers the separation is
   clean: GSM8K L24 = 0.12 [0.07, 0.19] vs multihop L24 = 0.45 [0.35, 0.56] (both n=200, intervals
   disjoint; same at L28). The "task-dependent core size" reading is supported.
2. **"GSM8K ≈0 at every layer" is refuted.** GSM8K itself leaks: 16/200 problems at L24, 17/200 at
   L28, 15/200 at L31 — problems base solves 0/200. Four of seven layers have intervals excluding
   zero. The audit-era correction (600b5f7: "every GSM8K ridge number really is 0.00") is itself
   wrong; the original unsourced "≈0.05" was closer to the truth than the correction that replaced
   it. The abstract's "recovers ≈0 of a reasoning LoRA's gain" was measured everywhere *except* the
   layers where it fails.

The unified statement both curves support: **≈0 through mid-stack, a late-stack leak on both tasks,
same shape, amplitude ~3.75× larger on multihop at the peak.** A shape replication with a
task-dependent amplitude, not a qualitative split.

## The measured curve

Recovery = (steer_acc − base) / (LoRA − base); budget = 0.650 − 0.000. Exact Clopper–Pearson 95%
intervals (`src/common/null_intervals.py`); they treat base/LoRA as known constants, so they cover
sampling error in the steered run only. Multihop reference: `steer_multihop_layers.json` (n=200,
budget 0.630).

| L | λ\* | GSM8K acc | hits/200 | GSM8K recovery [95%] | multihop recovery [95%] | disjoint? |
|--:|--:|--:|--:|---|---|---|
| 8 | 1.00e2 | 0.000 | 0 | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.03] | overlap |
| 12 | 3.16e2 | 0.000 | 0 | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.03] | overlap |
| 16 | 1.00e3 | 0.030 | 6 | 0.05 [0.02, 0.10] | 0.00 [0.00, 0.03] | overlap (reversed order) |
| 20 | 3.16e3 | 0.020 | 4 | 0.03 [0.01, 0.08] | +0.21 [0.13, 0.30] (contrast n=100, max_new=256 — **protocol differs**) | n/a |
| 24 | 3.16e3 | 0.080 | 16 | **0.12 [0.07, 0.19]** | **0.45 [0.35, 0.56]** | **disjoint** |
| 28 | 1.00e4 | 0.085 | 17 | **0.13 [0.08, 0.20]** | 0.38 [0.29, 0.48] | **disjoint** |
| 31 | 1.00e4 | 0.075 | 15 | 0.12 [0.07, 0.19] | 0.24 [0.16, 0.33] | overlap |

Artifact: `steer_results_layers.json`. Reference measurement: base = 0.000, LoRA = 0.650 at
max_new=512. **The LoRA reference sits outside the driver's advisory gate (0.36–0.46)**; the gate
did not abort. 0.650 matches published MetaMath-family GSM8K accuracy (~66%) and the brief predicted
references would rise at max_new=512 (the gate band appears calibrated to a shorter budget), so the
gate is read as stale rather than the run as wrong — recorded, not dropped.

Why nobody saw the leak before: L16's prior probe was n_eval=12 (`3919b6c`). At a true rate of 3%
the expected hits in 12 trials is 0.36 — a zero there was unsurprising. This is the underpowering
§1(b) of the brief argued from the Clopper–Pearson bounds, demonstrated concretely. The historical
all-layer **joint** injections read 0.00 even though single-layer L24 leaks — consistent only if
mid-stack injections corrupt the trajectory off-manifold and swamp the late gain, which is itself
evidence about where the map's output stops being useful.

## Ridge fit (step 2)

`sweep.json`, all 32 layers, λ ∈ logspace(−1, 7, 17). **L20 R²_te = 0.610 at λ\* = 3.16e3** —
acceptance gate (≥ 0.367, the `sweep_smoke.json` floor) passed; the multihop report's unsourced
"≈0.61" is vindicated and now has an artifact. L24 = 0.636. Curve: 0.952 @L0 falling to the
mid-stack minimum 0.559 @L14, recovering to 0.866 @L31. Both decisive layers sit at ~0.61–0.64, so
a steering null there could not have been blamed on fit quality — and the leak at L24 comes from a
map no better-fit than L20's. Multihop L20 remains the stronger fit (0.714).

## MLP rung (step 4) — the paradox replicates

`nonlinear_delta_gsm8k_L20.json` (19 min): **nonlinear_steer = 0.00 [0, 0.17]** (0/20 contrast,
max_new=256, same driver and protocol as multihop's `nonlinear_delta_multihop_L20*.json` — matched
by construction). Geometry: val cos **0.806** / R² **0.651** vs ridge's 0.631 / 0.330 on the same
residuals — the MLP fits the delta decisively better open-loop and recovers exactly nothing
closed-loop, reproducing the multihop paradox (0.822/0.675 vs 0.01 steer) on GSM8K. The ridge arm of
the same run also read 0.00 at n=20, consistent with the scan's 0.03 (expected hits at 3% in 20
trials: 0.6).

**Abstract clause verdict:** "and so do MLP …" now has its first artifact and the artifact supports
it *at L20*. Power caveat, flagged before the result was known: n=20 bounds a zero at [0, 0.17], so
this cannot distinguish "MLP is 0" from "MLP leaks like ridge does at L24 (0.12)" — sufficient where
ridge itself is 0.03, not sufficient at the leak layers. Before the clause is finalized in the
abstract it should be powered at `--n-contrast 100` and run at L24, where the question is live.

## Consequences for the papers (not yet applied — per brief §3, papers/ is untouched)

- The abstract's "recovers ≈0 of a reasoning LoRA's gain — and so do MLP, …" needs rescoping beyond
  what `section6_rescoped_claim.md` drafted: not only is the MLP clause artifact-gated, the ridge
  "≈0" itself is only true mid-stack. Honest form: *a fitted pointwise map transports at most
  0.13 [0.08, 0.20] of the gain at any single layer, all of it late-stack; the remaining ≥0.80 — the
  procedure core — does not transport at any layer or jointly.*
- The P2 axis reframes from divergence to **same mechanism, task-dependent amplitude**, which unifies
  the ladder with P4's independent finding (trajectory scaffold general, per-step work
  task-specific): the transportable late-stack component is the register share of the delta, large on
  scaffold-limited open-book composition, small on computation-limited arithmetic.
- Normalizing leak by the oracle sharpens the contrast where both exist (multihop L24: 0.45/0.78 ≈
  0.58 of recoverable; GSM8K L20: 0.03/0.75 ≈ 0.04; GSM8K's single-layer oracle at L24 was not run,
  so the L24 ratio needs that cell before it can be quoted).

## P5b — the leak is task-agnostic (cross-task transplant, measured 2026-08-10)

The multihop-fit maps (`maps_multihop/W_L*.pt`, fit on hop-composition CoTs, no arithmetic deltas
ever seen) injected into **GSM8K** via `--maps-suffix _multihop`, same protocol as the decisive run
(n=200, α=1.0, refs 0.000/0.650 supplied). Artifact:
`steer_transplant_multihop_maps_on_gsm8k.json`.

| L | native GSM8K map | foreign multihop map |
|--:|---|---|
| 20 | 0.03 [0.01, 0.08] | 0.01 [0.00, 0.04] |
| 24 | 0.12 [0.07, 0.19] | **0.09 [0.05, 0.16]** |
| 28 | 0.13 [0.08, 0.20] | **0.13 [0.08, 0.20]** (17/200 both — identical count) |

A map fit on the *wrong task* delivers ~75% of the native leak at L24 and 100% at L28; every
interval overlaps its native counterpart, and the task-fitted increment is ≤0.03 of budget, within
noise. **The late-stack leak carries essentially no task content** — it is a task-agnostic push
(format/emission register), and the task-specific component of the map transports ≈nothing even at
the layers where the map "works". This converts the L24 leak from an exception to the null into
direct evidence for the register/procedure decomposition, and it is the version of the claim the
paper should lead with. Caveats: both maps share the base model and LoRA-CoT training style, so
"task-agnostic" is within-model; whether even *conditioning* is needed (vs a fixed mean-δ vector) is
the remaining open cell.

## P5b — MLP at the leak layer: the paradox becomes an inversion (measured 2026-08-10)

`nonlinear_delta_gsm8k_L24_n100.json`, n_contrast=100, same protocol as the L20 cell:

| L24 | open-loop cos / R² | closed-loop recovery |
|---|---|---|
| MLP | **0.815 / 0.675** | 0.00 [0.00, 0.04] |
| ridge | 0.656 / 0.354 | **0.10 [0.05, 0.18]** |

**Disjoint intervals: the better-fitting estimator recovers significantly *less*.** At n=100 the
MLP zero now excludes ridge-level leak — the abstract's MLP clause is fully powered at the layer
where it is live. Read jointly with the transplant, both results say the transportable component is
the *unconditional* one: a foreign map delivers the leak (task-agnostic) and the estimator that best
captures the conditional structure delivers none of it (conditioning displaces the generic push).
This makes the mean-δ fixed-vector prediction nearly forced, but that control remains the direct
test.

## Remaining follow-ups (in value order)

1. **Fixed mean-δ control at L24** (both tasks): the direct conditional-vs-fixed test the two
   measured results predict. Needs a small driver flag (new code, TDD).
2. **Transcript check of the leak cells**: native L28 and transplant L28 both solve 17/200 — same
   problems? Solved-set overlap + problem-length profile (register hypothesis predicts short
   problems) once generations are dumped.
3. **α grid at L24/L28** (no new code): does GSM8K share multihop's narrow α=1.0 resonance?

## Wall-clock (for `docs/superpowers/specs/2026-08-07-workshop-papers-design.md`)

| step | estimate (brief) | measured |
|---|---|---|
| 0 preflight (§8 scope) | — | 52 s, 146 passed |
| 1 collect | 0.5–2 h | **24 min** |
| 2 fit | 10–30 min | **3 min** |
| calibration (§5, n=1 + n=5) | — | 5 min; marginal cost 5.2 s/generation |
| 3 layer sweep (7×200 + refs) | 3–5 h | **160 min** |
| 4 MLP rung | ~1 h | in flight |

Preflight note: the brief's step-0 command runs all of `tests/` (742 passed / 49 failed / 60 errors
here — dozens of unrelated stale-import failures, not three); the §8 *scoped* suite is the real
contract and passed 146/146 exactly. The brief's command and its cited baseline do not describe the
same run; fixed by using the §8 scope.

Determinism note: `results/attribution/accumulators` from the aborted 2026-08-06 collect (54/64
files, no `meta.json`) was deleted before this run to free 20 G; the rerun reproduced its trajectory
exactly and completed the save (64 files + `meta.json`, 25 G).
