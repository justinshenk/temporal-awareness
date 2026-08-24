# Q4 — E3 competition at fixed distance on Qwen2.5-7B-Instruct

**Verdict: the competition penalty is not detected on Qwen (its contrast includes zero; the
cross-family difference itself is suggestive but not significant, +0.055 [−0.015, +0.125]),
and the attention addendum shows an unambiguous family difference in mechanism signature:
under near-duplicate context Qwen's evidence share RISES where OLMo's was flat. Relative to
that mass advantage, near_dup underperforms by ≈0.10, close to OLMo's −0.085 penalty —
consistent with a masked, not absent, channel (flagged interpretive stacking below).**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · paired panel n = 365 (same
construction, drops, and seeds as the committed OLMo run: 4 overflow skips, 15 starved —
all applied to all arms at once) · artifacts `results/context_fatigue/qwen_e3_competition/`
and `qwen_e3_attention/` · driver `run_competition_sweep.py`. E3c closure arm pending
(running; will be reported separately).

## Main panel (free-generation, graded; parse rate 1.000, gold leaks 0)

| arm | shared options | n | accuracy | mean fill |
|---|---|---|---|---|
| `random` | 0.80 | 365 | **0.732** | 0.753 |
| `near_dup` | 3.75 | 365 | 0.701 | 0.740 |
| `disjoint` | 0.00 | 365 | 0.663 | 0.757 |

Paired gaps (10,000 draws): random − near_dup **+0.030 [−0.016, +0.074] ns**;
disjoint − near_dup −0.038 [−0.085, +0.008] ns; random − disjoint **+0.068 [+0.030,
+0.107] sig**. Joint fit within the panel: overlap β = +0.004 [−0.012, +0.021] ns, fill
β = +0.19 [−0.14, +0.51] ns.

OLMo's committed headline was random − near_dup = +0.085 [+0.030, +0.140] with overlap
carrying the joint fit. **CORRECTED (verification pass 2026-08-24):** the original text here
claimed Qwen's CI excludes OLMo's entire interval; it does not — they overlap on
[+0.030, +0.074], and the cross-family difference-in-differences is **+0.055
[−0.015, +0.125], not significant**. What is established: Qwen's own contrast includes zero
(the penalty is not detected on Qwen), while the cross-family *magnitude* difference is
suggestive, not demonstrated — the E2b rule (overlapping intervals are non-robustness, not
refutation) applies. The unambiguous family difference in this experiment is the
attention-direction inversion below, whose CIs are disjoint from OLMo's by an order of
magnitude. Qwen also operates far above OLMo's baseline here (0.66–0.73 vs 0.49–0.51).

**Anchor check (added in the same pass):** unlike OLMo, Qwen's low-competition arms do NOT
land on its own E1 `local` (0.630, n=192): `random` sits at 0.732 (+0.10, unpaired z≈2.4)
and `disjoint` at 0.663. Plausibly a real ICL effect rather than harness drift — same-task
DDXPlus context practices the probe's task where E1's MMLU filler does not, and Qwen shows
the larger random−disjoint gap consistent with exploiting that practice — but it means the
Q4 arms float on a practice benefit OLMo's did not, and any cross-family comparison of arm
levels (not just gaps) inherits it.

The one significant Qwen contrast, random > disjoint, was not pre-registered; it is
robust to clustering (pathology-clustered bootstrap +0.068 [+0.034, +0.113], spread over
21/44 pathologies) and matches OLMo's direction (+0.027 there). Candidate mechanism, in
line with the arm design's own ICL rationale: `disjoint` context shares zero of the
probe's candidate pathologies, so its in-context practice is maximally irrelevant to the
probe's option set. Reported as a suggested effect, not a claim.

## Attention addendum (all-layer pooled, `--head-layers 0..27`, same panel)

| arm | evidence share | question share |
|---|---|---|
| `near_dup` | **0.0225** | 0.2639 |
| `random` | 0.0155 | 0.2680 |
| `disjoint` | 0.0148 | 0.2670 |

Paired contrasts: near_dup − random = **+0.0071 [+0.0062, +0.0080] sig**; near_dup −
disjoint = +0.0077 [+0.0068, +0.0088] sig. On OLMo the same contrast was −0.00027 (null):
competition there cost accuracy *without* touching evidence mass. On Qwen the direction
inverts — under confusable context the model reallocates attention TO the local evidence
— and no accuracy cost appears.

**Interpretive note (flagged as such).** Q3's dose-response prices share at +10.4 accuracy
per unit. near_dup's +0.0071 mass advantage then predicts ≈+0.074 accuracy vs random;
observed is −0.030. The shortfall, ≈0.10, is close to OLMo's measured second-channel
penalty (−0.085). A consistent account: the competition channel exists in both families,
and Qwen masks it behaviorally by defending the evidence's attention mass under
confusability. This stacks two designs' coefficients and is not a within-experiment
estimate; the E3c closure arm (pending) bears on it — closing competitor mentions should
matter less on Qwen if the residual channel is already compensated.

## Notes

- Same panel as OLMo: n=365, both drop classes identical (4 overflow, 15 starved).
- Preflight (`qwen_e3_competition_preflight/`) validated arm construction (shared options
  0 / 1.2 / 3.7), grading on real replies, zero gold leaks.
