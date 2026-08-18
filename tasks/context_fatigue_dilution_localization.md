# EXECUTION BRIEF — Paper B: localizing attention dilution to *relevant* context

Written 2026-08-18. Scope: **Paper B (context fatigue) only.** No Paper A / attribution work is
in scope here.

> **Freeze conflict, read first.** `tasks/context_fatigue_worries.md` (2026-08-13) states
> *"Science is done — no more GPU-hours on this paper"* against an Aug 28 AoE deadline. Nothing in
> this brief is required for that submission. Treat it as **v2 / camera-ready / follow-up work**,
> to be started only after the Aug 28 artifact is out, unless a reviewer response forces E1 forward.
> E5 is the one item cheap enough (CPU + existing residual dumps) to consider before the deadline.

---

## 1. Problem statement

The paper's central claim is a **null**: with individually localized tasks, accumulating context
produces the full battery of context-rot signatures — entropy collapse, monotone attention drift off
the system prompt, halving of attention mass on the current query (r = −0.71 with fill at L24) — and
**no accuracy cost** across the first ~80% of the window.

The reading we want to support is the one the literature already points at, made precise:
**attention dilution is the operative mechanism of context rot, and it bites on *relevant* context.**
Where the answer-bearing information must be located among accumulating competitors (needle-in-a-
haystack, long documents, growing codebases), dilution degrades performance — as prior work reports.
Where each answer is locally self-contained (our harness; the shape of an agentic transcript of
finished, locally-scoped steps), the same mass drain is measurable but costless. Our null therefore
*corroborates and localizes* the dilution account rather than competing with it.

Three gaps stand between that reading and the evidence:

- **G1 — the contrast is cross-literature, not within-harness.** We never ran a buried-evidence
  condition in our own setup. The comparison is against other papers' models, items, prompts and
  metrics, so the first reviewer objection is "your task is simply too easy to degrade." We cannot
  currently answer it.
- **G2 — "mass is not the binding constraint" is untested.** Accuracy is flat over the range
  accumulation happened to traverse (current-query share ≈0.35 → ≈0.15). We never located a floor,
  so we cannot say whether accumulation stops far short of one or one step from it. This is the
  difference between "there is headroom" and "we did not reach the edge."
- **G3 — the one positive result is unattributed.** The ≥80% fill dip on the random stream
  (−0.141 [−0.249, −0.031], n=91) sits at ~93% of a 4k window. "Near the edge of the trained context
  length" is an untested alternative to "accumulation outruns ICL."

## 2. Agreed solution approach

Five experiments, ranked by decisiveness. E1 and E2 are the pair that converts the paper's null into
a measured mechanism; E3 is the novel contribution; E4 and E5 defend flanks.

All reuse the existing accumulation harness (`scripts/context_fatigue/run_ddxplus_mcq.py`,
`run_random_context.py`, `_cf_common.py`) and the validated attention reconstruction in
`src/probes/context_fatigue/attention_capture.py` (post-RoPE last-token attention, agrees with
`output_attentions` to max|Δ| ≈ 1.4e-3). Reuse the **overflow guard** from the deep-fill batch in
every condition — it must not be silently dropped, since truncating long items manufactures exactly
the dip we are attributing.

### E1 — Within-harness distance sweep (highest value)

Vary one thing: **where the answer-bearing information lives**, holding items, models, fill and
metric fixed.

| arm | evidence location |
|---|---|
| `local` | at the current query (current design; the control) |
| `back_k` | the current question's evidence block placed *k* turns back in the accumulated transcript, k ∈ {2, 5, 10, 20} |
| `split` | evidence split across two earlier turns at different depths (composition over the transcript) |

DDXPlus is the natural substrate: the case vignette detaches from the 5-option question cleanly, so
`back_k` moves a block that is genuinely required without changing the question text.

Fit accuracy ~ fill + distance jointly over pooled sessions. The claim predicts **distance carries the
coefficient and fill does not.**

### E2 — Causal attention-mass dose-response

Stop inferring the threshold from accumulation and measure it. Add a clamp hook that rescales
attention logits on a designated token span at inference, then renormalizes.

- **E2a (find the floor).** On cold-start contexts (low fill, natural query share ≈0.35), sweep the
  current-query span's post-softmax share down through {0.30, 0.20, 0.15, 0.10, 0.05, 0.02} and
  measure accuracy at each level. Expect a plateau then a cliff. Report the cliff location against
  the ≈0.15 that accumulation actually reaches — that gap *is* the headroom claim, as a number.
- **E2b (does mass rescue the dip?).** On the random stream at ≥80% fill, clamp query share back
  *up* to its cold-start level and re-measure. If the −14 point dip does not recover, the dip is not
  a dilution effect and G3 is half-closed from the mechanism side.

### E3 — Competition at fixed distance (most novel)

Needle setups confound *distance* with *competition*: burying the answer makes it both far away and
surrounded by plausible alternatives. Separate them. Keep the answer at the query (distance fixed,
fill fixed) and vary only the **confusability** of the accumulated context:

| arm | accumulated context |
|---|---|
| `unrelated` | random-subject MMLU stream (existing) |
| `same_subject` | prior cases from the current question's subject, distinct answers |
| `near_dup` | prior cases that are near-duplicates of the current question with *different* correct answers |

If accuracy falls with confusability at constant distance and fill, competition is isolated as the
variable — a distinction the current literature cannot make.

### E4 — Window-position control for the late dip

Re-run the deep-fill random-stream protocol at **identical absolute token counts** on a model with a
much longer trained window (Qwen-2.5-7B at 32k), so ~93% fill of the 4k protocol is a small fraction
of the model's window. If the dip travels with *fill fraction* it is accumulation; if it stays pinned
near the 4k-equivalent absolute position it is positional, and the paper must say so.

### E5 — Is mass even the right currency? (cheapest)

Attention mass can fall while extracted information holds constant (value vectors reorganize). Train
a linear probe to decode the current question's content (subject label, correct-option identity) from
the last-token residual, and track decodability against fill. **Flat decodability against halved mass**
is a direct demonstration that mass ≠ information transfer, and upgrades "enough, not maximal" from a
behavioural inference to a mechanistic one. Reuses the probe infrastructure from the
`within_task_fraction` work; runs on already-dumped residuals where available.

## 3. Files likely modified

- `src/probes/context_fatigue/attention_capture.py` — add the span-clamp hook (E2). Keep capture and
  intervention separate; the hook must be a no-op at scale 1.0.
- `src/probes/context_fatigue/context_assembly.py` — **new**; extract transcript construction
  (evidence placement, distractor selection, overflow guard) out of the run drivers so E1/E3 vary a
  parameter instead of forking a driver. The guard currently lives inline in the deep-fill path.
- `scripts/context_fatigue/run_distance_sweep.py` — **new** (E1).
- `scripts/context_fatigue/run_mass_clamp.py` — **new** (E2a + E2b).
- `scripts/context_fatigue/run_competition_sweep.py` — **new** (E3).
- `scripts/context_fatigue/run_random_context.py` — add `--window-model` / absolute-token mode (E4).
- `scripts/context_fatigue/run_query_decodability.py` — **new** (E5).
- `scripts/context_fatigue/analyze_null_statistics.py` — extend so every new number lands in
  `NULL_STATISTICS.md` under the existing provenance rule.
- `tests/context_fatigue/` — CPU tests per §7.

## 4. Non-goals

- **No Paper A / attribution work.** Nothing under `scripts/attribution/` or `src/probes/attribution/`.
- **No new model families.** The four-family replication is done and is not being extended; E1–E3 run
  on OLMo-2-7B-Instruct (the model with the full attention analysis) plus one confirmation family.
- **Do not revisit** the WildChat homogeneity analysis, the OLMo post-training dose-response, or the
  F90871 causal test. Those results stand as reported.
- **Do not re-litigate the entropy-collapse framing.** This brief is about the accuracy null and its
  mechanism only.
- **No new claim that prior context-rot work is wrong.** The framing throughout is corroborative:
  we are localizing a mechanism the literature already proposes, not displacing it.
- **Do not widen the accuracy claim** beyond localized tasks. Genuinely length-dependent single-task
  settings remain out of scope by design.

## 5. Operational constraints

- GPU box; `results/context_fatigue/` is gitignored, so every run writes a JSON artifact there and a
  committed report quotes it. No number enters any writeup without an artifact row.
- **Seeded throughout**; per-cell writes so a killed session preserves informative cells (the lesson
  from the F2 sweep that died mid-L28).
- **Preflight before any long run** — the standing rule from `tasks/lessons.md`: confirm the driver
  loads, produces one cell, and writes before committing hours.
- **Overflow guard mandatory** in every fill-dependent arm. Note in each report which items it skipped
  and in which direction that biases the result.
- Bootstrap CIs via `src/common/bootstrap_stats`, 10,000 draws, seed 42, **cases as the independent
  unit** — no head-level or turn-level pseudo-replication.
- Accumulators to CPU at the second allocation site (the OOM lesson from `47e12db`).
- Watch memory on E2: the clamp hook must not retain attention matrices across steps.

## 6. Acceptance criteria

Each experiment states what confirms it **and what falsifies it**. A falsifying outcome is a result,
not a failure, and goes in the paper.

**E1** — pooled n ≥ 150 cases per arm (≈10–15 sessions each, given tens of cases per window).
- *Confirms:* `local` flat with fill; `back_k` accuracy declines monotonically in k with the k=20 arm's
  95% CI excluding zero; in the joint fit, distance significant and fill not.
- *Falsifies:* `back_k` also flat at k=20 → our items/models lack the sensitivity to detect degradation
  at all, and the localized null is uninformative. **This must be reported if it happens** — it would
  undercut the paper's central claim, and E1 is worth running precisely because it can.

**E2a** — n ≥ 100 cases per clamp level, 6 levels.
- *Confirms:* a plateau across levels at or above the ≈0.15 accumulation reaches, then a cliff below
  it. Report the cliff share and the margin to 0.15 explicitly.
- *Falsifies:* accuracy declines smoothly from the natural share downward with no plateau, i.e. the
  cliff sits at or above 0.15 → mass **is** near-binding, and the "headroom" language must go.

**E2b** — n ≥ 91 (match the existing deep-fill pool).
- *Confirms:* clamping query share back up at ≥80% fill leaves the −14 point dip intact (CI on the
  rescue overlapping zero) → the dip is not dilution.
- *Falsifies:* the dip recovers → the late-window cost **is** dilution, which is a cleaner story than
  the current one and should be adopted.

**E3** — n ≥ 150 per arm.
- *Confirms:* accuracy declines with confusability at fixed distance and fill, `near_dup` CI excluding
  zero → competition isolated from distance.
- *Falsifies:* all three arms flat → competition alone is insufficient; distance/retrieval is required,
  which sharpens rather than weakens the localization claim.

**E4** — matched absolute token counts, n ≥ 91.
- *Confirms:* the dip reproduces at the same *fill fraction* on the long-window model → accumulation.
- *Falsifies:* no dip at matched absolute position in a large window → positional artifact; the paper's
  one positive result must be re-described.

**E5** — probe trained on held-out sessions, reported with held-out R²/accuracy.
- *Confirms:* decodability flat (or near-flat) across fill while measured query attention mass halves.
- *Falsifies:* decodability tracks mass downward → mass **is** the information currency and the
  behavioural null needs a different explanation.

Global: `NULL_STATISTICS.md` regenerated and every quoted number diffed against it before any of this
reaches a writeup.

## 7. Development process — test-forward

Write the test before the driver, in every case. Tests run on CPU with a tiny model
(`hf-internal-testing/tiny-random-*`) or synthetic tensors; no GPU in the test suite.

1. **Clamp hook (E2), before the driver.** Test that scale = 1.0 is bit-identical to the unhooked
   forward; that post-clamp attention rows sum to 1.0 within tolerance; that the requested span share
   is achieved within tolerance after renormalization; that the hook is removed cleanly on exit.
2. **Context assembly (E1/E3), before either driver.** Test that `back_k` places the evidence exactly
   k turns back and that the question text is byte-identical to the `local` arm; that the overflow
   guard skips rather than truncates and logs what it skipped; that `near_dup` never leaks the current
   answer into context.
3. **Analysis, before the runs.** Test the joint fit and the bootstrap on synthetic data with a known
   planted effect — a distance effect of known size must be recovered with a CI covering it.
4. Only then the GPU drivers, each behind a preflight that produces one cell end-to-end.

## 8. Test expectations

- `scale=1.0` no-op: max|Δ| vs unhooked forward < 1e-6.
- Clamp accuracy: achieved span share within 1e-3 of requested, rows sum to 1 within 1e-6.
- `back_k` placement: assert exact turn index of the evidence block for k ∈ {2, 5, 10, 20}; assert
  question span byte-equality against `local`.
- Overflow guard: on a synthetic set with 20% over-long items, assert those items are absent from
  results and present in the skip log, and that no result row has a truncated question.
- `near_dup` construction: assert zero overlap between any context item's correct option and the
  current item's correct option identity.
- Bootstrap: planted +0.10 accuracy difference on synthetic data recovered with 95% CI containing
  0.10 and excluding 0, seed-stable across two runs.
- Analysis regression: re-running the existing deep-fill artifact through the extended analyzer
  reproduces −0.141 [−0.249, −0.031] exactly.

## 9. Report back

One standalone report per experiment under `results/context_fatigue/`, quoting artifact filenames,
n per cell, the overflow-guard skip counts, and the acceptance/falsification verdict from §6 stated
explicitly — including when the falsifying branch is the one that fired.
