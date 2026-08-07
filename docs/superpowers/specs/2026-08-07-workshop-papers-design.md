# Two workshop papers for *Interpretability as a Science* (NeurIPS 2026) — design spec

**Date:** 2026-08-07 · **Deadline:** 2026-08-28 AoE (21 days) · **Branch:** `context-fatigue-datasets`

## 1. Problem statement

Two research threads on this branch are each written up as an extended abstract, and neither is
submittable:

- `register_vs_procedure_abstract.{md,tex}` — 6 findings, 4 pp, **zero figures**, **zero citations**,
  `neurips_2023.sty`, author name and email in the tex, and one claim in the abstract with no
  artifact behind it (`"and so do MLP, …"`).
- `context_fatigue_paper/context_fatigue.tex` — closer: NeurIPS-styled, 2 figures, real
  bibliography, body 4 pp. Still on `neurips_2023.sty` and still names the author.

Beyond mechanics, the flagship paper has one **substantive** defect. Its organizing claim is a
two-sided contrast — a *register* is low-rank, on-manifold and roughly pointwise; a *procedure* is
high-rank, off-manifold and time-dense — but only the procedure side was ever measured. Grepping
`2026-06-08-refusal-frontier.md` and `2026-06-14-lora-vs-loreft-commonsense.md` for any oracle,
lockstep, temporal or PCA-band run returns nothing. The register half is inferred from the fact that
a map happened to work on it. Separately, every procedure null is a bare `0.00` with no interval,
while the *context-fatigue* paper — same author, same submission batch — bounds its nulls properly.

At a venue whose stated topics are **measurement validity, falsifiability, and causal grounding**,
both of those are the objection, not a nitpick.

## 2. Venue

[Interpretability as a Science](https://interpscience.github.io/), NeurIPS 2026, Sydney, Dec 11–12.

| | |
|---|---|
| deadline | **Aug 28, 2026 AoE** |
| length | short ≤5 pp **or long ≤9 pp**; refs/appendix excluded |
| template | ICLR or NeurIPS format accepted; camera-ready must be NeurIPS |
| archival | non-archival |
| review | double-blind |
| dual submission | concurrent NeurIPS OK; **work under review at another workshop is prohibited** |

Consequences: (a) Interp4Discovery is **out** — the prohibition is mutually exclusive with it, and its
scope ("what do models know that we don't" — proteins, climate, astronomy) fits neither paper;
(b) the 9-page track means Paper A's submission **is** the ICLR 2027 draft, not a cut-down of it
(ICLR: abstract Sep 19, paper Sep 25 — note the new reciprocal-reviewing policy).

## 3. Agreed solution approach

### Paper A — *Register, Not Procedure* — 9 pp long paper (= ICLR draft)

Spine: register installs, procedure does not, here is the mechanism, here is the criterion.

| § | content | state |
|---|---|---|
| 1 Intro | steering and PEFT are one map `h ↦ h+(Wh+b)`; "is X steerable" is not one question | rewrite |
| 2 Related work | 4-part adversarial structure; see `docs/related_work_register_vs_procedure.md` | **from scratch — highest risk** |
| 3 Setup | ridge map, coherence-aware metric, contrast-set protocol | expand |
| 4 (R) Register installs | refusal Pareto: map 0.62 vs CAA/Arditi/CAST 0.00 | have |
| 5 (R) Divergent routes | LoRA vs LoReFT on commonsense; CKA 0.96→0.13 | have |
| 6 (P) Procedure does not | 5-rung null ladder, **with intervals** | have + S1 |
| 7 Mechanism | L20 oracle 0.75; distributed / compute-isn't-it / time-dense / variance-dense | have |
| 8 (G) Second procedure | MuSiQue: oracle +0.76, density replicates, plan-vs-execute in sign only | have + P5 |
| 9 (R) **Register battery** | same oracle/temporal/PCA-band drivers, run on a *register* task | **S2 — new** |
| 10 Criterion | 4 behaviours × 3 measured coordinates → installability | **S3 — new** |
| 11 (X) Cross-model | disposition carries; better aim transports worse | have |

Framing for this venue: lead §1 on evidentiary grounds — the field asks "can we steer X" as though it
were one question, and every null here is anchored to a positive control (the oracle), so a null means
*not installable*, not *we did not try hard enough*.

### Paper B — *Context Fatigue* — 5 pp short paper

Content is settled and committed. Work is mechanical: 2026 template, anonymize, fit to 5 pp, and one
framing paragraph aimed at measurement validity (the signatures replicate; the performance cost does
not exist). **Ship first and park it** — it protects against Paper A slipping.

### Strengthening work (approved)

- **S1 — bound every null (CPU, immediate).** `bootstrap_interval` / `clustered_rate_gap` already exist
  in `src/common/bootstrap_stats.py`. Every `0.00` in the ladder becomes `0.00, 95% CI [lo, hi], n`.
  A bounded null is a claim; a bare point estimate is not.
- **S2 — measure the register side (GPU, after P5).** Register `commonsense` as a third `TaskSpec` in
  `scripts/attribution/attribution_common.py`. The seam is five callables (`problems`, `prompt`,
  `score`, `format_gold`, `lens`), and `src/probes/attribution/commonsense_data.py` already exposes
  load/format/extract/score. Then run the **unmodified** drivers — `lockstep_patch_gsm8k`,
  `temporal_oracle_gsm8k`, `lockstep_pca_band` — against the existing commonsense donor LoRA
  (base 0.00 → 0.68). Hypothesis: recovery survives sparse temporal gating and low-rank truncation
  where GSM8K collapses. **Either outcome is reported as found**; a negative result here reshapes the
  paper and is more interesting than the expected one, not less.
- **S3 — the criterion table.** With S2 done: 4 behaviours (refusal, commonsense, GSM8K, MuSiQue) ×
  3 measured coordinates (δ-rank, ‖δ‖/‖h‖, temporal density) → binary installability, with the
  coordinates ordering the outcome. Turns §10 from a proposal into a predictor.

### S4 — §2 related work, restructured (CPU, first in the writing queue)

Full findings in **`docs/related_work_register_vs_procedure.md`** (literature check, 2026-08-07).
Two results drive the paper, not just the citations:

- **The map's form is not novel.** Input-conditional, matrix-valued affine steering is occupied
  ground as of 2026: conceptors (Postmus & Abreu), CLAS (Hsu et al., Apr 2026), INNSteer (Nguyen &
  Le, Jun 2026). CLAS is closest and is **rank-1 conditional** (fixed direction, learned magnitude)
  where ours is full-rank. Position the map as the **instrument, not the claim** — a scoped sentence
  in §3, promotable if no prior art for the closed-form donor-regression recipe turns up.
- **One contender must be rebutted explicitly.** *Weight Updates as Activation Shifts*
  (Adila et al., arXiv:2603.00425, Feb 2026) unifies weight updates with activation shifts, defines
  the same `δh_oracle = h_FT − h_base`, evaluates on GSM8K, and reports **within 0.2–0.9% of full
  finetuning**. It does not refute us — it trains an adapter on task loss over 8,790 GSM8K problems
  with **no donor**, intervening at **every block**, and uses its oracle only analytically. But the
  distinction (transport vs learn) must be argued in §2 and §6 or the null reads as refuted.

**Required scoping of the headline claim.** The abstract's "recovers ≈0 … and so do MLP, on-policy
DAgger, per-context, and task-loss (DAS) variants" implies *no* activation-space method installs
reasoning, which Adila et al. falsify. DAS is the exposed rung: task-loss-trained, yet rank-512 at a
**single layer** where their adapter spans every block. Reword to: *a procedure does not transport
through a fitted pointwise map at a layer; installing it requires distributed, temporally dense
intervention trained on the task* — which our own density measurements independently predict, making
their positive result our **confirming case** rather than a counterexample.

§2 order: (1) fixed-vector steering; (2) conditional/matrix steering — state the CLAS distinction;
(3) PEFT-as-activation-edit incl. Adila — the unification is not ours to claim; (4) the rebuttal
paragraph. All four contenders are concurrent (Feb–Jun 2026); say so where it matters.

### P5 — the pending GPU correction (mandatory, unchanged)

Brief: `tasks/gsm8k_ridge_layer_probe.md`. `collect_cot_residuals` → `fit_ridge_sweep` →
`steer_gsm8k --layers 8,12,16,20,24,28,31 --alphas 1.0 --n-eval 200` → `nonlinear_delta_gsm8k
--layer 20`. Closes the unmatched ladder axis (§8) and decides whether the abstract's
`"and so do MLP …"` clause is backed or must be cut. **Runs first** when the GPU returns.
Both §6 and §8 must be drafted to accept either outcome.

Note `tasks/current_task.md:122` marks P5 "IN FLIGHT" with a log at `.run_logs/p5_gsm8k_collect.log`
— that job is not running on this machine (no `.run_logs`, no CUDA). Correct the status line.

### Figures — six for Paper A, all CPU, all from artifacts present locally

Reuse before writing: `scripts/attribution/plot_{temporal_oracle,pca_band,das_subspace,
downstream_lesion,logit_lens,activation_similarity}.py` already exist and must be checked for reuse
first (project rule: no duplicate code).

| # | figure | source (verified present) |
|---|---|---|
| F1 | temporal-density knee, GSM8K + MuSiQue overlaid — the money figure | `temporal_oracle_L20.json`, `temporal_oracle_multihop_L20.json` |
| F2 | oracle layer sweep, L20 peak | `lockstep_multihop_single.json` |
| F3 | null ladder vs oracle, with S1 intervals | `short_arithmetic`, `local_refit_gsm8k`, `dagger_refit_gsm8k`, `das_subspace_L20` |
| F4 | variance-band cliff (top-64 = 55% energy, 0% recovery) | `lockstep_pca_band_L20.json` |
| F5 | gold-token lens by role — GSM8K crystallizes, multi-hop does not | `gold_token_lens_L20.json`, `gold_token_lens_multihop_L20.json` |
| F6 | refusal Pareto frontier | `refusal_frontier.json` |

A seventh (α-resonance / layer hump) needs the P2b JSONs, which exist **only on the GPU box** —
`results/` is gitignored, so retrieve `steer_multihop_{alpha_L20,layers}.json` and
`{nonlinear_delta,temporal_oracle}_multihop_L20_n100.json` when back on it.

### Provenance rule

The last three commits (`600b5f7`, `c3e2c62`, `089534e`) found four unsourced numbers, one of them in
the paper's abstract. Therefore: **maintain `numbers.md` beside each paper mapping every figure in the
tex to a committed artifact; no number enters the tex without a row.** Before drafting, finish the
audit sweep across `results/activation_weight_investigation.md` (line 71 repeats the MLP claim),
`paper_draft.md`, `consolidated_status.tex`, `docs/RESEARCH_PROGRAM.md`.

## 4. Files likely modified

**New:** `papers/register_vs_procedure/` (tex, bib, `numbers.md`, `figures/`);
`scripts/attribution/plot_null_ladder.py`, `plot_refusal_frontier.py`, `plot_oracle_layers.py`,
`plot_gold_token_roles.py` (only those not covered by existing plot scripts);
`scripts/attribution/null_intervals.py`; commonsense entries in `configs/attribution/`;
`tests/test_null_intervals.py`, commonsense cases in `tests/test_attribution_tasks.py`.

**Edited:** `register_vs_procedure_abstract.{md,tex}`, `context_fatigue_paper/context_fatigue.tex`,
`scripts/attribution/attribution_common.py` (commonsense TaskSpec only),
`results/attribution/2026-06-16-multihop-generality.md`, `results/activation_weight_investigation.md`,
`tasks/current_task.md`, `tasks/lessons.md`. NeurIPS 2026 `.sty` replaces `neurips_2023.sty`.

## 5. Non-goals

- **No third procedure.** Correct long-run, wrong for 21 days; name it as future work.
- **Do not re-run the multihop side** (P0–P4 stand) or **retrain any LoRA**, including GSM8K's.
- **Do not touch committed P4 gold-token-lens results.**
- **Do not rewrite the context-fatigue results** — that paper is content-complete.
- **No refactor of the drivers.** S2 adds a registry entry; if a driver needs changing, stop and re-plan.
- Not recovering the lost `steer_results.json` — unrecoverable and superseded by P5.
- Not fixing the three unrelated legacy test-collection errors (`test_tree_as_structures_system.py`,
  `test_batch_interventions.py`, `test_sample_position_mapping.py`) unless they block CI.

## 6. Operational constraints

- **No GPU until ~Aug 10**, then available near-continuously. All CPU work front-loaded.
- GPU order is fixed: **P5 first** (it corrects a live overclaim), **then S2**. ~1–2 days each.
- Long jobs run under `nohup` with a log in `.run_logs/` and are **actively monitored to completion**;
  on error, fix and restart immediately — never left unattended, never assumed successful.
- Seeded (42) throughout. `steer_gsm8k` ignores `--layers`/`--alphas` when naming output — **rename
  between runs**. Multihop drivers need `--n-eval 500` to match the cached 317 contrast indices.
- Do **not** reuse the contrast set's base/LoRA accuracies as references for a `max_new=512` run;
  they were measured at 256.
- Double-blind: strip author name, email, acknowledgements, and scrub GitHub/HuggingFace usernames,
  branch names and absolute paths from both PDFs.

## 7. Schedule

| days | resource | work |
|---|---|---|
| Aug 7–9 | CPU | **S4 §2 related work + §6 rebuttal draft (first)**, template migration, anonymization, **S1 null intervals**, six figures, provenance sweep |
| Aug 10–12 | CPU | Paper B to submittable 5 pp → parked |
| Aug 10–12 | GPU | **P5** (collect → fit → layer sweep → MLP); retrieve P2b JSONs |
| Aug 12–14 | GPU | **S2** register battery (commonsense TaskSpec → oracle / temporal / PCA-band) |
| Aug 12–24 | CPU | Paper A: §2 related work first, then §§1, 3–11; S3 criterion table |
| Aug 25–27 | CPU | provenance re-check, anonymity check, final builds, **submit Aug 27** (1-day buffer) |
| Sep | — | expand A for ICLR 2027 (abstract Sep 19, paper Sep 25) |

**Top risk:** §2 written from zero citations at 9 pp. It starts *before* the body, not after.
**Second risk:** P5 collapsing the divergence rewrites §8 — which is why it runs first.

## 8. Acceptance criteria

1. Two PDFs on the NeurIPS 2026 template, fully anonymized, ≤9 pp and ≤5 pp of main text, each
   self-contained without its appendix.
2. Paper A carries ≥6 figures and a real bibliography.
3. Every number in both papers traces to a committed artifact via `numbers.md`. Zero unsourced cells.
4. Every null in Paper A's ladder carries an interval and an n.
5. `nonlinear_delta_gsm8k_L20.json` exists; the abstract's "and so do MLP …" clause is either backed
   by it or cut.
6. GSM8K per-layer steering measured at L20 and L24; §8 states the matched comparison, whichever way
   it lands — if the divergence collapses, that is reported plainly, not soft-pedalled.
7. §9 reports the register battery and §10 the 4×3 criterion table — including if S2 refutes the
   expected pattern.
8. Full attribution + context-fatigue CPU suites pass unchanged.
9. §2 engages Adila et al., CLAS, INNSteer and conceptors explicitly; §6 carries the transport-vs-learn
   rebuttal; the headline null is scoped to pointwise single-layer transport rather than to activation
   space in general. No novelty claim for the map's *form* appears anywhere.
10. Submitted by Aug 27 AoE.

## 9. Development process — test-forward

New code is small and seam-shaped, and every piece of it is written test-first:

1. **Commonsense `TaskSpec`** — write `tests/test_attribution_tasks.py` cases first (registry returns
   the spec; `prompt`/`score`/`format_gold` round-trip on fixtures; contrast-set construction on a
   stub donor). Then register. No driver may change.
2. **`null_intervals.py`** — write `tests/test_null_intervals.py` first: known-input intervals against
   hand-computed values, an all-zero null yielding a `[0, hi]` interval with `hi > 0`, and clustered
   resampling using the problem as the unit (tokens are dependent within a chain).
3. **Plot scripts** — check the six existing `plot_*.py` for reuse before writing any. New ones get a
   smoke test that renders from a fixture JSON without network or GPU.

All tests CPU-only, no network, seeded.

## 10. Test expectations

- New: commonsense registry cases (~6), `test_null_intervals.py` (~6), plot smoke tests (~4).
- Unchanged and still passing: `tests/test_multihop_{data,prompts}.py`,
  `tests/test_attribution_tasks.py`, `tests/test_chain_token_roles.py`,
  `tests/common/test_bootstrap_stats.py`, `tests/probes/context_fatigue/test_null_statistics.py`,
  `tests/test_loreft_intervention.py`, `tests/test_commonsense_data.py` — **111 passing at spec time**
  (verified 2026-08-07); the count only goes up.
- The three legacy collection errors are pre-existing and out of scope; do not let them mask a new one.
