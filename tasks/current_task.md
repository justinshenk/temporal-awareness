# Current task — Paper B (context fatigue): dilution localization, E1–E5 (EXECUTION)

**Brief:** `tasks/context_fatigue_dilution_localization.md` (2026-08-18). That document is the
approved spec — §1 problem statement through §8 test expectations. This file tracks *execution
state* against it and records deviations found while reading the code.

**Scope decision (user, 2026-08-18):** run the **full E1–E5 program now**, overriding the brief's
own "start only after the Aug 28 artifact is out" gate. Submission prep for the Aug 28 workshop
deadline is explicitly **out of scope for me** — the user is handling it. Do not re-raise it.

**Environment:** `.venv/bin/python` (bare `python` has no transformers). transformers 5.3.0,
torch 2.9.1+cu128. Tests must pass **offline** — `HF_HUB_OFFLINE=1`, no `hf-internal-testing`
downloads; build tiny models from config instead (verified working, see below).

## Deviations from the brief found by reading the code

1. **`attention_capture.py` cannot serve OLMo-2 — the brief assumes it can.** §2 names
   `src/probes/context_fatigue/attention_capture.py` as the validated reconstruction to reuse, and
   §4 fixes E1–E3 on OLMo-2-7B-Instruct. But that module imports **Qwen2**'s
   `apply_rotary_pos_emb` and, critically, omits the **`q_norm`/`k_norm` (QK-norm)** that OLMo-2
   applies before RoPE. Running it on OLMo-2 would silently produce wrong attention.
   The working OLMo-2 recipe exists as a **duplicate inline class** (`Olmo2AttentionCapture`) in
   `scripts/context_fatigue/run_olmo_attention.py:62`. Two copies, one per family, neither shared.
   → **Unify** into a family-agnostic capture (introspect `q_norm`/`k_norm`/GQA off the module,
   resolve `apply_rotary_pos_emb` from the module's own package). Required by CLAUDE.md's
   no-duplicate-code rule and prerequisite to E2's clamp hook. Adds a file to the brief's §3 list.

2. **The overflow guard skips but does not log.** `run_random_context.py:104-108` `continue`s on
   over-long items with no record. §5 requires reporting "which items it skipped and in which
   direction that biases the result", and §8 requires them "present in the skip log". The skip log
   does not exist and must be built as part of extracting `context_assembly.py`.

3. **The DDXPlus driver has no overflow guard at all.** `run_random_context.py:104-108` has one
   (skip-not-truncate); `run_ddxplus_mcq.py` only stops on the fill target (line 265) and appends
   cases unguarded. E1's substrate is DDXPlus, and §5 makes the guard mandatory in every
   fill-dependent arm — so it must be *added* there, not just reused.

4. **Artifacts are not all under `results/context_fatigue/`.** The deep-fill random-stream data is
   at `results/random_context/` and `results/random_context_topbin/` (top level), while
   `results/context_fatigue/` holds `NULL_STATISTICS.md`, `null_statistics.json` and the
   wildchat/instruction dirs. E2b and E4 need the random-context artifacts; E5 needs residual
   dumps — location not yet confirmed.

## Verified facts (do not re-derive)

- Tiny OLMo-2 from config, offline, is a valid test fixture: `Olmo2Config(vocab_size=64,
  hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
  num_key_value_heads=4)`, `cfg._attn_implementation="eager"` → 24,864 params.
  `attn_implementation` must be set on the **config**, not passed to `forward`.
- The OLMo-2 capture recipe (q_norm/k_norm → RoPE → last-token scores → softmax) matches
  `output_attentions` last-token rows to **max|Δ| = 1.49e-8** on that fixture. This is the
  ground-truth comparison every capture test should use.
- Conversation shape in the harness: `conv = [{"role","content"}, ...]`, rendered by
  `_cf_common.render_prompt(tokenizer, conv, is_chat)`; fill measured as
  `len(tokenizer.encode(render_prompt(...))) / max_ctx`.
- Bootstrap API: `src/common/bootstrap_stats.py` → `Interval` (BaseSchema), `bootstrap_interval`,
  `pooled_rate_gap`, `clustered_rate_gap`. Tests monkeypatch `N_BOOT` down for speed.

## RESOLVED — E2 clamp mechanism: route (A), mask bias + bisection

Spiked on the tiny OLMo-2 fixture, 2026-08-18. Adding a bias to the additive `attention_mask` at
the `self_attn` pre-hook **works and is exact enough**: s₀ = 0.2362 → bias +1.0 gives 0.4539
(single-head logit-shift theory predicts 0.4567), +2.0 → 0.6905 (0.6956), −1.0 → 0.1028 (0.1021).
The residual is only mean-over-heads vs per-head logit shift; monotone in `b`, so bisection hits
an aggregate target to any tolerance. **bias = 0 is exactly bit-identical (max|Δ logits| = 0.0)**,
so route (A) gets §7.1's no-op requirement for free, with no reimplementation of the attention
forward — which also means the intervention is provably the model's own attention plus an offset.
Route (B) is dropped; the cost objection to (A) dissolves (~12 prefills/item ≈ 10 min for all of
E2a, not the blocker I assumed).

`self_attn` receives `attention_mask` as `[b, 1, q, k]` float32, min −3.4e38, max 0.0.

### Gap this exposed (fixed)
The capture **ignored `attention_mask` entirely**. It still matched `output_attentions` on plain
forwards, because the last query row of a causal mask is all zeros — so the bug was invisible in
every existing use. Under a mask-based intervention it would have reported the **unclamped**
attention while the model computed the clamped one: a clamp that silently measures nothing.
Fixed by adding the mask's last query row to the scores before softmax; pinned by
`test_capture_respects_attention_mask_bias`, which asserts agreement with `output_attentions`
*under* a bias and that the bias actually moved mass (non-vacuous).

## Superseded design notes (kept for the record)

Target: force the current-query span's post-softmax share to a requested value, then renormalize.
Adding a constant `b` to the span's pre-softmax logits gives exactly
`logit(s_target) = logit(s_0) + b`, so `b = logit(s_target) - logit(s_0)` — exact, and softmax
renormalizes for free. Two routes to apply it:

- **(A) via `attention_mask` additive bias** — non-invasive, no reimplementation of attention, so
  `scale=1.0` bit-identity is free. But the mask is shared across heads (`[b,1,q,k]`), so `b` is
  uniform over heads and `s_0` must be measured first → bisection on `b` (~10-15 extra prefills
  per item) to hit an aggregate target within tolerance.
- **(B) patched attention forward** — per-head exact `b` in a single pass, far cheaper at scale,
  but reimplements the forward, so `scale=1.0` no-op must be an explicit early-return to preserve
  bit-identity.

Cost matters: E2a is 6 levels × n≥100 cases. Leaning **(B)** with a hard early-return at scale 1.0.

## Phases (test-forward per brief §7 — test before driver, every time)

- [x] Recon: harness seams read, deviations logged, offline fixture + ground truth verified.
- [x] **§7.0 (added)** — attention capture unified across Qwen2/OLMo-2 and made mask-aware.
      `SelectiveAttentionCapture` now introspects `q_norm`/`k_norm`, derives head counts from
      projection widths (GQA-aware), resolves `apply_rotary_pos_emb` from the module's own family,
      and applies the additive mask. Agreement with `output_attentions`: **OLMo-2 1.49e-08,
      Qwen2/GQA exactly 0.0**. Duplicate `Olmo2AttentionCapture` deleted from
      `run_olmo_attention.py` (39 lines) and repointed at the shared class — API was identical, so
      it was a drop-in. 17 new tests; **484 pass** repo-wide; ruff clean on changed files.
- [x] **§7.1** — clamp hook (E2) done: `src/probes/context_fatigue/attention_clamp.py`
      (`SpanAttentionClamp`, `solve_span_scale`, `span_share`, `measure_span_share`). Biases the
      additive mask on the span's key columns, so the span's *odds* scale by exactly `e^b` and
      softmax renormalizes; no attention forward is reimplemented and nothing materializes an N×N
      matrix (satisfies §5's "must not retain attention matrices"). scale=1.0 is exactly bit-
      identical, context-manager exit removes hooks even when the body raises, and a `None`
      attention_mask raises loudly rather than silently no-opping. Solver hits all six E2a levels
      from s0=0.2517 to within ~1e-4 (scale 1.277 → 0.30, 0.060 → 0.02). 28 tests, all green.
      **Deviation:** placed in its own module, not appended to `attention_capture.py` as §3 lists —
      the stronger reading of §3's own "keep capture and intervention separate".
- [x] **§7.2** — `src/probes/context_fatigue/context_assembly.py` done: `assemble_transcript`,
      `OverflowGuard`, `select_competitors`, `ArmSpec`, plus `SkippedItem`/`AssembledTranscript`/
      `AssemblyReport` on `BaseSchema`. 35 tests green.
      **Design call:** every arm gives the evidence its **own user turn, `local` included**, so
      `local` is distance 0 rather than the brief's literal "inline at the current query". If
      `local` inlined while `back_k` split the evidence out, the arms would differ in turn
      structure *as well as* distance — two variables, which is exactly what E1 exists to avoid.
      Turn count and total text are now byte-matched across the whole ladder (pinned by
      `test_turn_count_is_matched_across_distances`).
      `OverflowGuard.report()` gives §9 its per-arm skip counts; `fits()` charges prompt +
      `max_new` + headroom, and records `n_tokens` vs `budget` so the log says *why*.
      Insufficient depth raises rather than clamping — a clamped distance would mislabel the arm.
      **Bug found and fixed while testing:** placing multi-block `split` evidence deepest-first
      is wrong. Each later, shallower insertion adds a user turn between the question and every
      block already placed, so a requested `(4, 12)` came out as `(4, 13)`. Insertion is now
      shallow-to-deep, with a parametrized regression test over 5 depth orderings.
- [x] **§7.3** — `src/probes/context_fatigue/dilution_analysis.py` done: `joint_fit`,
      `arm_accuracy_gap`, `final_bin_regression`, `Coefficient` (BaseSchema). 12 tests green.
      Linear probability model, not logistic: the paper quotes accuracy differences in percentage
      points, and OLS on a 0/1 outcome is already on that scale. Bootstrap resamples **cases**
      per §5. Tested in both directions — recovers a planted distance effect, recovers a planted
      fill effect, and finds nothing in pure noise (a fit that invented a fill effect would fake
      E1's headline). Planted +0.10 arm gap recovered with CI containing 0.10 and excluding 0,
      seed-stable across runs.
      **Published number reproduced exactly:** −0.141 [−0.249, −0.031], n=91, at the real
      N_BOOT=10000.

      **Provenance hazard found — FIXED 2026-08-19 at the driver.** That number comes *only* from
      `results/random_context_topbin/turns_pooled.csv`. `final_bin_stats()` called with no path
      falls back to `results/random_context/turns.csv`, which yields **−0.187 [−0.373, −0.001],
      n=31** — a different headline number from the same function call, decided by which files
      happen to be present. `analyze_null_statistics.py:37` does
      `TURNS = POOLED_TURNS if POOLED_TURNS.exists() else None`, so a fresh clone missing the
      pooled artifact would regenerate NULL_STATISTICS.md with the n=31 number and no indication
      the provenance changed. `final_bin_regression()` always names the artifact it read, and a
      test asserts that name. **Resolved:** `analyze_null_statistics.py` now passes `POOLED_TURNS`
      explicitly on all three calls and raises `FileNotFoundError` naming the missing file, so a
      clone without the pooled artifact stops instead of silently describing a different stream.
      The library default is left alone — `tests/probes/context_fatigue/test_null_statistics.py`
      pins its values, and the hazard was always the *driver* choosing a file by accident.
- [ ] **§7.4** — GPU drivers, each behind a one-cell preflight.
  - [x] **E1 distance sweep — CONFIRMED.** `scripts/context_fatigue/run_distance_sweep.py`,
        report `results/context_fatigue/E1_DISTANCE_SWEEP.md`, artifacts in
        `results/context_fatigue/e1_distance_sweep/`. n=192/arm, 0 overflow skips, mean fill
        0.688 identical across arms. local 0.464 → back_2 0.359 → back_5 0.292 → back_10 0.250 →
        back_20 0.276 (chance 0.200). Every gap's CI excludes zero; joint fit gives distance
        β=−0.0076 [−0.0117, −0.0035] significant and fill β=−0.0073 [−0.210, +0.187] not.
        `local` flat with fill. Effect saturates by k≈10 rather than deepening to k=20 — reported.
        Survives parsed-only; unparsed rate is a *fill* effect, not a distance effect.
  - [x] **E2a mass clamp — headroom claim NOT supported.** Report `E2A_MASS_CLAMP.md`, artifacts
        `e2a_mass_clamp/`. Natural cold-start share 0.258 @L24; plateau to 0.20 (no cost), then
        0.15 costs 16.4 pts [+0.036, +0.291]. Levels below 0.15 are reached only at −4.7 to −6.1
        nats (median scale 0.009 → 0.002) = near-ablation, excluded from interpretation; the model
        answers "A" 110/110 there. Cliff framing retired; plateau edge is 0.20 and degradation
        starts at the share accumulation reaches, so there is no margin.
  - [x] **E2b dip rescue — the dip did not reproduce.** Report `E2B_DIP_RESCUE.md`, artifacts
        `e2b_dip_rescue/` + `e2b_scoring_control/`. 26 sessions, n=108 top bin: top−rest =
        **−0.006 [−0.105, +0.092]** vs committed **−0.141 [−0.249, −0.031]**. Fill slope mildly
        positive. Rescue arm null (+0.021 [−0.124, +0.165], 2 flips/97) but a weak instrument
        (share restored only 0.092→0.123). **Scoring tested and excluded** as the explanation:
        dual-scored on the same forwards, 0/780 parse failures, forced-choice −0.006 vs
        generation +0.005. Intervals overlap the committed one, so this is non-robustness, not
        refutation — but the paper's one positive result is now in doubt (G3 worse than open).
  - [x] **E1b/E1c/E1d/E1e mechanism quartet — report `E1_MECHANISM.md`.** Verdict: the accuracy
        penalty is a **locality threshold**, not a distance gradient.
        - E1b (`e1_with_attention/`): evidence share falls 0.0408→0.0124 with distance, all gaps
          excluding zero, r=−0.83; question share flat. **Trap:** within-arm, higher evidence share
          predicts *lower* accuracy (−11.2 [−20.0, −2.6], stronger controlling for length) — it
          tracks difficulty, opposite sign to the causal effect.
        - E1c (`e1c_evidence_clamp/`): mass removal at local position reproduces **116%** of the
          distance penalty; local_clamped − back_20 = −0.029 [−0.126, +0.075] indistinguishable.
        - E1d (`e1d_evidence_rescue/`): mass restoration at back_20 recovers only 32%,
          +0.058 [−0.046, +0.161] n.s.; residual +0.121 [+0.017, +0.224] sig. Likely the
          instrument (uniform across-head bias can strip mass but not rebuild its pattern).
        - E1e (`e1e_dissociation/`): share is governed by **tokens not turns** (matched-token arms
          identical at 0.0104/0.0108; share~tokens sig, share~turns n.s.). Accuracy pays −0.156 to
          −0.193 at the first displacement and nothing after: +0.026 n.s. for +1240 tokens,
          +0.010 n.s. for 5→20 turns. **C2: evidence attention cut 64% at no accuracy cost.**
  - [x] **E1f evidence-share knee sweep — NO KNEE; the curve is a shallow gradient.** — `run_evidence_clamp.py --levels 0.036 …
        0.012` at local position, 192 probes. Artifacts `e1f_share_knee/`; folded into
        `E1_MECHANISM.md`. Ran to confirm a knee and **refuted it**: on the common subset present
        at every level (n=131) accuracy falls smoothly 0.473 (natural, share 0.0441) → 0.275
        (share 0.012), every adjacent step ≤0.038 and none excluding zero. The real resolution of
        the E1c/E1e tension is duller and better — both are the same shallow gradient, and E1e's
        single step was too small to detect. natural→0.012 gives +0.198 [+0.084, +0.313] against
        E1c's independently measured +0.207 [+0.103, +0.310] for the same share change: agreement
        to 0.01 across two separately-run experiments, the program's strongest internal check.
  - [x] **E2b exact-item replication — the dip is a single-bin artifact.** My e2b driver shares the
        committed driver's seed formula, so its sessions 0–11 present the **identical 344 questions**
        as `results/random_context/turns.csv`: subject match 344/344, gold agreement 1.000,
        accuracy 0.6221 both, **per-item agreement 1.000**, fill matched to 3e-4, and the dip
        reproduces exactly (−0.1874 [−0.3710, −0.0034]). So the harness is bit-identical and the
        disagreement is sample composition, not method.
        Fine bins locate it: 0.80–0.85 = 0.625 (n=40), **0.85–0.88 = 0.419 (n=31)**,
        0.88–0.93 = 0.703 (n=37). The committed run's max fill is **0.8784**, so its whole top bin
        *is* that trough. Same 12 sessions extended to 0.93 → −0.097 n.s.; 14 fresh sessions →
        **+0.090** (opposite sign); all 26 → +0.005 n.s.
        Ruled out: overflow-guard selection (1/781 skipped) and item length (86 vs 90 median tokens
        across the two deep bins). **G3 closed in the least convenient direction — the explanandum
        does not exist.**
  - [x] **E3 competition sweep — CONFIRMED, and it is a SECOND mechanism.** Report
        `E3_COMPETITION.md`. Paired n=365/arm, 0 gold leaks, 4 overflow skips, 15 starved.
        random 0.512 / disjoint 0.485 / **near_dup 0.427**; random−near_dup = +0.0849
        [+0.0301, +0.1397] sig, survives parsed-only. Shared-options β sig, fill β not — the same
        shape as E1's distance result. Both control arms agree with E1's `local` 0.464, so the
        harness did not drift; context length does not order the arms.
        **Attention addendum (`e3_attention/`) is the important part:** the 8.5-point accuracy gap
        comes with **no change in the evidence's attention mass** (−0.00027 [−0.00088, +0.00035]).
        E1f's dose-response (6.29 accuracy per unit share) predicts 2.0% of the observed effect;
        reproducing it through mass would need a 50× larger share change. So displacement acts
        through mass and competition does not — two mechanisms, which is why needle benchmarks
        (varying both at once) have never separated them.
        Not monotone in confusability: random sits *above* disjoint (n.s.), so the claim is
        "near-duplicate context is costly", not "cost rises with overlap".
        Original brief entry follows —
        driver `run_competition_sweep.py`, artifacts `e3_competition/`. The earlier "park it on
        power" note is superseded: the power objection applies to *graded* steps, and E3's
        near_dup vs random contrast is designed to be large. Paired over shared probes at n=384
        (2× E1) puts the CI half-width near 0.07.
        **Substrate changed from the brief's MMLU to DDXPlus**, on measurement: the MMLU
        instrument does not work (near_dup stem-jaccard 0.106 vs same_subject 0.102, only 16% of
        picks sharing an option), so a null there would mean "no near-duplicates exist", not "no
        competition effect". DDXPlus cases share a 46-pathology option universe and separate the
        arms 5× (0 / 0.75 / 3.65 shared options of 5) at ≤1.1% difference in context tokens.
        All three arms are DDXPlus, so ICL affordance is constant — keeping the brief's MMLU
        `unrelated` arm would have confounded competition with the ICL this paper credits for
        holding accuracy up.
  - [ ] ~~E4 window-position control~~ — **moot**, no dip to attribute
  - [ ] E5 query decodability — **recommend dropping.** Its premise (flat decodability against
        halved mass would show mass ≠ the information currency) is now answered the other way for
        displacement: E1c/E1f establish causally that mass *is* the currency there. And E3's
        attention addendum answers the interesting half of it more directly — competition costs
        accuracy at constant mass, so mass is not the *only* currency. E5 would now be
        confirmatory at best.

## Reporting convention deviation

`E1_MECHANISM.md` covers four runs in one report rather than §9's one-per-experiment, because
E1b–E1e are a single argument. E2a and E2b have their own reports as §9 specifies.

## THE BOX CHANGED — 32 GB, not 96 GB

`nvidia-smi` reports **RTX 5090, 32,607 MiB**. The Paper A task notes describe an RTX PRO 6000
(96 GB); that box is gone. `tasks/lessons.md:162` is exactly this lesson ("A config's capacity
assumption is a box assumption — recheck it when the box changes"). OLMo-2-7B bf16 weights sit at
~14.4 GB, leaving ~18 GB. E2's clamp needs `attn_implementation="eager"` for an additive mask,
which materializes [1, H, N, N] per layer — at 4k context that is ~1 GB transient in bf16, fine
here, but it must be checked before any batched variant.

## E1 bugs found and fixed (all three would have faked or masked the result)

1. **Options were not shuffled.** DDXPlus lists the differential in rank order, so gold was "A" in
   **71.4%** of probes while arms differed in letter bias (local answered "A" 35%, back_10 10%).
2. **`max_new=8` truncated before the letter**, and the unparsed rate tracked distance
   (25% → 48%), inflating the gap in the hypothesis's direction.
3. **26.7% of cases had <5 options**, 286 with a *single* option answerable without the vignette.

First run archived at `results/context_fatigue/e1_distance_sweep_VOID_option_bias/` with a
`VOID.md`. Lesson for `tasks/lessons.md`: when reusing a task from an existing driver, port its
*item construction* (shuffles, filters) as deliberately as its prompt — an MCQ whose gold letter
is 71% "A" looks like a working experiment right up until the arms differ in letter bias.

## Acceptance

Per brief §6 — each experiment states what confirms *and what falsifies* it; a falsifying outcome
is a result and gets reported. Per brief §9 — one standalone report per experiment under
`results/context_fatigue/`, quoting artifact filenames, n per cell, overflow-guard skip counts, and
the explicit verdict.

## E6 format erosion — execution state (2026-08-20)

Driver `run_format_erosion.py`, all arms n=40/depth, --max-new 256, OLMo-2-7B-Instruct.

- **Box changed again** (RTX PRO 4500 Blackwell, 32 GB). `/usr/local/bin/python` gone → venv
  resurrected via uv's standalone CPython 3.12.14 (`ln -sf` at /usr/local/bin/python; zero
  package reinstalls). `HF_HOME=/workspace/.cache/huggingface` (persistent mount) must be set
  explicitly — now in /root/.bashrc. Suite 221 green after repair.
- **mmlu arm (`e6_mmlu/`)**: compliance 0.875 → 0.000 at depth 3 (fill 0.147), never recovers;
  replies adopt bare-letter MCQ style. Enrichment flat-to-rising (1.50→2.14→1.47) → erosion is
  NOT attention-mediated (brief's falsification clause for the attention hypothesis fires).
  2 overflow skips.
- **Grader bug #4, fixed test-first**: leading bare-letter-line replies (`"B\n<prose>"`) fell
  through every answer fallback → scored unparsed+wrong → fabricated below-chance acc 0.075 at
  depth 3. Fixed in `instruction_checks.py` (lead-line fallback; 39 tests green). Re-graded from
  stored replies: acc 0.500/0.525 at depths 3/7, parse 1.00 everywhere. **Corrected story:
  compliance collapses, accuracy completely unharmed** (0.425 → 0.5-0.68 rising).
- **code arm run 3 (`e6_code/`)**: clean null confirmed without truncation — compliance
  0.875→1.000, enrich 1.51→3.07, symptoms 6.0 @depth 15. **Caveat: 32 overflow skips, ALL at
  depth 15 → n=8 there (shortest probes); interpret the ladder to depth 12 (fill 0.78, n=40).**
  Run 2 archived at `e6_code_VOID_truncation/`.
- **gsm8k arm (`e6_gsm8k/`)**: intermediate. Compliance holds ≥0.825 to depth 12, drops 0.600 at
  depth 15 (fill 0.480); failure mode is GSM8K's worked-solution prose style ("most likely
  diagnosis is: **X)** ... Reasoning:"). grounded_fraction 0.62→0.06: even compliant replies fill
  SUPPORTING with explanations, not vignette findings. 0 skips.
- **Reading**: each filler stamps its demonstrated answer shape on the reply; severity orders by
  the shape's applicability to the probe (mmlu >> gsm8k > code), at matched fill. "Unlearnable
  filler" framing retired (user, 2026-08-20): code is homogeneous and learnable but off-domain
  with no applicable answer shape.
- **Recovery arms (upclamp/refresh/both @ deepest depth)**: RUNNING. Key mechanistic test:
  flat enrichment predicts upclamp fails, refresh works.
- Parked pending user decision: filler-span attention addendum (does mmlu route generation-time
  attention to filler answer spans vs code at matched fill).
- Still queued: E1d all-layer re-run; E6_FORMAT_EROSION.md; numbers.md rows; lessons.md entry.
- **Recovery arms (`e6_mmlu_recovery/`) — BOTH restore compliance 0.000 → 1.000 at depth 42.**
  My prediction (flat enrichment → upclamp fails) was WRONG: upclamp (share 0.0196→0.1902) fully
  restores. Attention mass is a causal weight in the format contest (E5 down = lose, E6 up = win)
  even though its natural decline doesn't discriminate eroding from non-eroding arms. Costs:
  upclamp acc 0.675→0.425, both 0.275 (zero-sum mass, E1c arithmetic); refresh acc 0.500, no
  surgery. refresh's measured share (0.0934) is the restated copy (last-occurrence locator).
- **Span-recording infra added** (user approved "check what it looks at"): `locate_turn_spans` +
  `measure_multi_span_shares` in attention_clamp.py (measure_span_share now delegates),
  `--record-spans` → spans.csv in run_format_erosion.py. 44 clamp tests green. Three arms
  rerunning with identical seeds/depths → e6_{mmlu,code,gsm8k}_spans/ (also a free replication).
- Report drafted: `E6_FORMAT_EROSION.md` (recovery section filled; spans section pending).
- **Span addendum + exemplar-close (2026-08-20 evening).** spans re-runs replicate committed
  ladders exactly (max |Δ|=0.000). Answer-span enrichment gap (fa−fq): mmlu +1.28→+0.65,
  gsm8k +0.34→+0.10, code negative — ordering mirrors erosion. BUT closing the letters
  (fa_close, fa_matched) restores NOTHING (0.000 compliant); fq_close 0.132 (≈renormalization);
  rand1_close null. **Generation-time copying refuted; privileged letter-reading is
  epiphenomenal.** Surviving account: mode installed at prefill (task-vector style),
  self-sustaining; system-attention is an override input (upclamp works). My routing prediction
  was wrong — recorded in report. Clamp extended: span lists + scale=0 closure (21 tests).
- **Next decisive experiment: residual probes** (user asked; design agreed in chat): Probe 1
  instruction-presence decodability across depth (predict flat AUC); Probe 2 upcoming-format
  decodability pre-generation (localizes the mode). ~15 min GPU capture + sklearn.
- E1d all-layer RUNNING (b79cn8mr8).
- **Residual probes (2026-08-20 night) — BOTH POSITIVE.** Probe 1: instruction-presence transfer
  AUC = 1.000 at EVERY depth 0→42 (perm nulls ~0.5, p=0.000); layer-mean 0.985→0.791 (signal
  concentrates, peak drifts L1→L14). Three-level dissociation complete: attention flat →
  representation perfect → behavior zero. Probe 2: upcoming-compliance LOO-AUC 0.822 @stack L21
  (p=0.000, >0.8 from L18) within gsm8k matched cells — mode set pre-generation. Geometry:
  cos(mmlu, gsm8k mean-diffs) median +0.746 (+0.747 @L21) — shared axis (caveat: generic
  deep-context component). Dongre-comparable methodology (PCA→LDA, LOO, 200-perm nulls).
  Ops lesson: multi-thread BLAS on thousands of tiny fits thrashed (929 CPU-min, killed);
  single-thread finished in ~15 min.
- **Mode-vector steering RUNNING** @stack L21 (decoder 20): install/erase × real/random-control,
  meandiff vector. Report section stubbed.
- **Mode-vector steering, 3 rounds (2026-08-20 night) — FINAL: sufficient as a direction, not
  necessary as one.** R1 all-positions install invalid (rand identical); single-layer erase
  clean null. R2 install_last α1: compliance 0.875→0.05, acc intact — but α1 rand control
  missing (α3 rand overdosed; LESSON: norm-match controls per dose). erase_multi (11 layers)
  refuted necessity again with clean rand control (and caused 1-char replies, acc 0.447).
  R3 decode-time α1 with matched control: real 0.000 vs rand 0.525 compliant — DIRECTION-
  SPECIFIC install, replies reproduce the demonstrated registers verbatim in a zero-filler
  context; collateral acc cost 0.175 vs 0.375 (meandiff carries deep-context junk). α3 overdose.
  decode_time mode added to AdditionSteeringHook (tested; 61 safety tests green).
- Report E6_FORMAT_EROSION.md complete through steering. Owed items CLOSED 2026-08-21: paper
  restructured to three mechanisms (new \subsection{Precedent} folding E5+E6; title, abstract,
  intro, contributions, background instructions-vs-demonstrations paragraph, conclusion,
  limitations all updated), E1d "Mass restoration" paragraph now quotes the all-layer numbers
  (0.28 [0.07, 0.61] recovered, share-matched all-layer), numbers.md gained a §3.4 precedent
  section (E5+E6 rows) and the E1d all-layer rows, Anil et al. many-shot-jailbreaking entered
  related_work_citations.md and the bibliography. lessons.md entries committed in 704aaac.
  NOTE: no LaTeX toolchain on this box - the tex passed static checks (env/brace balance, all
  17 cites defined and used, refs resolve) but has NOT been recompiled; rebuild before
  submission. \workshoptitle is still the placeholder (user's call).
- **Round 4 erase (probe's own direction, 2026-08-20 late) — NULL in both contexts, maximally
  clean.** probe2_direction_L21.npy (LDA backprojected; export via analyze_format_probes.py
  --export-direction). mmlu d42 transfer: 0.000/0.684 identical to natural and rand, zero
  collateral. gsm8k d15 in-distribution: erase 0.575 vs rand 0.600 vs natural 0.600 — null.
  Four erasure strategies failed with clean controls. FINAL E6 mechanism claim: mode is
  readable (0.822), writable direction-specifically (0.000 vs 0.525), NOT erasable by any
  estimated linear direction → redundantly stored. Report complete. Driver gained --erase-only
  / --erase-context {mmlu,gsm8k}.
- **Iterative re-probing (CPU) corrects the redundancy story:** L21 linear code is rank-≈2
  (0.822 → 0.619 after one projection → 0.505 chance after two), NOT redundant. So the live
  erase null = behavior doesn't consume the linearly decodable component — the probe reads a
  SHADOW of the mode, not its carrier (nonlinear / other layers / re-derived). Second correlate
  causally retired. Report's final reading rewritten accordingly. Cheap open arm: live erase of
  both L21 directions (L21 linear code to chance in vivo) → airtight "decision doesn't route
  through this layer's linear code".

## E3c competitor closure (2026-08-21) — RESCUE; title claim revised

User-raised: constant evidence mass doesn't rule out the competitor-reading route. Built
`locate_phrase_spans` (test-first) + `--close-arms` on the competition driver; ran paired
n=365 (same panel as committed E3, anchors match: near_dup 0.419/0.427, random 0.512/0.512).
**Closing the context's option-name mentions recovers 59% of the competition penalty**
(+0.055 [+0.006, +0.104]; net vs size-matched rand-closure control +0.060 [+0.008, +0.112],
control itself −0.006 n.s.; residual to random +0.038 n.s.; parsed-only survives, net at
boundary [+0.000, +0.114]). So competition IS attention-mediated — competitor-side. Paper
revised: title now "Starved Evidence, Attended Impostors, Installed Precedent"; §4.3 retitled
+ closure paragraph; abstract/intro/conclusion/limitations updated; §4.5 gains the
one-intervention-two-outcomes contrast (same closure restores nothing in E6). Report
`E3C_COMPETITOR_CLOSE.md`; numbers.md §4.3 rows added. Open: is the ~40% residual prefill
interference or instrument slack (verbatim mentions only); per-layer localization of the route.
