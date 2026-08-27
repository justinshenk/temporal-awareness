# Qwen reproduction — execution todo (this box: A100-80GB, fresh, 2026-08-24)

Plan: /root/.claude/plans/checkout-tasks-qwen-reproduction-md-and-polished-adleman.md
Scope per user: OLMo entirely out of scope; Q0 gate waived; run Q1→Q7 here.

## Phase 0 — bootstrap
- [x] Install uv
- [x] `uv sync` (.venv from uv.lock; + `--extra dev` for pytest)
- [x] Download online: Qwen2.5-7B-Instruct, cais/mmlu (all), gsm8k (main), ddxplus test.csv (15G cache)
- [x] `pytest -q`: 669 passed; 9 failed = FileNotFoundError on OLMo artifacts in gitignored
      results/ (absent on fresh clone, expected; OLMo out of scope)
- [x] Offline bf16 load + generation smoke OK (28 layers, GQA 28q/4kv; default Qwen system
      prompt confirmed injected when none supplied)

## Phase 1 — queue (preflight → inspect → full run → monitor)
- [x] Q1 E1 distance sweep: CONFIRMED, fully monotone ladder 0.630→0.469, all paired gaps
      SIG; distance β −0.0061 SIG; local flat with fill; share falls with distance & tracks
      acc. Report: results/context_fatigue/QWEN_E1_DISTANCE_SWEEP.md. Parse rate 1.0.
- [x] Q2 E1c: CONFIRMED, sufficiency 106.7% (residual −0.010 [−0.057,+0.037] ≈ 0).
      Naturals: local 0.0390, back_20 0.0088. Report: QWEN_E1C_EVIDENCE_CLAMP.md.
- [x] Q3 E1f: CONFIRMED, smooth monotone dose-response, no knee, first SIG drop at 0.017,
      slope +10.4 ≈ Q1's +10.7. Report: QWEN_E1F_SHARE_SWEEP.md.
- [x] Q4: penalty not detected (DiD vs OLMo ns per verification pass); attention
      INVERSION established (near_dup raises evidence share); E3c closure null.
      Report: QWEN_E3_COMPETITION.md (verified+corrected by OLMo box).
- [x] Q5: graded compliance collapse, canary-ordered (prefix<suffix<forbid-never),
      accuracy rises. Report: QWEN_E5_SYSTEM_CLAMP.md.
- [x] Q6: applicability ordering + answer-reading signature confirm; erosion staged
      (marker at 3 turns, SUPPORTING partial); gsm8k/code never erode; refresh 1.000,
      upclamp 0.733; arm shares straddle Q5 thresholds. Report: QWEN_E6_FORMAT_EROSION.md.
- [x] Q7: accumulation null confirms (slope ns to full window); adherence canaries
      covered by committed run. Report: QWEN_RANDOM_CONTEXT.md. QUEUE COMPLETE.

Grader watch-item: extract_mcq_answer \b([ABCDE])\b fallback vs Qwen verbosity
(Q1/Q4/Q5/Q7; Q2/Q3 are letter-logit and safe). Validate on real replies each preflight.

## E7 format patch (tasks/format_patch_brief.md, interleaved on this box)
- [x] SpanActivationPatch instrument + 28 tests (commit 05f6030)
- [x] Driver run_format_patch.py Stage 1; preflight caught two real issues:
      rendered-length twin matching (isolated token counts misalign by template merge),
      and self-patch baselines (open-capture vs closed-baseline confound — unrelated
      control moved as much as the counterfactual patch). Commit d9d8beb.
- [ ] E7 mmlu arm (depth 42) RUNNING; then code arm (depth 15, fill-matched)
- [ ] Stage 2 bisection only after Stage-1 read
- NOTE: keep everything local (user 2026-08-24); ssh agent exists but no push for now

## E7 Stage-2 bisection (Qwen code cell, dd_full = -2.488; n=24/cell, A→B only)
- [x] Wave 1 position marginals: assistant_turns -0.287, user_turns +0.165 — neither
      role carries it; suspect distributed or chat-template glue tokens (excluded from
      both role subsets)
- [x] Wave 2 last_1/last_2/last_4 complete (2026-08-25 read): recency positions carry
      ~nothing of dd_full=-2.488 — dd_ab −0.075 / +0.0006 / −0.036. With Wave 1's role
      null, suspicion narrows to distributed/template-glue. Artifacts
      qwen_e7_bisect_pos_last_{1,2,4}. NOTE: E3c' preflight independently found 43% of
      context-body final-position mass on template glue/turn boundaries (OLMo) —
      convergent with the glue suspicion; see per-token program below.
- [x] Wave 3-4 + template + controls COMPLETE (report QWEN_E7_BISECTION.md). LOCALIZED:
      template glue positions carry −1.540 of dd_full −2.488 (62%; size-matched random
      control +0.016); layers 0-6/7-13/14-20/21-27 = −1.16/−1.59/−0.45/−0.11; crossed
      template x 7-13 = −1.051 (42%). Content subsets ~null (waves 1-2). The carrier is
      the transcript skeleton, lower half of the stack — converges with E3c' glue mass
      and E6' closure nulls.
- [x] OLMo follow-up CONFIRMS the channel (e7_bisect_pos_template_olmo, n=100): glue-only
      dd_ab +0.092 / dd_ba −0.090 vs full-patch +0.056 / −0.071, controls −0.011 both —
      OLMo's whole (small, non-inverted) effect rides the template positions. Folded into
      QWEN_E7_BISECTION.md, the paper (mode + Limitations), numbers.md. BISECTION THREAD
      CLOSED; open refinement: periodicity-matched content control.
      NOTE: "no push" instruction superseded 2026-08-25 — user asked to push; branch is
      being pushed after each fold-in commit.

## Review
Q1-Q7 queue complete 2026-08-24, all reports + artifacts pushed. E7 Stage 1 both
families pushed. Verification loop with OLMo box caught two report errors (lessons
captured). Bisection is the open thread.

## Per-token capture program (brief: tasks/per_token_capture_brief.md; EXECUTING 2026-08-25)
Box: A100-80GB, OLMo-2-7B-Instruct cached (Stages 1-3 target the OLMo numbers per brief).
- [x] Stage 0: library done test-first — `stacked_rows`/`mean_attention_row` (capture),
      `select_hot_token_spans`, `SpanAttentionClamp(window=all|prefill|decode)`,
      `PerHeadSpanAttentionClamp`, `solve_per_head_biases` (closed-form),
      `measure_span_share_by_head`, `solve_per_head_pattern` (iterative, all-layer).
      +44 tests; capture/clamp suite 121 green; drivers gained `--store-rows`
      (competition + distance sweeps, npz per probe: fp16 mean row + ids + span meta).
- [x] Stage 1: E3c′ COMPLETE (report E3C_HOT_CLOSE.md, artifacts e3c_hot_close/ incl.
      365 stored rows). Verdict: residual is NOT instrument slack — anchors reproduce
      (penalty +0.088 SIG, verbatim net +0.069 SIG vs committed +0.060); content-hot
      closure (0.087 mass, 11x verbatim's 0.0076) nets +0.003 ns; as-measured hot set =
      0.42 mass on template glue, closing it nets −0.175 SIG with parse 0.85 and
      prose-style replies → glue is load-bearing precedent structure (converges with E7
      Wave 1-2 nulls). Competition penalty follows token CONTENT, not received mass.
      All contrasts survive parsed-only.
- [x] Stage 2: E1d′ COMPLETE (report E1D_HEAD_PATTERN.md, artifacts e1d_head_pattern/).
      Per-head pattern restoration recovers 0.235 [−0.158, 0.588] vs uniform 0.375
      [0.167, 0.654] in-session (anchors committed 0.28); per-head − uniform −0.021 ns
      with faithful pattern install (err 0.004, bias_sd 1.22). Head-uniformity is NOT
      the instrument limitation → within-span token pattern or a second positional
      channel remain.
- [x] Stage 3: E6′ COMPLETE (report E6_CLOSE_WINDOWS.md, artifacts e6_close_windows/).
      Installation test NULL both windows: compliance 0.000 under prefill-only, decode-only,
      all-window fa closure (anchors reproduce: d0 0.875, fq_close 0.10≈0.132) — the mode
      does not route through exemplar-answer reading at all. Sharp positive: the fa
      channel's ACCURACY value is entirely prefill-borne (−0.175 SIG prefill-only, exactly
      0.000 item-for-item decode-only; all-window ≡ prefill-only item-for-item).
- [x] Stage 4: COMPLETE. 160 rows captured (e1_rows/rows, 32 probes x 5 arms, 1 session);
      `make_paper_figures.py --only appendix` now emits token_heatmap.pdf (real
      local-vs-back_10 pair, shares 0.0883 vs 0.0522 annotated from the stored rows; the
      bright periodic stripes are the template glue — the E3c' finding is visible raw).
      Builder registered conditionally so clones without rows still build.

## Artifact recovery / re-runs (from 2026-08-26 full-paper audit — see context_fatigue_paper/AUDIT_2026-08-26.md)

First step for everything below: check the A100 box's gitignored results/context_fatigue/ —
copy + commit whatever still exists; re-run only what is actually lost.

### Tier 1 — headline claims with no committed evidence
- [x] Recover or re-run OLMo E6 erosion program: `e6_code/`, `e6_gsm8k/`, `e6_mmlu/`,
      `e6_mmlu_recovery/`, `e6_exemplar_close/` + report `E6_FORMAT_EROSION.md`
      → RE-RUN 2026-08-27 (originals absent from this box). All headline anchors
      reproduce; enrichment gaps cell-exact. Code fill-0.778 cell settles at 0.900
      (tex says 1.00); skips now 0/0/7 (Appendix E says 2/32) — tex refresh needed,
      divergence list in the new E6_FORMAT_EROSION.md.
      (driver `run_format_erosion.py`; generation-only, ~40 probes x 3 streams x depths).
      Backs §4.4: 0.875→0.000 ladder, applicability ordering, reversal accuracies,
      enrichment ordering, Fig. 3 data, Appendix E skip counts (2 mmlu / 32 code).
      Also settles the code-arm value at fill 0.78 (tex now says 1.00 per numbers.md;
      superseded VOID predecessor is the only committed trace).
- [ ] Recover or re-run OLMo E6 probe captures + steering: `e6_format_probes/` (npz +
      probe_results.json), `e6_mode_steering{,_r2,_r3}/`, `e6_probe_dir_erase_*/`
      (drivers `run_format_probes.py`, `run_format_steering.py`). Backs probe AUC
      1.000 / 0.822, mode-vector cosine, install/erase asymmetry.
- [ ] Write + commit the rank≈2 iterative re-probe script (AUC 0.822→0.619→0.505 claim
      currently has NO committed analysis code, independent of the captures).

### Tier 2 — one arm and the original closure row
- [ ] Recover or re-run OLMo competition originals: `e3_competition/`, `e3_attention/`,
      `e3c_competitor_close/` (driver `run_competition_sweep.py`). The surviving
      `e3c_hot_close/` already reproduces random 0.512 / near_dup 0.425 / verbatim closure,
      so the unbacked pieces are the disjoint arm (0.485), the joint fit, the original
      59% closure row, and the cross-family DiD interval. A disjoint-arm + closure re-run
      on the existing panel suffices if the dirs are lost.
- [ ] Recover per-head CSVs: `e1_heads_all/`, `e3_heads_all/`, `head_structure.json`
      (backs all Appendix F numbers).

### Tier 3 — likely just file copies
- [ ] `results/random_context_topbin/turns_pooled.csv` (behind the n=699/1001 §4.1 nulls).
- [ ] ~~Qwen 32k adherence run `summary.json`/`turns.csv`~~ SKIPPED per user 2026-08-27:
      re-prefilling ~510 turns at up to 30k tokens costs hours for a floor-effect null whose
      claim already rests on the committed per-arm table in INSTRUCTION_ADHERENCE.md.
      Note: no 4k substitute exists — 4k canary numbers are E5-clamp, not accumulation.
- [ ] E5 raw dirs (`e5_neutral/`, `e5_system_clamp/`, profile) behind E5_SYSTEM_CLAMP.md.

### Cheap hygiene (do alongside any re-run)
- [ ] Commit a validation log/script asserting capture match on the real models
      (1.5e-8 OLMo / 0.0 Qwen GQA) — currently traces only to commit message ed9e365.
- [ ] Record the clinical system prompt's token count (tex claims 48) somewhere durable.
- [ ] Make drivers write the seed into summary.json (Qwen summaries record none).
- [ ] Note the two Qwen mmlu depth-42 passes disagree on natural compliance
      (0.0333 e6_mmlu vs 0.0667 e6_mmlu_recovery) — document, no action.
