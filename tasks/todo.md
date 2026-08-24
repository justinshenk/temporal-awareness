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
- [ ] Wave 2 last_1/last_2/last_4 RUNNING overnight (artifacts land in
      results/context_fatigue/qwen_e7_bisect_pos_last_*; commit tomorrow)
- [ ] Wave 3-4: layer blocks 0-6/7-13/14-20/21-27 (all positions)
- [ ] If role/recency marginals stay small: add 'template' complement mode to
      --patch-positions and test the glue tokens directly
- [ ] Winner: crossed cell + --random-control size-matched
- [ ] OLMo follow-up: 1-2 targeted cells only, n≈100 (~2 GPU-h each) once Qwen localizes

## Review
Q1-Q7 queue complete 2026-08-24, all reports + artifacts pushed. E7 Stage 1 both
families pushed. Verification loop with OLMo box caught two report errors (lessons
captured). Bisection is the open thread.
