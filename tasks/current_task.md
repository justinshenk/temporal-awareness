# Artifact recovery / re-runs (Tier 1 first) — started 2026-08-27

Source task: `tasks/todo.md` §"Artifact recovery / re-runs" (commit 93f0fe6).
Referenced audit `context_fatigue_paper/AUDIT_2026-08-26.md` is NOT in the repo (never pushed).

## Inventory outcome (this A100 box, 2026-08-27)

Everything named in Tiers 1–2 is **absent** from `results/context_fatigue/` — this box is the
fresh 2026-08-24 clone where OLMo was originally out of scope; only the newer OLMo re-runs
survive (`e1_rows`, `e1d_head_pattern`, `e3c_hot_close`, `e6_close_windows`,
`e7_bisect_pos_template_olmo`). Note: everything on disk under `results/context_fatigue/` is
actually **committed** (677 files) — recovery = re-run + commit, same as those dirs.
Tier 3: `random_context_topbin/` missing (a different binned `results/random_context/` exists —
check if it's the topbin source under another name); `instruction_adherence/` has only the .md;
OLMo `e5_neutral`/`e5_system_clamp` missing.

## Tier 1 re-run protocol (reconstructed)

Driver: `scripts/context_fatigue/run_format_erosion.py`, model default
`allenai/OLMo-2-1124-7B-Instruct` (cached), seed 42, n-probes 40. Depths taken from the
surviving Qwen mirrors' summary.json (Qwen reproduced the OLMo protocol exactly) and confirmed
against `context_fatigue_paper/numbers.md` (depth 42 recovery, code/gsm8k depths to 15):

1. `e6_mmlu`            — `--filler mmlu  --depths 0 3 7 14 21 28 35 42 --record-spans`
2. `e6_gsm8k`           — `--filler gsm8k --depths 0 2 4 6 9 12 15 --record-spans`
3. `e6_code`            — `--filler code  --depths 0 2 4 6 9 12 15 --record-spans`
4. `e6_mmlu_recovery`   — `--filler mmlu  --depths 0 42 --recovery`
5. `e6_exemplar_close`  — `--filler mmlu  --depths 0 42 --close-arms`

`--record-spans` folds the original separate `e6_*_spans/` re-runs (numbers.md line 146: exact
replication, max |Δ| = 0.000) into the main dirs — spans.csv lands next to turns.csv. After
committing, update numbers.md's `e6_*_spans/` citations to point at the merged dirs.

Then: `run_format_probes.py` → `e6_format_probes/`, `run_format_steering.py` →
`e6_mode_steering{,_r2,_r3}/` + `e6_probe_dir_erase_*`, and write the rank≈2 iterative
re-probe script (no committed analysis code for AUC 0.822→0.619→0.505).

Expected anchors (from committed numbers.md / E6_CLOSE_WINDOWS.md — new runs should land near):
- mmlu ladder 0.875 → 0.000 by depth 3 (fill 0.147); depth-0 accuracy 0.425, sys share ~0.19
- code compliant ~0.875→1.000 to depth 12; gsm8k ≥0.825 to depth 12, 0.600 at depth 15
- recovery at 42: natural 0.000/0.675, upclamp 1.000/0.425, refresh 1.000/0.500, both 1.000/0.275
- exemplar close: fa_close 0.000, fa_matched 0.000, fq_close ~0.132, rand1_close 0.000
- skip counts: 2 mmlu / 32 code (Appendix E)
- code arm at fill 0.78: settles 1.00-vs-0.78-row discrepancy (VOID predecessor committed)

## Progress
- [x] preflight e6_mmlu (depth0 1.000 compliant, collapse at depth 3 — signature present)
- [x] e6_mmlu full (~14 min): ladder 0.875 → 0.000 at all depths ≥3, fill 0.147 at d3;
      d0 acc 0.475 (anchor 0.425), d42 natural 0.000/0.650/share 0.0195 = E6' anchors
      exactly. spans.csv written. DIVERGENCE: 0 overflow skips vs Appendix E's "2 mmlu"
      — flag in report, paper skip count may need updating.
- [x] e6_gsm8k full (28 min): ≥0.825 through depth 12, 0.600 at depth 15 (fill 0.497 vs
      anchor 0.480) — anchors reproduce. 0 skips. spans.csv written.
- [x] e6_code full (35 min): high-compliance-throughout confirms (0.825 d0 → 0.975–1.000
      mid → 0.900 at d12/fill 0.778 → 0.939 at d15/fill 0.942). DIVERGENCES for paper
      update: fill-0.778 cell = 0.900 (tex says 1.00); skips 7 (Appendix E says 32).
      Qualitative claim (code never erodes) intact. spans.csv written.
- [x] e6_mmlu_recovery (38 min): natural 0.000/0.650 (anchor 0.675), upclamp 1.000/0.425
      EXACT, refresh 1.000/0.450 (anchor 0.500), both 1.000/0.275 EXACT. Install/erase
      asymmetry line intact.
- [x] e6_exemplar_close (22 min): fa_close/fa_matched/rand1_close 0.000 EXACT, fq_close
      0.100 (committed 0.132; E6' anchor run also measured 0.100).
- [x] enrichment gaps from spans.csv: mmlu +1.28→+0.65, gsm8k +0.34→+0.07,
      code −0.08→−0.17 — cell-exact vs numbers.md line 162.
- [x] E6_FORMAT_EROSION.md written (with divergence list for tex refresh), committed
- [x] e6_format_probes capture + analysis: Probe 1 AUC 1.000 all depths p=0.000
      (cell-exact), layer-mean 0.985→0.789 (anchor 0.791); Probe 2 LOO-AUC 0.875 L22
      (anchor 0.822 — labels from regenerated gsm8k stream); geometry median cos +0.744.
      Committed+pushed 3b75110 incl. 220MB npz captures per audit list.
- [x] analyze_probe_rank.py written+committed; L21 iterative removal 0.868→0.622→0.492
      reproduces rank≈2 (anchor 0.822→0.619→0.505).
- [x] steering r1 (install a1 null, 1-layer erase null), r2 (11-layer erase null,
      install_last_a3 partial 0.225), r3 (decode install real 0.000 vs rand 0.425 —
      anchor 0.525; acc 0.325 vs anchor 0.175), probe-dir erase mmlu 0.000/0.000 and
      gsm8k 0.500/0.600 — all four erase strategies null with clean controls.
      TIER 1 COMPLETE.
- [x] Tier 2 COMPLETE (commit 261464c): disjoint 0.485 exact, shares byte-level,
      closure 56% (comp_close 0.474 = hot_close), drains 0.0455/0.0022 exact, DiD ns.
- [x] Tier 3 random_context_topbin REBUILT (commit 3435cdf): nulls hold, tighter bounds.
- [x] Tier 3 E5 dirs RE-RUN: profile/neutral/main all cell-exact vs E5_SYSTEM_CLAMP.md.
- [x] Hygiene: seeds in summaries, 48-token row, depth-42 note, capture validation
      (measured 1.1e-06 / 3.1e-05 fp32 — historical 1.5e-8/0.0 not reproducible on
      the 2026-08 stack; Appendix B should cite measured bounds).
- [x] 32k adherence: SKIPPED per user.

## TASK COMPLETE 2026-08-27 — remaining work is the tex/numbers refresh from the
## divergence lists (E6_FORMAT_EROSION.md, E3_RECOVERY_RERUN.md, NULL_STATISTICS.md),
## which touches the split repo ~/Documents/breaking-down-context-rot on the user's side.
