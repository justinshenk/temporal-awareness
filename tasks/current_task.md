# Current task — Multi-hop generality test of the register-vs-procedure thesis (EXECUTION)

**Decision locked (was the open §0 in `multihop_generality.md`):** dataset = **MuSiQue**
(`dgslibisey/MuSiQue` on HF; schema verified: `paragraphs[{title,paragraph_text,is_supporting}]`,
`question_decomposition[{question,answer}]` with `#k` refs, `answer`, `answer_aliases`), framing =
**open-book** (gold *supporting* passages placed in the instruction; LoRA learns multi-hop
*composition* over given facts — the analogue of GSM8K's in-problem numbers — not parametric recall).
This diverges from the brief's earlier "closed-book 2WikiMultiHopQA" default; the code
(`multihop_prompts.py`) already committed to MuSiQue open-book and the user confirmed it. Brief
updated to match.

**Environment:** one RTX PRO 6000 Blackwell (96 GB), CUDA on, datasets 4.4.2 / torch 2.9 / peft 0.19
/ transformers 5.3. Base `NousResearch/Llama-2-7b-hf` (ungated). Heavy phases run here on GPU.

## Goal
Re-run the GSM8K *procedure* apparatus on a second, non-arithmetic multi-step procedure (MuSiQue) and
adjudicate **H_general** (procedure thesis generalizes: full-δ oracle recovers; pointwise-map ladder
≈0; temporal density sharp) vs **H_arith** (it was arithmetic-specific). Either is publishable.

## Reuse map (verified by reading the code)
- `attribution_common.py` centralizes the GSM8K seam: `prompt_token_ids → metamath_prompt(question)`,
  `gsm8k_problems` (HF "gsm8k"), `gsm8k_accuracy → numeric_match`. Parameterize via a **task registry**
  (default GSM8K, unchanged) — add `--task {gsm8k,multihop}` to the 5 drivers.
- Trainer helpers reused from `train_loreft_commonsense.py`: `load_frozen_base`, `collate_left_padded`,
  `linear_warmup_decay`. `encode_examples` is commonsense-specific (calls `commonsense_data.format_*`)
  → write a multihop encode. Prompt↔target join = prompt + `"\n"` + solution (per
  `metamath_fewshot_prompt`). LoRA recipe (LLM-Adapters): r32/α64/dropout .05, {q,k,v,up,down}_proj.
- Seams already done + tested (9 CPU tests): `multihop_prompts.py` (prompt, `resolve_decomposition`,
  `format_multihop_solution`, `extract_pred_answer`, `normalize_answer`, `answer_match`,
  `answer_span_gate`). KEEP `multihop_prompt(q, passages)` signature (test-pinned); refactor it to
  delegate to a single-`{instruction}` template so a one-arg driver prompt exists too.

## Phases (each gates the next; recovery = (acc−base)/(lora−base))
- **P0 — LoRA + gap (go/no-go).** New: `src/probes/attribution/multihop_data.py`,
  `configs/attribution/multihop_llama2.yaml`, `scripts/attribution/train_lora_multihop.py`. Train r32
  LoRA on ~20k answerable MuSiQue (open-book supporting passages). GATE: donor exact-match ≫ base
  closed-book on a ≤500 scan → need ≥~80 base-fail/donor-solve contrast problems. If gap too small,
  STOP and report (itself a finding).
- **P1 — oracle + L\*.** Task-parameterize drivers. all-layers oracle positive control (≈donor);
  single-layer sweep {0,4,…,28,31} → `L*`.
- **P2 — pointwise ladder @L\*.** ridge / MLP / on-policy DAgger recovery vs full-δ oracle.
- **P3 — temporal density @L\*.** periodic(k) knee + structural `answer_only`/`reasoning_only` split.
- **P4 (optional) — plan-vs-execute** (E1b analogue), only if P1–3 land H_general.

## Acceptance / outputs
Per-axis verdict vs the GSM8K numbers, honest n=2-procedures caveat. Writeup
`results/attribution/2026-06-16-multihop-generality.md` + JSONs/figures; update
`results/activation_weight_investigation.md` if the thesis generalizes. CPU tests for every new seam
pass with no network. All seeded (42); contrast set cached `multihop_contrast_set.json`.

## Progress
- [x] Recon: GPU, dataset (MuSiQue verified), trainer/driver contract mapped, decisions locked.
- [x] P0 code (data module, config, trainer) + CPU test + data-pipeline smoke.
- [x] P0 run: LoRA trained (`results/attribution/lora_multihop`), gate PASSED — base 0.000 /
      donor 0.634 on 500-scan, 317 contrast problems cached (`multihop_contrast_set.json`).
- [ ] P1 driver task-registry refactor + oracle + L*.
  - [x] Task registry in `attribution_common.py` (TaskSpec/get_task/task_accuracy/build_contrast_set);
        lockstep driver task-parameterized; 37 CPU tests pass.
  - [x] AC1 validate: all-layers lockstep == donor greedy, 3/3 per-problem match (`.run_logs/p1_validate.log`).
  - [x] Control (all-layers, n-contrast 100): acc=1.000, recovery=+1.000
        (`results/attribution/lockstep_multihop_control.json`) — positive control exact, as GSM8K.
  - [x] All 5 drivers task-parameterized (collect/steer/dagger/temporal-oracle too); shared
        task-aware `load_contrast` moved into `attribution_common`; multihop P3 gates
        (`answer_only`/`reasoning_only` via `answer_span_gate`) wired into the oracle driver;
        multihop config got P2/P3 keys (n_te, sweep, acc/maps/sweep/steer paths);
        `tests/test_attribution_tasks.py` added — 16 tests pass.
  - [x] Single-layer sweep {0,4,…,28,31} DONE (`lockstep_multihop_single.json`, n-contrast 100):
        0/4=+0.000, 8=+0.070, 12/16=+0.020, **20=+0.760**, 24=+0.780, 28=+0.890†, 31=+1.000†
        († degenerate: hook overwrites layer *output*, so L31 = all-layers control; GSM8K's 28/31
        were flagged ~degenerate the same way). **L\* = 20** — same layer, same magnitude as
        GSM8K's 0.75 → oracle axis REPLICATES. NOTE: always pass `--n-eval 500` for multihop so
        the scan aligns with the cached 317 indices (driver default is 60 → would misindex).
- [x] P2 ladder @L20 DONE: ridge R²_te@L20=+0.71 (λ*=3.16e3); ridge steer **+0.26 scan / +0.35
      contrast** (DIVERGES from GSM8K's ≈0.05 — partial linear transport); MLP **+0.00** despite
      better geometry (cos .822/R² .675 vs ridge .636/.270 — GSM8K paradox replicates); DAgger
      joint all-layer **0.00** all rounds (replicates). Oracle still beats all rungs by ≥0.4.
- [x] P3 temporal @L20 DONE (`temporal_oracle_multihop_L20.json`, 20 contrast): periodic_1=0.750
      (=oracle), periodic_2=0.050, k≥4=0.000; step_boundary(7%)=0.050; **reasoning_only=0.750 @
      frac 1.000** (skipping the answer span is free — mirrors GSM8K planning_only). NOTE:
      answer_only VACUOUS (frac 0.000 — unpatched base never emits "The answer is:", gate never
      fires) — not evidence, flagged in writeup.
- [x] Writeup complete (`2026-06-16-multihop-generality.md`): verdict = H_general on oracle +
      temporal axes (exact); ladder PARTIAL (MLP/DAgger replicate at 0, ridge diverges:
      +0.26 scan / +0.35 contrast vs GSM8K ≈0.05 — the wall is lower, not absent).
- [x] Committed on `context-fatigue-datasets`.
- [x] P2b/P3b follow-ups DONE (all four caveat-closers, scan refs base 0.000/donor 0.630):
  - α sweep @L20 (`steer_multihop_alpha_L20.json`): narrow resonance at α=1.0 —
    {0.25:+0.01, 0.5:+0.01, 0.75:+0.02, 1.0:+0.26, 1.25:+0.05, 1.5:0, 2.0:0}.
  - Layer sweep @α=1.0 (`steer_multihop_layers.json`): {8/12/16:0, 20:+0.26, **24:+0.45**,
    28:+0.38, 31:+0.24} — leak humps over the oracle plateau, peaks L24 (GSM8K ≈0 everywhere).
  - Contrast n=100 (`nonlinear_delta_multihop_L20_n100.json`): ridge +0.21 / MLP +0.01 —
    the n=20 read of +0.35 was small-n inflation; scan and contrast now agree ~0.21–0.26.
  - Temporal n=100 (`temporal_oracle_multihop_L20_n100.json`): periodic_2 0.060,
    reasoning_only 0.760 — n=20 reads confirmed.
  - Writeup P2b section + verdict updated; strand-5 entry added to
    `results/activation_weight_investigation.md`.
- [x] **P4 plan-vs-execute DONE** (`gold_token_lens_multihop_L20.json`, n=317 contrast, 19,970
      tokens, LoRA-TF sanity 0.950 — lower than GSM8K's 0.997 *by design*: GSM8K forces the donor's
      own greedy CoT, multihop forces the gold training target). Decisions taken: gold chain only,
      all 317 problems, GSM8K arm re-run as a regression check.
  - Roles built **by construction**, not by search: `src/probes/attribution/chain_token_roles.py`
    renders the chain and its role spans in one pass, and `multihop_data.gold_chain` now delegates
    to it so the supervised target and the lens labels cannot drift. Token roles via fast-tokenizer
    char offsets; teacher-forced ids asserted to round-trip. Span boundaries fall on the space that
    *opens* a role so leading-space tokens (`▁Danny`) can't straddle two roles.
  - Result: execute 0.725 > plan 0.671 (**+0.055 [+0.040,+0.069]**, bootstrap over problems) —
    H_plan in sign; but execute − all = +0.001 (spans 0) vs GSM8K's +0.133, and **nothing
    crystallizes** with depth (GSM8K computed digits went 18→7→0; every multihop role starts near
    rank 0). `final_answer` 0.933, lens rank 0 at every layer = pure copy. hop ≥2 − hop 1 =
    **+0.130** — the *inverse* of the naive composition prediction, because teacher-forcing supplies
    the earlier hops; so this contrast cannot test composition, only that supplying the trajectory
    makes the nominally-hard part easy.
  - GSM8K regression: reproduces the committed `gold_token_lens_L20.json` **exactly** (identical on
    every key), so the `computed_flags` → `chain_token_roles` refactor is behaviour-preserving; a
    20k-random-sequence parity check against the deleted legacy function also passed before removal.
  - Interval machinery extracted to `src/common/bootstrap_stats.py` (`clustered_rate_gap`) and
    re-used by `null_statistics.py` — tokens are dependent within a chain, so the resampling unit is
    the problem. New CPU tests: `tests/test_chain_token_roles.py` (19), `tests/common/test_bootstrap_stats.py` (7),
    plus contrast-interval + role-class-name tests in `tests/test_attribution_tasks.py`.
  - Writeup: P4 section + fourth verdict row + P4 caveats in `2026-06-16-multihop-generality.md`;
    strand-5 entry extended in `results/activation_weight_investigation.md`.
- [ ] **P5 — the missing GSM8K ridge layer probe (IN FLIGHT)**. Brief: `tasks/gsm8k_ridge_layer_probe.md`.
      Trigger: the "GSM8K ridge ≈0.05, ≈0 at every layer" baseline turned out to have **no artifact**
      — per-layer GSM8K ridge steering was only ever run at L0/L1/L14/L16/L31 (smoke, all 0.00) plus
      all-layer joint injections; **L20 and L24 were never probed**, which are exactly the layers
      where multihop leaks. Docs corrected in `600b5f7`; this run closes the measurement gap.
  - Pipeline: `collect_cot_residuals` (running, `.run_logs/p5_gsm8k_collect.log`) →
    `fit_ridge_sweep` → `steer_gsm8k --layers 8,12,16,20,24,28,31 --alphas 1.0 --n-eval 200`.
  - **Output-name hazard**: `steer_gsm8k` always writes `cfg.output.steer_json`
    (`steer_results.json`) regardless of `--layers`/`--alphas` — rename between runs.
  - **References**: let the first steer run measure base/LoRA at max_new=512 and supply those to
    later runs. Do NOT reuse the contrast set's 0.000/0.565 — those were measured at max_new=256.
  - Sanity gate: L20 R²_te should land ≈0.61 (the value the multihop report cites for GSM8K).
  - Either outcome is publishable and must be reported as found: GSM8K ≈0 at L20/L24 confirms the
    divergence; GSM8K also leaking **collapses** it and turns the ladder into a fourth replication.
- [x] Paper updated (`fad12a8`): multihop folded into `register_vs_procedure_abstract.{md,tex,pdf}`
      as finding (G); synthesis sharpened to "trajectory scaffold general, per-step work
      task-specific"; ladder axis stated as pending, with the P5 run named in Next steps. 3→4 pages.
- NOTE: `lockstep_contrast_set.json` for GSM8K was deleted at some point and rebuilt
  deterministically this session (113/200 KEEP, matching the original).

## Side thread — context-fatigue null statistics (DONE, committed)
Interval estimates for the extended abstract's nulls (`null_statistics.py` + analysis script;
outputs in gitignored `results/context_fatigue/NULL_STATISTICS.md` / `.json`, regenerate with
`uv run python scripts/context_fatigue/analyze_null_statistics.py`). Headlines folded into
`context_fatigue_paper/context_fatigue.tex` (rebuilt with tectonic; body 4 pp + refs p5):
flat-accuracy null bounded (declines >7.3/9.2 pts excluded at 95%, coherent/random); Table 2
redone per-case (pooled Δ=+0.045 [−0.003,+0.097] — marginal, stated honestly; sign positive
in every bin/layer); confidently-wrong gap quantified + new Fig. 2 (confidence 0.85→0.95 with
fill, r=+0.72, wrong answers alike, accuracy flat). 7 CPU tests in
`tests/probes/context_fatigue/test_null_statistics.py`.
