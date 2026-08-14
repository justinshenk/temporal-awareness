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
- [x] **P5 — the missing GSM8K ridge layer probe — DONE 2026-08-10** (updated brief:
      `tasks/p5_execution_brief.md`; report `results/attribution/2026-08-10-gsm8k-ridge-layer-probe.md`).
      Preflight 146/146 (§8 scope — the brief's full-`tests/` command shows ~49 unrelated stale
      failures; scoped suite is the contract). Collect 24 min (token-identical to the aborted 08-06
      run; old partial accumulators deleted first, 20 G). Fit 3 min: **L20 R²_te = 0.610** @λ*
      3.16e3 (gate ≥0.367 passed; the "≈0.61" the audit flagged unsourced was *right*). Decisive
      sweep 160 min, n=200, α=1.0, refs base 0.000 / LoRA 0.650 (outside the stale 0.36–0.46 gate —
      max_new=512 protocol; recorded). **Result: GSM8K leaks late** — L8/12 0.00 [0,.03], L16 0.05,
      L20 0.03 [.01,.08], **L24 0.12 [.07,.19]**, L28 0.13, L31 0.12 — vs multihop 0.45 [.35,.56]
      @L24 (disjoint CIs). Verdict: **shape replication, task-dependent amplitude (~3.75× @peak)**;
      "every GSM8K ridge number is 0.00" (600b5f7) refuted; divergence survives as degree. MLP rung:
      **0.00 [0,.17]** n=20 with cos .806/R² .651 vs ridge .631/.330 — paradox replicates; power at
      n=100 + L24 before the abstract clause is final. Docs updated (multihop report P2/P2b/verdict/
      caveats, strand 5); `steer_results_layers.json` renamed per output hazard; papers/ untouched
      per brief. Analysis artifact: claude.ai/code/artifact/70d09622-6746-4067-b517-0c35f3019bff
- [ ] **P5b — sharpening follow-ups** (register-share hypothesis):
  - [x] **Cross-task transplant DONE — the leak is task-agnostic.** Multihop-fit maps on GSM8K
        (`steer_transplant_multihop_maps_on_gsm8k.json`, n=200, refs supplied): L20 0.01, L24
        **0.09 [.05,.16]** (native 0.12), L28 **0.13** (= native exactly, 17/200 both). Foreign map
        delivers 75–100% of the native leak → the task-fitted component transports ≤0.03 (noise);
        the leak is a task-agnostic late-stack register push. Lead finding for the paper.
  - [x] **MLP @L24 n=100 DONE — paradox is an inversion with disjoint CIs**
        (`nonlinear_delta_gsm8k_L24_n100.json`): MLP cos .815/R² .675 → recovery **0.00 [0,.04]**;
        ridge cos .656/R² .354 → **0.10 [.05,.18]**. Better fit ⇒ *less* transport. Joint with the
        transplant: the transportable component is the *unconditional* one. Abstract MLP clause now
        powered at the live layer.
  - [ ] α grid @L24/28 **DIED, NO ARTIFACT — deferred** (checked 2026-08-13: process gone, GPU
        idle, log's last write 2026-08-10 17:48). Reached 9 of 12 cells and stopped before
        L28 α∈{1.25,1.5,2.0}; `steer_gsm8k` writes its JSON only at the end, so
        `steer_results.json` was never created and the 9 measured cells exist **only** in
        `.run_logs/p5b_alpha_grid.log` — unciteable under the provenance rule. Log reads (n=200,
        refs base 0.000 / LoRA 0.650): L24 {.25:0.000, .5:0.005, **.75:0.095 (+0.15)**, 1.25:0.000,
        1.5:0.000, 2.0:0.000}; L28 {.25:0.000, .5:0.000, .75:0.040 (+0.06)}. Note the peak sits at
        **α=0.75, not the α=1.0 of the headline sweep** (which read L24 +0.12) — so a re-run would
        likely *raise* GSM8K's leak, narrowing the multihop gap further. Resuming means re-running
        all 12 cells (~4–5 h); not an acceptance criterion, so it yields to S2. If revived: rename
        the output on completion.
  - [ ] Transcript check of leak cells (native/transplant L28 both 17/200 — same problems?);
        mean-δ fixed-vector control (needs small driver flag, TDD).
- [ ] **S2 — the register battery (IN PROGRESS 2026-08-13)**. Brief: `tasks/s2_execution_brief.md`;
      parent spec `docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3. Closes the
      paper's substantive hole: the register half of a two-sided contrast was never measured.
  - [x] Prereqs (both were missing on this box): `data/commonsense/` downloaded (170,420 train +
        boolq/piqa/ARC-Challenge test, schema verified); **no commonsense LoRA donor existed** —
        trained here, r32/α64, 20k subset, 3 ep, seed 42, final CE 0.0074.
  - [x] **Donor trained TWICE — the first save was destroyed by a disk quota.** Run 1
        (`.run_logs/s2_train_donor.log`) completed all 3 epochs (CE 0.0074) and then died mid-save:
        `adapter_model.safetensors` truncated at exactly 192 MiB (vs 224,395,264 B for 56,098,816
        fp32 params) and `adapter_config.json` never written. Detected because a small PNG write
        failed with `Errno 122` at the same moment. **My watcher could not have caught it**: it used
        `pgrep -f "train_lora_commonsense"`, a pattern matching its OWN command line, so it would
        never exit — replaced with a PID-based watch (`kill -0 <pid>`). Freed 48 G by deleting
        `accumulators/` + `accumulators_multihop/` (50 G; they feed only `fit_ridge_sweep` and
        `--project-k` steering — both `maps/` trees intact, multihop frozen by spec). 53G → 4.7G.
        Run 2 (`.run_logs/s2_train_donor2.log`) verified complete: 224,438,280 B, config present,
        r=32/α=64/{q,k,v,up,down}_proj.
  - [x] Seams committed (`62fda47`, 93 CPU tests): `commonsense` + `commonsense_format` TaskSpecs
        (the pair decomposes recovery into format installation vs answer selection — exact, since
        both share problems/prompt by identity and decoding is greedy); `commonsense_problems` in
        `commonsense_data.py`; `control_injection` (mean_delta / shuffle_positions) + `--control`
        on the oracle driver, giving a k-way task the empirical floor a procedure never needed;
        `configs/attribution/commonsense_llama2.yaml`.
  - [x] `lockstep_pca_band` task-parameterized — it was the ONE driver P1 missed (multihop's P3
        never used it). GSM8K path behaviour-preserving by construction (same loader, same score
        composition, same filename); user approved the driver edit.
  - [x] Floors measured on the ARC-Challenge scan: chance 0.25, **majority-class 0.288** (gold
        spread 144/137/117/102 over n=500) — conditional accuracy is read against 0.288.
  - [x] **Gap gate PASSED 2026-08-13** (`.run_logs/s2_gate.log`, ARC-Challenge scan n=500,
        max_new=32): **base 0.000 / donor 0.676** — the spec predicted 0.68 — with **338
        base-fail/donor-solve** contrast problems (floor was 80), cached to
        `commonsense_contrast_set.json`. AC1 PASS: all-layers lockstep reproduced the donor
        per-problem (3/3). NOTE: always pass `--n-eval 500` for commonsense so the scan aligns
        with the cached 338 indices (driver default 60 would misindex) — same hazard as multihop's 317.
  - [x] **Oracle layer sweep DONE** (`lockstep_commonsense_single.json`, n-contrast 100,
        max_new 32): L0/4/8 = 0.000, L12 = 0.050, **L16 = 0.830**, **L20 = 0.990**, L24 = 0.990,
        L28/31 = 1.000 (degenerate tail as always). **L\* = 20** by the same earliest-plateau rule,
        but the curve is NOT the procedures': onset is earlier and far sharper (L16 0.830 vs
        multihop's 0.02) and the plateau is essentially total (0.990 vs GSM8K 0.75 / multihop 0.76).
        First quantitative separation between register and procedure **on the oracle axis itself** —
        previously the two sides were only compared on the ladder.
  - [x] **Floors DONE, after two rounds of correction.** First round was VOID: the controls averaged
        δ over *all* positions, and with prompt ~15:1 over generation the diverse per-token shifts
        cancel (‖mean δ‖ 29 → 11), so the injection was a **no-op** — decoded generations were
        byte-identical to base. Because the contrast set is base-fails/donor-solves, base scores
        0.000 on it by construction, so a no-op scores 0.000 automatically and "the floor is 0.000
        not 0.25" was a tautology. Fixed (`generated_rows`, threaded `prompt_len`), plus a
        `random_matched` control (random directions at the true per-token norms). Final at L20,
        n=100: oracle **0.990**, `mean_delta` **0.820**, `random_matched` **0.000** (byte-identical
        to base — a random shift at 30–45% of residual norm does not move the model, while the
        mean-δ direction at that magnitude installs format *and* answer).
  - [x] **`mean_delta` is a per-step ORACLE statistic, NOT a fixed vector** (corrected same day).
        `lockstep_generate` re-runs `capture_residuals` every step, so the 0.820 still needs a live
        donor forward per step; it collapses *positional* variation only. The real fixed-vector test
        (`global_register_vector.py`, estimate once / no donor at inference) reads **0.000**
        per-problem AND pooled over 100 disjoint problems, at α ∈ {0.5, 1, 1.5, 2} — and the
        per-problem vector through the *same* additive hook also reads 0.000, so the gap is the
        **injection mechanism**, not the vector (per-problem cosines to pooled: mean .883, min .820).
        **Net: collapsing across positions is nearly free (0.99→0.82); collapsing across TIME
        destroys it (→0.000).** A CAA-style fixed vector installs nothing on this register.
  - [x] **S2c collect + fit DONE 2026-08-14** (brief: `tasks/s2d_execution_brief.md`). The
        2026-08-13 run marked "IN FLIGHT" had **died leaving nothing** — its log stopped at
        `fit=200 held-out=60` before the first `[train] 10/200` line, and neither
        `accumulators_commonsense/` nor `maps_commonsense/` existed. Third silent loss in a week.
        Relaunched with a disk preflight (a `--n-fit 2` smoke writes the *full-size* 25 G tree, so
        it doubles as the quota test that the truncated donor save needed) and a PID watcher.
    - Collect: 1,200 train / 360 held-out CoT tokens, 64 accumulators + meta.json
      (`.run_logs/s2c_collect2.log`).
    - Fit: **L20 R²_te = 0.8934 @ λ*=1.00e2** (`sweep_commonsense.json`), peak 0.9946 at L0.
    - **Why this gate mattered:** the fit uses **1,200** tokens where GSM8K's used **34,893** — same
      4096² map, 29× less data — so a steering null had to be shown not to be an underdetermined
      fit. R²_te 0.89 settles that: the map is not data-starved.
    - **But 0.89 is NOT comparable to GSM8K's 0.610.** `r2_te` divides by the *uncentred* Σ‖δ‖²
      (`gram_accumulator.py:61`), so it credits a map for merely reproducing δ's constant component
      — and commonsense δ is constant-dominated (per-problem means cosine 0.883 to pooled). From the
      measured norms the constant alone explains ≈29²/42² ≈ **0.48**, leaving ≈0.80 conditional.
      Fixed: `GramAccumulator` now streams the first moment `d_sum`, exposes `constant_r2()`, and
      `fit_ridge_sweep` records `r2_const_te` + `r2_te_centred` per layer, where
      `r2_te_centred = (r2_te − r2_const)/(1 − r2_const)` — an identity, so no refit is needed
      (verified in `tests/test_gram_accumulator.py`). **Both commonsense and GSM8K need a re-collect
      to populate it**; until then the cross-task R² comparison stays confounded.
  - [x] **S2c steer DONE 2026-08-14 — and it is the paper's register arm.** First attempt read
        0.000 at every layer and was **VOID**: the generations showed a destroyed model
        (`\end​​​​` repetition at α=1.0, byte-identical to base at α=0.5), caused by the fit-window
        mismatch above. After `--fit-positions all`:
    - **Format compliance** (`steer_commonsense_allpos.json`, n=500, α=0.75, refs base 0.004 /
      donor 1.000): L8 0.010, L12 0.016, L16 0.140, **L20 0.972**, L24 0.998, L28/31 1.000. Against
      GSM8K 0.03/0.12 and MuSiQue 0.26/0.45 on the same instrument, this is the two-sided contrast
      measured on one axis: **a register transports through a fitted pointwise map; a procedure
      does not.**
    - **Answer selection: none.** n=60 contrast, L20, α=0.75: accuracy 0.267 with `answer1` on
      45/60 — and gold `answer1` is 16/60 = 0.267, i.e. accuracy equals the constant's base rate
      exactly (replicated at n=40: 0.200 with gold-`answer1` 8/40).
    - **Magnitude-controlled** (`.run_logs/s2_alpha_selection{,2}.log`, n=60): α 0.75/0.90/1.00 →
      format 0.85/0.85/0.77 with the constant policy throughout; **α=1.24, which magnitude-matches
      the donor (‖δ_donor‖ 45.60 vs ‖W·a‖ 36.88), collapses format to 0.083**; α=1.5 → 0.000. There
      is no α that installs the register *and* selects.
    - **Geometry** (`.run_logs/s2_geometry.log`, n=30, last prompt position): cos(map, donor)
      **+0.788** (sd 0.013) at 62% of the donor's norm; cos(few-shot, donor) 0.380; cos(few-shot,
      map) 0.332. Two different routes into the same surface behaviour. **Confound open:** the
      few-shot prompt is 324 tokens longer, so `δ_few` needs a length-matched no-format preamble
      subtracted before it can be called a format direction.
  - [x] **The contrast set is ~two-thirds FORMAT, not capability** (`.run_logs/s2_format_rescue*.log`,
        n=100). Tests the brief's judgment call (c), which asserted this and never measured it:
        base zero-shot **0.000** (0/100 format) → base 4-shot **0.630** (100/100 format) → donor
        1.000. **Scrambled-label control: also exactly 0.630**, so the exemplars supply format and
        nothing about the task. Consequence: the oracle's 0.990, the L16 0.830 onset and
        `mean_delta`'s 0.820 substantially measure **format installation**, and §9 may not describe
        them as recovering a capability base lacks.
  - [x] **S2d `fixed_vector` DONE — PUSHBACK item 5 only PARTIALLY answered.** Frozen per-problem
        vector through the lockstep path, L20, n=100:
        **0.000** (`lockstep_commonsense_single_fixed_vector_per_problem.json`), generations
        base-like rather than destroyed. But the run's own cosine diagnostic reads **0.544** (min
        0.304, max 0.803, 3,023 steps) between the frozen vector and the live running mean, so
        direction *and* loop both varied — not the single-variable contrast intended. The
        deployable claim is solid (three independent nulls); the narrow "is 0.820 an early-step
        artifact" question still needs the pushback's construction: record `mean_delta`'s
        **final-step** vector, re-inject it from step 1. Side finding: cosine as low as 0.304
        between successive running means means the required shift **rotates within a 7-token
        generation** — measured evidence against "the register is one direction".
  - [ ] **S2d controls — built and tested 2026-08-14, awaiting the GPU** (80 CPU tests pass):
    - `--control fixed_vector` — **the discriminator PUSHBACK item 5 asks for.** `mean_delta`'s
      0.820 is recomputed from a live donor forward every decode step, and the early steps are
      near-oracle *by construction*: at step 1 there are no generated rows so the statistic falls
      back to the whole sequence; at step 2 the "mean" **is** the true δ of the first generated
      token; at step 3 the mean of two. Those tokens are the trigger phrase, i.e. the span that
      decides the score. This mode freezes the donor's whole-trajectory mean and injects it through
      the *identical* delivery path, so only the loop varies. `per_problem` is the direct comparison
      to 0.820; `pooled` is the CAA claim, estimated off a disjoint slice.
    - `--control random_constant` — **the floor `random_matched` never supplied.** `random_matched`
      draws an *independent* direction per position where `mean_delta` injects one *coherent*
      constant; independent draws partly cancel downstream where a coherent shift accumulates. This
      is one random direction at ‖mean generated δ‖, matched by construction.
    - `--control-positions {all,generated,prompt}` — separates re-encoding the prompt (~150 tokens)
      from steering the generation (~7). `mean_delta` is *sub-oracle* at generated positions (norm
      29 vs the true 42) yet scores 0.820, so the prompt is a live suspect. Also bears on PUSHBACK
      item 4's length confound.
    - Harness: every driver JSON now carries the first 8 decoded generations, and sweep JSONs are
      written **after each cell** — the α grid reached 9 of 12 and left no artifact because the
      write was at the end.
  - [ ] Then: PCA band; `shuffle_positions` corrected; then STOP experimenting and assemble the paper.
- [ ] **FUTURE — cross-model replication (deferred 2026-08-14, not for this session).** Answers the
      "n=1 model" objection, which no amount of extra Llama-2 measurement can.
  - **Zero code change required.** The only architectural assumption in the whole apparatus is
      `model.model.layers[i]` (the Llama layout), which holds unchanged for Llama-3.x, Mistral and
      Qwen2/2.5. A second model is **one new YAML + a trained donor**. Every existing config is
      Llama-2-7b.
  - **Recommended: Llama-3.2-3B** (d=3072 × 28 layers). Changes family *and* scale — a stronger
      generality test than a size-matched swap — and the Gram accumulator tree is 12.7 G against
      Llama-2-7b's 25 G (the pole scales as d²). Qwen2.5-3B is cheaper still (2048 × 36 → 7.2 G).
      Mistral-7B-v0.1 is the weakest use of GPU time: same cost as work already done, and it only
      rules out Llama-2 idiosyncrasies.
  - **Budget:** donor ~30 min; gate+contrast ~15 min; oracle sweep ~20 min; floors ~20 min;
      collect/fit/steer ~30 min → **~2 h for the register arm**. A procedure arm (MetaMath donor +
      256–512-token generations) is ~5–6 h more.
  - **Two traps to write into the brief.** (a) **Sweep at relative depth, not absolute layer** —
      L*=20 of 32 is 62.5% depth, i.e. L17–18 on a 28-layer model and L22–23 on a 36-layer one; the
      literal `{0,4,…,28,31}` grid would miss the plateau onset and read as a failed replication.
      (b) **Use base weights, not `-Instruct`** — the contrast-set protocol needs base ≈ 0.00, which
      holds because no base model emits `"the correct answer is X"` unprompted; an instruct
      checkpoint may partially comply and collapse the contrast set.
  - [ ] **PUSHBACK — local review session, 2026-08-13, after pulling `820788f`. Address before
        paper assembly; items 1–3 gate drafting, 4–5 gate specific sentences.**
    1. **S2c is load-bearing for §1, not a caveat-closer.** With fixed vectors at 0.000 per-problem
       AND pooled, the register is time-dense at the trajectory level too — on the temporal axis,
       register and procedures now MATCH (fixed vector → 0.000 is the register's analogue of
       periodic:2 → 0.00). The measured separation is the oracle's onset/ceiling only. If the
       commonsense ridge map reads ≈0, the register side of the two-sided contrast rests on refusal
       alone and §1 must be rewritten as graded rather than categorical — not softened.
    2. **`2026-08-13-register-battery.md` has stale blocks that contradict its own corrections.**
       The "two-sided contrast, on one instrument" paragraph still claims "83% of the oracle
       survives having no temporal structure at all" — refuted by the CORRECTION block directly
       above it (`mean_delta` re-captures every decode step; it has temporal structure). It also
       compares a position-ablation (`mean_delta`) to the procedures' time-ablation (`periodic:2`),
       which is not the same axis. And the scope note below it still says the pooled vector is "not
       yet run" — it ran, 0.000. Do ONE consolidation rewrite of this file before drafting from it.
    3. **`numbers.md` has NO rows for the corrected S2 results** (oracle 0.990, `mean_delta` 0.820,
       `random_matched` 0.000, fixed vectors 0.000 per-problem/pooled). Under the provenance rule,
       none of today's headline numbers may enter the tex until rows exist.
    4. **Generation-length confound on the oracle-ceiling comparison.** Commonsense target ~7 tokens
       at max_new=32 vs procedure chains at 256–512: a 0.99-vs-0.75 ceiling gap could partly be
       "fewer tokens to preserve", not register-ness. NOTE the *onset* claim is immune (L16 0.830 vs
       MuSiQue 0.020 is within-task-across-layer, length held constant) — lead with onset, caveat
       the ceiling. First reviewer objection otherwise.
    5. **`mean_delta`'s 0.820 may be a short-target artifact.** It averages over generated positions
       of the CURRENT sequence, so at the first generated steps the "mean" over 1–2 positions ≈ the
       true per-token δ — and those early tokens are exactly the trigger phrase, i.e. the register.
       Cheap discriminator before "collapsing across positions is nearly free" becomes a paper
       sentence: apply the FINAL-trajectory mean at every step (still live-donor), or exclude the
       current position from the running mean. If either still reads ~0.8, the claim stands.
    6. **The GSM8K oracle layer-sweep re-run (the F2 blocker) is missing from the remaining-work
       list above.** It is cheap, on the critical path for F2 and every "same layer" sentence — the
       S2 report itself repeats "the same layer as both procedures", unsourced for GSM8K per
       `numbers.md` — and it shares the box with everything else. Slot it before "STOP
       experimenting"; write to a non-colliding filename per the known output hazard.
    7. **Priority if time runs short:** S2c and the GSM8K sweep are load-bearing; the corrected
       `shuffle_positions` and the PCA band are the first cuts.
- [ ] OLD BRIEF (superseded, kept for context): `tasks/gsm8k_ridge_layer_probe.md`.
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
