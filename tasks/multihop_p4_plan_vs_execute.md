# P4 — plan-vs-execute on MuSiQue (the E1b analogue) — EXECUTION BRIEF

Status: **brief written, awaiting decisions (§0) — no code yet.**
Parent: `tasks/current_task.md` (P0–P3 done, verdict H_general / ladder PARTIAL).

## 1. Problem statement

GSM8K's **E1b** (`results/attribution/2026-06-13-compute-vs-communicate-L20.md`) teacher-forced base
on the LoRA's correct CoT and lensed the *gold* next token, split by token role. Result: base predicts
genuine computed-result digits at **96.8%** TF-acc — *higher* than copied digits (89.5%) — and those
results crystallize with depth (lens rank 18→7→0 over L20–24) while copied digits are rank 0 already
at L20. Verdict: **base's deficit is multi-step trajectory control, not per-step computation.**

Does that hold for a non-arithmetic procedure? On MuSiQue open-book the per-step act is not arithmetic
but **retrieval/composition over given passages**, and the plan is not "which operation next" but
"which sub-question next". P4 asks: given the correct chain, can base *execute* each hop, so that its
0.000 free-generation accuracy is again a planning failure?

P4 also repairs P3's one dead axis. P3's `answer_only` gate was **vacuous** (frac 0.000 — unpatched
base never emits `The answer is:`, so the gate never fired) and `reasoning_only` therefore patched
everything (frac 1.000). Teacher-forcing removes that failure mode by construction: the chain is
supplied, so every token role is present and scoreable.

**H_plan** (E1b replicates): TF-acc(hop answer) ≥ TF-acc(sub-question); base executes hops but cannot
lay the chain. **H_exec** (diverges): TF-acc(hop answer) low even on correct rails — base lacks the
per-hop retrieval/composition itself, and multihop's deficit is not GSM8K's.

Both are publishable; H_exec would be the *second* multihop divergence after the ridge leak, and would
sharpen the P2b story (a lower wall + an execution deficit are a coherent pair).

## 2. Solution approach

### 2.1 Token roles — the seam that has to be invented

GSM8K's role classifier keys on `=` (`gold_token_lens_gsm8k.computed_flags`): a result span opens on a
token containing `=` and runs over space/digit tokens. **MuSiQue's chain has no such delimiter** —
`format_multihop_solution` renders `Step i: <sub-question> <answer>.` with only a space between plan
and answer. Roles must therefore be recovered from the *known* hop strings by character-offset
alignment, not by an online state machine. (This is also why P4 is a teacher-forced lens and **not** a
causal gate: "am I inside the answer span" is not decidable online without look-ahead. See §4.)

Four mutually exclusive roles, assigned per character span then mapped to tokens:

| role | span | GSM8K analogue | prediction under H_plan |
|---|---|---|---|
| `sub_question` | `Step i: ` … up to the answer | planning tokens | LOW |
| `hop_answer` | the hop's answer at the end of each Step line | **computed (result of `=`)** | HIGH |
| `final_answer` | after `The answer is: ` | **copied digit** | HIGHEST, rank 0 @ L20 |
| `scaffold` | `Step`, `i`, `:`, `.`, `\n`, the marker itself | format tokens | high |

A token spanning a role boundary takes the role of its **first** character (Llama tokenizes the
leading space with the following word, so ` Danny` lands wholly inside `hop_answer`).

**Extra split (multihop-specific, no GSM8K counterpart): `hop_answer` by hop index.** Hop 1's answer is
findable from the question alone; hop ≥2's requires the `#k` composition. `TF-acc(hop 1) ≫ TF-acc(hop
≥2)` would localize the deficit to *composition* rather than retrieval — the sharpest possible P4
result, and one GSM8K could not ask.

### 2.2 Worked example (sample 0, a 2-hop problem)

Gold chain appended to the prompt after `### Response: Let's think step by step.` + `"\n"`:

```
Step 1: Who is the performer of Mary's Prayer? Danny Wilson.
Step 2: What record label is Danny Wilson signed to? Virgin Records.
The answer is: Virgin Records
```

Labels emitted by the new classifier:

- `Step 1:` → `scaffold`; ` Who is the performer of Mary's Prayer?` → `sub_question`;
  ` Danny Wilson` → `hop_answer` (hop_index 1); `.` → `scaffold`
- `Step 2:` → `scaffold`; ` What record label is Danny Wilson signed to?` → `sub_question`
  (note `#1` already resolved to `Danny Wilson` by `resolve_decomposition`);
  ` Virgin Records` → `hop_answer` (hop_index 2); `.` → `scaffold`
- `The answer is:` → `scaffold`; ` Virgin Records` → `final_answer`

Scored positions are `t ∈ [prompt_len-1, seq-2]`, gold is `full_ids[t+1]`, and the role recorded for a
position is the role of **its gold token** — identical convention to `gold_token_lens_gsm8k.gold_ranks`.
For this problem that is ~45 rows: ~6 `hop_answer`, ~22 `sub_question`, ~13 `scaffold`, ~4
`final_answer`. Across the contrast set that is a few thousand rows, vs GSM8K's 2784 (n=95 computed) —
so the discriminating class is *larger* here, not smaller.

### 2.3 Driver

Task-parameterize `scripts/attribution/gold_token_lens_gsm8k.py` with `--task {gsm8k,multihop}`,
exactly as P1 did for the other five drivers (filenames kept `_gsm8k` per that precedent). New
`TaskSpec` seam `token_roles(tok, full_ids, gold, chain_start) -> list[role]`; `summarize` groups by
role label generically instead of GSM8K's hardcoded four classes. GSM8K's committed classes
(`all` / `digit` / `computed` / `copied digit`) are recomposed from the labels
(`digit = computed ∪ copied_digit`) so the published table is reproduced exactly — see §7.

Everything else is reused verbatim: `load_base_and_lora`, `load_contrast`, `generate_cot_ids`,
`PerTokenResidualCapture`, `LogitLens`, `gold_ranks`.

## 3. Files likely modified

**New**
- `src/probes/attribution/chain_token_roles.py` — pure string/offset logic (no torch, no `datasets`),
  CPU-testable like `multihop_prompts.py`: `gold_chain_roles(...)`, `anchor_generated_chain(...)`,
  `gsm8k_result_roles(...)` (the ported `=` state machine).
- `tests/test_chain_token_roles.py` — see §8.

**Modified**
- `scripts/attribution/gold_token_lens_gsm8k.py` — `--task`, role-driven `classify`/`summarize`,
  task-aware output path (`gold_token_lens_multihop_L20.json`).
- `scripts/attribution/attribution_common.py` — `TaskSpec.token_roles` (5th seam) + both registrations.
- `results/attribution/2026-06-16-multihop-generality.md` — new P4 section + verdict line.
- `results/activation_weight_investigation.md` — strand-5 entry extended.
- `tasks/current_task.md` — progress.

## 4. Non-goals

- **No causal gate.** A `hop_answer_only` / `plan_only` gated-oracle arm is *not* in scope: with no
  lexical delimiter the span boundary is not online-decidable, so such a gate would either leak
  look-ahead or approximate the boundary — worse evidence than the teacher-forced lens. P3 already
  ran the online-decidable gates (`periodic(k)`, `step_boundary`).
- **No retraining, no new LoRA, no new contrast set.** Reuse `results/attribution/lora_multihop` and
  the cached 317-problem `multihop_contrast_set.json`.
- **No change to the multihop chain format** (`format_multihop_solution`) — adding a delimiter would
  invalidate the trained donor and every P0–P3 number.
- **No change to committed GSM8K numbers.** The refactor is behaviour-preserving for `--task gsm8k`.
- Not touching the context-fatigue paper strand.

## 5. Operational constraints

- One RTX PRO 6000 (96 GB), CUDA; seed 42 from config; `uv run python -m scripts...`.
- Cost is dominated by chain generation, not the forwards (2 teacher-forced forwards per problem are
  ~free). Est. 20–35 min for the multihop arm at n=317; ~5 min if gold chains are used (no generation).
- `load_contrast` reads `n_eval` from the cache, so the P1 `--n-eval 500` hazard does not apply here.
- Logs to `.run_logs/p4_*.log`, monitored to completion; CPU tests must pass with no network.

## 6. Acceptance criteria

1. `tests/test_chain_token_roles.py` passes offline; existing 37 + 16 attribution tests still pass.
2. **GSM8K regression**: `--task gsm8k --layer 20 --n-contrast 20` reproduces the committed
   `gold_token_lens_L20.json` (computed 0.968 / copied 0.895 / all 0.835, LoRA-TF 0.997).
3. Multihop run produces `gold_token_lens_multihop_L20.json` with LoRA-TF sanity ≥ 0.95 (wiring check:
   the donor was trained on this exact format, so near-perfect agreement is expected; a low value means
   the chain join or prompt is wrong, and the run is invalid).
4. Per-role table (n, TF-acc, median final rank, per-layer lens rank at {20,22,…,31}) + the hop-index
   split, with every class non-empty (the anti-vacuity requirement P3 failed).
5. A stated verdict, H_plan vs H_exec, with the GSM8K numbers side by side and the honest caveats of
   §9 written into the report.

## 7. TDD / test-forward order

1. Write `tests/test_chain_token_roles.py` against the *specified* API — red.
2. Implement `chain_token_roles.py` — green.
3. Port GSM8K's `computed_flags` into it and assert equality with the current function on a
   hand-built `= 48` sequence — proves the refactor is behaviour-preserving before touching the driver.
4. Refactor driver + `TaskSpec`; re-run the GSM8K arm; diff against `gold_token_lens_L20.json`.
5. Only then run the multihop arm.

## 8. Test expectations

- 2-hop and 4-hop gold chains → exact per-token role lists; roles **exhaustive and mutually exclusive**
  (every scored position gets exactly one label).
- A hop answer that is a substring of its own sub-question (e.g. answer `Virgin Records` inside
  `What did Virgin Records …`) is anchored at the **line-final** occurrence, not the first.
- A `#k` title literal (the `#9 Dream` case `_resolve_ref` guards) survives labeling unchanged.
- `final_answer` span covers exactly the text after `The answer is: ` to end-of-chain.
- Token-to-role mapping respects the leading-space rule (` Danny` is one token, wholly `hop_answer`).
- Anchoring a *generated* chain returns `None` when a hop answer is absent or out of order (so the
  driver can drop and count it) — only if §0 Q1 selects the generated-chain arm.
- GSM8K parity: `= 48` → `4`,`8` flagged computed; the `The answer is: 48` restatement is not.

## 9. Caveats to state in the report

- Open-book means every hop answer is **verbatim in the prompt**, so `hop_answer` conflates retrieval
  with computation in a way GSM8K's post-`=` digits do not. The hop-index split is the partial control;
  the honest framing is "locate-and-compose", not "compute".
- Teacher-forcing isolates execution from planning by construction: a high `hop_answer` TF-acc shows
  base executes *given correct rails*, never that base could lay them (same caveat as E1b).
- n = 2 procedures; per-token counts are not independent within a problem.

## 0. Open decisions — ANSWERED (2026-08-06)

- **Q1 chain source → gold chain only.** Teacher-force the gold `format_multihop_solution` chain;
  exact offset labels, no anchoring failures, no drop rate. Justified because the donor was *trained*
  on this exact format (unlike GSM8K, whose gold CoT is out-of-format for MetaMath — the reason E1b
  had to use the generated chain). ~5 min run, no generation. Divergence from E1b stated in §9.
- **Q2 GSM8K regression re-run → YES.** Rebuild the GSM8K contrast set (deterministic, file-order, no
  seed) and re-run `--task gsm8k` to prove the refactor reproduces `gold_token_lens_L20.json`.
  Adapter is the HF hub `LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32`; `lockstep_contrast_set.json`
  is gone locally so the 200-problem scan must be rebuilt (~15–20 min).
- **Q3 n → all 317 contrast problems.**

## 10. Outcome (2026-08-06) — all acceptance criteria met

1. **PASS** — `tests/test_chain_token_roles.py` 19 tests offline; `tests/test_attribution_tasks.py`
   grew to 10 and the wider attribution/multihop/context-fatigue CPU suites (60 tests) pass.
   Added `tests/common/test_bootstrap_stats.py` (7) for the clustered-interval helper.
2. **PASS, exactly** — the GSM8K arm reproduces the committed `gold_token_lens_L20.json` on every
   key (computed 0.968 / copied 0.895 / digit 0.906 / all 0.835, LoRA-TF 0.997, all lens-rank
   medians incl. the 18→7→0 crystallization). A 20,000-sequence CPU parity check against the
   legacy `computed_flags` was run before that function was deleted.
3. **PASS** — LoRA-TF sanity **0.950** on 317 problems / 19,970 tokens. Lower than GSM8K's 0.997
   because the chains differ in kind: GSM8K forces the donor's own greedy CoT (TF-acc ≈1 by
   construction), multihop forces the gold training target the donor approximates.
4. **PASS** — full per-role table with the hop-index split; **every class non-empty**, the
   anti-vacuity requirement P3's `answer_only` gate failed.
5. **PASS** — verdict **H_plan, weakly**, with the GSM8K columns alongside and all §9 caveats in
   the report's caveat paragraph.

**Beyond the brief:** the decisive differences are reported as 95% bootstrap intervals resampling
**problems** rather than tokens (`src/common/bootstrap_stats.clustered_rate_gap`, extracted from
`null_statistics.py` and re-used there). Without clustering, a 0.055 gap over 19,970 dependent
tokens would have looked ~3x more certain than it is. Headline: execute − plan
**+0.055 [+0.040, +0.069]**; execute − all +0.001 (spans 0); hop ≥2 − hop 1 +0.130; copy − execute
+0.207.
