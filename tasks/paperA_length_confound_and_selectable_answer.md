# EXECUTION BRIEF — Paper A: the length confound, and a procedure with a selectable answer

Written 2026-08-18. Scope: **Paper A (register vs procedure) only.** No context-fatigue work here —
see `tasks/context_fatigue_dilution_localization.md` for Paper B.

Both experiments are drawn from the task families Wu et al. (2024, ReFT, arXiv 2404.03592) already
ran, so each lands beside published LoRA/LoReFT numbers on the same base family.

---

## 1. Problem statement

### G1 — "register vs procedure" is confounded with "short vs long generation"

Every register arm we have is short-output; every procedure arm is long-output:

| arm | side | `max_new` |
|---|---|---|
| refusal frontier | register | short |
| commonsense (S2) | register | **32** |
| MuSiQue multi-hop | procedure | **256** |
| GSM8K | procedure | **512** |

The paper's central mechanism — recovery requires patching >=94% of decode steps, every <=50%
periodic gate returning 0.00 — is measured only on the two long-output arms. As the evidence
currently stands, **"time-dense trajectory state" and "long generation" are the same variable**, and
a reviewer can read the temporal-density curve as a statement about output length rather than about
trajectory dependence. Nothing in the current design separates them.

The missing cell is **long generation with no procedure**. Wu et al. §4.4 supplies exactly that:
instruction-following on Ultrafeedback, where the target is a sustained disposition (helpfulness,
format, register) over hundreds of tokens with no chain to execute. Their LoReFT reaches an
85.60 Alpaca-Eval win-rate at 0.0039% of parameters, beating full finetuning — so a strong donor is
known to exist.

### G2 — both procedures tested require the answer to be produced, never selected

GSM8K computes its answer; MuSiQue copies its answer out of an in-context passage. In both, the
final token is unreachable without running the chain. **AQuA is the case where it isn't**: five-option
multiple-choice algebra, so elimination can reach the answer without executing the trajectory.

Wu et al. Table 2 (LLaMA-1 7B / 13B), LoReFT minus LoRA:

| dataset | answer form | gap 7B | gap 13B |
|---|---|---|---|
| **AQuA** | **selected (5-way MC)** | **+2.5** | **+5.1** |
| MAWPS | produced (short) | -2.8 | -1.2 |
| SVAMP | produced (short) | -5.3 | -0.4 |
| GSM8K | produced (long CoT) | -11.5 | -9.4 |

The one arithmetic task where an activation-space edit beats LoRA is the one whose answer is
selectable. That is our register/procedure boundary appearing *inside* a single task family, and it
predicts our apparatus should recover materially more on AQuA than on GSM8K.

## 2. Agreed solution approach

### E1 — Instruction-following: long generation, no procedure

Point the existing apparatus at a register whose outputs are as long as GSM8K's.

- **Donor:** LoRA on Ultrafeedback, matching the multihop recipe (r=32, alpha=64, dropout 0.05,
  targets `[q_proj, k_proj, v_proj, up_proj, down_proj]`) so the donor is comparable to the two
  procedure donors rather than to the commonsense one.
- **Ladder:** the same rungs as every other arm — global ridge map, per-layer probe at L20/L24,
  lockstep oracle, and the **temporal gate sweep**, which is the measurement that matters.
- **Prediction (state before running):** the map installs the register despite `max_new` in the
  hundreds, and — decisively — the **temporal gate is flat**: sparse periodic gates recover most of
  the effect, unlike GSM8K/MuSiQue where every <=50% gate collapses to 0.00. If the register
  survives sparse gating at long output length, generation length is not the barrier and trajectory
  dependence is.

**The one genuinely new piece is the scoring seam.** `TaskSpec.score` is `(completion, gold) -> bool`;
Alpaca-Eval is a pairwise LLM-judged win-rate and does not fit that signature. Decide before writing
code — do not let an external judge become a load-bearing dependency:

- **Primary (required, local, checkable):** register adherence scored by rule against the donor's
  format, read **as a function of decode position** so the claim is length-resolved rather than
  pooled. This mirrors the existing `commonsense_format` seam, which already reads a task for format
  compliance instead of correctness.
- **Secondary (optional):** Alpaca-Eval win-rate with a judge, reported as corroboration only. If the
  judge is unavailable the experiment still stands on the primary metric.

### E2 — AQuA: a procedure whose answer is selectable

A drop-in fifth task through the existing `TaskSpec` seams.

- **Donor:** LoRA trained on the AQuA slice of MATH10K, same recipe as above. (The GSM8K arm borrows
  a public rank-16 MetaMath adapter; AQuA has no equivalent, so Phase 0 trains one — the same
  workflow `train_lora_multihop.py` already follows.)
- **Task seams:** `problems` from `deepmind/aqua_rat`; `prompt` reusing the arithmetic CoT template;
  `score` = exact match on the selected option letter; `format_gold` = the letter.
- **Ladder:** contrast set -> collect -> fit -> per-layer steer at **L20 and L24** (the two layers
  where the multi-hop linear leak appears) -> lockstep oracle -> temporal gate.
- **Prediction:** AQuA recovery sits **above GSM8K's ~0 and below the register arms**, with a
  *shallower* temporal gate than GSM8K — because the selected answer is partly reachable without the
  trajectory. If instead AQuA reads ~0 like GSM8K, the boundary tracks the task family rather than
  what the final token requires, and the "selectable answer" reading dies.

## 3. Files likely modified

- `scripts/attribution/attribution_common.py` — add `aqua` (and, for E1, `instruct_register`) to
  `TASKS`. Keep every driver task-agnostic; nothing outside the registry should learn a task name.
- `src/probes/attribution/aqua_data.py` — **new**; loader, prompt, option-letter extraction.
- `src/probes/attribution/instruct_data.py` — **new**; Ultrafeedback loader and the rule-based
  register scorer, position-resolved.
- `scripts/attribution/train_lora_aqua.py`, `scripts/attribution/train_lora_instruct.py` — **new**,
  both modelled on `train_lora_multihop.py`.
- `configs/attribution/aqua_llama2.yaml`, `configs/attribution/instruct_llama2.yaml` — **new**,
  modelled on `multihop_llama2.yaml` (note `accum_device: cpu` — see §5).
- `tests/attribution/` — per §7.
- No changes to `ridge_steering_map.py`, `lockstep_oracle.py`, `temporal_gate.py`. **If either
  experiment needs an edit to those, stop and re-plan** — the whole point is that the apparatus is
  held fixed across tasks.

## 4. Non-goals

- **No Paper B / context-fatigue work.**
- **No new model family.** Llama-2-7B throughout, as with every existing arm.
- **Not** re-running MAWPS/SVAMP. The full four-dataset ladder is attractive but is a separate,
  larger piece of work; these two experiments each answer a specific open question on their own.
- **Not** running LoReFT itself on a procedure. Wu et al. already did (GSM8K 26.0 vs LoRA 37.5);
  our arms are donor-anchored transport, which is a different question.
- **Do not** let an external LLM judge become required for E1's verdict.
- **Do not** touch the refusal, commonsense, GSM8K or MuSiQue artifacts. Their numbers are settled.

## 5. Operational constraints

- GPU box. `results/attribution/` is gitignored; every run writes a JSON artifact there and a
  committed report quotes it. **No number enters the paper without a row in `numbers.md`.**
- **Preflight before any long run** (the standing rule): confirm the driver loads, produces one cell,
  and writes, before committing hours.
- `accum_device: cpu` for both new configs. The multihop config records why: 2 splits x 32 layers x
  3 x 4096^2 f64 ~ 26 GB, which OOMs beside bf16 models on a 32 GB card. E1's long sequences make
  this worse, not better.
- Per-cell writes, so a killed session preserves informative cells (the lesson from the F2 sweep that
  died mid-L28).
- Seeded (seed 42); bootstrap CIs over **problems** as the resampling unit via
  `src/common/bootstrap_stats`.
- Every steered run must record `N_GENERATION_RECORDS` decoded samples. Read the generations before
  believing any verdict — E1 especially, where a rule-based register scorer can be gamed by
  degenerate output that satisfies the rule.
- Both arms need their **floor**: AQuA is 5-way (chance 0.20), so a garbled intervention still scores
  at chance. Run `--control` as the commonsense arm does.

## 6. Acceptance criteria

Each states what confirms it **and what falsifies it**. A falsifying outcome is a result, not a
failure, and goes in the paper.

**E1 (instruction-following).** n >= 200 held-out prompts.
- *Confirms:* the map installs the register at long output length, and the temporal gate is
  materially flatter than GSM8K's — sparse periodic gates retain most of the effect where GSM8K's
  collapse to 0.00. Length is then decoupled from trajectory dependence, and G1 closes.
- *Falsifies:* the register also requires >=94% of decode steps. Then the temporal-density result is
  about **generation length**, not trajectory state, and the paper's central mechanism claim must be
  restated. **This is the outcome most worth knowing and the reason to run E1 first.**

**E2 (AQuA).** n >= 200 eval, contrast set >= 100 base-fail/donor-solve problems.
- *Confirms:* recovery strictly between GSM8K (~0) and the register arms, with a shallower temporal
  gate than GSM8K. The boundary then tracks what the final token requires, not the task family.
- *Falsifies (a):* AQuA reads ~0 like GSM8K -> the selectable-answer reading dies; the boundary is
  coarser than proposed, which is a cleaner and more conservative claim.
- *Falsifies (b):* AQuA recovers as fully as a register -> "procedure" is not one thing at all, and
  multiple-choice framing alone converts a procedure into a register. That would be the most
  interesting result available here and should be pursued, not buried.

Global: `numbers.md` updated with a row per new figure before anything reaches the tex.

## 7. Development process — test-forward

Write the test before the driver. Tests run on CPU with a tiny model
(`hf-internal-testing/tiny-random-*`) or synthetic tensors; no GPU in the test suite.

1. **Task seams first, before any training.** For `aqua`: `problems` returns `(question, gold)` pairs
   of the declared shape; `prompt` is byte-stable across calls; `score` accepts the donor's own
   correct completions and rejects a shuffled-label control; `format_gold` round-trips.
2. **Register scorer (E1), before the driver.** Assert it rejects degenerate output that trivially
   satisfies the rule (empty, repeated token, prompt echo), and that the position-resolved variant
   agrees with the pooled one when position is collapsed.
3. **Registry wiring.** `get_task("aqua")` resolves; an unknown key still raises with the known-task
   list; no driver imports a task module directly.
4. **Analysis before the runs.** Temporal-gate summarisation and the bootstrap on synthetic data with
   a planted effect of known size, recovered with a CI covering it.
5. Only then the GPU drivers, each behind a preflight producing one cell end-to-end.

## 8. Test expectations

- `get_task("aqua")` returns a `TaskSpec` with all four seams non-None; `get_task("nope")` raises
  `KeyError` listing known tasks.
- AQuA `score`: exact-match on option letter; 1.0 on donor-correct completions, ~0.20 on a
  shuffled-label control (5-way chance), 0.0 on empty completions.
- AQuA prompt: byte-identical across two calls with the same question; no trailing-whitespace drift.
- Register scorer: 0.0 on empty / single-repeated-token / prompt-echo outputs; monotone
  position-resolved curve on a synthetic sequence that degrades at a known index.
- Bootstrap: planted +0.10 recovery difference recovered with 95% CI containing 0.10 and excluding 0,
  seed-stable across two runs.
- Config regression: loading both new configs yields `accum_device == "cpu"` and `seed == 42`.

## 9. Order of work today

E2's donor is cheaper and its seams are a genuine drop-in; E1 needs a scoring decision before any
code. Realistic sequencing:

1. **Now — decide E1's primary metric** (§2). One paragraph written into this file before code.
2. **Now — E2 seams + tests** (CPU, no GPU): `aqua_data.py`, the `TASKS` entry, tests from §7.1/§7.3.
3. **Then — E2 Phase 0**: train the AQuA donor, then build the contrast set. Preflight first.
4. **Then — E1 donor training** launched behind E2's, since E2's ladder can run while it trains.
5. **Tomorrow — the ladders**: collect -> fit -> per-layer steer (L20, L24) -> oracle -> temporal
   gate, E2 first.

**Realistically E2 Phase 0 and both seam suites are a today job; the ladders are not.** Do not start a
multi-hour collect without the preflight, and do not report a verdict from a run whose generations
have not been read.

## 10. Report back

One standalone report per experiment under `results/attribution/`, quoting artifact filenames, n per
cell, the control/floor value, decoded generation samples, and the acceptance-or-falsification
verdict from §6 stated explicitly — including when the falsifying branch is the one that fired.
