# S2 — the register battery — execution brief

**Date:** 2026-08-13 · **Parent spec:** `docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3
**Deadline pressure:** submission Aug 27; this is Paper A §9 and feeds §10. GPU is free.

## 1. Problem statement

Paper A's organizing claim is a two-sided contrast — a **register** is low-rank, on-manifold and
roughly pointwise; a **procedure** is high-rank, off-manifold and time-dense — but only the
procedure side was ever measured. GSM8K and MuSiQue have oracle sweeps, temporal-density knees,
PCA-band cliffs and a five-rung null ladder. No register task has *any* of them: the register half
is inferred from the single fact that a ridge map installs refusal tone. At a venue whose stated
topics are measurement validity and falsifiability, an unmeasured half of a two-sided contrast is
the reviewer's first objection.

S2 measures it: run the **unmodified** procedure drivers against a **register** donor and report
what comes out.

## 2. Agreed solution approach

Register `commonsense` as a third `TaskSpec` in `scripts/attribution/attribution_common.py`. The
seam is four callables (`problems`, `prompt`, `score`, `format_gold`; `lens=None` — there is no P4
here). `src/probes/attribution/commonsense_data.py` already exposes every piece:
`format_prompt` (`"%s\n"`), `extract_answer` (word after `"the correct answer is"`),
`score_predictions`, `load_commonsense_json`. Then run the existing drivers unchanged.

Per the spec, the axes are **split across two registers** because commonsense cannot carry the
temporal one (its supervised span is ~7 tokens, so `periodic:2` and `periodic:4` are nearly the
same intervention — degenerate by construction):

- **commonsense** → oracle layer sweep (`lockstep_patch_gsm8k`) + δ-rank / off-manifold fraction
  (`lockstep_pca_band`).
- **refusal** → temporal density (later, separate run).

This brief covers the **commonsense** arm only.

### Prerequisites (not present on this box — discovered 2026-08-13)

1. `data/commonsense/` did not exist. **DONE** — `download_commonsense_data.py` fetched
   `commonsense_170k.json` (170,420 items) + boolq/piqa/ARC-Challenge test splits. Schema verified
   uniform: `{instruction, input, output: "the correct answer is X", answer: "X"}`.
2. **No commonsense LoRA donor exists.** Only `lora_multihop` is on disk; the `loreft_commonsense/`
   directory holds a similarity JSON, not an adapter. So the donor must be trained:
   `train_lora_commonsense.py` on the pinned recipe (r32/α64/dropout .05, {q,k,v,up,down}_proj,
   lr 3e-4, 3 epochs, 20k subset, seed 42) — the same subset and supervised signal as the LoReFT
   arm, so the existing LoRA-vs-LoReFT comparison stays addressable. ~30–60 min GPU.

### Three judgment calls, stated before the fact

**(a) Primary eval set = ARC-Challenge, not boolq.** The contrast-set protocol keeps
base-fails/donor-solves problems and reads recovery as accuracy on that set. On a **binary** task
(boolq true/false; piqa solution1/solution2) a *meaningless* perturbation that merely garbles the
output still scores ~50% by coin flip, so a partial intervention cannot be distinguished from a
broken one. ARC-Challenge is 4-way (chance 25%) and is the hardest of the three. boolq and piqa are
reported as secondary, with their chance floors stated.

**(b) A scrambled-δ control is mandatory here, unlike GSM8K.** Because a wrong answer is one of
k choices rather than an arbitrary number, every commonsense number needs an empirical floor:
inject the δ **from a different problem** and re-measure. On GSM8K a garbage injection scores ~0
because it cannot produce the right integer by accident; on a 4-way task it scores ~25%. Without
this control, "recovery survives low-rank truncation" is unfalsifiable. This is one extra driver
flag, TDD'd, and it is the difference between a result and an artifact.

**(c) Base scores 0 for a format reason, and §9 must say so.** Llama-2 base will not emit
`"the correct answer is X"` unprompted, so its accuracy under this scoring is ~0 by
*format non-compliance*, not by incapacity — the base model plainly knows some of these answers.
The donor's δ is therefore substantially a **format register**. This does not invalidate S2 (a
register is exactly what we claim installs), but it does mean §9 must state that the commonsense
contrast set measures installation of a *response format plus its answer selection*, and it is the
mirror image of the multihop `answer_only` gate that came out vacuous for the same class of reason.
Reporting it as though base were incapable would be the same error the 08-06 audit caught.

### S2c — the ridge map on base, and the format/answer decomposition (user, 2026-08-13)

Judgment call (c) said base scores ~0 for a *format* reason and that §9 must disclose it. Better:
**measure it instead of disclosing it.** If the claim is that a ridge map installs a register, and
the thing base is missing is largely a format register, then fit the map on the commonsense donor's
δ and steer base with it — the register arm of the ladder that GSM8K (0.03 @L20) and multihop
(0.26 @L20) already have. That axis is currently missing on the register side entirely, which is
the same hole S2 exists to close.

This runs on **existing, already task-parameterized drivers** — `collect_cot_residuals` →
`fit_ridge_sweep` → `steer_gsm8k`, all of which take `--task` — so it needs no new driver code.

The payoff is that a register task splits the outcome in two, where a procedure cannot:

- **format compliance** — does the steered base emit `"the correct answer is …"` at all,
  independent of whether the answer is right;
- **conditional accuracy** — given it complied, is the answer correct, versus the k-way chance floor.

Measured **without touching a driver**, by registering a second spec `commonsense_format` whose
`score` asks only whether the trigger was emitted. Greedy decoding is deterministic and seeded, so
the two runs see identical generations and the pair is an exact decomposition of the same eval.

The three outcomes are all publishable and say different things:

| format | conditional acc | reading |
|---|---|---|
| ~1.0 | ~chance | the map installs the *register* and nothing else — the cleanest possible statement of the paper's thesis, and it retro-explains the procedure leak as a pure register push |
| ~1.0 | ≫ chance | the register carries answer selection too; §10's coordinates must separate disposition from knowledge |
| low | — | the map does not install even a format; the register side is weaker than the refusal result implies, and §4/§9 need rewriting |

It also closes a loop with P5b. The transplant showed multihop-fit maps deliver 75–100% of GSM8K's
native leak — "a task-agnostic late-stack register push". If a **commonsense-fit** map, which has
never seen arithmetic or multi-hop composition, steers GSM8K to that same ~0.12 leak, that
identifies the transported component as register directly rather than by elimination. One extra
cheap steer run against the existing GSM8K contrast set.

## 3. Files likely modified

**New:** `configs/attribution/commonsense_llama2.yaml` (attribution-side config: adapter path,
eval split/n_eval/max_new, output paths, contrast cache) · commonsense cases in
`tests/test_attribution_tasks.py`.

**Edited:** `scripts/attribution/attribution_common.py` — **two registry entries (`commonsense`,
`commonsense_format`) + a `commonsense_problems` loader in `commonsense_data.py`** · the control
injections in `src/probes/attribution/lockstep_oracle.py` plus a `--control` flag in the oracle
driver (smallest possible surface) · `tasks/current_task.md`.

**Scope discovery, 2026-08-13 — `lockstep_pca_band.py` is not task-parameterized.** P1 gave
`--task` to seven drivers; this one was missed because multihop's P3 never used it. It hardcodes
`gsm8k_problems`, `metamath` prompting via a bare `prompt_token_ids`, and
`numeric_match`/`extract_pred_number` scoring, so it cannot see a commonsense problem at all. The
spec's §5 says to stop and re-plan if a driver needs changing, hence this note. The change is the
same mechanical `--task` addition already applied to its seven siblings, not a redesign — but it is
a driver edit, and the δ-rank/off-manifold axis (acceptance criterion 5) cannot run without it.

**Written by runs (gitignored):** `results/attribution/lora_commonsense/`,
`commonsense_contrast_set.json`, `lockstep_commonsense_{control,single}.json`,
`lockstep_pca_band_commonsense_L*.json`, `.run_logs/s2_*.log`.

## 4. Non-goals

- **No driver refactor.** S2 adds a registry entry. If a driver needs changing beyond the one
  scrambled-δ flag, STOP and re-plan (spec §5).
- No retraining of the GSM8K, multihop, or LoReFT donors; no touching committed P0–P5 artifacts.
- No temporal-density run on commonsense — degenerate by construction; that axis is refusal's.
- No third procedure. No paper writing in this brief's scope.
- Not reviving the dead α grid.

## 5. Operational constraints

- Seed 42 throughout. Generation is unbatched (~35–70 tok/s); commonsense answers are ~7 tokens, so
  the eval is cheap (spec estimates < 1 h for the whole battery).
- `--n-eval` must match whatever the contrast cache was built at, or indices misalign (the multihop
  317-index lesson).
- Long runs under `nohup` with a log in `.run_logs/`, **actively monitored to completion**; on
  error, fix and restart immediately.
- One problem end-to-end before launching any full set (`lessons.md:58`).
- Every reported cell names its JSON (`lessons.md:60`).

## 6. Acceptance criteria

1. `commonsense` resolves from the task registry; `gsm8k` and `multihop` behaviour is bit-identical
   (existing tests unchanged and passing).
2. A commonsense LoRA donor exists and clears a gap gate on a ≤500 scan: base ≈0.00, donor ≥~0.60
   (spec cites 0.68), with ≥80 contrast problems. If the gap fails, STOP and report.
3. All-layers lockstep control reproduces the donor per-problem (AC1), as it did for both procedures.
4. Single-layer oracle sweep over {0,4,…,28,31} with an `L*`, reported against GSM8K's 0.75 @L20 and
   multihop's 0.76 @L20.
5. PCA-band recovery curve (δ-rank / off-manifold fraction), reported against GSM8K's cliff
   (top-64 = 55% energy, 0% recovery).
6. Every commonsense cell carries its scrambled-δ floor and its chance level.
7. Verdict reported **as found**. The spec's hypothesis is that recovery survives low-rank
   truncation where GSM8K collapses; a refutation reshapes §9 and §10 and is more interesting, not
   less. It does not get soft-pedalled.
8. **S2c:** a ridge map fit on the commonsense donor, steered onto base, reported as the
   format-compliance / conditional-accuracy pair against the k-way chance floor — the register
   arm of the ladder that both procedures already have.
9. Writeup `results/attribution/2026-08-13-register-battery.md`; `tasks/current_task.md` updated.

## 6b. Measured floors (ARC-Challenge, n=500 scan, verified 2026-08-13 on CPU)

Gold distribution over the 500-problem scan: answer2 144 / answer3 137 / answer1 117 / answer4 102.
So the two floors any commonsense cell must clear are **chance = 0.25** and, more demandingly,
**majority-class = 0.288** — a degenerate policy that always emits `answer2` scores 0.288 while
being perfectly format-compliant. Conditional accuracy is read against 0.288, not 0.25.

## 6c. Run plan (fixed order; each gates the next)

All runs `nohup` + `.run_logs/s2_*.log`, actively monitored. `--max-new 32` everywhere (config
`eval.max_new`); the driver default of 256 would waste ~8× the budget on ~7-token answers.

1. **Gap gate + contrast set** — `lockstep_patch_gsm8k --task commonsense --validate --n-eval 500`.
   Builds and caches `commonsense_contrast_set.json` and prints base/donor on the scan. GATE:
   base ≈0.00, donor ≥~0.60, ≥80 contrast problems. Also runs AC1 (all-layers lockstep reproduces
   the donor per-problem). If the gate fails, STOP and report.
2. **Oracle sweep** — `--mode single --layers 0,4,8,12,16,20,24,28,31 --n-eval 500 --n-contrast 100`.
   Compare `L*` and its magnitude against GSM8K 0.75 @L20 and multihop 0.76 @L20.
3. **Floors at `L*`** — the same command twice more with `--control mean_delta` and
   `--control shuffle_positions`. `mean_delta` is the register hypothesis stated as an
   intervention: if it holds up where the procedures collapsed, that is the result.
4. **PCA band** — `lockstep_pca_band --task commonsense --layer L* --n-pca 100 --n-contrast 100`.
   Against GSM8K's cliff (top-64 = 55% energy, 0% recovery).
5. **S2c ridge arm** — `collect_cot_residuals --task commonsense` → `fit_ridge_sweep` →
   `steer_gsm8k --task commonsense`, then the same steer under `--task commonsense_format` for the
   decomposition. **Rename outputs between runs** (`steer_json` is fixed in the config).
6. **S2c transplant (optional, cheap)** — commonsense-fit maps steered onto the GSM8K contrast set
   at L24, against P5b's native 0.12 and multihop-transplant 0.09.

## 7. Development process — test-forward

1. Write `tests/test_attribution_tasks.py` commonsense cases **first**: registry returns the spec;
   `prompt` round-trips `"%s\n"`; `score` accepts the trigger phrasing and rejects a bare answer
   token; `problems` returns `(instruction, answer)` pairs at the right length with `skip`/`seed`
   honoured; unknown split fails loudly.
2. Then register the spec. No driver may change.
3. Scrambled-δ flag: test that the permutation is seeded, total (no fixed points where avoidable),
   and that it is a no-op when disabled — before wiring it into the driver.
4. All new tests CPU-only, no network (fixtures, not the 96 MB file), seeded.

## 8. Test expectations

- New: ~6 commonsense registry cases, ~3 scrambled-δ cases.
- Unchanged and passing: the 63 in the scoped suite verified this session
  (`test_attribution_tasks`, `test_multihop_{data,prompts}`, `test_chain_token_roles`,
  `common/test_bootstrap_stats`, `probes/context_fatigue/test_null_statistics`,
  `test_commonsense_data`). The count only goes up.
- The three legacy collection errors stay out of scope and must not mask a new one.
