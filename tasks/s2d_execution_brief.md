# S2d — isolate the injection mechanism, and finish the register arm — execution brief

**Date:** 2026-08-14 · **Parent:** `tasks/s2_execution_brief.md` · **Spec:**
`docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3
**Two lanes in parallel:** GPU relaunches the dead S2c collect; CPU builds the controls that
adjudicate a claim already committed to the repo.

## 1. Problem statement

Three things, in the order they hurt.

**(a) S2c is dead, not in flight.** `tasks/current_task.md:221` and commit `820788f` both record the
commonsense ridge collect as running. It is not: no process, GPU at 2 MiB / 0 %, and
`.run_logs/s2c_collect.log` stops at `fit=200 held-out=60 problems (max_new=32)` — before the first
`[train] 10/200` progress line, so it died inside the first few problems. Neither
`results/attribution/accumulators_commonsense/` nor `maps_commonsense/` exists. This is the third
silent loss in a week (α grid 08-10, donor save 08-13, this), and all three drivers write their
output only at the very end.

**(b) The 0.820-vs-0.000 gap is attributed to "the injection mechanism", but the mechanism is not
isolated — and the two paths are closer than the commit implies.** Traced by reading the code:

- Lockstep capture forwards run with `OverwriteResidualHook.enabled == False`, so during the
  injected forward layer *L*'s natural output **is** `a`. Overwriting with `a + v` is therefore
  exactly *adding v at every position*.
- `AdditionSteeringHook(last_token=False)` adds `v` at every prefill position and at every cached
  decode step.

Same vector, same footprint, same coverage. What actually differs is only: **(i)** lockstep
re-estimates the constant at *every decode step* from a live donor forward on the evolving
trajectory — a closed loop — and **(ii)** at step 1 there are no generated rows, so
`generated_rows` falls back to *all* rows (verified on CPU) and the first injected constant is the
diluted all-position mean. The claim that survives is stronger and more interesting than the one
written down, but it is untested.

**(c) The floor for the constant-vector claim is not the tight one.** `.run_logs/s2_delta_norms.log`
measures per-token ‖δ‖ at L20: **~28–30 on prompt positions, ~41–43 on generated ones**, against a
base residual norm of ~90. So `mean_delta` injects a coherent constant of norm ≈29 *everywhere* —
roughly matched on the prompt, and **weaker than the oracle on the generated positions**, yet it
scores 0.820. Two gaps follow:

1. `random_matched` draws an **independent** direction per position; `mean_delta` is a **coherent**
   constant. Independent noise over ~100 positions partially cancels downstream where a coherent
   shift accumulates, so `random_matched` is not the matched floor for a constant-vector claim. The
   matched floor is *one random direction held constant at norm ≈29*.
2. Nobody has asked whether the 0.820 comes from re-encoding the **prompt** or from steering the
   **generation**. Since `mean_delta` is sub-oracle at generated positions, the prompt is a live
   suspect and the answer changes what the number means.

Provenance side-note: those δ-norm figures are quoted in `2026-08-13-register-battery.md` but exist
only in a log. By this repo's own rule they are an uncitable cell until an artifact carries them.

## 2. Agreed solution approach

### Lane A (GPU, immediate) — relaunch S2c

`collect_cot_residuals --task commonsense` → `fit_ridge_sweep` → `steer_gsm8k --task commonsense`,
then the same steer re-scored under `commonsense_format`. No code change; the drivers are already
task-parameterized.

**Preflight, in this order** (each one is a lesson already paid for):
1. **Disk.** The collect writes 64 `GramAccumulator` state dicts, each three 4096² float64 matrices
   = ~402 MB, so **~26 GB** lands in one burst at the end. `fallocate` a 26 GB file first and delete
   it — a direct test of the quota that truncated the donor save, not an inference from `df`.
2. **One end-to-end pass.** `--n-fit 2 --n-te 1 --out-suffix _smoke`, timed. It exercises the full
   write path at full size, so it is both the smoke test and the disk test.
3. **PID watcher.** `cmd & PID=$!` then poll `kill -0 "$PID"`. Never `pgrep -f` on a string that
   appears in the watcher's own command line.

### Lane B (CPU, TDD) — three control modes and two pieces of harness

All in `src/probes/attribution/lockstep_oracle.py` + flags on `lockstep_patch_gsm8k.py`. No new
driver, no refactor of the existing modes.

| new mode / flag | question it answers |
|---|---|
| `fixed_vector` — inject a **precomputed constant** through the lockstep path | is the 0.820-vs-0.000 gap the closed loop, or the delivery path? If this reads ~0.82 the fixed-vector null is an artifact and "a CAA-style vector installs nothing" is **wrong**. |
| `random_constant` — one random direction, constant norm ‖mean gen δ‖, all positions | the by-construction-matched floor for a coherent constant shift |
| `--control-positions {all,generated,prompt}` | does the 0.820 come from re-encoding the prompt or from steering the generation? |

Plus two pieces of harness that make the recurring failure structural rather than aspirational:

- **Persist generations.** `lockstep_eval` decodes text, scores it, throws it away; both retractions
  this week were caught by bespoke dump scripts written afterwards. Store the first N
  `(prompt_len, generation, gold, ok)` records in every driver's JSON.
- **Write sweep JSON incrementally**, one cell at a time. The α grid reached 9 of 12 cells and left
  no artifact because the write is at the end.

The `fixed_vector` run additionally reports the cosine between the lockstep-time running mean and
the donor-trajectory mean — never measured, and it is the remaining "is it the same vector?" doubt.

## 3. Files likely modified

**Edited:** `src/probes/attribution/lockstep_oracle.py` (three modes + position scoping) ·
`scripts/attribution/lockstep_patch_gsm8k.py` (flags, generation records, incremental write) ·
`tests/test_attribution_tasks.py` or a new `tests/test_lockstep_controls.py` ·
`tasks/current_task.md` (correct the S2c status line) ·
`results/attribution/2026-08-13-register-battery.md` (results + the two corrections).

**Written by runs (gitignored):** `accumulators_commonsense/`, `maps_commonsense/`,
`sweep_commonsense.json`, `steer_commonsense*.json`,
`lockstep_commonsense_single_{fixed_vector,random_constant}*.json`, `.run_logs/s2d_*.log`.

## 4. Non-goals

- No change to `mean_delta`, `shuffle_positions` or `random_matched` semantics — the committed
  numbers must stay reproducible. New modes are additive.
- No change to the GSM8K or multihop paths; their artifacts stay bit-identical.
- No retraining of any donor. No PCA band, no refusal temporal axis in this brief.
- No paper writing.

## 5. Operational constraints

- Seed 42. `--n-eval 500` on every commonsense run or the 338 cached indices misalign.
- `--max-new 32`; the driver default of 256 wastes ~8× on ~7-token answers.
- `steer_gsm8k` always writes `cfg.output.steer_json` — **rename between runs**.
- Every long run under `nohup` with a log in `.run_logs/`, watched by **PID**, monitored to
  completion. Verify the artifact exists and is the expected size before calling anything done.
- Decode and read generations for every new cell before its number is written anywhere.

## 6. Acceptance criteria

1. `accumulators_commonsense/` holds 64 accumulators + `meta.json` at the expected byte size, and
   `maps_commonsense/` holds 32 maps. S2c's ridge R²_te at L20 is reported with its λ*.
2. The steer pair is reported as **format compliance / conditional accuracy** against chance 0.25
   and majority-class 0.288.
3. `fixed_vector` at L20, n=100, is reported with its generations, and the writeup states plainly
   whether the fixed-vector null survives.
4. `random_constant` at L20, n=100, gives `mean_delta`'s 0.820 a matched floor.
5. The `--control-positions` triple (all / generated / prompt) at L20, n=100.
6. Every new cell carries its JSON path and a decoded generation sample in that JSON.
7. Existing 65 scoped CPU tests still pass; new tests added for every new mode.
8. `tasks/current_task.md` no longer claims S2c is in flight.

## 7. Development process — test-forward

1. Tests first for `fixed_vector` (returns `a + α·v` at the requested positions and nothing else),
   `random_constant` (one direction reused across positions; seeded; norm equals the target), and
   `control_positions` (positions outside the scope are left exactly at `a`).
2. Then the modes; then the driver flags.
3. A no-op regression: `mean_delta` / `shuffle_positions` / `random_matched` at
   `--control-positions all` must return exactly what they return today.
4. All new tests CPU-only, seeded, no network.

## 8. Test expectations

- New: ~4 `fixed_vector`, ~3 `random_constant`, ~3 position-scoping, ~2 generation-record, ~2
  incremental-write cases.
- Unchanged and passing: the 65 in the scoped suite (verified 2026-08-14 before any edit).
