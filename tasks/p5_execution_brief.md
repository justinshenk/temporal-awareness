# EXECUTION BRIEF — P5: the GSM8K ridge layer probe (GPU machine)

**For:** an autonomous coding agent on the GPU host. You will not have the conversation that
produced this. Everything you need is here or in the repo.
**Written:** 2026-08-07 · **Branch:** `context-fatigue-datasets` · `git pull` before starting.
**Deadline context:** a paper submits **2026-08-28**; this run gates two of its claims.

---

## 1. Problem statement

The paper claims a *divergence* between two procedures on its "pointwise ladder" axis: on MuSiQue
multi-hop QA a ridge steering map leaks ~¼ of the recovery budget at L20 and ~½ at L24, "where every
GSM8K ridge measurement is 0.00."

Two independent problems with that claim, both found in audits, neither yet fixed by measurement.

**(a) GSM8K was never probed at those layers.** Per-layer GSM8K ridge steering exists only at
L0/L1/L14/L16/L31 (smoke runs, all 0.00) plus all-layer joint injections. **L20 and L24 — the exact
layers where multihop leaks — have no measurement at all.** The cited "≈0.05" has no artifact; the
0.05s in the corpus are the PCA-band oracle and the lesion control, different experiments.

**(b) The existing GSM8K nulls are too underpowered to support the comparison anyway.** Exact
Clopper–Pearson bounds were computed on 2026-08-07 (`results/attribution/null_bounds.json`):

| rung | recovery | 95% interval | n |
|---|---:|---|---:|
| global primal-ridge map | 0.000 | [0.000, **0.217**] | 30 |
| per-context local refit | 0.000 | [0.000, 0.217] | 30 |
| on-policy DAgger | 0.000 | [0.000, 0.231] | 30 |
| DAS (all ranks) | 0.000 | [0.000, 0.168] | 20 |

Multihop's ridge recovers **+0.21–0.26**. GSM8K's null ceiling is **+0.217**. **They overlap.** At
n=30 you would need to have observed 3–4 successes to detect multihop-level leakage, and observing
0/30 does not exclude a true rate that would produce them. So the divergence is *not currently
established*, and this run is what settles it.

At `n_eval=200` a zero gives a ceiling of **0.018 accuracy ≈ 0.034 of budget** — about 6× below
multihop's leak. That resolves it decisively either way.

**(c) A third gap, same class.** The paper's abstract asserts the null holds "and so do MLP,
on-policy DAgger, per-context, and task-loss (DAS) variants." DAgger, local-refit and DAS each have a
committed artifact. **MLP does not** — no GSM8K nonlinear run was ever committed. Step 4 below either
backs that clause or forces its removal.

---

## 2. Agreed solution approach

Rebuild the GSM8K ridge maps (the accumulators are gone) and run the **same per-layer protocol
multihop got**, so the two curves are matched layer-for-layer. Four steps, in order.

```bash
# 0. PREFLIGHT — do not skip, see §5
uv run pytest tests/ -q --ignore=tests/common/test_tree_as_structures_system.py \
  --ignore=tests/inference/test_batch_interventions.py \
  --ignore=tests/intertemporal/test_sample_position_mapping.py

# 1. COLLECT  (~0.5-2 h; the accumulators are 64 x 4096^2 f64 ~ 8.6 GB on-GPU)
uv run python -m scripts.attribution.collect_cot_residuals \
  --config configs/attribution/metamath_llama2_gsm8k.yaml

# 2. FIT  (~10-30 min) -> sweep.json + maps/W_L*.pt
uv run python -m scripts.attribution.fit_ridge_sweep \
  --config configs/attribution/metamath_llama2_gsm8k.yaml

# 3. THE DECISIVE RUN  (~3-5 h) - measures its own base/LoRA refs at max_new=512
uv run python -m scripts.attribution.steer_gsm8k \
  --config configs/attribution/metamath_llama2_gsm8k.yaml \
  --layers 8,12,16,20,24,28,31 --alphas 1.0 --n-eval 200
mv results/attribution/steer_results.json results/attribution/steer_results_layers.json

# 4. THE MISSING MLP RUNG  (~1 h) - needs only maps/W_L20.pt from step 2
uv run python -m scripts.attribution.nonlinear_delta_gsm8k \
  --config configs/attribution/metamath_llama2_gsm8k.yaml \
  --layer 20 --n-contrast 20 \
  --out results/attribution/nonlinear_delta_gsm8k_L20.json
```

**Bound the result.** Do not report a bare `0.00`. Use the module added 2026-08-07:

```python
from src.common.null_intervals import bounded_null_from_rate
bounded_null_from_rate(steer_acc, n_eval, base_acc=..., lora_acc=...).render()
```

It raises on a non-positive budget — if it does, the donor scored at or below base and that arm
carries no recovery claim; report it on the accuracy scale and say why.

**Also retrieve, while you are on this machine.** These are gitignored, exist only here, and a figure
depends on them. Copy them somewhere they can be transferred:
`steer_multihop_alpha_L20.json`, `steer_multihop_layers.json`,
`nonlinear_delta_multihop_L20_n100.json`, `temporal_oracle_multihop_L20_n100.json`.

---

## 3. Files likely modified

**New (commit the JSONs; maps and accumulators are large and gitignored):**
`results/attribution/steer_results_layers.json`, `…/sweep.json`,
`…/nonlinear_delta_gsm8k_L20.json`, `…/maps/W_L*.pt`, `…/accumulators/*`.

**Edited:** `results/attribution/2026-06-16-multihop-generality.md` (P2/P2b tables, the ‡ footnote,
verdict, caveats), `results/activation_weight_investigation.md` (strand 5),
`tasks/current_task.md` (P5 progress).

**Do NOT edit** `register_vs_procedure_abstract.{md,tex}` or anything in `papers/`. Those carry
drafted-but-unapplied rewordings that depend on your result; report your numbers and let the paper
edit happen in one pass. `papers/register_vs_procedure/section6_rescoped_claim.md` documents the
pending wording and both branches of the MLP clause — **read it before writing your report** so your
conclusions land in the vocabulary the paper will use.

---

## 4. Non-goals

- **No new code.** Every driver exists and is task-parameterised. If something seems to need
  writing, stop and re-plan — you have misread the interface.
- **Do not re-run the multihop side.** Its numbers stand; only GSM8K's are missing.
- **Do not retrain the GSM8K LoRA** or rebuild its contrast set.
- **Do not touch the committed P4 gold-token-lens results.**
- **Do not batch `task_accuracy`.** It would cut step 3 from ~5 h to ~30 min, but it sits under all
  five verified drivers and perturbing a validated apparatus before a deadline is a bad trade. The
  time is available.
- Not recovering the lost `steer_results.json` — unrecoverable and superseded by this run.
- **S2 / S5 / the leak diagnostic are NOT in scope.** They are blocked on a `commonsense` TaskSpec
  that does not exist yet (`grep commonsense scripts/attribution/attribution_common.py` → 0). That
  code is being written on the other machine and will arrive by `git pull`.

---

## 5. Operational constraints

- **Calibrate before launching.** Per `tasks/lessons.md:58` — written after a 40-minute failure —
  run **one** problem end-to-end through step 3 and time it before starting the full set. Generation
  is unbatched (`task_accuracy` loops `model.generate` at batch size 1), so throughput is
  bandwidth-bound at ~35–70 tok/s and the estimates above are *estimates*.
- **Output-name hazard.** `steer_gsm8k` always writes `cfg.output.steer_json`
  (`results/attribution/steer_results.json`) regardless of `--layers`/`--alphas`. **Rename between
  runs** or the next invocation silently overwrites this one.
- **Reference hazard.** Let step 3 measure base/LoRA itself at `max_new=512` and feed those to any
  later run. **Do NOT reuse the contrast set's 0.000 / 0.565** — measured at `max_new=256`, a
  different protocol.
- **Off-manifold generations do not emit EOS** and run to the 512 cap, which is why step 3 sits at
  the upper end of its estimate. Expected, not a fault.
- Every phase under `nohup` with a log in `.run_logs/`, **actively monitored to completion**. On
  error: fix and restart immediately. Never leave a job unattended; never assume success without
  reading the output.
- Seeded (42) throughout. `HF_HOME=/workspace/.cache/huggingface/`.

---

## 6. Acceptance criteria

1. `sweep.json` exists with a per-layer λ\* and an R²_te at L20. **Do not gate on ≈0.61** — that
   figure appears in the report with no artifact behind it (same class of error as the 0.05). The
   only committed GSM8K sweep, `sweep_smoke.json`, gives L20 R²_te = **0.367** at λ\* = 3.16e3;
   being a smaller fit that is a floor, and whatever you measure *becomes* the citable number.
   Multihop's L20 is 0.714.
2. A GSM8K per-layer steering curve at α=1.0 over {8,12,16,20,24,28,31}, `n_eval=200`, with
   base/LoRA references measured under the same protocol — **L20 and L24 finally measured**.
3. Every reported rate carries an exact interval and its n. No bare `0.00` anywhere.
4. `nonlinear_delta_gsm8k_L20.json` exists, giving the MLP rung its first artifact.
5. The multihop report's P2b layer table gains a real GSM8K row and the ‡ footnote is replaced by
   the measurement.
6. **Whichever way it lands, it is reported as found.** This is the point of the run:
   - **GSM8K ≈0 at L20/L24 with a tight ceiling** → the divergence is real and the paper's
     "task-dependent core size" claim stands, now properly powered.
   - **GSM8K also leaks at L20/L24** → the divergence **collapses**, the "task-dependent core"
     revision is withdrawn, and the ladder becomes a *fourth replication* rather than a divergence.
     This is the outcome that most changes the paper and it **must not be soft-pedalled**.
   - **A ceiling that still overlaps +0.21** → underpowered again; say so plainly and state what n
     would be needed rather than presenting it as a null.
   - **MLP ≈0** → the abstract's clause is vindicated. **MLP non-zero** → the abstract is wrong and
     must be corrected; say so explicitly in the report.

---

## 7. Development process — test-forward

No new code is expected, so this is a regression contract rather than a TDD cycle:

1. Run the suite **before** starting (step 0 above) and record the pass count.
2. If you find yourself writing code, stop — that is a signal you have misread a driver interface.
   If a genuine defect blocks the run, write the failing test **first**, then fix.
3. Run the suite again before reporting. The count must not drop.

---

## 8. Test expectations

- Baseline as of 2026-08-07 on the CPU machine: **146 passing** across
  `tests/common/`, `tests/test_attribution_tasks.py`, `tests/test_chain_token_roles.py`,
  `tests/probes/context_fatigue/`, `tests/test_multihop_{data,prompts}.py`.
- **Three pre-existing collection errors are unrelated and out of scope** — do not "fix" them and do
  not let them mask a new one: `tests/common/test_tree_as_structures_system.py`
  (missing `core_diversity`), `tests/inference/test_batch_interventions.py`,
  `tests/intertemporal/test_sample_position_mapping.py`. All are stale imports from older
  feature-geometry work.
- No new tests expected. If you add code, it needs tests, CPU-only, no network, seeded.

---

## 9. Report back

Write `results/attribution/2026-08-XX-gsm8k-ridge-layer-probe.md` containing: the per-layer curve
with intervals; the L20 R²_te; the MLP rung; an explicit statement of which §6 branch the divergence
question landed in; and the wall-clock each phase actually took, so the estimates in
`docs/superpowers/specs/2026-08-07-workshop-papers-design.md` can be replaced with measurements.

State plainly anything you could not complete and why. Partial results reported honestly are worth
more than a tidy narrative — the last three commits on this branch exist because prose outran its
artifacts, and this run is the correction.
