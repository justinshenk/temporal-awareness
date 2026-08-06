# P5 — the missing GSM8K ridge-steering layer probe — EXECUTION BRIEF

## 1. Problem statement

The multihop generality report's one *divergent* axis (P2/P2b) claims the linear ridge map leaks
~¼ of the recovery budget at L20 and ~½ at L24 on MuSiQue, "where GSM8K is ≈0 at every layer".
Tracing that baseline on 2026-08-06 found **no artifact behind it**:

| GSM8K ridge-steering artifact | coverage | result |
|---|---|---|
| `steer_results_smoke.json` (HEAD) | per-layer **L1, L14, L31**, α ∈ {0.5, 1.0}, n_eval=50 | 0.000 |
| same file @ `3919b6c` | per-layer **L0, L16, L31**, α=1.0, n_eval=12 | 0.000 |
| `short_arithmetic.json`, `local_refit_gsm8k.json`, `dagger_refit_gsm8k.json` | all-layer **joint** injection | 0.000 |
| `steer_results.json` (the config's full-run output) | **never committed, not on disk** | — |

**L20 and L24 — the exact layers where the multihop leak appears — were never probed per-layer on
GSM8K.** The cited "≈0.05" has no artifact at all (the 0.05s in the corpus are the PCA-band oracle
and the lesion control, different experiments). Every GSM8K ridge number really is 0.00, so the
direction of the claim is likely right, but it is an inference from five other layers, and the docs
asserted it as a measurement. Docs corrected in `600b5f7`; this brief closes the measurement gap.

This matters because the divergence is the finding that revises the paper's headline from
"a procedure never installs" to "the procedure *core* does not install; its size is task-dependent."

## 2. Solution approach

Rebuild the GSM8K ridge maps (the accumulators are gone locally) and run the **same** per-layer
steering protocol multihop got, so the two curves are matched layer-for-layer.

1. **Collect** — `collect_cot_residuals`, cfg defaults (n_fit 200 / n_te 60 / max_new 512) →
   per-layer train+held-out `GramAccumulator`s in `results/attribution/accumulators`.
2. **Fit** — `fit_ridge_sweep` → λ\* per layer by held-out R²_te, `maps/W_L*.pt` + `sweep.json`.
   Sanity: L20 R²_te should land ≈0.61 (the value the multihop report cites for GSM8K); a wild
   departure means the refit is not the same object the old claim referenced.
3. **Layer sweep @ α=1.0** — `steer_gsm8k --layers 8,12,16,20,24,28,31 --alphas 1.0 --n-eval 200`.
   This is the matched comparison and the decisive run.
4. **α grid @ the leak layers** — only after step 3, and scoped by what it shows (see §6).

**References:** the first steer run measures base/LoRA itself at max_new=512 and those values are
then supplied verbatim to later runs. Do *not* reuse the contrast-set numbers (base 0.000 /
lora 0.565): those were measured at max_new=256, so they are not the same protocol.

**Output-name hazard:** `steer_gsm8k` always writes `cfg.output.steer_json`
(`results/attribution/steer_results.json`) regardless of `--layers`/`--alphas` — the multihop repro
block carries the same warning. **Rename between runs** or the second overwrites the first.

## 3. Files likely modified

- **New results:** `results/attribution/steer_results_layers.json`, `…/sweep.json`, `maps/W_L*.pt`,
  `accumulators/*` (the last two are large + gitignored; only the JSONs get committed).
- **Edited:** `results/attribution/2026-06-16-multihop-generality.md` (P2/P2b tables + ‡ footnote +
  verdict + caveats), `results/activation_weight_investigation.md` (strand-5), and afterwards
  `register_vs_procedure_abstract.{md,tex}` + rebuilt `.pdf`.
- No source changes expected — this is a measurement gap, not a code defect.

## 4. Non-goals

- **No new code.** Every driver already exists and is task-parameterized; if something needs
  writing, stop and re-plan.
- **Do not re-run the multihop side.** Its numbers stand; only GSM8K's are missing.
- **Do not retrain the GSM8K LoRA** or rebuild its contrast set (rebuilt today, 113/200).
- **Do not touch the committed P4 gold-token-lens results.**
- Not attempting to recover the lost `steer_results.json` — it is unrecoverable and superseded.

## 5. Operational constraints

- One RTX PRO 6000 (96 GB), currently idle. Accumulators are 64 × 4096² f64 ≈ 8.6 GB on-GPU.
- Long: collection is the pole (~hours), then 7 steering cells × 200 problems × 512 max_new.
- Every phase runs under `nohup` with a log in `.run_logs/` and an active `Monitor` — per the
  standing rule that long jobs are watched to completion and failures fixed immediately.
- Seeded (42) throughout; `HF_HOME=/workspace/.cache/huggingface/`.

## 6. Acceptance criteria

1. `sweep.json` exists with a per-layer λ\* and R²_te at L20. **Do not gate on "≈0.61"** — that
   number was in the multihop report but has no artifact either (found 2026-08-06, same class of
   error as the 0.05). The only committed GSM8K sweep, `sweep_smoke.json`, gives L20 R²_te =
   **0.367** at λ\* = 3.16e3; being a smaller fit, that is a floor. The refit should land at or
   above it, and *it* becomes the citable GSM8K number — replacing a figure the report asserted
   without a source. Multihop's L20 is 0.714 (verified in `sweep_multihop.json`).
2. A GSM8K per-layer steering curve at α=1.0 over {8,12,16,20,24,28,31}, n_eval=200, with base/LoRA
   references measured under the same protocol — i.e. **L20 and L24 finally measured**.
3. The multihop P2b layer table gains a real GSM8K row; the ‡ footnote is replaced by the
   measurement or, if the result is unexpected, by the corrected reading.
4. **Whichever way it lands, it is reported as found**:
   - GSM8K ≈0.00 at L20/L24 → the divergence is confirmed and the paper's sharpened claim stands.
   - GSM8K also leaks at L20/L24 → the divergence **collapses**, the "task-dependent core size"
     revision is withdrawn, and the P2 axis becomes a *replication* rather than a divergence.
     This is the outcome that would most change the paper, and it must not be soft-pedalled.

## 7. Test expectations

No new CPU tests (no new code). The existing attribution suites must still pass unchanged.

## 8. Caveats to carry into the writeup

- Cross-run in *time* even after this: the multihop maps were fit on a different day from these,
  though both use the identical driver, seed and protocol.
- GSM8K CoTs are ~250 tokens vs multihop's ~60, so the fit corpora differ in token count
  (GSM8K's is larger) even at equal problem counts — a difference in the maps' training data, not
  in the evaluation protocol.
- α=1.0 is the multihop resonance peak; if GSM8K reads 0.00 there, a null at other α values is
  implied but not measured unless step 4 runs.
