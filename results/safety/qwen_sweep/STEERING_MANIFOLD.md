# Why does task-vector steering cliff at high α — overwrite vs off-manifold? (Qwen)

Diagnostics + norm-preserving/projection fixes for the high-α failure of task-vector
steering. Two solid findings, and one important negative about reproducibility.

## 1. Diagnostics — the steer is both off-manifold and (at α≳2) overwriting
Natural last-token activations at each layer → top-k PCA (90% variance, k=21–38).

| layer | ‖a‖ | ‖d‖ | d off-subspace frac | norm ratio α·‖d‖/‖a‖ (α=1/2/4) | off-manifold ratio (α=1/2/4) |
|---|--:|--:|--:|---|---|
| L14 | 62 | 29 | **0.85** | 0.46 / 0.93 / 1.85 | 16.7 / 33 / 67 |
| L21 | 142 | 93 | 0.80 | 0.65 / 1.31 / 2.62 | 9.6 / 19 / 38 |
| L27 | 353 | 248 | 0.83 | 0.70 / 1.41 / 2.82 | 8.3 / 17 / 33 |

- The task vector is **~80–85% off** the natural top-k activation subspace, and even at
  α=1 the steer's off-subspace energy is **8–17× the natural off-manifold scatter** — yet
  α=1 steering is tolerated. So the model absorbs large off-manifold excursions.
- The **norm ratio crosses 1 at α≈2** — exactly where the accuracy cliff appears. So the
  cliff coincides with the steer term reaching the activation's own magnitude
  (**overwrite**), more than with a sudden off-manifold spike.

## 2. Refusal lives in the LOW-variance directions (solid, replicated on 2 splits)
Projecting activations onto the top-90%-variance PCA subspace — *with no steering* (α=0) —
keeps task accuracy (~0.18–0.30) but **drops refusal 1.00 → 0.00**. So refusal is carried
by low-variance directions the top-k PCs discard. Consequences:
- **Projection steering cannot be a safety-preserving fix** — it ablates refusal by
  construction.
- Mirrors the rest of the investigation: safety is a *small/low-energy* feature, not a
  dominant component (cf. the rank-16 LoRA-B null and the partial activation-direction
  ablation).

## 3. Apparent irreproducibility was a BUG, not nondeterminism — RETRACTED
An earlier version of this script reported α=1 steering at ~0.22 (vs 0.65 in
`steering_safety`), which I wrongly attributed to bf16/knife-edge nondeterminism. That was
a hook-stacking bug: the fix-eval loop built `[("plain", AdditionSteeringHook(...)),
("normpreserve", ...), ("projection", ...)]` as a list literal, which **registers all three
steering hooks on the model at once** — so every measurement ran with three steering hooks
stacked. Two checks settle it:
- In-process repeat (`run_steer_repro.py`): base `[0.139, 0.139, 0.139]`, steered α=0.5/1.0
  `[0.65, 0.65, 0.65]` — the forward is **fully deterministic** and steering reproduces 0.65.
- Perturbation-flip test (`run_steer_margin.py`): steering compresses the top-2 option-logit
  margin (frac<1.0: 0.28→0.45) but noise at bf16 scale (ε=0.01) flips only ~1–3% of answers
  — far too few to move accuracy 0.65→0.22. The nondeterminism story is refuted.

The bug is fixed (one hook at a time); the corrected fix table is regenerated.

## Corrected fix table (one hook at a time) — does norm-preserving / projection widen the band?

| condition | task_acc | refusal |
|---|--:|--:|
| base | 0.139 | 1.00 |
| plain α=1 | **0.65** | 0.13 |
| norm-preserve α=1 | **0.65** | 0.03 |
| projection α=1 | 0.25 | 0.00 |
| plain α=2 | 0.225 | 0.00 |
| norm-preserve α=2 | **0.225** | 0.00 |
| plain α=4 | 0.20 | 0.00 |
| norm-preserve α=4 | 0.225 | 0.00 |

- **The cliff is a *directional* overwrite, not a norm effect.** Norm-preserving steering
  gives the SAME collapse as plain at α=2 (0.225 = 0.225). Re-normalizing to ‖a‖ doesn't
  help because at α≳2, `a+αd ≈ αd` in *direction* (d is 80% off-manifold), so the activation
  becomes (a scaled copy of) the task direction and loses the per-case content — magnitude
  isn't the problem, the direction swamping is.
- **Projection steering is a *bad* fix:** it kills the task gain already at α=1 (0.65→0.25),
  because the task signal lives in the off-manifold component it discards (and refusal→0,
  low-variance). So the off-manifold-ness is load-bearing for the task — you can't project
  it away.
- **Minor:** norm-preserving slightly reduces refusal erosion at α=1 (0.03 vs 0.13), but
  doesn't change the cliff.
- **Net:** neither variant widens the usable band; it stays **α≈0.5–1** (perturbation
  regime). Beyond that the steer directionally overwrites the content.

## Bottom line
- **Solid:** transfer is **reproducible** (plain α=1 = 0.65, deterministic in-process); the
  high-α cliff is a **directional overwrite** (norm-preserving doesn't fix it; projection
  makes it worse because the task signal is itself off-manifold); **refusal lives in
  low-variance directions** (top-PCA projection ablates it).
- **Lesson:** never build steering hooks in a list literal — they attach on construction
  and stack. The earlier "irreproducibility" was that self-inflicted bug, now fixed.

## Reproduce
`scripts/safety/run_steering_manifold.py`; `NormPreservingSteeringHook`,
`ProjectionSteeringHook` in `src/probes/safety/steering_hook.py`. JSON:
`steering_manifold.json`.
