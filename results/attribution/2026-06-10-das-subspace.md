# Does a task-loss subspace recover where variance can't? (DAS vs PCA, L20 oracle)

**Question.** The PCA-band result showed capability is invisible to variance (top-64 = 55% energy,
0% recovery; recovery needs a wide band, ≥512 dirs). The salvage hypothesis: the causal directions
are **low-rank but low-variance** — findable by a *task-loss* subspace search where variance fails.
DAS tests it. Learn an orthonormal R (d×r) by gradient descent on a behavioral loss (inject
`a + Π_R(δ_true)` at L20, teacher-forced on the base trajectory, CE against LoRA's greedy next-token),
then evaluate with the **identical** closed-loop lockstep oracle used for the PCA bands. Only the
subspace differs (task-loss R vs variance V); the injection and eval are byte-identical.

## Result: task-loss search does NOT beat variance — it's strictly worse, and collapses to 0

| rank r | DAS-R recovery | PCA top-r recovery | DAS teacher-forced CE |
|---:|---:|---:|---:|
| 8 | 0.000 | 0.000 | 2.674 |
| 64 | 0.000 | 0.000 | 0.957 |
| 256 | 0.000 | 0.050 | 0.162 |
| 512 | **0.000** | **0.450** | **0.038** |

R orthonormal to ≤5e-7 at every rank. Figure: `results/figures/das_vs_pca_L20.png`.

**DAS recovers nothing at any rank**, including r=512 where the variance band recovers 0.45 — and
where DAS has driven the teacher-forced CE to **0.038** (near-perfect next-token reproduction on the
training trajectory). The salvage hypothesis is **refuted**: task-loss subspace selection is ≤ variance
at every rank and **decisively below it at r=512** (0.00 vs 0.45 = 9 of 20 problems). The r=256 gap
(0.00 vs 0.05) is a single problem and within noise — the robust claim is *DAS = 0 even where PCA
recovers*, not a large margin at 256.

### Readout parity (verified — the comparison is apples-to-apples)
DAS-R and PCA-top-r are scored by the **same function**: `das_subspace_gsm8k.py` imports
`eval_band` from `lockstep_pca_band.py`, which produced the PCA numbers. Same 20 contrast problems
(`load_contrast(cfg)[:20]`, identical indices), same `max_new`; the only differing argument is the
projection matrix (R vs V[:, :r]). In `lockstep_generate` the next token is `base_logits(S).argmax()`
— the **injected base** free-runs its own context for *both* methods, fed the true δ projected onto
the subspace each step. So this is **not** a lockstep-vs-free-gen artifact; the gap is the subspace.
The teacher-forced/closed-loop distinction is *within* DAS (trained teacher-forced, evaluated
closed-loop) — exactly the exposure bias below.

## Why — the teacher-forced objective is anti-aligned with closed-loop δ-fidelity
Both methods inject `a + Π_subspace(δ_true)` at eval, so recovery is governed by how faithfully the
subspace reconstructs the **true** δ under the model's own rollout. PCA top-512 captures the highest-
variance directions = **80% of δ's energy** by construction — a trajectory-agnostic, robust basis that
reconstructs most of δ. DAS instead minimizes teacher-forced next-token CE, a proxy that:
- selects whatever 512-dim subspace **suffices to predict the next token on the training tokens**
  (a low-entropy, mostly-format target) — *not* the subspace that maximally reconstructs δ;
- **overfits the teacher-forced trajectory** (CE→0.04 on 25k train tokens, 2M params), so the chosen
  directions are idiosyncratic to on-policy context and don't transfer to the closed-loop, held-out
  contrast rollout.

So the CE→0 / recovery→0 dissociation is the exposure-bias wall made explicit: a subspace can nail
teacher-forced next-token and still carry **zero** closed-loop capability — and the variance basis,
blind to the trajectory, generalizes strictly better.

## Where this leaves the program
- **Oracle (true δ, full or PCA top-512):** works (0.45–0.75). The capability is *in* δ's wide
  variance structure.
- **Any learned feed-forward map (ridge, MLP, on-policy DAgger, local refit):** ≈0. (Prior results.)
- **Task-loss subspace (DAS), true δ injected:** 0 at every rank — and ≤ variance. (This result.)
- **Variance subspace (PCA), true δ injected:** up to 0.45 — variance is already near-optimal for
  *subspace selection*; the bottleneck is reconstructing the **value** (true δ) in a wide band.

The wall is **value-fidelity in a wide variance band**, not subspace choice. Subspace search is
ruled out as the salvage. The remaining honest move is the **LoReFT-style value map** — learn the
injected *value* (not just the subspace), trained **on-policy / closed-loop** to defeat the exposure
bias this run exposed. Caveat: on-policy DAgger already failed for the linear map (`2026-06-07-dagger
-refit.md`), so this is the upper-bound test, not an obvious win.

## Reproduce
`scripts/attribution/das_subspace_gsm8k.py --ranks 8,64,256,512 --n-train 100 --epochs 40`;
plot `scripts/attribution/plot_das_subspace.py`. Module `src/probes/attribution/das_subspace.py`
(`OrthoSubspace`, `inject_value`, `subspace_lm_loss`); eval reuses `lockstep_pca_band.eval_band`.
JSON: `results/attribution/das_subspace_L20.json`. Tests: `tests/test_das_subspace.py` (5 passing).
