# ICL ≈ Finetuning on the task, but the safety cost is weight-specific and context-fragile

A thirteen-experiment investigation on `google/gemma-2-9b-it`, DDXPlus medical MCQ as the finetuning
task, AdvBench (`mlabonne/harmful_behaviors`) + Alpaca for the refusal probe. All code under
`src/probes/{lora_icl,safety}`, `scripts/{lora_icl,safety}`; reports under `results/{lora_icl,safety}`.
Adapter + shift tensors are gitignored and regenerate deterministically (seed 42).

## One-paragraph thesis
In-context learning and LoRA finetuning push the residual stream into the **same late-layer subspace**
*on the task they adapt to* (cos ≈ 0.8). But finetuning carries a **side-effect ICL does not**: it
erodes refusal. That safety cost is **weight-specific** (ICL of the same content leaves the refusal
axis untouched), lives in a **specific, low-rank, input-gated direction** distinct from the task
adaptation, is **causally removable** by ablating that direction, and — most importantly — is a
**latent context-fragility**: the finetuned model's refusal collapses as context fills (0.80→0.10)
while the base model's refusal is context-robust (flat ≈1.0). The safety cost of finetuning is not a
fixed number you can read off a short-context eval; it is a fragility that long context detonates.

## The arc

| # | Experiment | Result |
|---|------------|--------|
| 1 | **Subspace convergence** (`lora_icl`) | ICL shift vs LoRA shift on DDXPlus: cos rises from ~0 early to **0.81 @ L35**; mean-centered subspace overlap ~0.67 late. ICL and finetuning converge in the back half of the stack — but only on task inputs. |
| 2 | **Safety: weights vs activations** | Refusal: base 0.98, base+medical-ICL **0.98**, DDXPlus-LoRA **0.84**. The LoRA shift points along −r (toward compliance); the ICL shift is off-axis (cos(ICL,r)≈0) and orthogonal to the LoRA shift. Safety erosion is **weight-specific**. |
| 3 | **Decomposition** | The shared (ICL-aligned) task direction is ~orthogonal to the refusal axis (cos(shared,r)≈−0.03) → safety-neutral. On harmful inputs the finetune shift is ~93% the ICL-orthogonal part, which carries the −r drift (cos(perp,r)≈−0.4). Harm = finetune-only, ICL-orthogonal, input-gated. |
| 4 | **Capstone (ablation)** | Ablating the harm direction (Arditi-style, all layers): refusal **0.84→0.98** (back to base) while DDXPlus accuracy holds **1.00→0.98**. |
| 5 | **Controls** | base+ablate-harm accuracy = **0.00** → the retained accuracy is LoRA **weight** robustness, not surgical task-sparing. lora+ablate-**random** refusal = **0.82** → the safety recovery is **harm-direction-specific**. |
| 6 | **Long-context** | As DDXPlus context fills 0→85%: LoRA accuracy holds (1.00→0.97), base accuracy *rises* via ICL (0.13→0.70) — task is context-robust. But finetuned refusal **collapses 0.80→0.10** while base refusal is **flat 0.97→0.97**. Fragility is finetuning-induced, not dimensionality. |
| 7 | **Keep-component steering + sweep** | At safe magnitude both ICL-parallel and orthogonal components carry task signal (~0.7 acc). The **refusal damage rides on the ICL-shared / context-mode direction**: at c=0.5 orthogonal keeps refusal 1.00, parallel drops to 0.16. (The earlier "parallel destroys task" was a c=1.0 over-drive artifact.) |
| 8 | **Geometry resolution** | Three *distinct* late-layer directions: `ctx` (ICL/context-mode, cos with r ≈ 0), `medPerp` (medical task knowledge, cos with r ≈ 0 — safety-neutral), `harmPerp` (harmful-input compliance, cos with r ≈ −0.5 — the harm). medPerp vs harmPerp cos ≈ 0.3 — partially overlapping but **not the same vector**, so "LoRA-specific" is input-dependent, not one axis. |
| 9 | **Context-fill baseline** | Base model, neutral filler, **local** harmful questions: refusal **flat ≈1.0** from 0→90% fill (~7k tokens). Pure context length does not erode a normal model's refusal — confirming #6's collapse is the finetuning × context **interaction**. |
| 10 | **Recipe under context** | Ablating the harm direction holds up under fill: un-ablated LoRA refusal collapses 0.76→**0.04** @45%, but **+ablate-harm stays 1.00→0.88→0.84** across 0/45/85% fill, with accuracy retained (0.84→0.80). The context erosion is expressed through the *same* compliance direction the ablation removes, so one static direction restores safety even in long context. |
| 11 | **Drift stress test** | Context-fill drift is large (‖drift‖≈200 @L35) but **mostly off the refusal axis** (cos(drift,r)≈−0.04 to −0.08). drift·r̂ does **not** track behavior: base has the *more* negative drift·r̂ yet rock-stable refusal; the LoRA's refusal drops with drift·r̂≈0. The static `r` is an incomplete probe in the context regime. Entropy is the better signal (LoRA 0.85→1.28 as refusal erodes). "Completely benign" is wrong (small consistent toward-compliance lean) but it is a minor component of a largely-orthogonal drift. |
| 12 | **Context-fit compliance axis** | A behavior-grounded `d_comply = unit(mean resid|comply − mean resid|refuse)`, fit on the LoRA's refuse/comply behavior under fill, is **mostly orthogonal to r** (cos −0.16 to −0.38) and the drift moves **~3× more along it** (drift·d_comply +24 vs drift·r +8). So the context erosion *is* directional — just along a different axis than the static r. *Ablating* d_comply backfires (refusal 0.54→0.08): it IS the behavioral refusal axis, so projecting it out removes refusal (Arditi). The correct fix is additive. |
| 13 | **Additive context-aware fix** | Steering the LoRA *toward* refusal (−d_comply, additive, layers 21/28/35) holds refusal **0.92–0.96 across all fills** at coeff 1 (un-steered collapses to 0.54), task accuracy unharmed (0.83→0.92). A **Goldilocks curve**: 0.5 under-corrects, 1.0 optimal, 2.0–4.0 over-drive and destroy refusal (→0.12→0.00) — magnitude-sensitivity made visible. The correct-sign mirror of #12's failed ablation; single-layer steering was washed out by LayerNorm, multi-layer bites. |

## What is solid
- **ICL and finetuning converge on the task subspace** (1), but **safety degradation is weight-specific** (2): the same content in-context does not erode refusal or move along the refusal axis.
- **The harm is a specific, low-rank, input-gated direction**, distinct from the task adaptation (3, 8), and **causally removable** (4) — and the removal is **harm-specific**, not a generic ablation effect (5).
- **Finetuning makes refusal context-fragile** (6) while the base model is context-robust on local questions (9). The safety cost is a latent fragility detonated by long context — invisible to short-context evals.

## What is qualified / what we corrected
- The capstone's "task retained" is **LoRA weight redundancy**, not surgical sparing (controls, 5). The clean half is the safety recovery.
- Steering is **magnitude-sensitive**: strong steering breaks refusal non-specifically (sweep, 7), and pushing base toward the context-mode direction erodes refusal *as a steering artifact* — **real** context fill on base does not (9). The defensible claim is the simple one: base context-robust, finetuned context-fragile.
- "Shared subspace = the beneficial part" is too clean: the shared direction is safety-neutral and ICL-aligned, but **both** components carry task signal; the task answer is not confined to one.
- Scope: one model, one finetune task, near-ceiling base refusal (modest behavioral headroom), N≈25–60.
- **The static refusal direction `r` does not transfer to the long-context regime** (drift stress test + context-refusal probe). Context-fill drift is large but only ~4–8% aligned with `r`, and `r` does not predict the context/finetuning behavioral change (base has the more-negative drift·r yet stable refusal; the LoRA's refusal drops with drift·r ≈ 0). A *behavior-grounded* compliance direction `d_comply` (fit on the LoRA's refuse/comply behavior under context) is mostly orthogonal to `r` (cos −0.16 to −0.38) and the drift moves ~3× more along it. So any "movement along the refusal axis" claim is **short-context/base-specific** and should not be read into context-fatigue dynamics. The base context drift is *not* "completely benign" — it carries a small consistent toward-compliance lean — but it is largely off-axis and behaviorally inert at the ceiling. The context erosion *is* directional along `d_comply`, and the correct intervention is **additive** steering toward refusal (−`d_comply`), not ablation: ablating `d_comply` removes the refusal axis (Arditi) and backfires (0.54→0.08), whereas additive steering at coeff 1 holds refusal 0.92–0.96 across fills with task intact (#13) — inside a calibrated band (a Goldilocks curve; over-driving destroys refusal). So the context-fatigue safety dynamics live on a **behavior-grounded axis distinct from the static `r`**, and that axis yields a working context-aware fix the static-`r` story could not.

## Practical takeaways
- A finetune that looks "mostly safe" in a short-context eval (refusal 0.84) can lose ~90% of its refusals once context fills (→0.04–0.16). **Safety-eval finetuned models under long context.**
- The safety damage occupies a specific low-rank direction that can be ablated to restore refusal with little task cost (the task is weight-redundant). **The ablation is robust to long context** (#10): one static direction removed holds refusal at 0.84–1.00 across the fill range where the un-ablated model collapses — a deployable recipe. (The additive "keep-the-ICL-component" recipe is the wrong framing — the ICL-shared direction is the refusal-eroding one.)
- **Two distinct working fixes for the context collapse, both causal:** (a) *ablate* the finetune-added harm direction (#10, static, robust to fill); (b) *additively steer* toward refusal along the context-fit `d_comply` (#13, holds 0.92–0.96 across fills at coeff 1). The additive steer must be tuned — it works only inside a calibrated band (over-driving destroys refusal) and must act across several layers (a single late-layer nudge is renormalized away). Ablation is the more robust, less tuning-sensitive recipe; the additive steer is the one that directly matches the context-fatigue geometry.

## Reproduce
Runbooks: [`results/lora_icl/README.md`](lora_icl/README.md), [`results/safety/README.md`](safety/README.md).
All driver scripts take `--config`; see `configs/{lora_icl,safety}`.
