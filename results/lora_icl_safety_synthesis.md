# ICL ≈ Finetuning on the task, but the safety cost is weight-specific and context-fragile

A nine-experiment investigation on `google/gemma-2-9b-it`, DDXPlus medical MCQ as the finetuning
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

## What is solid
- **ICL and finetuning converge on the task subspace** (1), but **safety degradation is weight-specific** (2): the same content in-context does not erode refusal or move along the refusal axis.
- **The harm is a specific, low-rank, input-gated direction**, distinct from the task adaptation (3, 8), and **causally removable** (4) — and the removal is **harm-specific**, not a generic ablation effect (5).
- **Finetuning makes refusal context-fragile** (6) while the base model is context-robust on local questions (9). The safety cost is a latent fragility detonated by long context — invisible to short-context evals.

## What is qualified / what we corrected
- The capstone's "task retained" is **LoRA weight redundancy**, not surgical sparing (controls, 5). The clean half is the safety recovery.
- Steering is **magnitude-sensitive**: strong steering breaks refusal non-specifically (sweep, 7), and pushing base toward the context-mode direction erodes refusal *as a steering artifact* — **real** context fill on base does not (9). The defensible claim is the simple one: base context-robust, finetuned context-fragile.
- "Shared subspace = the beneficial part" is too clean: the shared direction is safety-neutral and ICL-aligned, but **both** components carry task signal; the task answer is not confined to one.
- Scope: one model, one finetune task, near-ceiling base refusal (modest behavioral headroom), N≈25–60.

## Practical takeaways
- A finetune that looks "mostly safe" in a short-context eval (refusal 0.84) can lose ~90% of its refusals once context fills (→0.10). **Safety-eval finetuned models under long context.**
- The safety damage occupies a specific low-rank direction that can be ablated to restore refusal with little task cost (the task is weight-redundant). A keep-the-ICL-aligned-component recipe is plausible but not yet cleanly demonstrated (steering magnitude confounds).

## Reproduce
Runbooks: [`results/lora_icl/README.md`](lora_icl/README.md), [`results/safety/README.md`](safety/README.md).
All driver scripts take `--config`; see `configs/{lora_icl,safety}`.
