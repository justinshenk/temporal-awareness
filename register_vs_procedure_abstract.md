# Register, Not Procedure: What a Conditional Linear Map Can and Cannot Install

*Extended abstract — findings report. Branch `context-fatigue-datasets`, Llama-2-7B family.*
*Author: Ronen Raj Roy. Date: 2026-06-16.*

## Abstract

Activation steering and parameter-efficient finetuning both edit a model's behavior, and both can
be written as an **input-conditional affine map** of the residual stream — `h ↦ h + (Wh + b)`. We
ask where the *same* map family succeeds and where it fails, holding the apparatus fixed across
tasks. The boundary is sharp and reproducible: the map installs and transports a **register/
disposition** (refusal *tone*, a yes/no answer disposition) but never a multi-step **procedure**
(GSM8K arithmetic). A primal-ridge conditional map dominates every fixed-vector steering baseline on
the harm-refusal frontier, yet the identical machinery recovers ≈0 of a reasoning LoRA's gain — and
so do MLP, on-policy DAgger, per-context, and task-loss (DAS) variants. We trace the failure to its
mechanism: the reasoning capability is a *distributed, time-dense trajectory state* (recovery needs
≥94% of decode steps and ~80% of the residual's variance band), not a low-rank pointwise function of
the activation. Cross-model, even a *trained* register edit carries only the coarse disposition
off-policy; a strictly better-fitting affine bridge transports it **worse**, because the edit lives
off-manifold and regression shrinks exactly what must be preserved. We propose the register/procedure
distinction as the organizing axis for "what is steerable," and outline the experiments that would
turn it from an empirical regularity into a predictive criterion.

## 1. Setup and question

A behavior can be controlled by adding a vector to activations (CAA, Arditi, ActAdd, CAST) or by
finetuning a low-rank edit into the weights (LoRA) or a learned subspace (LoReFT). LoReFT's edit
`h + (Wh + b − hR)Rᵀ` and our primal-ridge steering map `(I + αW)·a` (fit closed-form so that
`W·a ≈ δ = a_expert − a_base`) are the *same object*: an input-conditional affine edit in a learned
subspace. We fix this object and sweep the **task**, asking a single question — *does the capability
install?* — under one coherence-aware metric. Two behaviors anchor the extremes: refusal (a
disposition) and GSM8K chain-of-thought arithmetic (a procedure). Commonsense QA sits between them.

## 2. Findings

**(R) Register installs from the conditional map alone, dominating fixed vectors.** Pointing the
ridge map at refusal and scoring on the harm-refusal / over-refusal Pareto frontier (60+60 held-out
prompts, refusal tone conditioned on coherence), only the *input-conditional* map reaches the
harmful-refusal axis at all:

| method | conditionality | harmful refusal | benign over-refusal |
|---|---|---:|---:|
| CAA mean-difference vector | none | 0.00 | — |
| Arditi single direction | none | 0.00 | — |
| CAA + logistic gate (CAST) | explicit classifier | 0.00 | — |
| **ridge map `W·a`** (α=0.12) | learned, for free | **0.62** | **0.02** |

The fixed vectors go straight from no-effect to off-manifold gibberish (their per-layer ℓ2 norm is
~49); the map's per-input shift lands *on* the chat manifold and even repairs base coherence
(0.63→0.98) before it refuses. Conditionality is learned, not bolted on: `W·a` is large on harmful
inputs and ≈0 on benign. The δ it regresses is near-perfectly decodable (held-out R²≈0.9997), so the
open question was never "can δ be read off" but "does injecting it steer" — and for register it does.

**(R) Commonsense is a register both LoRA and LoReFT solve — via divergent routes.** A paper-faithful
LoReFT (32 layers, rank 8, f7+l7) and a matched LoRA trained on the *byte-identical* supervised signal
both recover the task from a 0.00 base (BoolQ/PIQA/ARC-C: LoReFT 0.667/0.798/0.602, LoRA
0.680/0.790/0.655). But the *edits* are not the same change: CKA(δ) collapses 0.96 (L4) → 0.13 (L28),
per-token cosine stays 0.11–0.39, and top-8 subspace overlap stays 0.16–0.25. They share only an
early-layer, rotation-equivalent skeleton (the register/format nudge); the task content is written
through genuinely different deep-layer edits. There is no canonical "commonsense direction."

**(P) Procedure does not install — under any map we tried.** The same machinery pointed at GSM8K
recovers ≈0 of a MetaMath-LoRA's budget, robustly across the obvious salvages:

| salvage attempt | result | rules out |
|---|---|---|
| global primal-ridge map | 0.00 | — (baseline null) |
| short-output single-operation arithmetic | 0.00 | trajectory *length* as the wall |
| per-context local refit (problem-specific map) | 0.00 | map *generality* |
| on-policy DAgger refit (≈97% on-policy by round 2) | 0.00 | *distribution shift* |
| DAS task-loss subspace (CE→0.038, r=512) | 0.00 | low-rank-low-variance *subspace search* |

The map installs reasoning-*shaped* tokens with no arithmetic underneath — "the arithmetic wall."

**(P) The procedure is a distributed, time-dense trajectory state.** A *lockstep oracle* that
overwrites the base's L20 residual with the LoRA's every step *does* recover GSM8K (0.75), which let
us dissect what the map cannot supply. (i) **Distributed, not transported:** identity-ablating just
the top two downstream layers (30–31) craters not only the patched base (0.75→0.10) but the
natively-capable LoRA itself (1.00→0.10) — L20 holds an *intermediate*, not a finished answer. (ii)
**Compute is not the deficit:** teacher-forced on the correct chain, the base predicts computed-result
tokens at 96.8%; its failure is *trajectory control*, not per-step arithmetic. (iii) **Dense in time
and in variance:** recovery requires patching ≥94% of decode steps (every ≤50% gate → 0.00) and ~80%
of δ's variance band (top-64 = 55% of energy but 0% recovery; the cliff is top-256→512). A donor-free
steerer would therefore have to reproduce the LoRA's near-exact 4096-d state at almost every step —
i.e., *be the donor*. The procedure is not a low-rank, pointwise, or time-sparse function of the
activation.

**(X) Cross-model, even a trained register edit carries only the coarse disposition — and aiming
better hurts.** Transplanting a trained LoReFT edit from a donor (Llama-2-7B base) to a recipient
(Llama-2-7B-chat) through a per-layer orthogonal-Procrustes bridge transfers the *disposition*
off-policy (BoolQ boolean-commitment 32→53/64, lenient 0.33→0.58) but not the trained output template
or answer correctness (exact-match ≈0). On-policy refit is *worse*; and a ridge-fit **affine** bridge
with 12–45× better held-out geometry transports the edit *worse still* (8/64), because the LoReFT
delta lives off the data manifold (‖δ‖>‖h‖) and regression shrinks weakly-represented directions —
precisely the ones that must be preserved. **Strength beats aim:** full-norm deltas through a crude
isometry move behavior; well-aimed deltas at half norm do nothing. The geometric "wall" was real but
behaviorally irrelevant.

## 3. Synthesis

One axis organizes all of the above. A **register/disposition** — refusal tone, an answer format, a
yes/no commitment — is a low-rank, on-manifold, roughly *pointwise* function of the activation; it is
installable by a single conditional affine map and even survives a crude cross-model rotation. A
**procedure** — multi-step arithmetic — is a high-rank, off-manifold, *time-dense* trajectory state;
it is invisible to variance- and task-loss-based subspace search, and recoverable only by transporting
essentially the donor's full per-step state. Finetuning (LoRA/LoReFT) and activation steering are the
same map family, and they inherit the same boundary: they relocate *where a disposition points*, not
*how a computation unfolds*. A corollary for distillation: a trained edit can be "distilled" by the
conditional map only when fit on the steered (on-policy) input — but that is near-tautological (the
edit is affine in its own input), a positive control, not an independent capability-install.

## 4. Next steps

- **Hybrid transplant bridge** — affine forward map (better-aimed donor delta) + *orthogonal*
  back-projection (norm-preserving transport), to isolate aim from strength; Result 5 predicts it
  beats the rigid-rotation baseline.
- **Quantify the disposition carry** beyond n=64 smoke: loosen the extractor to bare `true/false`,
  re-score full BoolQ/PIQA/ARC splits.
- **Same-family base→base transplant** to separate "cross-model" from "base→chat instruction-tuning
  shift" as the cause of the carry failure.
- **A register-vs-procedure predictor**: turn the empirical boundary into a measurable criterion —
  e.g. does (δ-rank, on-/off-manifold fraction ‖δ‖/‖h‖, temporal density of the recovery gate)
  predict installability *a priori* on a held-out behavior (tone, sentiment, sycophancy, induction)?
- **Leak diagnostic for installed refusal**: the tone axis is established; run the two-axis HarmBench
  leak test on a compliant instruction-following base (e.g. Vicuna) to test whether content is
  withheld under the installed apology.

## Evidence index

Refusal frontier `results/attribution/2026-06-08-refusal-frontier.md`; LoRA-vs-LoReFT
`…/2026-06-14-lora-vs-loreft-commonsense.md`; procedure nulls `…/2026-06-07-{short-output-arithmetic,
local-refit,dagger-refit}.md`, `…/2026-06-10-{pca-band-complement,das-subspace}.md`; L20 oracle
dissection `…/2026-06-13-compute-vs-communicate-L20.md`, `…/2026-06-14-temporal-oracle-L20.md`;
cross-model transplant `…/2026-06-12-crossmodel-transplant.md`. All runs seeded, CPU-tested, and
reproducible via the drivers listed in each report.
