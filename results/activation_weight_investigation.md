# Manipulating LLM behavior: activations vs weights — investigation overview

A consolidated index of one investigation (branch `context-fatigue-datasets`; strands 1–4 on
`google/gemma-2-9b-it`, strand 5 on `Llama-2-7b`): **what can you change by manipulating
activations vs weights, and where
do in-context learning (ICL) and finetuning converge or diverge** — stress-tested under long context
and many-shot. Each strand has a detailed report; this doc is the map and the unifying thesis.

## Unifying thesis

"**Where a behavior lives** (written into weights vs carried by activations/context)" and "**how best
to control it**" are related but *distinct* axes — and **long context and many-shot are the stress
tests that separate them.** Two behaviors anchor the extremes:

- **Refusal-harm is weight-written, low-rank, and activation-ablatable.** Finetuning erodes it; the
  same content in-context does not. The harm occupies a specific input-gated direction you can project
  out to restore refusal — and that fix survives long context.
- **Sycophancy is base-intrinsic, distributed, and in-context-policy-driven.** No finetune needed; a
  single steering direction only weakly moves it, but many-shot demonstrations control it almost
  completely.

And across both: **ICL and finetuning converge on a shared task *direction*** (the format/function
component), even when they differ on side-effects and on sparse mechanism.

## The four strands

### 1. Refusal: weights vs activations, and context-fragility
[`lora_icl_safety_synthesis.md`](lora_icl_safety_synthesis.md) · [`safety/README.md`](safety/README.md)
- Safety erosion is **weight-specific** (finetune erodes refusal; ICL of the same content does not).
- The harm is a **low-rank, input-gated direction**, causally **ablatable** to restore refusal with
  task intact — and the ablation **holds under long context** (a deployable recipe).
- Finetuned refusal is **context-fragile** (collapses as context fills) while base refusal is
  context-robust. The static refusal direction `r` does **not** transfer to long context; a
  behavior-grounded `d_comply` does, and **additive steering along it restores refusal**.

### 2. Sycophancy: distributed and in-context-driven
[`sycophancy/README.md`](sycophancy/README.md)
- Base model is **already highly sycophantic** (caves ~50–56% under one confident pushback); `d_syco`
  is a clean axis (sep 55.5).
- **Additive steering toward holding ground is modest and narrow** (0.51→0.38, over-drives by coeff 2).
- **Many-shot priming dominates:** a context of cave/hold demonstrations drives caving 0.00 ↔ 0.98 at
  matched length — the in-context channel controls sycophancy where a static direction barely does.

### 3. Literature grounding (verified deep-research)
[`activation_vs_weight_literature.md`](activation_vs_weight_literature.md) ·
[`icl_mechanisms_literature.md`](icl_mechanisms_literature.md)
- Taxonomy: activation-steerable behaviors (RepE, ActAdd, CAA, ITI) vs weight-edited knowledge
  (ROME/MEMIT, task arithmetic, finetuning-erodes-safety). Steering **stacks on top of** finetuning.
- The single-vs-multi-direction refusal debate (Arditi vs Pan/Joad/Engels) — our context-`r`-doesn't-
  transfer result is a data point on it.
- ICL writes a compact **task/function vector** into activations (Hendel, Todd); induction heads are a
  developmental precursor to function-vector heads.

### 4. ICL ↔ weights bridge, and function-vector extraction
[`lora_icl/2026-06-02-antonym-fv-vs-lora.md`](lora_icl/2026-06-02-antonym-fv-vs-lora.md) ·
[`lora_icl/2026-06-02-fv-extraction.md`](lora_icl/2026-06-02-fv-extraction.md)
- **ICL and LoRA converge on a shared task direction:** cos 0.81 @L35 (DDXPlus), **0.77 @L35
  (antonyms)** — the latter on a genuine ICL task where the LoRA provably generalizes.
- That shared direction is the **format/function** component. The **sparse Todd-style FV isolates a
  *different*, near-orthogonal *label-mapping* sub-direction** (Min et al., made mechanistic).
- **Method lesson:** the DDXPlus FV attempt was a null because DDXPlus isn't an ICL task (labels
  inert). The first antonym FV null was a *zero-shot-corruption artifact* (single-head AIE ≈ 0 by
  construction); with Todd-faithful shuffled-label corruption, FV heads appear (early-middle layers,
  AIE up to +0.94 nats). The "distributed across heads" claim was retracted.

### 5. Register vs procedure: what a conditional affine map can install
[`attribution/2026-06-16-multihop-generality.md`](attribution/2026-06-16-multihop-generality.md) ·
[`../register_vs_procedure_abstract.md`](../register_vs_procedure_abstract.md)
- The ridge map `W·a` (the same object as a LoReFT edit) installs a **register/disposition**
  (refusal tone dominates fixed-vector baselines) but not a **procedure** (GSM8K: ≈0 under every
  salvage — MLP, DAgger, local refit, DAS); the procedure is a distributed, temporally dense
  trajectory state (lockstep oracle 0.75 @L20, needs ~every decode step).
- **The thesis generalizes beyond arithmetic (n=2 procedures):** on MuSiQue open-book multi-hop QA
  the oracle axis replicates exactly (+0.76 @L20, same layer/magnitude) and so does temporal
  density; the pointwise ladder is PARTIAL — MLP/DAgger stay 0, but the *linear* rung leaks
  ~¼ of the budget @L20 and ~half @L24 (α=1.0-resonant, layer-humped over the plateau) vs ~5%
  anywhere on GSM8K. Sharpened claim: the *procedure core* does not install; its size is
  task-dependent.

## Cross-cutting taxonomy (where our two behaviors land)

| | Refusal-harm | Base sycophancy |
|---|---|---|
| Where it lives | weight-written (finetune-induced) | base-intrinsic |
| Geometry | low-rank, input-gated direction | clean axis but **distributed** mechanism |
| Best activation control | ablate / additive-steer the direction (works) | weak (narrow band, over-drives) |
| Dominant lever under context | the static low-rank direction | **in-context demonstrations (many-shot)** |
| Under long context | finetuned fragile; base robust; ablation holds | high, ~flat under neutral length |

## What is solid vs qualified
- **Solid:** weight-specific safety erosion; harm direction ablatable + context-robust; base
  sycophancy high with a clean axis; many-shot bidirectional control of sycophancy; ICL↔LoRA shared
  task direction (two tasks, ~0.8).
- **Qualified / corrected:** steering is magnitude-sensitive (Goldilocks bands throughout); the static
  `r` doesn't transfer to long context; the sycophancy context-fatigue *rise* did not replicate across
  subsets (within noise); the antonym sparse-FV isolates mapping not format (and the earlier
  "distributed" reading was a corruption artifact, retracted). Scope: one model; small N; near-ceiling
  base refusal.

## Reproduce
Each sub-report lists its driver + config under `scripts/{lora_icl,safety,sycophancy}` and
`configs/{...}`; all seeded (42). Adapters and shift/direction tensors are gitignored and regenerate.
