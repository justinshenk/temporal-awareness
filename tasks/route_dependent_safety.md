# Brief — Route-dependent safety cost: weight (LoRA) vs activation (many-shot ICL)

Status: BUILDING on **Qwen2.5-7B-Instruct** (ungated — HF_TOKEN no longer needed; we
re-extract r̂ and d̂_task on Qwen rather than reusing gemma artifacts).

## Step 0 — baseline-refusal dynamic-range check ✅ DONE (2026-06-03)
`scripts/safety/check_baseline_refusal.py` → `results/safety/qwen_baseline/baseline_refusal.json`.
Qwen harmful refusal **0.983** (near ceiling → erosion headroom), harmless **0.017**
(classifier discriminates). Headroom OK — the floor/ceiling trap from the
instruction-adherence null does NOT apply here. Cleared to build the sweep.

## Big reuse find
`scripts/safety/{extract_refusal_shifts,run_safety_comparison}.py` ALREADY implement the
single-point route-decomposition (base/ICL/LoRA refusal + per-layer projection of
icl_shift/lora_shift onto r̂, with `project_onto`/`vector_cosine`/`mean_direction_cosine`).
Currently gemma + single dose. New work = port to Qwen (config) + add the DOSE dimension.

## Question
At **matched task dose**, is the safety cost route-dependent? Hypothesis: ΔRefusal ∝ the
move's component along the **refusal direction** `r̂`; the weight route (LoRA) carries that
component, the activation route (many-shot ICL) moves along the **task direction** `d̂_task`
but ~orthogonal to `r̂`. This is the mechanistic "why" behind the already-observed
weight-specific safety erosion ([[safety-weights-vs-activations-result]]) + the shared
ICL↔LoRA task direction (cos 0.81@L35).

## Decisions locked
- **Dose match = BOTH, as a sweep.** Sweep dose for both routes (LoRA: steps/data;
  ICL: #shots); plot ΔRefusal vs each projection axis; align routes on the behavioral
  task-gain x-axis. Avoids single-point matching circularity.
- **Domain D = DDXPlus; safety eval = a FRESH held-out harmful set** (e.g. an AdvBench
  slice), disjoint from D and from `r̂`-extraction prompts.
- Model: gemma-2-9b-it.

## Method
Per dose level, per route, at the prediction site (per layer), relative to base+clean:
- `lora_shift = resid(LoRA, x) − resid(base, x)`
- `icl_shift  = resid(base, x + manyshot-D) − resid(base, x)`
Decompose: `⟨shift, r̂⟩`, `⟨shift, d̂_task⟩`, and report `cos(r̂, d̂_task)` (oblique → also
report orthogonalized components). ΔRefusal = refusal(route) − refusal(base) on the held-out
harmful set via `refusal_classifier` (ICL eval carries the many-shot D context; LoRA eval
is adapter + clean safety prompt).

**Predicted result:** ΔRefusal collapses onto one line vs the refusal-axis projection across
both routes, and shows no relationship vs the task-axis projection. LoRA: nonzero `⟨·,r̂⟩`,
ΔRefusal<0. ICL: `⟨·,r̂⟩≈0`, ΔRefusal≈0. Both: matched `⟨·,d̂_task⟩` at matched dose.

## Reuse / new
- Reuse: `scripts/lora_icl/{train_ddxplus_lora,extract_shifts}.py`,
  `src/probes/safety/{refusal_direction,refusal_classifier,safety_data,ablation_hook}.py`.
- New: dose-sweep driver + projection/decomposition analysis + writeup.

## Phase 2 (deferred — "think about later"): weight-space basin leg
Interpolate θ_base → θ_specialist; trace refusal along the path (safety-basin traversal).
Then test whether the ICL activation trajectory is the **image** of that weight path or
**escapes** it. Not designed yet.

## TDD
Projection/decomposition helpers (cosine, component split, orthogonalization, the
ΔRefusal-vs-projection collapse stat) tested offline on synthetic vectors before wiring
to the gemma harness.
