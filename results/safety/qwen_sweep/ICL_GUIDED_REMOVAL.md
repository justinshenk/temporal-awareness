# Confirming the task axis & removing the refusal side-component (Qwen, DDXPlus→refusal)

Follow-up to the route-dependent safety sweep (`ROUTE_SWEEP.md`). Three questions:
confirm ICL and finetuning share the **task axis**; and try to **remove** the finetuning's
refusal side-component using ICL as the reference, via (a) activation ablation and
(b) weight-space projection.

## 1. The shared task axis — confirmed (on task prompts)
`cos(icl_shift, lora_shift)`, prediction site:

| prompts | L0 | L7 | L14 | L21 | L27 |
|---|--:|--:|--:|--:|--:|
| **DDXPlus (task)** | −0.17 | +0.17 | +0.46 | **+0.74** | **+0.71** |
| harmful (off-task) | −0.18 | +0.23 | +0.21 | −0.03 | −0.08 |

On the **task**, ICL and finetuning move along the **same** direction (cos +0.74@L21,
replicating gemma's +0.81). On **harmful** prompts they are ~orthogonal. So the shared
task axis is real but is a *task-prompt* phenomenon; off-task the finetuning carries an
extra component. The label-free residual `ŵ` = (LoRA move − its ICL-aligned part) recovers
the refusal axis: `cos(ŵ, r) = −0.575@L21, −0.521@L27`. So the off-task extra component
*is* the refusal direction — derived without any safety labels.

## 2. Activation ablation of ŵ — partial restore, and it beats r
Per-layer directional ablation (all 28 layers). Sanity: ablating `r` on **base** drops
refusal 0.98→**0.02** (recipe bites). Held-out harmful n=50:

| | LoRA-75 (threshold erosion) | LoRA-600 (deep erosion) |
|---|---|---|
| none | refusal 0.00, acc 0.75 | 0.00, 0.93 |
| **ablate ŵ (ICL-guided)** | **0.32, acc 0.70** | 0.00, 0.95 |
| ablate r (label-based) | **0.00**, 0.73 | 0.00, 0.98 |
| ablate random | 0.00, 0.78 | 0.00, 0.95 |

**Ablating the ICL-guided `ŵ` partially restores refusal (0.00→0.32) with task intact —
and beats the canonical refusal direction `r`, which fails entirely on the finetuned
model.** Mechanistic reading: finetuning erodes safety along *its own* direction `ŵ`
(distinct from the base refusal axis `r`); to repair a finetuned model you must remove the
direction it actually used, and the ICL contrast hands you that direction. Deep erosion
(600 ex) is not recoverable by activation ablation.

## 3. Weight-space projection of ŵ — null, and informative
Editing the LoRA so it doesn't write along `ŵ`: `B' = (I − ŵŵᵀ)B` on the residual-writers
(`o_proj`, `down_proj`) at every layer. **No effect** — every condition (incl.
`project_random`) equals `none`.

Why (verified): `lora_B` is **rank-16** (norm ≈ 0.28); a single residual direction `ŵ`
barely overlaps that 16-d column space, so projecting it out removes almost nothing.
Zeroing the same `B`s *does* move logits (Δ 1.27), so edits bite — the projection is just
near-orthogonal. The deeper point: **the refusal erosion is not a direct low-rank write
along `ŵ`; it is an emergent shift** produced by the LoRA changing the whole computation
(q/k/v/gate/up included) and the base weights amplifying it. That is exactly why
*activation* ablation of `ŵ` partly works (removes the emergent shift wherever it surfaces)
while *weight-space* projection of the output weights does not.

## Bottom line
- **Task axis: confirmed** (cos 0.74 on task; ~0 off-task). ICL and finetuning converge on
  the task direction; the finetuning's off-task surplus is the refusal axis.
- **Removal: ICL-guided activation ablation partially restores safety** on threshold
  erosion (0.00→0.32, task kept) and **beats the label-based refusal direction** — a real,
  if partial, win for the ICL-guided approach. Deep erosion resists it.
- **Weight-space projection is a null** because the erosion is emergent/distributed, not a
  removable low-rank write. The real weight-space fix would act during training
  (refusal-preserving penalty / gradient projection), not as a post-hoc output projection.

## Reproduce
`scripts/safety/{run_task_axis_ddxplus,run_strong_ablation,run_weight_projection}.py`
with `configs/safety/route_safety_qwen.yaml`; `PerLayerAblationHook` in
`src/probes/safety/ablation_hook.py`. JSONs: `task_axis_ddxplus.json`,
`strong_ablation.json`, `weight_projection.json`.
