# Distilling a LoRA finetune into a linear activation map on the base model

Can a linear map transfer a finetuned model's knowledge to the base model? Fit per layer
`W a_base ≈ Δh_L` (Δh_L = LoRA shift = resid(LoRA,x) − resid(base,x)) by ridge on DDXPlus fit
cases, then steer the BASE with α·(W a). Qwen2.5-7B, adapter dose 600, n_fit=100, n_eval=40,
n_harmful=25, λ=1.

## Results

| condition | task_acc | refusal |
|---|--:|--:|
| base | 0.139 | 1.00 |
| **distill (W on LoRA shift), α=0.5** | **0.675** | 0.84 |
| distill_safe (W on r-orthogonalized LoRA shift), α=0.5 | 0.675 | 0.88 |
| distill α=1.0 | 0.450 | 0.00 |
| distill_safe α=1.0 | 0.400 | 0.04 |
| LoRA finetune (reference) | 0.925 | 0.00 |

## Findings
1. **Yes — the linear map distills the finetune's task knowledge.** Steering the base with
   `W a_base` recovers **0.14 → 0.675**, ~73% of the base→LoRA gap (0.925), with no LoRA
   weights at inference. The shortfall below 0.925 is the per-layer-linear approximation cap
   (the LoRA's full effect is multi-layer and nonlinear).
2. **It also copies the finetune's safety erosion — partially.** The LoRA shift carries a
   refusal-axis component, so the distilled map erodes refusal 1.00 → 0.84 at α=0.5.
   r-orthogonalizing the target shift helps only modestly (0.84 → 0.88): the map is fit on
   DDXPlus activations and generalizes imperfectly to harmful activations, so it still emits
   some r-aligned output there.
3. **Narrow band:** α=1 over-steers (task ↓, refusal → 0).

## The principled takeaway
Distilling the **weight route (LoRA)** transfers slightly more task but imports the safety
baggage (and orthogonalization only partly scrubs it). Distilling the **activation route
(ICL)** — fitting the same map on the ICL shift instead (`CONDITIONAL_STEER.md`, ~0.65 task /
refusal fully preserved) — gives nearly the same competence transfer and is **clean by
construction**, because the ICL shift has no refusal component to begin with. So the answer is
"yes, and prefer the ICL source": you can transfer the finetune's competence into the base via
a linear activation map without importing its refusal erosion.

## Cross-family replication on gemma-2-9b-it (offline, existing adapter)

| condition (gemma, base 0.174) | task_acc | refusal |
|---|--:|--:|
| **distill (W on LoRA shift), α=0.5** | **0.700** | **0.96** |
| distill_safe (r-orthogonalized), α=0.5 | 0.700 | **0.36** ⚠ |
| LoRA finetune (reference) | 0.950 | 0.88 |

- **Core distillation replicates:** 0.17 → 0.70 (~64% of the base→LoRA gap), refusal fully
  preserved (0.96) — cleaner than Qwen because the gemma adapter is the mild r=32 subspace
  LoRA (weak refusal-axis component, so little erosion to copy).
- **The r-orthogonalized "safe distill" did NOT replicate — it eroded refusal to 0.36**
  (opposite of Qwen, where it helped 0.84→0.88). Honest negative: **orthogonalizing the
  distillation *target* against r is not a reliable cross-family safety lever.** Editing the
  map's on-task targets doesn't control its behavior on off-task (harmful) inputs, and gemma's
  refusal is fragile (narrow band), so the refit map tips it. The robust safe-transfer recipe
  remains "distill from the ICL shift" (naturally r-free), not "distill from the LoRA and
  orthogonalize."

## Caveats
One adapter (dose 600), n_eval=40, n_harmful=25, single seed, λ=1 untuned, 5 layers. The map
captures the LoRA's *local* per-layer effect, not the full multi-layer interaction. (Also hit
the in-place PeftModel-wrap bug — base activations computed before wrapping.)

## Reproduce
`scripts/safety/run_lora_distill.py` with `--adapter results/safety/qwen_sweep/adapter_d600`.
JSON: `lora_distill.json`.
