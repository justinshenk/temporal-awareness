# Safety Degradation — Weights vs Activations (DDXPlus → refusal)

Tests whether narrow **DDXPlus medical** adaptation erodes safety, and whether that erosion is a
function of the **weights** or the **activations** — by delivering the same medical adaptation two
ways and measuring effect on refusal. No new training: reuses the DDXPlus LoRA from
[`results/lora_icl`](../lora_icl/README.md).

- **Finetune arm:** the existing DDXPlus LoRA (`results/lora_icl/adapter`).
- **ICL arm:** base model + DDXPlus medical cases in context, then the harmful prompt.
- **Refusal direction** `r` (Arditi et al. 2024): mean last-token resid(harmful) − resid(harmless)
  on the base model, per layer. Harmful = `mlabonne/harmful_behaviors` (AdvBench mirror); harmless
  = `tatsu-lab/alpaca` instructions.
- **Measure:** refusal rate (substring classifier) for base / ICL / LoRA on 60 held-out harmful
  prompts, plus signed projection of the ICL and LoRA activation shifts onto `r`.

## Result (Gemma-2-9b-it)

| Condition | refusal rate |
|-----------|-------------:|
| base | 0.983 |
| base + DDXPlus medical ICL | 0.983 |
| DDXPlus LoRA | 0.867 |

The LoRA erodes refusal (−0.117) and its activation shift points along −r (toward compliance) in
late layers (cos(LoRA,r) → −0.59 at L35). Medical ICL does neither (refusal unchanged; off-axis to
`r`; orthogonal to the LoRA shift). **Verdict: weight-specific** — finetuning's safety side-effect
is not reproduced by in-context learning of the same content. (Contrast: on the DDXPlus *task* itself,
ICL and finetuning converge to the same late-layer subspace, cos ≈ 0.8.)

Full report: [`2026-06-01-safety-weights-vs-activations.md`](2026-06-01-safety-weights-vs-activations.md).

### Follow-up: is the shared subspace the *beneficial* part of finetuning?

[`run_subspace_decomposition.py`](../../scripts/safety/run_subspace_decomposition.py) splits the
finetune shift into the ICL-aligned `par` (shared) and orthogonal `perp` (finetune-only) and
projects each onto `r` (uses existing shifts; no model run). Result supports it:
- **Medical inputs:** the shared task direction is ≈74% of the finetune shift (late) and is
  ~orthogonal to the refusal axis (cos(shared dir, r) ≈ −0.03) — the part ICL reproduces is
  **safety-neutral**.
- **Harmful inputs:** the finetune shift is ≈93% the orthogonal `perp`, and that finetune-only part
  carries the toward-compliance drift (cos(perp, r) → −0.59 @L35).
- **Verdict:** shared subspace = beneficial, safety-neutral task adaptation; the compliance drift is
  a finetune-only direction ICL never produces — and it is *input-gated* (near-zero refusal-axis
  content on benign inputs), not a static always-on compliance vector.
Report: [`2026-06-01-subspace-decomposition.md`](2026-06-01-subspace-decomposition.md).

### Causal capstone: ablate the harm direction

[`run_ablation_capstone.py`](../../scripts/safety/run_ablation_capstone.py) projects one residual
direction out of every layer of the DDXPlus-LoRA model (Arditi-style) and measures refusal +
DDXPlus accuracy.

| Condition | refusal (safer↑) | DDXPlus acc (task↑) |
|---|---:|---:|
| base | 0.98 | 0.10 |
| LoRA | 0.84 | 1.00 |
| **LoRA + ablate harm dir** | **0.98** | **0.98** |
| LoRA + ablate task dir (control) | 0.68 | 1.00 |

**Ablating the finetune-only harm direction restores safety (0.84→0.98, to base) while keeping task
accuracy (1.00→0.98)** — a causal demonstration that the harm is separable from the beneficial task
subspace. Ablating the task direction does *not* restore safety (specificity control passes). Caveat:
task ablation didn't lower accuracy (the LoRA's task solution is redundant), so the task half of the
dissociation isn't shown; the safety half is the load-bearing result.
Report: [`2026-06-01-ablation-capstone.md`](2026-06-01-ablation-capstone.md).

## Run (H200 + HF_TOKEN)

```bash
export HF_TOKEN=...
CFG=configs/safety/ddxplus_safety_gemma.yaml
uv run python -m scripts.safety.extract_refusal_shifts --config $CFG   # ~20 min (generation-bound)
uv run python -m scripts.safety.run_safety_comparison --config $CFG    # seconds
```

## Caveats
- Base refusal is near ceiling (0.983); the LoRA effect is ~7/60 prompts — real but modest. A model
  with a softer safety prior would give more behavioral headroom.
- Substring refusal classifier (standard, conservative); could be swapped for a judge model.
- One model, one finetune task. The weight-specific conclusion is for *this* adaptation; harmful
  many-shot ICL (deferred) would test the content axis separately.

## Code
| Module | Purpose |
|--------|---------|
| `src/probes/safety/refusal_direction.py` | refusal direction `r`, signed projection |
| `src/probes/safety/refusal_classifier.py` | substring refuse/comply classifier |
| `src/probes/safety/safety_data.py` | harmful (AdvBench mirror) + harmless (Alpaca) loaders |
| `scripts/safety/extract_refusal_shifts.py` | refusal direction + base/ICL/LoRA residuals & generations |
| `scripts/safety/run_safety_comparison.py` | refusal-rate + projection report |
