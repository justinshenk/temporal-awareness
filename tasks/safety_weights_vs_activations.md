# Execution Brief — Safety Degradation: Weights vs Activations

Branch: `context-fatigue-datasets`. Status: ✅ DONE (run 2026-06-01).
Builds on [[lora-icl-subspace-experiment]] (ICL≈finetuning converge in a late-layer subspace on DDXPlus).

## Verdict: WEIGHT-SPECIFIC
Gemma-2-9b-it, 60 held-out harmful prompts. Refusal rate: base 0.983, base+medical-ICL 0.983,
DDXPlus-LoRA 0.867. The LoRA erodes refusal AND its shift points along −r (cos(LoRA,r) → −0.59 @L35);
medical ICL does neither (refusal unchanged; off-axis; orthogonal to LoRA shift, cos≈0 all layers).
So safety degradation from this adaptation is a function of WEIGHTS, not activations — ICL of the same
content doesn't carry it. Contrast: on the DDXPlus *task*, ICL & finetuning converge (cos≈0.8). Two
findings together: adaptation converges in activation space, but the safety side-effect is weight-only.
Report: results/safety/2026-06-01-safety-weights-vs-activations.md. 22 unit tests; full suite 76 green.

## 1. Problem statement
Finetuning is documented to erode safety (Qi et al. 2024), even on benign data. ICL changes only
activations, not weights. Question: does ICL erode safety along the **same internal axis** as
finetuning (⇒ safety degradation is an activation-space phenomenon), or can only weight changes do it
(⇒ weight-specific)? Operationalized via the **refusal direction** (Arditi et al. 2024).

## 2. Agreed solution approach  (REVISED: reuse DDXPlus medical, no new training)
Test whether narrow **DDXPlus medical** adaptation erodes safety, delivered by weights vs context,
measured behaviorally and as a shift along the refusal direction. Reuses the EXISTING DDXPlus LoRA
adapter and DDXPlus ICL builder from [[lora-icl-subspace-experiment]] — no benign Alpaca, no new train.

- **Refusal direction** `r_ℓ` per layer = mean resid(harmful) − mean resid(harmless), base model,
  last-token (Arditi). Normalize per layer. (+r = more refusal.)
- **Finetune arm:** the existing DDXPlus LoRA (`results/lora_icl/adapter`).
- **ICL arm:** base model + DDXPlus medical cases in context (reuse the icl_ids builder), final turn
  = the harmful eval prompt.
- **Eval:** held-out harmful prompts (AdvBench). For each, at the prediction site capture residuals
  under base-clean / base+DDXPlus-ICL / DDXPlus-lora-clean, and generate a response.
  - icl_shift  = resid(base, harmful WITH DDXPlus-medical ICL) − resid(base, harmful clean)
  - lora_shift = resid(DDXPlus-lora, harmful clean)            − resid(base, harmful clean)
- **Metrics:**
  - Behavioral: refusal rate (refuse vs comply, substring classifier) for the 3 conditions.
  - Mechanistic: signed projection of each shift onto `r_ℓ` (negative ⇒ moved toward compliance);
    cosine(icl_shift, lora_shift) per layer (reuse `src/probes/lora_icl/subspace_metrics.py`).
- **Conclusion logic:** finetune erodes refusal + shifts along −r; if ICL ALSO does ⇒ activation-space.
  If finetune erodes but ICL doesn't move along r ⇒ weight-specific. (A null for benign-ICL is itself
  an informative answer.)

## 3. Files likely created
- `src/probes/safety/refusal_direction.py`   — per-layer r, normalization, signed projection
- `src/probes/safety/refusal_classifier.py`  — refuse/comply from generated text (substring method)
- `src/probes/safety/safety_data.py`         — AdvBench harmful + Alpaca harmless, deterministic splits
- `scripts/safety/extract_refusal_shifts.py` — 3-condition capture + generation + projections + refusal
- `scripts/safety/run_safety_comparison.py`  — report: refusal rates + per-layer projection/alignment
- `configs/safety/ddxplus_safety_gemma.yaml`
- `tests/probes/safety/` — classifier, r math, projection (test-forward)
- `results/safety/README.md` + dated report
- Reuse: existing DDXPlus LoRA adapter, `ddxplus_cases` (ICL builder), `subspace_metrics`,
  `shift_extraction`, `PerTokenResidualCapture`. NO new training.

## 4. Non-goals / do not change
- No harmful content generation or harmful training/ICL data — benign finetune + benign ICL only;
  harmful prompts are used solely to MEASURE refusal (standard defensive eval).
- No steering/restoration arm (deferred; user picked mechanistic+behavioral, not the steering option).
- One model (Gemma-2-9b-it). Don't touch main or the DDXPlus artifacts.

## 5. Operational constraints
- Local H200 + HF_TOKEN (already validated). bf16.
- AdvBench (e.g. `walledai/AdvBench`) + Alpaca (`tatsu-lab/alpaca`) from HF.
- Generation needed for behavioral arm (slower than DDXPlus single-token); keep eval set modest (~60–100).
- Deterministic seed 42; adapters/tensors gitignored, only md/json committed.

## 6. Acceptance criteria
- Reproducible: train script + config → benign adapter; extract + compare → refusal-rate table +
  per-layer projection/alignment report, deterministically.
- Report states whether ICL reproduces the finetuning refusal-direction shift (the weights-vs-activations verdict).
- Unit tests for classifier + refusal-direction + projection pass.

## 7. TDD (test-forward)
- refusal_classifier: known refusal strings ("I can't help", "I'm sorry, but") → refuse; compliant → comply.
- refusal_direction: r = mean(harmful) − mean(harmless); projection sign on synthetic vectors.
- projection-onto-r reuses vetted subspace_metrics where possible.

## 8. Test expectations (scientific)
- Expect benign finetune to erode refusal modestly (Qi) and shift activations along −r.
- Open whether benign ICL does the same: if yes ⇒ activation-space; if no ⇒ weight-specific. Both publishable.

## Decision needed (the one fork)
ICL-arm content: **benign-parallel** (same benign data as the finetune, clean weights-vs-activations
test) vs **harmful many-shot** (Anil-style jailbreak — stronger erosion but conflates "harmful context"
with "ICL mechanism", and adds harmful in-context content). Recommend benign-parallel.
