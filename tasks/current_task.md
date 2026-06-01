# Execution Brief — LoRA vs ICL Activation-Subspace Comparison (DDXPlus)

Branch: `context-fatigue-datasets`. Status: ✅ DONE — full run executed 2026-06-01.

## Result
ICL and LoRA finetuning move Gemma-2-9b-it's prediction-site residual into a SIMILAR subspace,
localized to the mid-to-late layers. Mean-shift cosine: ~0 early (L0-7), rising to +0.66 (L21),
peak +0.81 (L35, ~0.85 depth), +0.74 (L41). Subspace overlap (mean-centered PCA-5) also high late
(~0.65-0.68), so the per-case variation aligns too — not just a shared format offset. Random null
±0.017, so late-layer alignment is ~40-50x chance. Scope was one real adapter, no controls (user's
call). Caveat: single task can't fully separate task-subspace from MCQ-format-subspace; cross-task
run would settle it. Report: results/lora_icl/2026-06-01-subspace-comparison.md.

## Built (2026-06-01)
- `src/probes/lora_icl/{subspace_metrics,shift_extraction,ddxplus_cases}.py` (+ `__init__` auto-export)
- `scripts/lora_icl/{train_ddxplus_lora,extract_shifts,run_subspace_comparison}.py`
- `configs/lora_icl/ddxplus_gemma_lora.yaml`, `results/lora_icl/README.md` (runbook)
- `tests/probes/lora_icl/` — 34 tests: metrics, shift assembly, case builder, compare_layer,
  + offline PEFT-hook plumbing smoke (tiny random Llama, verifies hooks fire under LoRA wrap)
- deps: added `peft`, `accelerate` via uv; `.gitignore` excludes adapters + shift tensors
- Verified: ruff clean on new files; full `tests/probes/` green (56); scripts import-check OK
- To run live: `export HF_TOKEN=...` then follow `results/lora_icl/README.md` (4 steps)


## 1. Problem statement
Test whether **LoRA finetuning** on DDXPlus and **in-context learning (ICL)** (accumulating
DDXPlus cases in the prompt) push the model's residual stream along the **same low-dimensional
direction / subspace**. The LoRA adapter trained ~2026-05-31 was never committed and is gone;
we retrain it reproducibly and build the comparison harness.

## 2. Agreed solution approach
Compare two **activation shifts**, both measured relative to *base-model-clean*, at the
prediction site (last token before the answer letter), per layer ℓ, on a held-out DDXPlus
test split:
- **ICL shift**   uᵢ,ℓ = resid_ℓ(base, case i WITH accumulated context) − resid_ℓ(base, case i clean)
- **LoRA shift**  vᵢ,ℓ = resid_ℓ(LoRA, case i clean)               − resid_ℓ(base, case i clean)

Primary metric (per user): **direction cosine / subspace angle**
1. Mean-direction cosine: cos(mean_i uᵢ,ℓ, mean_i vᵢ,ℓ) per layer.
2. Principal angles between top-k PCA subspaces of {uᵢ,ℓ} and {vᵢ,ℓ} (Grassmann overlap).
Secondary corroboration: probe-transfer (train separator on ICL shift, test on LoRA shift).

Controls / nulls (branch culture demands them):
- Random high-d direction null (≈0 ± 1/√d).
- **Shuffled-label LoRA** (trained on DDXPlus with permuted answers) — should NOT align.
- Cross-task ICL (MMLU context) vs DDXPlus-LoRA — should align less.
- Full per-layer profile (where, if anywhere, alignment peaks).

## 3. Files likely created/modified
- `configs/lora_icl/ddxplus_gemma_lora.yaml`        — committed LoRA hyperparams
- `scripts/lora_icl/train_ddxplus_lora.py`          — PEFT/TRL trainer (+ shuffled-label flag)
- `scripts/lora_icl/extract_shifts.py`              — compute ICL & LoRA shifts (reuse PerTokenResidualCapture)
- `scripts/lora_icl/run_subspace_comparison.py`     — metrics + markdown writeup
- `src/probes/lora_icl/subspace_metrics.py`         — cosine, principal angles, probe transfer
- `src/probes/lora_icl/shift_extraction.py`         — shift assembly / sign conventions
- `tests/probes/lora_icl/test_subspace_metrics.py`  — TDD (test-forward)
- `pyproject.toml`                                  — add peft, trl, accelerate
- `results/lora_icl/<date>-subspace-comparison.md`  — writeup; adapter + .pt gitignored
- Reuse unchanged: `src/probes/ddxplus.py`, `src/probes/extraction.py`, `_cf_common.py`

## 4. Non-goals / do not change
- Do NOT touch `main` or the existing v1–v6 task_position artifacts.
- LoRA only (no full finetune); one base model first (Gemma-2-9B-IT) before generalizing.
- Not re-deriving within_task_fraction probe; not a behavioral-accuracy paper (geometry-first).
- No weight-space-only comparison — both arms reduce to activation-space shifts (apples-to-apples).

## 5. Operational constraints
- Local NVIDIA H200 (143 GB) — Gemma-9B LoRA fits in bf16 comfortably.
- Need `peft`/`trl`/`accelerate` added to deps.
- Gemma is gated → requires `HF_TOKEN` in env.
- Deterministic seed 42; adapters + activations (.pt) gitignored, only metrics+writeup committed.
- DDXPlus eval split for extraction must be DISJOINT from LoRA train split.

## 6. Acceptance criteria
- `train_ddxplus_lora.py` + committed YAML regenerate an adapter from scratch.
- `extract_shifts.py` + `run_subspace_comparison.py` regenerate the per-layer cosine +
  principal-angle table and writeup from the adapter, deterministically.
- Writeup reports the metric vs all controls (random null, shuffled-label LoRA, cross-task).
- Unit tests for metric functions pass.

## 7. TDD (test-forward)
Write `test_subspace_metrics.py` FIRST:
- principal angles of identical subspace → 0; orthogonal subspaces → 90°.
- cosine: parallel → 1, antiparallel → −1, two random high-d vectors → ≈0.
- shift sign convention on a tiny toy (shift of zero context vs itself = 0).

## 8. Test expectations (scientific)
- If H holds: ICL/LoRA cosine significantly > random null at mid layers; small principal angles.
- Shuffled-label LoRA control near null (alignment is task-specific, not a generic finetune artifact).
- A clean null is still a publishable result (cf. the v1–v6 intervention nulls).

## Confirmed decisions (2026-06-01)
1. Base model = `google/gemma-2-9b-it`.
2. `HF_TOKEN` not yet set — user will add it before training (training step blocked until then;
   code + unit tests can be built/run without it).
3. LoRA = **r=32, α=64, dropout=0.05, target q/k/v/o + gate/up/down, heavier (more steps)**.

## Implementation order (test-forward)
1. Add `peft`, `trl`, `accelerate` to pyproject.
2. `tests/probes/lora_icl/test_subspace_metrics.py` FIRST → then `subspace_metrics.py`.
3. `shift_extraction.py` (+ toy test) → `train_ddxplus_lora.py` (+ shuffled-label flag, YAML).
4. `extract_shifts.py` → `run_subspace_comparison.py` → writeup.
5. Run training only after HF_TOKEN is set.
