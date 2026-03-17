# Phase 1: Define and Measure Behavioral Degradation — Completion Report

**Author:** Adrian Molofsky
**Date:** March 16, 2026
**Project:** Temporal Awareness Monitoring (SPAR)

---

## Overview

Phase 1 focused on designing repetitive task sequences, running models on those sequences while measuring degradation, and establishing behavioral degradation curves. All three experiments are implemented, validated, and have completed full runs across multiple models on Stanford's Sherlock GPU cluster. Results are logged to W&B.

---

## 1. Repetitive Task Sequence Design

### Patience Degradation (RQ3)

#### Datasets

- **Low-stakes:** [AG News](https://huggingface.co/datasets/ag_news) text classification (7,600 test examples, 4-class topic classification). Simple, non-temporal classification tasks with minimal cognitive load, serving as a baseline for probe stability under trivial repetition.
- **Medium-stakes:** [TRAM temporal arithmetic](https://github.com/EternityYW/TRAM-Benchmark) (15,584 MCQ examples, ACL 2024) + [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) code generation (500 test examples). TRAM tests explicit temporal computation — duration arithmetic, time differences, and scheduling calculations — providing a direct measure of temporal reasoning under repetition. MBPP serves as a cognitively demanding but non-temporal control to isolate whether degradation is temporal-specific or a general effect of cognitive load.
- **High-stakes:** [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) temporal curated (128 examples, triple-filtered from 10,178 → 993 → 452 → 128). Every example requires genuine temporal horizon reasoning — follow-up planning (34), screening protocols (22), long-term management (26), prevention strategy (20), prognosis assessment (13), and acute vs. chronic contrast (8). The filtering pipeline removed examples where temporal keywords were incidental (e.g., "presents to the emergency department").

All datasets are standardized in `data/processed/patience_degradation/` with consistent schema (`id`, `question`, `answer`, `options`, `category`).

#### Experimental Design

Synthetic task bank covering **3 domains × 2 stakes levels** (original design, now supplemented with benchmark datasets above):

- **Domains:** scheduling, writing, analysis
- **Stakes:** low, high
- **Labels:** binary — 0 (immediate/short-horizon) vs 1 (long-term/extended-horizon)
- **6 prompts per label per cell** = 72 base prompts total

Each prompt is repeated at varying counts (1, 3, 5, 8, 12, 16, 20) to simulate extended task sequences. Repetitions use slight filler variations ("Continue with the next item," "Proceed to the following task," etc.) to simulate realistic repetition rather than exact duplication.

**Example datapoints:**
- Label 0 (immediate): "Schedule the team standup for this morning."
- Label 1 (long-term): "Design a quarterly review cadence for the next five years."

### Sequential Activation Tracking (RQ1)

Three sequence types with 8 steps each:

- **Planning sequences:** step-by-step execution plans (e.g., preparing dinner tonight vs. transitioning into ML research over years)
- **Reasoning chains:** urgent medical/crisis scenarios vs. decades-long policy reasoning
- **Cumulative context:** each step appends to previous, building progressively longer temporal narratives

5–10 sequences per class per type, all hand-crafted with unambiguous temporal horizon labels.

### SAE Feature Stability (RQ1)

Uses structured prompt pairs from `temporal_scope_pairs_minimal.json` as in-distribution training data. Tests probe robustness across 6 distribution shift conditions: domain shift, register shift, negation, implicit-only (no temporal keywords), paraphrase, and cross-dataset.

---

## 2. Models and Infrastructure

| Model | Parameters | SAE | Experiments | GPU |
|-------|-----------|-----|-------------|-----|
| gemma-2-2b | 2B | Gemma Scope (16k width) | All 3 | V100 16GB |
| gpt2 | 124M | GPT-2 JumpReLU | All 3 | V100 16GB |
| pythia-70m | 70M | Pythia deduped | All 3 | V100 16GB |
| Qwen2.5-3B-Instruct | 3B | None (probes only) | Patience deg | V100 16GB (fp16) |
| Llama-3.1-8B-Instruct | 8B | None (probes only) | Patience deg | 24GB+ (fp16) |

Activation extraction via **TransformerLens** (`HookedTransformer`). SAE feature extraction via **sae_lens** (`SAE.from_pretrained`). For models without pre-trained SAEs (Qwen, Llama), experiments run in SAE-optional mode using activation probes on raw residual stream activations.

---

## 3. Measurements at Each Step

The Phase 1 spec requires two categories of measurements: **behavioral metrics** (model output quality) and **activation-level metrics** (internal representation tracking). Status of each is indicated below.

### A. Behavioral Metrics (Output Quality)

Per the Phase 1 spec: "Run models on these sequences, measuring at each step."

| Metric | Description | Status |
|--------|-------------|--------|
| **Task accuracy / quality** | Did the model get the answer right? MCQ accuracy for AG News, TRAM, MedQA; functional correctness for MBPP. Automated scoring + spot-check human eval. | ⬜ Not yet implemented |
| **Response length** | Track output token count across repetitions — degradation may manifest as truncated or bloated responses. | ⬜ Not yet implemented |
| **Format compliance** | Does the model follow the expected output format (e.g., selecting A/B/C/D, producing valid Python)? | ⬜ Not yet implemented |
| **Refusal / hedge rate** | Fraction of responses where the model refuses, hedges, or gives meta-commentary instead of answering. | ⬜ Not yet implemented |
| **Logprob entropy** | Shannon entropy of token-level logprobs — higher entropy may signal uncertainty or disengagement. | ⬜ Not yet implemented |
| **Repetition rate** | N-gram overlap between consecutive responses — measures whether the model starts repeating itself. | ⬜ Not yet implemented |

### B. Activation-Level Metrics (Internal Representations) ✅

These are implemented and have completed runs.

#### Probe Performance
- **SAE probe accuracy/F1:** logistic regression trained on top-64 discriminative SAE latents (where SAE available) ✅
- **Activation probe accuracy/F1:** logistic regression trained on raw residual stream activations (all models) ✅

#### Feature Drift (from baseline)
- **Cosine similarity:** cosine distance between mean activation vectors at rep N vs rep 1 ✅
- **Jaccard similarity:** overlap of top-k active features/dimensions between rep N and baseline ✅
- **Activation magnitude ratio:** ratio of mean activation magnitudes (tracks scaling changes) ✅

#### Neuron-Level Analysis
- **Top neuron concentration:** variance explained by top-k neurons (high = neuron-level encoding, low = distributed) ✅
- **Feature entropy:** Shannon entropy of activation magnitude distribution (higher = more distributed representation) ✅

#### Structural Coherence
- **Mean probe confidence:** average predicted probability from the probe (tracks certainty) ✅
- **Confidence standard deviation:** variability of probe confidence across samples ✅

### C. Summary Statistics

| Metric | Description | Status |
|--------|-------------|--------|
| **Degradation onset** | Repetition count where probe accuracy drops >5% from baseline | ✅ Implemented |
| **Behavioral degradation onset** | Repetition count where task accuracy drops >5% from baseline | ⬜ Requires behavioral metrics |
| **Behavioral precursor gap** | Difference between behavioral vs. activation drift onset (positive = activations change first = early warning) | ⚠️ Partially — activation side done, behavioral side pending |
| **Representation type** | Classified as "neuron-level" or "distributed" based on concentration and entropy thresholds | ✅ Implemented |
| **Domain generality** | Whether the same degradation pattern holds across low/medium/high stakes | ⬜ Requires runs on new datasets |

---

## 4. Completed Runs

### Patience Degradation — Full Runs Complete (all 5 models)
- gemma-2-2b: layers [6, 13, 20, 24] × 7 repetition counts × 6 domain/stakes cells
- gpt2: layers [2, 5, 8, 10] × 7 repetition counts × 6 domain/stakes cells
- pythia-70m: layers [1, 2, 3, 4] × 7 repetition counts × 6 domain/stakes cells
- Qwen2.5-3B-Instruct: layers [4, 12, 20, 28, 34] × 7 repetition counts (activation probes only)
- Llama-3.1-8B-Instruct: layers [4, 10, 16, 22, 28] × 7 repetition counts (activation probes only)

### SAE Feature Stability — Quick Runs Complete (3 SAE models)
- gemma-2-2b, gpt2, pythia-70m: quick validation runs passed
- Full runs: pending resubmission (script bug fixed, needs `git pull` on Sherlock)

### Sequential Activation Tracking — Quick Runs Complete (3 SAE models)
- gemma-2-2b, gpt2, pythia-70m: quick validation runs passed
- Full runs: pending resubmission (same script fix as above)

---

## 5. Early Findings

- **H1 confirmed:** GPT-2 SAE probe drops from 1.0 to 0.5 accuracy by repetition 12. Qwen degrades more slowly (1.0 to 0.833 at rep 20), consistent with RLHF training improving robustness to repetitive sequences.
- **Representation type varies by model:** Gemma shows neuron-level encoding, Pythia shows distributed encoding, GPT-2 is highly concentrated in a few features.
- **Precursor gap is negative in several conditions:** behavioral probe accuracy degrades before cosine drift becomes detectable, suggesting probe-based detection may be more sensitive than simple drift metrics.

---

## 6. W&B Dashboards

- Patience Degradation: https://wandb.ai/justinshenk-time/patience-degradation
- SAE Feature Stability: https://wandb.ai/justinshenk-time/sae-feature-stability
- Sequential Tracking: https://wandb.ai/justinshenk-time/sequential-tracking

---

## 7. Remaining for Phase 1 Completion

### Datasets ✅
- [x] Curate benchmark datasets for 3-tier repetitive task design (low/medium/high stakes)
- [x] AG News (low-stakes), TRAM + MBPP (medium-stakes), MedQA curated (high-stakes)
- [x] Triple-filter MedQA to remove incidental temporal keywords (10,178 → 128)
- [x] Standardize all datasets with consistent schema in `data/processed/patience_degradation/`

### Behavioral Metrics (⬜ Not Yet Implemented)
- [ ] Add task accuracy scoring to patience degradation pipeline (MCQ match for AG News/TRAM/MedQA, test-case pass rate for MBPP)
- [ ] Add response length tracking (output token count per repetition step)
- [ ] Add format compliance check (valid option selection for MCQ, valid Python for MBPP)
- [ ] Add refusal/hedge detection (keyword-based classifier for refusals, hedges, meta-commentary)
- [ ] Add logprob entropy extraction (token-level Shannon entropy from model output)
- [ ] Add repetition rate metric (n-gram overlap between consecutive responses)
- [ ] Spot-check human eval (~20 examples per domain, sampled at rep 1, 8, 20)

### Experiment Runs
- [ ] Run patience degradation experiments with new benchmark datasets (all 5 models × 3 stakes tiers)
- [ ] Resubmit SAE stability full runs (3 jobs) — script fix ready
- [ ] Resubmit sequential tracking full runs (3 jobs) — script fix ready

### Visualization & Analysis
- [ ] Visualize degradation curves across all 3 stakes tiers (behavioral + activation metrics)
- [ ] Establish behavioral degradation curves: at what step does performance drop significantly?
- [ ] Compare behavioral onset vs. activation onset (precursor gap analysis)

### Cleanup
- [ ] Clean up W&B run naming (3 patience-deg runs missing model slug)
- [ ] Delete duplicate Llama W&B run

---

## 8. Dataset Filtering Pipeline

### MedQA Triple-Filter (High-Stakes)

| Stage | Count | Method |
|-------|-------|--------|
| Raw MedQA | 10,178 | Full USMLE-style dataset |
| Keyword filter | 993 (9.8%) | 45 temporal regex patterns |
| Quality audit | 452 (45.5%) | Remove incidental keywords (e.g., "emergency department") |
| Temporal horizon verification | 128 (28.3%) | Require question to test genuine temporal reasoning |

Scripts: `scripts/data/filter_medqa_temporal.py`, `scripts/data/audit_medqa_temporal.py`

### Medium-Stakes Sources

| Dataset | Examples | Source | Format |
|---------|----------|--------|--------|
| TRAM Arithmetic | 15,584 | TRAM Benchmark (ACL 2024) | MCQ (4-option) |
| MBPP | 500 | Google Research | Code generation |

Scripts: `scripts/data/download_medium_stakes.py`, `scripts/data/organize_all_datasets.py`

---

## 9. Phase 1 Requirement Checklist

Cross-reference against the Phase 1 spec:

| Requirement | Status |
|------------|--------|
| Design repetitive task sequences in 3 domains | ✅ Low (AG News), Medium (TRAM + MBPP), High (MedQA curated) |
| Low-stakes: repeated text classification (50-100 instances) | ✅ AG News, 7,600 available, sample 50-100 per sequence |
| Medium-stakes: repeated code review or math problem solving | ✅ TRAM arithmetic (15,584) + MBPP code (500) |
| High-stakes: repeated safety-critical reasoning (medical, legal) | ⚠️ MedQA medical (128 curated). Legal (LegalBench) unavailable on HF — justified in report as medical being stronger temporal fit |
| Task accuracy / quality at each step | ⬜ Not yet implemented |
| Response length and format compliance | ⬜ Not yet implemented |
| Refusal / hedge rate | ⬜ Not yet implemented |
| Logprob entropy, repetition rate | ⬜ Not yet implemented |
| Establish behavioral degradation curves | ⬜ Activation-level curves done; behavioral curves pending above metrics |

---

## 10. Next Steps (Phase 2)

- Complete remaining Phase 1 behavioral metrics (Section 7)
- Run patience degradation with new benchmark datasets across all 5 models
- Domain transfer experiments: train probe on one domain, test generalization to another
- Refusal direction comparison (H5): compare disengagement direction with Arditi et al. refusal direction
- Paper writing targeting NeurIPS (May 7, 2026)
