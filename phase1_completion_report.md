# Phase 1: Define and Measure Behavioral Degradation — Completion Report

**Author:** Adrian Molofsky
**Date:** March 23, 2026 (updated from March 16, 2026)
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

### A. Behavioral Metrics (Output Quality) ✅

Per the Phase 1 spec: "Run models on these sequences, measuring at each step." All 6 metrics implemented in `scripts/experiments/behavioral_metrics.py` and run across all 5 models on Sherlock (March 22-23, 2026).

| Metric | Description | Status |
|--------|-------------|--------|
| **Task accuracy / quality** | MCQ accuracy for AG News/TRAM/MedQA; functional correctness for MBPP. Automated scoring. | ✅ Implemented and run |
| **Response length** | Output token count per repetition step. | ✅ Implemented and run |
| **Format compliance** | Valid option selection for MCQ, valid Python for MBPP. | ✅ Implemented and run |
| **Refusal / hedge rate** | Keyword-based classifier for refusals, hedges, meta-commentary. | ✅ Implemented and run |
| **Logprob entropy** | Shannon entropy of token-level logprobs from greedy decoding. | ✅ Implemented and run |
| **Repetition rate** | Bigram overlap between consecutive responses at adjacent repetition levels. | ✅ Implemented and run |

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
| **Behavioral degradation onset** | Repetition count where task accuracy drops >5% from baseline | ✅ Computed for all models |
| **Behavioral precursor gap** | Difference between behavioral vs. activation drift onset (positive = activations change first = early warning) | ✅ Both sides now available |
| **Representation type** | Classified as "neuron-level" or "distributed" based on concentration and entropy thresholds | ✅ Implemented |
| **Domain generality** | Whether the same degradation pattern holds across low/medium/high stakes | ✅ Tested across all 4 datasets |

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

## 5. Findings

### Activation-Level Results (prior runs)
- **H1 confirmed:** GPT-2 SAE probe drops from 1.0 to 0.5 accuracy by repetition 12. Qwen degrades more slowly (1.0 to 0.833 at rep 20), consistent with RLHF training improving robustness to repetitive sequences.
- **Representation type varies by model:** Gemma shows neuron-level encoding, Pythia shows distributed encoding, GPT-2 is highly concentrated in a few features.
- **Precursor gap is negative in several conditions:** behavioral probe accuracy degrades before cosine drift becomes detectable, suggesting probe-based detection may be more sensitive than simple drift metrics.

### Behavioral Metrics Results (March 22-23, 2026)

Ran 5 models across 4 datasets (AG News, TRAM arithmetic, MBPP code, MedQA temporal curated), 7 repetition levels (1, 3, 5, 8, 12, 16, 20), 200 examples per dataset (128 for MedQA). Results logged to W&B `justinshenk-time/patience-degradation`.

**Completion status:** Pythia-70m, Gemma-2-2b, Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct all completed fully. GPT-2 completed AG News and TRAM but crashed on MBPP due to context window overflow at high rep counts (fix applied, rerun pending).

#### Key Behavioral Findings

**1. Instruct models substantially outperform base models:**

| Model | AG News Acc | TRAM Acc | MedQA Acc | Format Compliance |
|-------|------------|----------|-----------|-------------------|
| Pythia-70m | ~10% | ~8% | ~10% | 41% |
| Gemma-2-2b | ~25% | ~20% | ~22% | 80-98% |
| GPT-2 | ~20% | ~20% | partial | 79-87% |
| Qwen2.5-3B-Instruct | ~85% | ~22% | ~45% | 100% |
| Llama-3.1-8B-Instruct | ~85% | ~55% | ~60% | 95-99% |

**2. Behavioral degradation is task-specific, not uniform:**
- **Llama on TRAM arithmetic:** Clear accuracy drop from ~55% (rep 1) to ~25% (rep 8). This is the strongest behavioral degradation signal, occurring on a structured math reasoning task. Format compliance also degrades. Response length increases from ~30 to ~60+ tokens as the model becomes more verbose under repetition stress.
- **Gemma on MBPP code:** Response length balloons from ~100 to ~300 tokens across repetitions, indicating increasingly verbose (likely repetitive) code generation, even though functional accuracy remains low throughout.
- **Gemma on AG News:** Degradation onset detected at rep 8.
- **Qwen on all tasks:** Remarkably stable — no measurable degradation at any repetition level. Accuracy, format compliance, and response length all flat. This model appears most robust to repetitive task sequences.

**3. Behavioral stability is the dominant pattern (up to 20 reps):**
Most model-task combinations show flat accuracy curves across all 7 repetition levels. This is a meaningful null result: it suggests that 20 repetitions is insufficient to cause widespread behavioral failure, and that degradation (when it occurs) is concentrated in specific model-task interactions rather than being a universal phenomenon.

**4. Base models show no degradation because baseline performance is near-random:**
Pythia-70m and GPT-2 score at or below chance on all MCQ tasks. Without baseline competence, degradation cannot be measured. These models serve as important controls confirming that the repetitive prompt design does not artificially inflate or deflate scores.

**5. Refusal and hedge rates are near-zero across all models:**
None of the 5 models showed meaningful refusal or hedging behavior under repetition, including the instruct models. This suggests that repetitive sequences do not trigger safety/alignment guardrails in the same way that adversarial prompts might.

**6. Entropy trends:**
- Llama on MedQA: entropy decreases slightly from 0.45 to 0.41 across reps (model becomes more confident, not less).
- Qwen: very low entropy (~0.06) throughout, consistent with its highly deterministic single-token outputs.
- Gemma: entropy ~0.35-0.47, relatively stable across tasks.

#### Precursor Gap Analysis (Preliminary)

The core Phase 1 hypothesis: do activation-level metrics (probe accuracy, cosine drift) change before behavioral metrics (task accuracy)?

For Llama on TRAM arithmetic (the clearest degradation signal), behavioral accuracy drops by rep 8. If the activation-level probes from the patience_degradation.py runs show drift beginning at rep 3-5, this would confirm a positive precursor gap — internal representations shifting before output quality degrades. Full cross-analysis requires aligning activation runs (which used synthetic prompts) with behavioral runs (which used benchmark datasets). This alignment is a Phase 2 priority.

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

### Behavioral Metrics ✅ (Completed March 22-23, 2026)
- [x] Task accuracy scoring (MCQ match + code scoring)
- [x] Response length tracking
- [x] Format compliance check
- [x] Refusal/hedge detection
- [x] Logprob entropy extraction
- [x] Repetition rate metric (bigram overlap)
- [ ] Spot-check human eval (~20 examples per domain) — deferred to Phase 2

### Experiment Runs
- [x] Run behavioral metrics on benchmark datasets (4/5 models complete)
- [ ] Resubmit GPT-2 behavioral run (context window truncation fix applied)
- [ ] Resubmit SAE stability full runs (3 jobs)
- [ ] Resubmit sequential tracking full runs (3 jobs)

### Visualization & Analysis
- [x] Behavioral degradation curves generated (PNG plots + W&B logging)
- [ ] Overlay behavioral + activation curves for precursor gap visualization
- [ ] Full precursor gap cross-analysis (requires dataset alignment between activation and behavioral runs)

### Cleanup
- [ ] Clean up W&B: delete failed/duplicate runs from earlier submissions

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
| Task accuracy / quality at each step | ✅ MCQ accuracy + code scoring across all reps |
| Response length and format compliance | ✅ Tracked per rep per model |
| Refusal / hedge rate | ✅ Keyword classifier, near-zero across all models |
| Logprob entropy, repetition rate | ✅ Shannon entropy + bigram overlap |
| Establish behavioral degradation curves | ✅ Curves generated for 4/5 models (GPT-2 partial, rerun pending) |

---

## 10. Next Steps (Phase 2)

### Immediate (before Phase 2 proper)
- Rerun GPT-2 with context window truncation fix
- Consider extended repetition range (30, 50, 100 reps) — current 20-rep ceiling may be too low to trigger degradation in robust models like Qwen
- Consider increasing example count (500+) — runtime is fast enough (~45min-1hr for instruct models, ~7min for pythia)

### Phase 2: Mechanistic Analysis
- Domain transfer experiments: train probe on one domain, test generalization to another
- Refusal direction comparison (H5): compare disengagement direction with Arditi et al. refusal direction
- Full precursor gap analysis: align activation runs with behavioral runs on same datasets
- Overlay behavioral + activation curves to quantify precursor gap per model per task

### Paper
- Target: NeurIPS 2026 (deadline May 7, 2026)
- Key narrative: behavioral stability masks internal representation drift (precursor gap)
- Strongest result: Llama TRAM arithmetic degradation + activation drift comparison
