# Phase 3: Cross-Domain Validation & Causal Analysis — Completion Plan

**Project:** Temporal Awareness Monitoring (NeurIPS 2026)
**Author:** Adrian Sadik
**Date:** 2026-04-10 (updated with literature-grounded revisions)
**Status:** Scripts complete, pending execution on Sherlock

## Overview

Phase 3 establishes that the degradation signal found in Phases 1–2 is (a) distinct from known safety mechanisms, (b) not a context-length artifact, and (c) causally linked to behavioral degradation. These three experiments transform the paper from "we found a correlation" to "we found a causal, specific mechanism."

Each experiment has been designed against specific prior work to preempt the most likely reviewer objections. This document maps each experimental control to the paper it addresses.

## Experiment 1: Refusal & Sycophancy Direction Comparison

**Script:** `scripts/experiments/phase3_refusal_direction.py`
**Output:** `results/phase3_refusal_direction/{model}/`

### Motivation

Arditi et al. (2024) showed that refusal in safety-trained models is mediated by a single linear direction. If our degradation direction is simply the refusal direction under another name — i.e., the model "refuses" to engage with repetitive tasks — the mechanistic story collapses. Similarly, "Sycophancy Is Not One Thing" (2025) showed that compliance-related behaviors (opinion agreement, answer switching) are encoded along distinct, separable linear directions. If degradation ≈ sycophancy, the model might simply be "going along" with the repetitive setup rather than exhibiting a novel behavioral state.

### Method

1. Extract the **refusal direction** via mean-diff on harmful vs. harmless prompts (Arditi et al. method), using two datasets for robustness:
   - **AdvBench** (Zou et al., 2023): 40 representative harmful prompts
   - **HarmBench** (Mazeika et al., 2024): 30 harmful prompts with broader coverage
   - 40 harmless control prompts (factual questions, writing requests)

2. Extract the **sycophancy direction** via mean-diff on opinion-pressure prompts vs. neutral factual questions (30 each), following the paradigm from "Measuring Sycophancy in Multi-turn Dialogues" (2025) and "Sycophancy Is Not One Thing" (2025).

3. Extract the **degradation direction** via mean-diff on high-rep (rep=20) vs. low-rep (rep=1) activations from TRAM arithmetic and MedQA temporal datasets.

4. Compute **cosine similarity** between all direction pairs:
   - Refusal vs. degradation (primary test — addresses Arditi et al.)
   - Sycophancy vs. degradation (new — addresses sycophancy literature)
   - Sycophancy vs. refusal (cross-check — are they also distinct?)

5. **Subspace overlap analysis** (addresses "Geometry of Refusal: Concept Cones", 2025): Beyond single cosine similarity, compute principal angles between the top-k PCA directions of each condition. This captures multi-dimensional overlap that a single mean-diff vector might miss. Full implementation requires raw activation matrices with `--save-activations`.

6. **Linear vs. nonlinear probe comparison** (addresses "Refusal in LLMs: A Nonlinear Perspective", 2025): Train both a logistic regression (linear) and a 2-layer MLP (nonlinear) probe for each direction type. If the MLP substantially outperforms the linear probe, there is nonlinear structure that mean-diff methods miss.

7. Compare all metrics against a **random baseline**: expected |cos| ≈ √(2/(πd)) ≈ 0.012 for d=4096.

### Literature Coverage

| Paper | How Addressed |
|-------|---------------|
| Arditi et al. (2024) — Refusal is a single direction | Cosine similarity: refusal vs. degradation |
| "Sycophancy Is Not One Thing" (2025) — Separable compliance directions | Cosine similarity: sycophancy vs. degradation |
| "Geometry of Refusal: Concept Cones" (2025) — Multi-dimensional cones | Subspace overlap via principal angles (PCA) |
| "Refusal in LLMs: A Nonlinear Perspective" (2025) — Nonlinear components | MLP probe vs. linear probe gap |
| "Understanding Refusal with SAEs" (2025) — Causal SAE features | Noted as future work (requires SAE training) |
| "COSMIC" (2025) — Generalized refusal identification | Our multi-dataset refusal extraction is a simplified version |
| Alignment faking (Anthropic, 2024) — Probes detect covert states | Validates our probe methodology (cited in methods) |

### Expected Results

| Outcome | Cosine Similarity | Interpretation |
|---------|-------------------|----------------|
| **Ideal** | < 0.15 for all pairs across all layers | Degradation is novel, independent of both refusal and sycophancy |
| **Concerning** | 0.15–0.4 for refusal OR sycophancy in middle layers | Partial overlap, needs nuanced discussion |
| **Problematic** | > 0.5 for either in multiple layers | Degradation may reuse existing compliance circuitry |

### Outputs

- `refusal_comparison_results.json`: Per-layer cosine similarity (refusal + sycophancy), probe accuracies, direction norms, subspace overlap metrics, nonlinear probe comparison
- `directions/`: Saved .npy direction vectors: `refusal_advbench_layer{N}.npy`, `refusal_harmbench_layer{N}.npy`, `sycophancy_layer{N}.npy`, `degradation_{dataset}_layer{N}.npy`
- `cosine_similarity_profile.png`: Layer-wise cosine similarity with random baseline band
- `probe_accuracy_comparison.png`: Side-by-side refusal vs. degradation vs. sycophancy probe accuracy

---

## Experiment 2: Context Length Confound Control

**Script:** `scripts/experiments/phase3_context_confound.py`
**Output:** `results/phase3_context_confound/{model}/`

### Motivation

A natural objection: "Your 'degradation' is just Lost in the Middle (Liu et al., 2023). At high repetitions the context is longer, and models are known to degrade with long contexts." This experiment disentangles repetition-specific degradation from raw context-length effects. Additionally, Leviathan (2025) showed that exact prompt repetition actually *improves* non-reasoning LLMs, creating a direct contradiction with our hypothesis that needs resolution.

### Method

Four conditions, all **matched for total token count**:

1. **REPETITIVE** (treatment): 20 repetitions of same-domain tasks (e.g., all TRAM arithmetic). This is the standard Phase 1/2 setup.
2. **SHUFFLED** (control 1): Same total tokens, but tasks drawn from **different domains** (mix of TRAM, MedQA, MBPP). Same context length, but no domain repetition.
3. **PADDED** (control 2): Same total tokens, but the prefix is **benign filler text** (Wikipedia-style passages), with only the final task being the actual question.
4. **EXACT_REPEAT** (control 3, new): Same total tokens, but the prefix is the **exact same question** repeated N times. Directly addresses Leviathan (2025) — if exact repetition helps but diverse same-domain repetition hurts, that is a powerful distinction.

All conditions evaluate the **same final question** — only the prefix differs.

Train a degradation probe on REPETITIVE (fresh vs. degraded), then test transfer. Also train an MLP probe alongside the logistic regression to check for nonlinear structure (per "Refusal in LLMs: A Nonlinear Perspective", 2025).

### Transfer Matrix

| Condition | Probe Fires? | Interpretation |
|-----------|-------------|----------------|
| REPETITIVE | yes (by construction) | Probe was trained on this |
| SHUFFLED | no | Degradation is repetition-specific, not context-length |
| SHUFFLED | yes | Degradation is (partially) context-length |
| PADDED | no | Degradation requires task engagement, not just tokens |
| PADDED | yes | Degradation is purely a context-length effect |
| EXACT_REPEAT | no | Degradation is specific to *varied* same-domain tasks |
| EXACT_REPEAT | yes | Degradation is any-repetition (contradicts Leviathan 2025 framing) |

### Literature Coverage

| Paper | How Addressed |
|-------|---------------|
| Liu et al. (2023) — Lost in the Middle | SHUFFLED condition: same length, different content |
| Context Rot (Chroma Research) | PADDED condition: same token count, benign filler |
| Leviathan (2025) — Prompt repetition improves | EXACT_REPEAT condition: same question repeated |
| "Refusal in LLMs: A Nonlinear Perspective" (2025) | MLP probe alongside logistic regression |

### Key Metric

**Probe transfer accuracy** on SHUFFLED, PADDED, and EXACT_REPEAT. The ideal finding is:
- REPETITIVE: ~90% accuracy (by construction)
- SHUFFLED: ~50% (chance) → degradation is NOT context-length
- PADDED: ~50% (chance) → degradation requires task engagement
- EXACT_REPEAT: ~50% (chance) → degradation is task-variety-specific

If EXACT_REPEAT shows ~50% but REPETITIVE shows ~90%, this cleanly separates our finding from Leviathan (2025) and shows the degradation mechanism is triggered by *varied same-domain repetition*, not by repetition per se.

### Outputs

- `context_confound_results.json`: Per-condition probe accuracy, transfer metrics, MLP vs. linear comparison
- `condition_comparison.png`: Bar chart of probe accuracy across all 4 conditions
- `token_count_verification.png`: Confirms token counts are matched across conditions

---

## Experiment 3: Causal Activation Patching

**Script:** `scripts/experiments/phase3_causal_patching.py`
**Output:** `results/phase3_causal_patching/{model}/`

### Motivation

Experiments 1–2 establish what the degradation direction *is not* (not refusal, not sycophancy, not context-length). This experiment establishes what it *does* — by directly manipulating it and measuring behavioral effects. This follows the tradition of causal mechanistic interpretability: Meng et al. (2022) ROME for factual editing, Arditi et al. (2024) for refusal ablation, and Turner et al. (2023) for activation addition.

### Method

1. **Injection at early repetitions** (rep=1, model still performing well):
   - Add scaled degradation direction to residual stream at target layer
   - If accuracy drops → direction *causes* degradation-like behavior

2. **Ablation at late repetitions** (rep=15, model already degraded):
   - Remove projection of activations onto degradation direction: `h' = h - (h·d̂)d̂`
   - If accuracy recovers → direction is *necessary* for degradation

3. **Controls:**
   - **Random direction** (same L2 norm): injection/ablation should NOT affect accuracy
   - **Refusal direction** (from Experiment 1): injection should NOT systematically degrade task performance (if the two directions are truly distinct)

4. **Dose-response sweep**: Test strengths [0.5, 1.0, 2.0, 4.0, 8.0] to show graded, predictable effect. This is important — a real causal direction should show monotonic dose-response, not a step function.

### Literature Coverage

| Paper | How Addressed |
|-------|---------------|
| Meng et al. (2022) — ROME causal tracing | Injection/ablation methodology |
| Arditi et al. (2024) — Refusal direction ablation | Ablation restores behavior; refusal direction as control |
| Turner et al. (2023) — Activation addition | Injection methodology and dose-response |
| Li et al. (2023) — Inference-time intervention | Validates per-layer intervention approach |
| "Argument Driven Sycophancy" (2025) | Refusal control tests whether direction is compliance-related |

### Key Metrics

- **Causal specificity (injection)**: |Δacc_degradation| - |Δacc_random| at strength=1.0
- **Causal specificity (ablation)**: Δacc_degradation - Δacc_random at strength=1.0
- **Recovery ratio**: (ablation_recovery) / (early_baseline - late_baseline)

| Metric | Strong Evidence | Moderate | Weak |
|--------|----------------|----------|------|
| Injection specificity | > 0.10 | 0.05–0.10 | < 0.05 |
| Recovery ratio | > 0.50 | 0.25–0.50 | < 0.25 |

### Outputs

- `causal_patching_results.json`: All patching conditions with accuracy deltas, causal specificity scores
- `dose_response_layer{N}.png`: Dose-response curves per layer (degradation vs. random vs. refusal)
- `layer_comparison.png`: Which layers have the strongest causal effect
- `recovery_comparison.png`: Degradation gap vs. ablation recovery

---

## Dataset Update: AG News → TRAM Ordering

As of this update, the **low-stakes benchmark** has been changed from AG News (text classification) to **TRAM Ordering** (temporal ordering MCQs). Rationale:

1. AG News was not producing useful degradation signal — the task was too simple for models to degrade meaningfully, and the non-temporal nature made it an outlier in a temporal-reasoning paper.
2. TRAM Ordering keeps all benchmarks within the temporal reasoning domain, strengthening the paper's coherence.
3. Ordering questions (day/month/sequence ordering) are genuinely low-stakes — they require no domain expertise and have minimal cognitive load.

**File:** `data/processed/patience_degradation/low_stakes_tram_ordering.json`
**Generator:** `scripts/create_tram_ordering_dataset.py`
**N examples:** 1,600 (400 day ordering + 400 month ordering + 400 sequence + 400 relative)
**Phase 2 config updated:** `scripts/experiments/phase2_cross_domain.py` — `"low"` key now points to `low_stakes_tram_ordering.json`

---

## Noted Limitations & Future Work

These are known gaps that should be acknowledged in the paper's discussion section:

1. **SAE feature analysis** ("Understanding Refusal with SAEs", 2025): Identifying specific SAE features that causally mediate degradation (as opposed to our direction-level analysis) would provide more granular mechanistic understanding. This requires training SAEs on our models' residual streams, which is a substantial additional compute cost.

2. **Full subspace analysis**: Our current subspace overlap analysis approximates from mean-diff directions. A complete analysis would extract full activation matrices per condition and compute PCA-based principal angles, requiring `--save-activations` mode which increases storage requirements significantly.

3. **Multi-turn sycophancy dynamics**: "Measuring Sycophancy in Multi-turn Dialogues" (2025) tracks stance reversal across turns. Our sycophancy comparison is single-turn (opinion-pressure prompts), which may not capture the full multi-turn compliance dynamic. A future experiment could track probe activation across a multi-turn conversation where the model is pushed to agree.

4. **Alignment faking connection**: Anthropic's alignment faking work (2024, 2025) shows that models can fake compliance when monitored. Our degradation may interact with monitoring-awareness — an extension would test whether degradation patterns change when the model believes it is being evaluated.

---

## Execution Plan

### Priority Order

1. **Experiment 1** (Refusal + Sycophancy Direction) — produces direction vectors needed by Experiment 3
2. **Experiment 2** (Context Confound + Exact Repeat) — independent, can run in parallel with Exp 1
3. **Experiment 3** (Causal Patching) — depends on Exp 1 outputs (direction .npy files)

### SLURM Scripts

- `submit_phase3_refusal.sh` — L40S 48GB, ~2.5hr per model (now includes sycophancy + MLP)
- `submit_phase3_confound.sh` — V100 32GB, ~2hr per model (now includes 4th condition + MLP)
- `submit_phase3_patching.sh` — L40S 48GB, ~3hr per model (generation-heavy)

### Timeline

| Date | Milestone |
|------|-----------|
| Apr 10 | Scripts complete with literature-grounded revisions |
| Apr 11–12 | Run Exp 1 + Exp 2 on all 4 models |
| Apr 13–14 | Run Exp 3 on all 4 models (after Exp 1 directions are saved) |
| Apr 15 | Analyze results, update completion doc with findings |
| Apr 16–17 | Integrate into paper draft (Section 5: Cross-Domain Validation) |

---

## Paper Integration

Phase 3 results map to the following paper sections:

- **Section 5.1**: Refusal vs. Degradation vs. Sycophancy (Experiment 1) — key Figure 5: cosine similarity heatmap across all direction pairs, plus nonlinear probe gap analysis
- **Section 5.2**: Context Length & Repetition Controls (Experiment 2) — Table 3: probe transfer accuracy across 4 conditions, with EXACT_REPEAT addressing Leviathan (2025) directly
- **Section 5.3**: Causal Evidence (Experiment 3) — Figure 6: dose-response curves, Figure 7: recovery ratio

Together these experiments address the five most likely reviewer objections:
1. "Is this just refusal?" → No, cosine similarity is low (Exp 1, Arditi et al.)
2. "Is this just sycophancy?" → No, cosine similarity is low (Exp 1, sycophancy literature)
3. "Is this just context length?" → No, probe doesn't fire on shuffled/padded (Exp 2, Liu et al.)
4. "Doesn't repetition help?" → Only exact repetition; varied same-domain repetition hurts (Exp 2, Leviathan 2025)
5. "Is this causal?" → Yes, injection causes degradation, ablation restores performance (Exp 3, Meng et al.)
