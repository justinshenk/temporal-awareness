# Gap Analysis: Temporal Awareness Monitoring → Top NeurIPS Paper

**Date:** 2026-04-10
**Deadline:** May 7, 2026 (27 days)
**Author:** Adrian Sadik
**Purpose:** Identify what's missing, what's achievable, and what will differentiate this from an ordinary submission

## Current State Summary

| Phase | Status | Key Finding |
|-------|--------|-------------|
| Phase 1 | Complete | Llama-3.1-8B-Instruct shows clear degradation by rep 8 on TRAM; Qwen2.5-3B remarkably stable; base models (GPT-2, Pythia) too weak for meaningful baseline |
| Phase 2 | Scripts complete, partial runs | Cross-domain probe transfer, precursor gap detection framework built |
| Phase 3 | Scripts complete, pending execution | Refusal/sycophancy comparison, context confound control, causal patching |
| Phase 4 | **Not implemented** | Intervention experiments (context refresh, steering, prompt restructuring) |

## The 7 Gaps That Separate This From a Top Paper

### Gap 1: Phase 4 Intervention Experiments — CRITICAL
**Status:** Zero implementation. No scripts, no framework.
**Why it matters:** The research plan promises "at least one intervention shown to delay or reverse degradation." Without this, the paper is purely observational — "we found a thing" vs. "we found a thing and showed how to fix it." Reviewers at NeurIPS strongly favor papers with practical implications.

**What's needed:**
Three intervention strategies, tested when the degradation probe crosses threshold:

1. **Activation steering** (highest priority — most mechanistically interesting):
   - Use the degradation direction from Phase 3 Exp 1 to steer activations *away* from degradation during inference
   - This is the causal patching experiment in reverse: instead of post-hoc ablation, do online steering
   - Infrastructure exists: `src/inference/interventions/intervention_factory.py` has `steering()` already
   - Key metric: does continuous anti-degradation steering maintain accuracy past the degradation cliff?

2. **Context refresh** (most practically relevant):
   - At the degradation threshold, summarize the conversation so far and restart with a compressed context
   - Tests whether degradation is carried in the hidden state or the context tokens
   - Simple to implement: just truncate + summarize at step N, measure if accuracy recovers

3. **Prompt restructuring** (simplest):
   - At the threshold, reframe the task (change instructions, add "this is important", etc.)
   - Tests whether degradation is about the task representation or attention fatigue
   - Weakest mechanistically but most practically applicable

**Effort estimate:** 3-4 days for scripts + 2 days for execution.
**ROI:** Very high. Without Phase 4, the paper has no "so what" for practitioners.

### Gap 2: Cross-Model Direction Transfer — HIGH IMPACT
**Status:** Phase 2 does cross-domain transfer within a model. No cross-model transfer exists.
**Why it matters:** If the degradation direction is universal across architectures (Llama, Qwen, DeepSeek), that's a finding about LLMs in general, not just one model family. If it's model-specific, that's also interesting — it implies degradation is an artifact of specific training procedures.

**What's needed:**
- Extract degradation direction from Model A (e.g., Llama)
- Apply it to Model B (e.g., Qwen) — does the probe still detect degradation?
- Requires aligned dimensionality or CKA-style representational similarity
- For models with same d_model (4096 for Llama/Qwen), direct cosine similarity between their degradation directions is meaningful
- For DeepSeek (d_model=3584), use CKA or train a projection

**Effort estimate:** 1 day to add to Phase 2, runs with existing compute.
**ROI:** Very high. "Degradation is a universal LLM phenomenon" is a much stronger paper than "Llama does this."

### Gap 3: Base Model vs. Instruction-Tuned Comparison — HIGH IMPACT
**Status:** Phase 1 tested GPT-2 and Pythia (base), but they were too small to be meaningful. Phase 2 uses 4 instruction-tuned models only.
**Why it matters:** This directly tests whether RLHF/instruction-tuning *creates* the degradation mechanism. If base Llama-3.1-8B shows no degradation but Llama-3.1-8B-Instruct does, that's a headline result: "alignment training introduces a disengagement mechanism."

**What's needed:**
- Add `meta-llama/Llama-3.1-8B` (base, not Instruct) to the model configs
- Run Phase 1 behavioral metrics + Phase 2 activation analysis
- Compare degradation curves and probe accuracy between base and instruct

**Effort estimate:** 1 day to configure, 1 day to run on Sherlock.
**ROI:** Extremely high. This is potentially the most publishable finding in the entire paper. If RLHF creates the degradation direction, that connects to the alignment faking literature and has immediate safety implications.

### Gap 4: Activation Trajectory Geometry — NOVEL CONTRIBUTION
**Status:** Infrastructure exists (`src/common/token_trajectory.py`, `src/common/math/trajectory_metrics.py`) but is NOT used for degradation trajectory analysis.
**Why it matters:** Current approach treats degradation as binary (fresh vs. degraded). But degradation is a *continuous process* — the activations move through a trajectory in activation space as repetitions increase. Characterizing this trajectory (linear? curved? sudden phase transition?) would be genuinely novel.

**What's needed:**
- Extract activations at every repetition count (1, 3, 5, 8, 12, 16, 20), not just endpoints
- Project onto top-2 PCA components and visualize the trajectory
- Compute trajectory metrics: velocity (how fast activations move per rep), curvature (is the path straight?), and acceleration (does movement speed up before behavioral degradation?)
- Test whether the trajectory is predictable: given activations at reps 1-5, can you forecast where they'll be at rep 20?

**Key finding to aim for:** "Activation trajectories follow a characteristic two-phase pattern: a slow drift period followed by a rapid transition through a critical manifold, predictable 5+ steps before behavioral degradation appears." This would directly validate H2.

**Effort estimate:** 2-3 days for analysis script + visualization.
**ROI:** Very high for novelty. This is the kind of finding that gets a paper remembered.

### Gap 5: Early Detection Quantification — COMPLETES H2
**Status:** Phase 2 has "precursor gap detection" but it's not properly quantified. H2 states "Linear probes can detect activation drift toward degradation ≥5 steps before behavioral metrics show it" but there's no dedicated experiment proving this.
**Why it matters:** Early detection is the most practically useful finding. If you can predict degradation before it happens, you can intervene proactively.

**What's needed:**
- For each model × dataset, align behavioral accuracy curve with probe confidence curve
- Compute the "precursor gap": at what repetition does probe confidence cross 0.7 vs. when does accuracy drop >5%?
- Report this gap in steps (e.g., "probe detects degradation 4.2 ± 1.3 steps before accuracy drops")
- This can be computed from existing Phase 2 results — no new model runs needed

**Effort estimate:** 1 day for analysis script.
**ROI:** High. This is a deliverable promised in the plan and a selling point in the abstract.

### Gap 6: Attention Pattern Analysis — MECHANISTIC DEPTH
**Status:** Zero implementation. Only residual stream activations are analyzed.
**Why it matters:** A reviewer will ask: "You found a direction in the residual stream, but what *circuit* produces it? Which attention heads change behavior?" Without this, the mechanistic story is incomplete — you've found *where* in the residual stream the signal lives but not *how* it gets there.

**What's needed:**
- Extract attention patterns (not just residual stream) at each layer during degradation
- Compute attention entropy per head across repetitions
- Identify "degradation heads" — heads whose attention patterns change most between early and late reps
- Check if degradation heads attend less to task-relevant tokens (connects to Lost in the Middle)
- Potentially: path patching through specific heads to test circuit-level causality

**Effort estimate:** 3-4 days for full implementation.
**ROI:** Medium-high. Deepens the mechanistic story but isn't strictly necessary for the core claims.

### Gap 7: Connection to In-Context Learning Circuits — THEORETICAL DEPTH
**Status:** Not addressed anywhere.
**Why it matters:** Degradation under repetitive tasks could be related to ICL circuits being "saturated." There's a growing literature on how ICL works mechanistically (Olsson et al. 2022 "induction heads", Von Oswald et al. 2023 "transformers learn in-context by gradient descent"). If degradation correlates with specific ICL circuit behaviors (e.g., induction head attention patterns flatten under repetition), that connects your finding to foundational mechanistic work.

**What's needed:**
- Identify induction heads in your models
- Track their attention patterns across repetitions
- Test whether degradation onset correlates with induction head attention entropy increasing

**Effort estimate:** 2-3 days.
**ROI:** Medium. Strengthens the theoretical contribution but is secondary to the empirical findings.

## Model Selection Assessment

**Current 4 models are well-chosen for diversity:**
| Model | Why Included | What It Tests |
|-------|-------------|---------------|
| Llama-3.1-8B-Instruct | Primary case, well-studied | Standard instruction-tuned behavior |
| Qwen3-8B | Cross-family comparison | Generality beyond Meta models |
| Qwen3-30B-A3B | MoE architecture | Does sparse activation change degradation? |
| DeepSeek-R1-Distill-Qwen-7B | Reasoning-distilled | Does reasoning training affect degradation? |

**What's missing:**
| Model | Why Needed | Priority |
|-------|-----------|----------|
| Llama-3.1-8B (base) | Tests whether RLHF creates degradation | **CRITICAL** |
| Llama-3.1-70B-Instruct or Qwen3-32B | Tests scaling: do larger models degrade less? | Medium |
| A DPO-trained model vs RLHF | Tests whether training objective matters | Low |

**Recommendation:** Add Llama-3.1-8B (base) as a 5th model. This is the single highest-ROI model addition. You don't need more models — 5 with clear architectural diversity is sufficient for NeurIPS. The base vs. instruct comparison within the same architecture is the cleanest possible experiment.

## Priority-Ordered Action Plan (27 Days)

### Week 1 (Apr 10-16): Execute Phase 3 + Start Critical Gaps

| Day | Task | Notes |
|-----|------|-------|
| Apr 11 | Submit Phase 3 Exp 1 + Exp 2 on Sherlock | Parallel, ~4hr each |
| Apr 11 | Add Llama-3.1-8B (base) to model configs | 1hr config work |
| Apr 12 | Submit Phase 3 Exp 3 (needs Exp 1 directions) | After Exp 1 completes |
| Apr 12 | Submit base Llama Phase 1 behavioral run | Overnight job |
| Apr 13 | Implement Phase 4 intervention scripts | Steering + context refresh |
| Apr 14 | Implement early detection quantification | Analysis of Phase 2 data |
| Apr 14 | Implement cross-model direction transfer | Add to Phase 2 framework |
| Apr 15 | Submit Phase 4 intervention experiments | After Phase 3 directions |
| Apr 16 | Implement trajectory geometry analysis | Visualization + metrics |

### Week 2 (Apr 17-23): Execute Remaining + Attention Analysis

| Day | Task | Notes |
|-----|------|-------|
| Apr 17 | Analyze Phase 3 results | Direction comparison, confound control |
| Apr 18 | Analyze Phase 4 intervention results | Steering effectiveness |
| Apr 19 | Implement attention pattern analysis (optional) | If time permits |
| Apr 20-21 | Compile all results, generate figures | Key paper figures |
| Apr 22-23 | Write results section draft | Phase 2-4 results |

### Week 3 (Apr 24-30): Paper Writing

| Day | Task | Notes |
|-----|------|-------|
| Apr 24-25 | Write introduction, related work, methods | Core sections |
| Apr 26-27 | Write discussion, conclusion | Implications, limitations |
| Apr 28 | Internal review, iterate on figures | Quality pass |
| Apr 29-30 | Full paper revision | Polish |

### Week 4 (May 1-7): Final Polish + Submit

| Day | Task | Notes |
|-----|------|-------|
| May 1-3 | Advisor review + revisions | Incorporate feedback |
| May 4-5 | Supplementary materials, code release prep | Appendix, reproducibility |
| May 6 | Final proofread | Formatting, references |
| May 7 | Submit | Deadline |

## What Would Make This a "Best Paper" Candidate

The findings that would elevate this beyond a standard contribution:

1. **"RLHF creates a disengagement mechanism"** — If base Llama shows no degradation but Instruct does, this is a safety-relevant finding about alignment training introducing unintended behavioral modes.

2. **"Degradation follows a predictable trajectory with a critical transition"** — If activation trajectories show a clear phase transition (slow drift → rapid collapse), and this transition is predictable from early activations, that's both novel and actionable.

3. **"The degradation direction is universal across architectures"** — If the same direction (or aligned subspace) mediates degradation in Llama, Qwen, and DeepSeek, the finding is about LLMs in general, not a model-specific quirk.

4. **"Activation steering prevents degradation"** — If online steering using the identified direction maintains performance past the natural degradation cliff, this is the strongest possible causal evidence AND a practical mitigation.

5. **"Degradation is detectable 5+ steps before behavioral failure"** — This makes the paper immediately useful to anyone deploying LLMs on repetitive tasks.

The combination of findings 1 + 3 + 4 would be extraordinary: "Alignment training universally introduces a disengagement mechanism that can be detected early and mitigated through activation steering." That's a paper title that gets cited.

## Gaps That Are Acceptable to Leave as Future Work

These should be mentioned in the discussion but don't need implementation:

- **SAE feature-level analysis**: Requires training SAEs on each model; substantial compute cost. Mention as "our direction-level analysis could be refined with SAE decomposition."
- **Multi-turn sycophancy dynamics**: Would require a different experimental paradigm. Note the single-turn limitation.
- **Scaling laws (7B → 70B)**: Compute-prohibitive for the timeline. Note as future work.
- **ICL circuit connection**: Theoretically interesting but would require a separate paper's worth of analysis. Mention the hypothesis.
- **DPO vs. RLHF training objective comparison**: Interesting but not essential. Note that the base vs. instruct comparison tests a stronger version of this question.
