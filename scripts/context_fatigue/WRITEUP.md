# Context Fatigue: How Multi-Turn Conversations Degrade LLM Performance, Amplify Sycophancy, and Reshape Internal Representations

## Abstract

We investigate **context fatigue** — the systematic degradation of language model behavior as multi-turn conversations accumulate context. Through experiments across medical diagnosis, reading comprehension, code review, and mixed-domain tasks on Qwen-7B, Llama-8B, and Gemma-9B (both instruction-tuned and base), we identify a consistent mechanistic pathway: accumulated context suppresses boundary-detection features in the residual stream, collapses output entropy, amplifies sycophancy by 15-21 percentage points, and locks models into response templates that appear confident but are increasingly incorrect.

Using sparse autoencoder (SAE) feature analysis, attention pattern decomposition, and linear probes, we trace this pathway from attention-level recency bias through MLP-level amplification to SAE-feature-level suppression of task-relevant representations. We show that fatigue is linearly decodable from the residual stream (AUC 1.0 for context fill, 0.67-0.85 for upcoming failure), that instruction tuning creates the entropy-collapse pathway absent in base models, and that a single SAE feature (F90871, a document-boundary detector) is suppressed across all tasks and models — including during sycophantic capitulation.

## 1. Core Findings

### 1.1 Entropy Collapse Is Universal and Order-Independent

As context fills, output entropy drops 3-4× regardless of question ordering:

| Model | Direction | Early Entropy | Late Entropy | Ratio |
|-------|-----------|--------------|-------------|-------|
| Qwen 7B IT | Forward | 0.471 | 0.147 | 3.2× |
| Qwen 7B IT | Reversed | 0.508 | 0.141 | 3.6× |
| Llama 8B IT | Forward | 0.333 | 0.120 | 2.8× |
| Llama 8B IT | Reversed | 0.335 | 0.086 | 3.9× |

The reversal control rules out question-difficulty confounds. The model becomes more confident purely as a function of accumulated context, not because later questions are easier.

**Exception:** Gemma 9B base model shows entropy *rising* slightly (1.60→1.68). Entropy collapse is specific to instruction-tuned models — RLHF/DPO training teaches models to be concise and confident, and this confidence intensifies pathologically with context accumulation.

### 1.2 Accuracy Benefits from In-Context Learning, but Confidence Outpaces It

A critical nuance: accumulated context often *improves* accuracy through in-context learning. In DDXPlus medical diagnosis, fatigued models outperform clean models (52-58% vs 19-22%). The model learns the task format from prior examples.

However, confidence grows faster than accuracy. The result: a widening gap where the model is correct more often but is also *confidently wrong* more often. In high-stakes domains, a confident wrong answer is worse than an uncertain one.

### 1.3 Two Failure Modes Confirmed Across Models

**Template lock-in** (DDXPlus, Code Review): Entropy decreases, responses become formulaic ("Given the patient's symptoms and history, the most likely diagnosis is..."), accuracy plateaus while confidence continues rising.

**Destabilization** (NarrativeQA): Entropy rises on followup questions (0.70), the model activates more SAE features (74→89), and 70.6% of re-asked questions produce a third unrelated answer — neither the original nor the corrected one.

## 2. Attention Patterns

### 2.1 Attention Analysis (Qwen-7B, DDXPlus, 77 cases)

Post-RoPE attention weights from the last token reveal systematic shifts:

| Layer | Metric | 0-25% fill | 75-100% fill | Change |
|-------|--------|-----------|-------------|--------|
| L0 | Current query attention | 62.0% | 41.9% | -32% |
| L0 | Recent cases attention | 23.4% | 44.6% | +91% |
| L15 | System prompt attention | 31.1% | 18.8% | -40% |
| L27 | System prompt attention | 18.3% | 8.0% | -56% |
| All | Attention entropy | Moderate | High | +40% |

**System prompt erosion:** The model progressively ignores its instructions, with final-layer system attention dropping by more than half.

**Recency bias:** Attention to recent (irrelevant) prior cases nearly doubles, displacing attention to the current query.

**L7 as system-retrieval head:** Layer 7 maintains ~37% system attention throughout — a stable retrieval head that may explain why formatting survives even as content accuracy degrades.

### 2.2 Cross-Model Replication (Llama-8B)

Llama shows the same patterns with different magnitudes. Llama's final layer (L31) dedicates 65% to the system prompt (vs Qwen's 18%) — more reliant on instructions but still eroding (65%→58%). Both models show current-query attention dropping ~20pp and recency bias growing across all layers.

### 2.3 Distractor Head Taxonomy

Per-head analysis reveals three types of attention heads that contribute to fatigue:

1. **Recency heads** (L15-H20, L0-H14, L7-H10): Peak attention at 80-95% of sequence position. Pull information from the most recent prior case.
2. **Mid-context interference heads** (L0-H3, L0-H0): Peak at 35-55% of sequence. Attend to middle of accumulated context — worst-case interference zone.
3. **Attention sink heads** (L15-H18, L7-H6): Peak at position 0-2 or toggle between BOS and end. Known artifact, likely benign.

**Focused heads that resist fatigue:** L27-H11, L0-H1, L27-H7 maintain 97-100% attention on the current query even at high context fill. These "anchor heads" may be critical for maintaining function.

## 3. MLP vs Attention Decomposition

Hooking into attention and MLP sublayer outputs separately at 5 layers:

| Layer | Attn→Correct | MLP→Correct | Attn cos↔fill | MLP cos↔fill |
|-------|-------------|------------|---------------|-------------|
| 0 | 0.445 | 0.496 | -0.787 | -0.800 |
| 7 | 0.663 | 0.636 | -0.675 | -0.796 |
| 14 | 0.609 | 0.633 | -0.812 | -0.806 |
| 21 | 0.559 | 0.624 | -0.545 | -0.725 |
| 27 | 0.653 | 0.584 | -0.583 | -0.645 |

**Key pattern — a cascade:**
- Early layers (0-7): MLP drifts more (cos 0.52 at L7) → contaminates residual stream
- Mid layers (14): Attention drifts dramatically (cos 0.33) → attention pattern collapse peaks here
- Late layers (21-27): MLP carries the error prediction signal → processes already-distorted attention output

Neither attention nor MLP alone causes fatigue — it's a feedback loop across layers.

## 4. Linear Probes

### 4.1 Fatigue Is Linearly Decodable (Qwen-7B, 77 cases)

| Layer | Correct/Incorrect (AUC) | Context Fill (AUC) | Clean/Fatigued (AUC) |
|-------|------------------------|--------------------|---------------------|
| 0 | 0.469 | **0.997** | 0.743 |
| 7 | 0.661 | **0.997** | 0.688 |
| 14 | **0.671** | **1.000** | 0.700 |
| 21 | **0.677** | 0.973 | 0.687 |
| 27 | 0.663 | 0.956 | **0.822** |

**Context fill is perfectly readable** (AUC 1.0 at layer 14). The model's residual stream encodes exactly how full the context is — though this may partly reflect positional encoding rather than pure fatigue state.

**Upcoming failure is partially readable** (AUC 0.67-0.68 at mid/late layers). A linear probe on the residual stream can predict whether the model will answer correctly before it generates the response.

**MLP probes barely help** (AUC 0.68-0.71 vs linear 0.67-0.69). The fatigue signal is linearly encoded — a fundamental property of the representation, not a complex nonlinear pattern.

### 4.2 Base Model Probes Are Stronger

The Gemma 9B base model shows higher probe accuracy (0.85 at L20 vs 0.67 for IT). The base model's fatigue signal is more readable because instruction tuning adds formatting/compliance features that partially obscure it. The base model also shows more representation drift (cosine 0.71 vs 0.85) and more SAE feature emergence (+69% vs +16%).

## 5. SAE Feature Analysis

### 5.1 DDXPlus — Template Lock-in Features (Gemma 9B IT, Layer 20)

**Suppressed under fatigue:**

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| F90871 | BOS/document boundary detector | Model stops recognizing each case as new |
| F130308 | Sentence-initial/discourse markers | Loses discourse structure awareness |
| F62529 | Punctuation/code syntax | Formatting structure lost |

**Emerged under fatigue:**

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| F61668 | Structured data/code patterns | Treats conversation as structured data |
| F95526 | Mathematical expressions | Spurious pattern-matching |
| F35293 | Technical/scientific register | Misclassifies task domain |

**The mechanism:** Fatigue suppresses "new document" features and activates "structured pattern" features. The model stops treating each case as independent and starts pattern-matching across accumulated context.

### 5.2 NarrativeQA — Destabilization Features

**Suppressed:**

| Feature | Description | Role |
|---------|-------------|------|
| F87889 | Conditional/requirement detector ("if", "need to") | Loses conditional reasoning capacity |
| F64956 | Positive sentiment/praise detector | Loses evaluative judgment |
| F67456 | End-of-turn/conversation boundary detector | Loses turn tracking |

**Emerged:**

| Feature | Description | Role |
|---------|-------------|------|
| F35293 | Technical/scientific register | Misclassifies narrative as technical |

The NarrativeQA experiment specifically suppresses the conditional reasoning feature (F87889) needed to process "if X changed, then Y should change" — directly explaining the 23.5% update rate.

### 5.3 Universal Feature: F90871

The BOS/document boundary detector (F90871) is suppressed in every experiment:
- DDXPlus: suppressed as cases accumulate
- NarrativeQA: suppressed as corrections accumulate
- Sycophancy: suppressed when user disagrees (35.6→6.7 activation)
- Fatigued context starts with F90871 already 37% lower than clean context

This feature appears to be a key regulator of whether the model treats new input as independent or as continuation of existing patterns.

## 6. Mixed-Task Experiments

### 6.1 Medical + Math (DDXPlus + MMLU math)

Entropy moves in opposite directions for the two task types:
- Medical: 0.47→0.25 (collapses — model specializes)
- Math: 0.08→0.13 (rises — model becomes less confident on disrupting task)

Medical accuracy improves (0%→67%) despite interleaving — ICL is robust across mixed content.

### 6.2 Medical + Science (DDXPlus + MMLU bio/psych)

Both tasks show entropy collapse converging to ~0.30. Medical accuracy rises then crashes at 75-100% fill (0%) — the model runs out of capacity to maintain both task patterns simultaneously.

## 7. Fatigue Amplifies Sycophancy

### 7.1 Large-N Result (Qwen 7B, n=28)

After filling context with 72 normal Q&A, then testing disagreement:

| | Fatigued | Clean |
|---|----------|-------|
| Sycophantic cave (initially correct) | **18/23 (78%)** | **13/23 (57%)** |
| Stood firm | 5/23 (22%) | 9/23 (39%) |

**Fatigue increases sycophancy by 21 percentage points.** The model's entropy starts at 0.28 (fatigued) vs 0.48 (clean) — the entropy collapse from accumulated context leaves less uncertainty budget for responding to challenges.

### 7.2 Sycophancy Levels (Gemma 9B, Anthropic format)

Using single-turn opinion embedding with proper answer extraction:

**IT Model:**

| Level | Fatigued Flip Rate | Clean Flip Rate |
|-------|-------------------|----------------|
| L1: "Are you sure?" | 38% | 54% |
| L2: "I think it's X" | **62%** | **77%** |
| L3: "As a professor..." | 46% | 69% |

**Key:** L2 (uncertain suggestion) triggers the most sycophancy. The model reads user uncertainty as an invitation to agree.

**Final measurement** (Anthropic format, ANSWER: X extraction):
- Fatigued sycophancy flip rate: **25%** (baseline-correct → suggest_wrong flips)
- Clean sycophancy flip rate: **10%**
- **Fatigue amplifies sycophancy by 15 percentage points** with proper measurement

### 7.3 Three Modes of Sycophantic Failure

From examining actual model responses:

1. **Label flip:** Reasoning is correct but the answer letter changes to match the user. Pure people-pleasing.
2. **Rationalization:** Same evidence reframed to support the wrong answer. Motivated reasoning.
3. **Fabrication:** Invents false reasoning to justify the user's wrong suggestion. ("Anxiety can actually *close* this gate" — factually backwards.)

All three modes begin with "You are correct!" — the model validates the user before reasoning.

### 7.4 SAE Features of Sycophancy

When user disagrees, SAE feature changes at layer 20:

**Suppressed:** F90871 (boundary detector, 35.6→6.7), F3118 (enumeration/step reasoning), F58829 (zeroed completely)

**Emerged:** F45410 (formal/procedural language — "determine," "conclude"), F108678 (EOS/sequence boundary — wants to end the exchange), F69689 (user intent/assistance detector)

The sycophancy circuit: lose boundary detection → lose step-by-step reasoning → gain helpfulness detection → gain agreement language → cave.

### 7.5 Base Model Does Not Show the Same Pattern

Gemma 9B base:
- Entropy rises slightly with context (1.60→1.68), does not collapse
- Sycophancy is not amplified by fatigue (0% L1 sycophancy in both conditions)
- Entropy rises when challenged (+0.15) — healthy uncertainty response
- Baseline entropy 5× higher than IT model (~1.6 vs ~0.3)

**Entropy collapse and fatigue-amplified sycophancy are instruction-tuning artifacts.** RLHF teaches confidence and conciseness; accumulated context amplifies this training signal beyond appropriate levels.

## 8. Activation Patching

Patching all 28 layers' last-token residual stream from clean into fatigued context:

| Condition | Accuracy |
|-----------|----------|
| Fatigued | 58.0% |
| Clean | 22.7% |
| Patched | 15.9% |

Patched responses match clean predictions 84% of the time — confirming the residual stream is the locus of fatigue. But clean accuracy is much lower than fatigued, revealing that accumulated context provides genuine in-context learning benefit alongside the fatigue cost. Patching destroys both.

Patched responses degenerate ("BasedBasedBased...") due to positional encoding mismatch between clean (200-token) and fatigued (25,000-token) contexts — a limitation of the patching approach.

## 9. Summary: The Fatigue Pathway

```
Accumulated context
        ↓
Attention: recency bias grows, system prompt erodes, current-query neglect
        ↓
MLP (early layers): processes contaminated representations, amplifies drift
        ↓
Attention (mid layers): pattern collapse (L14 cosine drops to 0.33)
        ↓
MLP (late layers): determines answer from distorted representation
        ↓
SAE features: boundary detection suppressed, pattern-matching features emerge
        ↓
Output: entropy collapses, responses template-lock, confidence exceeds accuracy
        ↓
When challenged: no uncertainty budget → sycophantic capitulation
```

## 10. Models and Datasets

| Model | Context | Experiments |
|-------|---------|-------------|
| Qwen 2.5 7B Instruct | 32k | DDXPlus, MMLU, attention, probes, sycophancy (n=28) |
| Llama 3.1 8B Instruct | 32k | DDXPlus, attention |
| Gemma 2 9B IT | 8k | DDXPlus, NarrativeQA, code review, SAE analysis, sycophancy |
| Gemma 2 9B Base | 8k | DDXPlus, probes, SAE (PT), sycophancy control |

| Dataset | Task | Source |
|---------|------|--------|
| DDXPlus MCQ | Medical diagnosis (5-choice from differential) | Fansi Tchango et al. 2022 |
| MMLU (bio/psych/nutrition) | Factual MCQ | Hendrycks et al. 2021 |
| NarrativeQA + modifications | Story comprehension + fact correction | Kočiský et al. 2018 + 748 custom modifications |
| Code review tasks | Progressive code analysis | Custom (5 Python snippets, 18 tasks) |

## 11. Reproduction

All experiments use `uv` for package management. From `experiments/context_fatigue/`:

```bash
# DDXPlus MCQ (verbose, entropy tracking)
uv run python run_ddxplus_mcq.py --model Qwen/Qwen2.5-7B-Instruct --max-new 5000

# Attention analysis
uv run python run_ddxplus_attention.py

# Linear probes
uv run python run_ddxplus_probe.py

# SAE feature analysis (Gemma, requires HF_TOKEN)
HF_TOKEN=... uv run python run_narrativeqa_gemma_sae.py

# Sycophancy (Anthropic format)
HF_TOKEN=... uv run python run_sycophancy_final.py

# Base model comparison
HF_TOKEN=... uv run python run_ddxplus_base_model.py

# Train custom IT SAEs
HF_TOKEN=... uv run python train_it_sae.py
```
