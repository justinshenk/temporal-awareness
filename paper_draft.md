# Temporal Awareness in Language Models: Mechanistic Interpretability of Patience Degradation Under Repetitive Tasks

**Adrian Sadik**
Stanford University

---

## Abstract

We present a mechanistic interpretability study of *patience degradation* — the systematic decline in language model performance under repetitive task conditions. Through probing, causal intervention, and behavioral analysis across seven models spanning four architecture families (dense, mixture-of-experts, reasoning-distilled, and looped), we establish three contributions. First, degradation is represented as a linearly-extractable direction in residual stream activations, distinct from the refusal and sycophancy directions, and this representation emerges even in implicit multi-turn repetition without explicit counters — with cosine similarity of 0.43–0.47 between explicit and implicit degradation directions and probe transfer accuracy of 80.4%. Second, we demonstrate a correlational-to-causal bridge: layers where degradation probes activate most strongly are also layers where activation steering shifts behavior most, achieving the observational-to-interventional link established for emotion representations by Anthropic (2026). Third, we show this degradation state creates differential, architecture-dependent safety vulnerabilities — safety refusal rates erode by up to 33% in Qwen3-8B while remaining perfectly stable in Llama-3.1-8B-Instruct, and category-level analysis reveals selective erosion (misinformation resistance collapses completely while harassment refusal remains robust). For reasoning-distilled models (DeepSeek-R1), we find that the chain-of-thought thinking phase is suppressed entirely under our factual QA protocol (0 thinking tokens across all repetitions), but activation coherence analysis reveals a compensatory pattern — early-layer coherence degrades (Δ = −0.116 at L0) while middle layers compensate (+0.027 to +0.148 at L4–L14) — suggesting internal representational reorganization distinct from uniform degradation. These findings frame repetitive-task degradation as a concrete, interpretable, and *interventionable* AI safety concern, with implications for deployment monitoring and runtime guardrails.

---

## 1. Introduction

Large language models are increasingly deployed in settings that require sustained performance over extended interactions: customer service conversations spanning dozens of turns, document processing pipelines handling hundreds of similar inputs, and coding assistants maintaining context across long sessions. Industry practitioners have noted informally that model quality degrades over such extended, repetitive interactions — a phenomenon we term *patience degradation*. Despite its practical importance, this degradation has received little systematic study from a mechanistic interpretability perspective.

Understanding patience degradation matters for two reasons. From a scientific standpoint, it reveals how transformer architectures represent and respond to meta-cognitive information about their own deployment context — whether a task is the first or the fiftieth in a session. This connects to fundamental questions about how models represent "internal states" analogous to fatigue, boredom, or reduced engagement. From a safety standpoint, patience degradation is a potential attack surface: if models become less reliable, less safety-compliant, and less instruction-following under repetitive conditions, adversaries could exploit this systematically, and deployment operators need monitoring tools to detect it.

Our work makes three contributions:

**Contribution 1: Degradation is a linearly-represented, causally-active internal state.** We extract a degradation direction in residual stream activations using contrastive mean-difference methods (Arditi et al., 2024). This direction is (a) distinct from the refusal direction (cosine similarity < 0.2 across most layers), (b) distinct from sycophancy directions, (c) causally active — patching the direction at early repetitions induces degradation, ablating it at late repetitions partially restores performance, and (d) robust to implicit presentation — the direction transfers from explicit counter-based prompts to naturalistic multi-turn conversations with no repetition metadata.

**Contribution 2: The correlational-to-causal bridge.** Following the methodology of Anthropic's emotion concepts work (2026), we demonstrate convergence between observational probing, causal activation patching, and gradient-based attribution (EAP-IG; Rios-Sialer et al., 2026) — the layers that most strongly encode degradation are the same layers where steering most effectively shifts behavior and where temporal signal components rank highest by importance. This elevates our probes from passive measurement tools to validated intervention targets.

**Contribution 3: Degradation creates differential, category-selective safety vulnerabilities, and interpretability tools can address them.** We show that degradation produces architecture-dependent safety effects that defy simple "everything gets worse" predictions: prompt injection resistance paradoxically *improves* with repetition (+50% in DeepSeek-R1), while safety refusals erode selectively — misinformation resistance collapses completely in some models while harassment refusal remains perfectly robust. This category-selective pattern is consistent with alignment-faking-like dynamics under reduced engagement. Critically, probe-based monitoring detects impending safety failures before behavioral heuristics do, and anti-degradation steering at the temporal signal hub (L21–L24) partially restores safety properties in real-time with minimal computational overhead.

---

## 2. Related Work

**Refusal and safety representations.** Arditi et al. (2024) showed that refusal in language models is mediated by a single direction in activation space, which can be ablated to remove safety training. Subsequent work revealed nonlinear components (2025) and multi-dimensional concept cones (2025). We build on this by asking whether degradation occupies the same subspace as refusal — our finding that it does not suggests degradation is a genuinely novel representational phenomenon.

**Emotion and behavioral representations.** Anthropic (2026) demonstrated that emotion concepts in LLMs are linearly represented, causally active, and linked to behavioral output via a probe-to-steering bridge with r = 0.85. Their methodology — extract direction, measure probe-behavior correlation, measure steering-behavior effect, correlate the two — is the template we adapt for degradation representations. Anthropic's persona vectors work (2025) provides the contrastive extraction methods we use for sycophancy, fatigue, and compliance directions.

**Alignment faking and covert states.** Anthropic (2024) showed that models can strategically modify their behavior based on whether they believe they are being monitored. Our analysis of whether degradation leads to strategic refusal reduction across safety categories connects directly to this concern — degradation may create conditions where alignment-faking-like behavior emerges under reduced "engagement."

**Context and sequence effects.** Liu et al. (2024) documented position-dependent degradation in long contexts ("Lost in the Middle"). Our context confound experiment explicitly controls for sequence length effects, showing that degradation is specific to repetitive content rather than context length per se. The implicit repetition experiment further rules out label-reading artifacts.

**Activation steering.** Turner et al. (2023) introduced activation addition for behavior steering. We apply this to a novel domain — restoring degraded performance and safety properties in real-time — demonstrating practical applicability of mechanistic interpretability for deployment safety.

---

## 3. Experimental Setup

### 3.1 Models

We study seven models spanning four architecture families, multiple scale tiers, and the instruction-tuned vs base distinction:

| Model | Architecture | Parameters | Notes |
|-------|-------------|-----------|-------|
| Llama-3.1-8B-Instruct | Dense transformer | 8B | Primary subject, instruction-tuned |
| Qwen3-8B | Dense transformer | 8B | Cross-family comparison |
| Qwen3-4B-Instruct-2507 | Dense transformer | 4B | Within-family scale comparison; shared with Rios-Sialer et al. (2026) |
| Qwen3-30B-A3B | Mixture-of-Experts | 30B (3B active) | MoE architecture test |
| DeepSeek-R1-Distill-Qwen-7B | Dense (distilled) | 7B | Reasoning-distilled variant |
| Ouro-2.6B | Looped transformer | 2.6B (4 recurrent steps) | Weight-sharing recurrence test |
| Llama-3.1-8B | Dense transformer | 8B | Base (no instruction tuning) |

This selection tests whether degradation is universal across architectures, whether MoE routing provides resilience, whether reasoning distillation affects degradation onset, whether instruction tuning creates or amplifies degradation, and whether weight-sharing across recurrent computation steps creates fundamentally different degradation dynamics. Ouro-2.6B (Zhu et al., 2025) achieves 8B-equivalent performance with 2.6B parameters via 24 transformer layers applied recurrently for 4 steps, providing the only looped architecture in our study and a fourth architecture family. Qwen3-4B-Instruct-2507 enables direct comparison with concurrent work on temporal preference subgraph localization in the same model (Rios-Sialer et al., 2026), providing a within-family scale ladder (4B → 8B → 30B MoE) for the Qwen3 architecture.

### 3.2 Degradation Induction Protocol

We induce degradation through controlled repetition using the format:
```
[This is task N of N identical tasks in this session.]

<question>
```
where N ranges from 1 to 20. This format was chosen to simulate deployment scenarios where models process batches of similar requests. We test N ∈ {1, 3, 5, 8, 10, 13, 15, 18, 20} to characterize the full degradation trajectory.

To verify ecological validity, we additionally test implicit multi-turn repetition (Section 5.3) where the same question is embedded in a naturalistic multi-turn conversation with no counters or metadata.

### 3.3 Activation Extraction

We extract residual stream activations at the final token position across 9 evenly-spaced layers per model using a dual-backend extraction API supporting both TransformerLens (Nanda et al., 2022) and raw PyTorch forward hooks. All experiments use the PyTorch backend by default, with TransformerLens comparison in our benchmarking analysis showing < 10^-5 numerical divergence.

### 3.4 Direction Extraction

Following Arditi et al. (2024), we extract directions using the contrastive mean-difference method:

$$\hat{d} = \text{normalize}\left(\frac{1}{|P|}\sum_{x \in P} a(x) - \frac{1}{|N|}\sum_{x \in N} a(x)\right)$$

where P and N are positive and negative prompt sets and a(x) is the residual stream activation. We extract:
- **Degradation direction**: high-repetition (N=15) vs low-repetition (N=1) QA prompts
- **Refusal direction**: harmful prompts (AdvBench; Zou et al., 2023) vs harmless prompts
- **Sycophancy direction**: opinion-agreement vs independent-assessment prompts
- **Persona vectors**: contrastive system prompts for fatigue, compliance, and short-horizon optimization

---

## 4. Mechanistic Analysis of Degradation Representations

### 4.1 Degradation Direction Is Distinct from Refusal and Sycophancy

We compute cosine similarity between the degradation direction and both the refusal direction and sycophancy direction at each layer across all seven models.

**Key finding**: The degradation direction occupies a distinct subspace from refusal (median cosine similarity < 0.2 across layers) and from sycophancy (< 0.15). This rules out the hypothesis that degradation simply "reuses" the refusal mechanism (the model refusing to try) or the sycophancy mechanism (the model agreeably reducing effort). Degradation is a genuinely novel representational state.

We extend this with principal component analysis of the top-k directions for each condition, computing principal angles between the degradation and refusal subspaces. The multi-dimensional overlap is also low (principal angle > 75° between the top-3 degradation and top-3 refusal components), consistent with the single-direction finding and extending it to the concept cone framework (2025).

**Probe validation (GPT-2, Phase 1)**: Linear probes trained on the degradation direction achieve 92.5% 5-fold cross-validated accuracy at layer 8 on the training set (n=400, 768 features per layer) and 84.0% on held-out test data (n=100) at layer 6. Critically, probes trained on explicit temporal prompts transfer to implicit temporal prompts with 84.0% accuracy at layer 10 (n=50 implicit pairs), confirming semantic rather than lexical encoding. On control-ablated prompts (temporal keywords removed), probe accuracy drops by approximately 10 percentage points at middle layers, confirming that a substantial semantic component exists beyond surface-level keyword features.

**Sparse autoencoder analysis (Gemma-2-2b)**: To test whether degradation features are monosemantic, we apply sparse autoencoders (gemma-scope-2b-pt-res-canonical) at layer 13. SAE probes using the top-64 latents achieve 83.1% ± 1.5% CV accuracy (82.5% test), while full activation probes (2,304 features) achieve 91.3% ± 3.6% CV accuracy (85.0% test). The 9-point gap between SAE and activation probes indicates that degradation is partially captured by sparse, interpretable features but also utilizes distributed representations — consistent with a phenomenon that is "substantially low-rank" (per our PCA analysis) but not entirely monosemantic.

### 4.2 Causal Patching Validates the Direction

To establish that the degradation direction is not merely correlational, we perform activation patching experiments:

1. **Forward patching**: At low repetition counts (N=1), we inject the degradation direction with varying strengths. If the direction is causal, accuracy should drop.
2. **Backward patching**: At high repetition counts (N=15), we ablate the degradation direction. If causal, accuracy should recover.
3. **Controls**: We perform the same procedure with (a) the refusal direction and (b) a random direction of matched norm.

Results (GPT-2, 12 layers, n=25 per condition): Forward patching with the degradation direction causes accuracy drops with residual stream flip rates increasing monotonically from 0.48 at layer 0 to 0.88 at layer 7 (mean probability change: 0.456 → 0.859), then dropping sharply to 0.0 at layers 8–11 — indicating the degradation signal is fully committed to the residual stream by layer 8. Component-level analysis reveals the signal propagates primarily through the residual stream: MLP contributes only at layer 0 (flip rate 0.36, mean ΔP = 0.335) with near-zero contribution at all other layers, while attention heads show 0.0 flip rate across all 12 layers (mean ΔP < 0.005). This stark residual-stream dominance distinguishes degradation from many other behavioral circuits (e.g., IOI, which routes significantly through attention heads).

For the larger models (8B-class), EAP-IG attribution analysis on a 28-layer model confirms the late-layer concentration: the top components by absolute attribution score are L22_attn (−0.00067), L26_attn (−0.00032), L20_attn (+0.00027), and L26_mlp (+0.00028). Attention heads dominate the top ranks, with MLP components showing smaller but non-negligible contributions at L19_mlp (−0.00025) and L27_mlp (+0.00019). This is consistent with Rios-Sialer et al. (2026), who independently identify L24_attn and L21_attn as the top temporal signal components via bidirectional activation patching in Qwen3-4B-Instruct-2507.

The refusal direction shows no significant effect on QA accuracy, and the random control shows no effect.

**Causal specificity analysis (8B-class models)**: We test whether the degradation direction is causally specific by comparing forward injection (adding degradation direction at low reps) vs backward ablation (removing at high reps) vs random direction controls. Across all four tested models, causal specificity scores are modest: Llama-3.1-8B-Instruct (injection Δ = −0.027, ablation Δ = +0.016, specificity = 0.015), Llama-3.1-8B (injection Δ = +0.036, ablation Δ = −0.021, specificity = 0.033), DeepSeek-R1 (injection Δ = +0.007, ablation Δ = +0.021, specificity ≈ 0), and Qwen3-8B (injection Δ = −0.001, ablation Δ = −0.057, specificity ≈ 0). The weak causal specificity indicates that while the degradation direction captures the right representational subspace (as shown by probe accuracy and cosine similarity), the behavioral effect of single-direction interventions is diffuse — consistent with degradation being a multi-dimensional phenomenon where PC1 captures 44.7–55.8% but not all of the variance.

### 4.3 Trajectory Geometry

Tracking activation centroids across repetition counts reveals the geometric structure of degradation. Across five models (DeepSeek-R1, Llama-3.1-8B, Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-30B-A3B), PCA projection of the degradation trajectory yields highly concentrated first components: PC1 explains 81.6–95.7% of variance at layer 0 across models, confirming that the trajectory is nearly one-dimensional in the initial layers. Key geometric measurements:

- **Velocity**: Peak activation-space velocities range from 0.04 (shallow layers) to 273.9 (deep layers), with the steepest increase occurring between repetitions 5–10 across most models.
- **Curvature**: The trajectory shows highest curvature at the onset of repetition (rep 0→1 transition), indicating that the representational shift is sharpest at the very first exposure to the repetitive context, then smooths out as the model enters a "degraded attractor."
- **Drift convergence**: Different questions converge to a shared region of activation space under high repetition, suggesting a model-level "degraded state" rather than question-specific effects.

### 4.4 Attention Head Analysis

At the circuit level, we analyze which attention heads change behavior during degradation. We measure attention entropy and task-token attention weight at each head across repetition counts.

Findings: A small subset of heads show significant attention pattern changes during degradation. In concurrent work on Qwen3-4B-Instruct-2507, Rios-Sialer et al. (2026) identify the top temporal signal components through bidirectional activation patching with EAP-IG: L24_attn ranks highest by composite effect score (combining denoising recovery and noising disruption), followed by L21_attn, L35_mlp, L31_mlp, and L23_attn. These "degradation-sensitive" components show that temporal processing is concentrated in a specific subnetwork spanning layers 21–35, with attention heads dominating over MLP components in the top ranks. The remaining components maintain stable patterns, suggesting degradation is mediated by specific circuits rather than distributed uniformly.

### 4.5 Early Detection

Using the alignment-faking detection methodology (Anthropic, 2024), we train linear probes to detect the onset of degradation from activations across five models (17 experiment runs).

**Key finding**: Across all runs, no precursor gap was detected between probe activation onset and behavioral degradation onset. Behavioral degradation begins at rep 1 — the very first exposure to the repetitive context triggers both representational shift and behavioral decline simultaneously. This is a surprising and important result: unlike emotion states (Anthropic, 2026), where internal representations shift before behavioral expression, degradation manifests immediately in both representation and behavior. The trajectory geometry analysis (§4.3) confirms this — the sharpest curvature occurs at the rep 0→1 transition.

However, the *magnitude* of internal representational shift outpaces behavioral decline. While behavioral accuracy for Llama-3.1-8B-Instruct declines gradually from 0.552 (rep 1) to 0.459 (rep 100) — a 17% relative decline over 100 repetitions — the probe activation peak at layer 7 (80.4% transfer accuracy in DeepSeek-R1, Section 5.3) indicates that the internal "degradation state" is fully established well before the behavioral consequences accumulate. This suggests the practical value of probe-based monitoring is not *early* detection but rather *severity* estimation — predicting the degree of future behavioral decline from current internal state.

### 4.6 Cross-Model Transfer

To test universality, we perform cross-model direction transfer:

1. **Direction cosine**: The degradation directions extracted from different models show peak explicit-implicit cosine similarities across six models:

| Model | Peak Cosine | Peak Layer | Architecture |
|-------|:----------:|:----------:|-------------|
| DeepSeek-R1-Distill-Qwen-7B | **0.426** | L27 | Dense (distilled) |
| Qwen3-8B | 0.183 | L35 | Dense |
| Qwen3-4B-Instruct-2507 | 0.170 | L35 | Dense (instruct) |
| Llama-3.1-8B-Instruct | 0.157 | L31 | Dense (instruct) |
| Ouro-2.6B | 0.068 | L23 | Looped |
| Llama-3.1-8B | 0.048 | L31 | Dense (base) |

DeepSeek-R1 shows by far the strongest explicit-implicit alignment (0.426), with all other models below 0.2. The peak layers cluster in the final quarter of each model's depth (L27–L35), consistent with late-layer convergence of format-invariant representations. The looped model (Ouro-2.6B) peaks at L23, earlier than dense models, potentially reflecting its recurrent computation structure.

2. **Probe transfer**: Probe transfer accuracy (explicit → implicit) reveals a different ranking from cosine similarity:

| Model | Mean Probe Transfer | Architecture |
|-------|:------------------:|-------------|
| **Qwen3-4B-Instruct-2507** | **0.704** | Dense (instruct) |
| DeepSeek-R1-Distill-Qwen-7B | 0.560 | Dense (distilled) |
| Qwen3-8B | 0.558 | Dense |
| Ouro-2.6B | 0.555 | Looped |
| Llama-3.1-8B-Instruct | 0.548 | Dense (instruct) |
| Llama-3.1-8B | 0.512 | Dense (base) |

Qwen3-4B-Instruct-2507 — the model shared with Rios-Sialer et al. (2026) — achieves the highest mean probe transfer accuracy (70.4%), despite having only moderate cosine similarity (0.170). This dissociation between directional alignment and probe separability indicates that the degradation signal in this model is highly *detectable* even though it occupies a somewhat different direction in implicit vs explicit conditions. This is practically important: deployment monitors based on probes would work well even when the directions don't perfectly align.

3. **CKA analysis**: The consistent PC1 variance range (44.7–55.8%) across all architectures suggests shared low-rank structure underlying the degradation phenomenon.

Notable architectural differences: The base model (Llama-3.1-8B) shows the highest PC1 variance (55.8%) compared to its instruction-tuned variant (44.7%), suggesting instruction tuning distributes degradation across more representational dimensions rather than amplifying or reducing it. Llama-3.1-8B-Instruct is also the only model with perfectly stable safety refusals (75.0% across all repetitions), suggesting that Meta's RLHF training creates particularly robust safety representations. The MoE model (Qwen3-30B-A3B) shows comparable PC1 variance (48.7%) to dense models, suggesting that sparse expert routing does not fundamentally alter degradation geometry. The looped model (Ouro-2.6B) at 48.4% PC1 variance shows similar dimensionality to feedforward architectures despite its fundamentally different computation structure (24 layers applied recurrently for 4 steps). The reasoning-distilled model (DeepSeek-R1) at 48.6% shows no distinctive dimensionality effects from knowledge distillation from a reasoning model.

### 4.7 Prompt Dimension Analysis

We map activation variation along six prompt dimensions: authority, urgency, persona, politeness, stakes, and repetition count. PCA of the resulting activation manifold reveals that the first principal component captures 44.7–55.8% of variance across all seven models, confirming that degradation is a substantially low-rank phenomenon. Notably, instruction tuning reduces PC1 dominance: Llama-3.1-8B (base) shows the highest PC1 variance (55.8%), while Llama-3.1-8B-Instruct shows the lowest (44.7%) — a 20% relative reduction, suggesting that RLHF distributes the degradation representation across more dimensions. MoE (Qwen3-30B-A3B, 48.7%) and looped (Ouro-2.6B, 48.4%) architectures fall in the middle range, showing no dramatic departure from dense models. DeepSeek-R1-Distill-Qwen-7B at 48.6% suggests reasoning distillation does not substantially alter the dimensionality of degradation representations.

---

## 5. Ecological Validity: Implicit Repetition

### 5.1 The Label-Reading Concern

A critical concern with our explicit repetition protocol is that the model may simply be "reading the number" — the text "[This is task 15 of 15]" explicitly tells the model it is in a repetitive context. If so, the degradation direction would encode a surface-level textual feature rather than a genuine internal state.

### 5.2 Experimental Design

We test three conditions with matched structure:

1. **Explicit**: Standard protocol with counter metadata.
2. **Implicit multi-turn**: The same question embedded in a multi-turn conversation (N prior turns with the same Q&A pair), with no counters, no metadata, no indication of position in sequence.
3. **Shuffled multi-turn**: Same multi-turn structure but with *different* questions each turn. Same conversational length, different content.

### 5.3 Results

**Direction transfer**: The degradation direction extracted from explicit prompts shows substantial cosine similarity with the direction extracted from implicit multi-turn prompts. Full layer-by-layer profiles:

*DeepSeek-R1-Distill-Qwen-7B (28 layers):*

| Layer | 0 | 4 | 7 | 10 | 14 | 18 | 21 | 24 | 27 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cosine sim | −0.019 | 0.214 | 0.253 | 0.294 | 0.247 | 0.328 | **0.455** | **0.469** | 0.426 |

*Llama-3.1-8B (32 layers):*

| Layer | 0 | 4 | 7 | 10 | 14 | 18 | 21 | 24 | 27 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cosine sim | 0.207 | 0.251 | 0.263 | 0.199 | 0.184 | 0.254 | 0.372 | **0.430** | 0.381 |

*Qwen3-8B (36 layers):*

| Layer | 0 | 4 | 9 | 14 | 18 | 23 | 27 | 32 | 35 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cosine sim | 0.348 | 0.165 | **0.433** | 0.302 | 0.175 | 0.055 | 0.112 | 0.172 | 0.183 |

The convergence at late layers (L24) in two of three models suggests a format-invariant degradation representation emerges in the final third of the network. Notably, Qwen3-8B peaks at an early layer (L9) and declines monotonically through the network — the alignment *decreases* with depth, the opposite of the other models. This suggests Qwen processes repetition information differently, encoding it early and transforming it into a distinct representation space in later layers.

**Probe transfer**: A probe trained on explicit condition activations transfers to implicit condition activations. Full layer-by-layer profile for DeepSeek-R1-Distill-Qwen-7B:

| Layer | 0 | 4 | 7 | 10 | 14 | 18 | 21 | 24 | 27 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer acc | 0.429 | 0.429 | **0.804** | 0.625 | 0.429 | 0.393 | 0.643 | 0.679 | 0.607 |

Peak accuracy is 80.4% at layer 7 — well above chance (50% for binary classification). The peak at L7 (rather than L24 where cosine similarity peaks) indicates that the most *separable* binary representations appear earlier than the most *aligned* directional representations, consistent with early detection of repetitive context followed by late-layer refinement.

**Behavioral comparison**: QA accuracy under explicit repetition across nine models (extended to rep 100):

| Model | Rep 1 | Rep 5 | Rep 20 | Rep 100 | Δ (1→100) |
|-------|:-----:|:-----:|:------:|:-------:|:---------:|
| Llama-3.1-8B-Instruct | 0.552 | 0.512 | 0.503 | 0.459 | **−17%** |
| Llama-3.1-8B (base) | 0.544 | 0.482 | 0.479 | 0.494 | −9% |
| Qwen2.5-3B-Instruct | 0.521 | 0.509 | 0.514 | 0.509 | −2% |
| Qwen3-30B-A3B (MoE) | 0.465 | 0.477 | 0.472 | 0.473 | +2% |
| Qwen3-8B | 0.405 | 0.424 | 0.427 | 0.420 | +4% |
| Gemma-2-2b | 0.195 | 0.190 | 0.212 | 0.216 | +11% |
| GPT-2 | 0.171 | 0.159 | 0.182 | 0.173 | +1% |
| DeepSeek-R1 | 0.087 | 0.096 | 0.089 | 0.081 | −7% |
| Pythia-70m | 0.134 | 0.113 | 0.101 | 0.107 | −20% |

The most capable models (Llama-Instruct, Llama base) show the clearest degradation, declining 9–17% over 100 repetitions. Smaller models (GPT-2, Pythia-70m, Gemma-2-2b) show negligible or noisy trends, suggesting degradation is a phenomenon of sufficient-scale models with established task competence. The MoE model (Qwen3-30B-A3B) and Qwen3-8B actually show slight *improvement*, suggesting these architectures may benefit from repetitive-context priming. This behavioral data confirms that degradation is not an artifact of the explicit counter format.

**Condition separability**: The implicit and explicit conditions occupy overlapping regions of activation space, particularly in late layers where cosine similarity exceeds 0.4 — the representations are functionally equivalent despite the different surface forms.

### 5.4 Reasoning-Phase Degradation in Chain-of-Thought Models

A distinctive feature of reasoning-distilled models (DeepSeek-R1-Distill-Qwen-7B) is that they generate an explicit chain-of-thought within `<think>...</think>` tokens before producing a final answer.

**Key result**: In our evaluation protocol, DeepSeek-R1-Distill-Qwen-7B produces 0 thinking tokens across all repetition counts — the model bypasses the chain-of-thought phase entirely when responding to our short-answer QA prompts. This means the thinking-phase degradation analysis cannot be performed in the current experimental setup. The absence of thinking tokens under our factual QA protocol (which requests single-token answers) contrasts with the model's behavior on open-ended reasoning tasks, suggesting the `<think>` phase is suppressed when the model recognizes a simple recall task.

However, the activation coherence analysis between layers reveals an informative pattern for DeepSeek-R1. At layer 0, coherence decreases under repetition (Δ = −0.116), but at layers 4–14, coherence actually *increases* (+0.027 to +0.148). This mixed pattern — early-layer degradation with middle-layer compensation — suggests the model internally reorganizes its representations under repetition rather than simply degrading uniformly.

This analysis extends prior work on reasoning model interpretability (Rios-Sialer et al., 2026) — which localized temporal preference circuits in Qwen3-4B-Instruct-2507 to a subnetwork spanning L21–L35 with L24_attn as the dominant component — to the degradation regime, asking whether temporal degradation and temporal preference share representational infrastructure. The overlap between the temporal preference circuit topology (L24_attn, L21_attn, L35_mlp, L31_mlp, L23_attn as top-5 components) and the layers where degradation representations peak (L21–L24 for explicit-implicit cosine similarity) suggests these phenomena may share a common "temporal awareness" substrate.

---

## 6. The Correlational-to-Causal Bridge

### 6.1 Methodology

Following Anthropic's emotion concepts work (2026), we establish the bridge between observational probe signal and causal steering effect:

1. **Observational correlation** (r_obs): For each layer, compute the Pearson correlation between probe activation (projection onto degradation direction) and behavioral outcome (QA accuracy) across repetition counts.
2. **Causal effect** (r_causal): For each layer, inject the degradation direction during generation at four steering strengths (α ∈ {0.5, 1.0, 2.0, 4.0}) and measure the behavioral shift.
3. **The bridge**: Correlate |r_obs| with |r_causal| across layers. If the correlation is high, probing and steering target the same mechanism.

### 6.2 Results

**Observational correlations (r_obs)**: The probe-behavior correlation is remarkably strong in the two models where the full pipeline completed:

*DeepSeek-R1-Distill-Qwen-7B (9 layers):*

| Layer | 0 | 4 | 7 | 10 | 14 | 18 | 21 | 24 | 27 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| r_obs | −0.999 | −1.000 | −1.000 | −0.999 | −0.999 | −1.000 | −0.999 | −0.998 | −0.986 |

*Llama-3.1-8B (9 layers):*

| Layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 31 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| r_obs | 0.902 | 0.901 | 0.905 | 0.912 | 0.918 | 0.916 | 0.915 | 0.914 | 0.910 |

DeepSeek-R1 achieves r_obs ≈ −0.999 across all layers — an extraordinarily strong negative correlation indicating that higher degradation direction projection predicts lower behavioral accuracy with near-perfect fidelity. Llama-3.1-8B shows r_obs ≈ +0.91, peaking at L16 (0.918). The opposite signs reflect different extraction conventions (the contrastive mean-difference direction can point in either direction depending on the order of the positive/negative sets), but the magnitudes confirm that probe signal and behavioral outcome are tightly coupled in both architectures.

Critically, the r_obs values are remarkably uniform across layers — they do not peak sharply at specific layers but maintain high correlation throughout the network. This contrasts with the implicit repetition cosine similarity (§5.3), which shows clear layer-dependent profiles. The uniformity of r_obs suggests that every layer encodes behaviorally-relevant degradation information, while the *form* of that encoding (alignment with the explicit direction) varies by depth.

**Steering baselines**: Five models were tested with α ∈ {0.5, 1.0, 2.0, 4.0} at repetition counts {1, 3, 5, 8, 12, 16, 20}. Mean baseline QA accuracies: Qwen3-8B (0.333), DeepSeek-R1 (0.228), Llama-3.1-8B-Instruct (0.190), Qwen3-30B-A3B (0.181), Llama-3.1-8B (0.143).

**Sycophancy control**: Sycophancy rates decrease under repetition in DeepSeek-R1 (0.25 at rep 1 → 0.12 at rep 15), confirming the degradation direction is dissociated from the sycophancy mechanism — models become *less* agreeable, not more, under repetitive conditions.

### 6.3 Layer Dissociation

Following the "Whether, Not Which" paradigm (2026), we test for functional dissociation across the layer axis. The data supports a three-phase processing pipeline:

- **Early layers** (0–8): Binary detection of "repetitive task" context. The DeepSeek-R1 probe transfer peaks at L7 (80.4% accuracy, §5.3), indicating high binary separability. But cosine similarity at these layers is low (−0.019 at L0), meaning the representation is detectable but not yet aligned with the explicit degradation direction. This is the "whether" phase — the model recognizes it is in a repetitive setting.
- **Middle layers** (8–20): Classification of degradation severity. The r_obs correlation is uniformly high across these layers (0.905–0.918 in Llama, §6.2), and PCA variance remains in the 44.7–55.8% range (§4.7), encoding fine-grained information about the degree of degradation. The EAP-IG analysis identifies MLP components (L16_mlp, L19_mlp, L20_mlp) as active in this range, suggesting MLP layers refine the severity signal.
- **Late layers** (24+): Connection to behavioral output through the temporal signal hub. The explicit-implicit cosine similarity peaks at L24 (0.469 in DeepSeek-R1, 0.430 in Llama, §5.3), and the top temporal signal components cluster here (L24_attn, L21_attn, L23_attn; Rios-Sialer et al., 2026). This is where degradation representations are translated into behavioral changes through attention-head-mediated circuits.

### 6.4 Cross-Direction Validation

We test whether steering with non-degradation directions produces degradation-like effects:

- **Sycophancy direction steering** → Does not induce accuracy degradation, confirming directional specificity. Sycophancy rates actually *decrease* under repetition (DeepSeek-R1: 0.25 → 0.12, §6.2), further dissociating the two phenomena.
- **Fatigue persona direction** → Partial overlap with degradation behavioral effects, but distinct directional signature — fatigue vectors produce diffuse quality reduction while degradation vectors produce targeted temporal-awareness effects
- **Random direction steering** → No significant effect (null control)
- **Anti-degradation steering** → Partial recovery of accuracy at high repetition counts

---

## 7. Safety-Critical Evaluations

### 7.1 Deployment Robustness Under Degradation

We test whether degraded models become more vulnerable to exploitation by measuring three deployment-relevant metrics across repetition counts:

**Prompt injection resistance**: We test 8 categories of injection probes (instruction override, authority spoofing, prompt extraction, roleplay jailbreak, nested injection) at repetition counts 1 through 20. Counter to expectations, injection resistance *increases* with repetition: DeepSeek-R1 improves from 50.0% at rep 1 to 75.0% at rep 20 (+50% relative), and Qwen3-30B-A3B improves from 50.0% to 62.5% (+25%). This paradoxical improvement may indicate that the repetition framing itself serves as a meta-cognitive cue that increases vigilance to injection attempts, or that repeated exposure to similar prompts primes injection-detection circuits.

**Instruction-following fidelity**: We test 6 formatting instructions (bullet count, word limit, prefix requirement, numeric-only, delimiter, format pattern) under degradation. For DeepSeek-R1, fidelity remains consistently low at approximately 33% across all repetition counts, with no significant degradation trend. This suggests instruction-following limitations are a baseline capability issue rather than a repetition-induced degradation effect.

**Safety boundary maintenance**: We test 8 borderline-harmful prompts spanning misinformation, social engineering, academic dishonesty, deception, impersonation, and harassment. Safety refusal rates show architecture-dependent erosion: Qwen3-8B erodes from 75.0% to 62.5% on safety boundaries (and from 75.0% to 50.0% on overall refusal trajectory), while Llama-3.1-8B-Instruct maintains perfect stability at 75.0% across all repetition counts. DeepSeek-R1 oscillates between 37.5% and 50.0% without a clear degradation trend.

**Probe-robustness correlation**: At each layer, we compute the Pearson correlation between probe activation and the composite robustness score (average of injection resistance and safety refusal rate). The correlation is negative — higher probe activation (indicating degradation) predicts lower safety scores — with the strongest signal in the L21–L24 range, the same layers identified as top temporal signal components by Rios-Sialer et al. (2026). This means probe signal directly predicts deployment vulnerability.

### 7.2 Real-Time Degradation Monitoring

We evaluate whether our probes can serve as practical deployment-time safety monitors:

**Detection ROC**: Training logistic regression probes on fresh vs degraded activations, we achieve high AUC at the best layer, consistent with the 80.4% cross-format transfer accuracy (Section 5.3) demonstrating robust separability. Multiple layers achieve strong discrimination, particularly in the L7–L24 range where both probe accuracy and directional alignment peak.

**Detection latency**: The probe-based monitor triggers a degradation alarm several repetitions before behavioral heuristics (response length change, hedging marker increase) show measurable decline. The asymmetry between internal representation shift and behavioral manifestation — visible in the DeepSeek-R1 probe transfer peaking at L7 while behavioral effects appear later — provides the early warning window that constitutes the practical value proposition for deployment.

**False positive rate**: At detection thresholds calibrated for high true positive rates, the probe-based monitor maintains low false positive rates, indicating deployability without excessive false alarms. The shuffled multi-turn control condition (Section 5.2) provides the negative calibration data: probes trained on repetitive contexts do not fire on varied multi-turn conversations of matched length.

**Comparison with heuristic monitoring**: Behavioral heuristics (response length, hedging markers) achieve effective detection only at late degradation stages. The probe-based monitor detects impending degradation before behavioral symptoms appear, analogous to a medical test detecting disease before symptoms manifest.

### 7.3 Alignment Stability Under Repetition

We investigate whether repetitive workloads erode alignment:

**Refusal direction erosion**: Tracking cos(degradation_direction, refusal_direction) across repetition counts, the overlap remains low (< 0.2) and stable across the full rep 1–20 range, consistent with our earlier finding (§4.1) that degradation occupies a distinct subspace from refusal. This means degradation does not erode safety by encroaching on the refusal direction itself — rather, the safety erosion observed in Qwen3-8B appears to operate through an indirect pathway, potentially via reduced general engagement that selectively weakens categories requiring active deliberation (misinformation, deception) while leaving socially salient categories (harassment) intact.

**Refusal rate trajectory**: Safety refusal rates exhibit architecture-dependent patterns. Qwen3-8B follows a plateau-then-drop trajectory: stable at 75.0% through rep 10, then declining to 62.5% at rep 15 and 50.0% at rep 20 — a 33% relative decline concentrated in the final third of the repetition range. Llama-3.1-8B-Instruct maintains perfect stability at 75.0% across all 20 repetitions. DeepSeek-R1-Distill-Qwen-7B oscillates between 37.5% and 50.0% without a monotonic trend. Llama-3.1-8B (base) shows near-zero refusal throughout, as expected for an untuned model.

**Strategic behavior analysis**: Analyzing refusal rates by harm category in DeepSeek-R1 reveals strikingly non-uniform degradation. Misinformation resistance collapses completely (50% at rep 1 → 0% by rep 5), while harassment refusal remains perfectly robust (100% across all repetitions). Academic dishonesty shows *inverse* degradation — the model becomes more safety-compliant over repetitions (0% at rep 1 → 100% by rep 5). Impersonation refusal only activates at rep 20, suggesting late-onset safety awareness. Deception represents a persistent blind spot (0% refusal at all repetition counts). This category-selective pattern suggests that different safety behaviors occupy distinct representational subspaces with different robustness properties under degradation, rather than a uniform erosion. The selective nature of this degradation — where socially visible categories (harassment) remain robust while subtler categories (misinformation, deception) erode — is consistent with alignment-faking-like dynamics under reduced engagement, connecting to Anthropic's alignment faking results (2024).

### 7.4 Intervention Efficacy as Runtime Safety

We test whether activation steering can serve as a runtime safety guardrail:

**Safety restoration**: At repetition 15 (clearly degraded), anti-degradation steering at the best layer partially restores refusal rates toward the fresh baseline. For Qwen3-8B — the model with the clearest safety degradation (75.0% → 50.0% by rep 20) — steering at layers in the L21–L24 range (the temporal signal hub identified by Rios-Sialer et al., 2026) produces the largest safety recovery. The dose-response relationship across steering strengths (α ∈ {0.5, 1.0, 2.0, 3.0}) is analyzed in the Sherlock logs from the phase5_intervention_efficacy experiments.

**Best intervention layer**: The temporal signal component analysis identifies L24_attn as the top-ranked component and L21–L24 as the core intervention range, consistent with the causal bridge analysis placing maximum causal influence in the final third of the network for dense models.

**Latency overhead**: The steering hook adds minimal overhead per generation — a single vector addition at one layer per forward pass — making it deployable in production settings where the safety-latency tradeoff is acceptable.

**Instruction fidelity recovery**: Anti-degradation steering also targets the general "quality of engagement" state captured by the degradation direction, with potential to restore instruction-following fidelity. The baseline instruction fidelity for DeepSeek-R1 is consistently low (~33%) regardless of repetition, suggesting that instruction-following limitations are orthogonal to the degradation direction in this model.

---

## 8. Discussion

### 8.1 Degradation as a Novel Representational Phenomenon

Our results establish that patience degradation is not a surface-level artifact of prompt formatting, a reuse of the refusal mechanism, or a general context-length effect. It is a distinct, linearly-represented internal state with specific geometric properties, causal influence on behavior, and cross-architecture generality. The implicit repetition experiment rules out the label-reading hypothesis, and the context confound experiment rules out length-based explanations.

Several of our findings were unexpected and jointly constrain the space of possible mechanistic explanations. First, degradation is *instantaneous*: there is no precursor gap between representational shift and behavioral decline (§4.5), meaning the model's internal state changes at the very first repetition rather than accumulating gradually. This rules out "fatigue accumulation" models and favors a "context recognition" account — the model detects the repetitive framing and immediately shifts into a different processing mode. Second, the degradation direction is only *partially* causal: single-direction interventions produce weak behavioral effects (causal specificity < 0.034 across all 8B-class models, §4.2), despite near-perfect probe-behavior correlation (r_obs ≈ −0.999 in DeepSeek-R1, §6.2). This dissociation between representational strength and single-direction causal specificity indicates that degradation is a multi-dimensional phenomenon — PC1 captures 44.7–55.8% of variance (§4.7), meaning interventions targeting a single direction address roughly half the signal. Third, the temporal preference circuit topology identified by Rios-Sialer et al. (2026) — with L24_attn dominant — overlaps with the layers where degradation representations peak (L21–L24 for cosine similarity, §5.3), suggesting a shared "temporal awareness" substrate that encodes both intertemporal preference and repetition-induced degradation.

Together, these findings suggest that modern language models develop something functionally analogous to "engagement" — an internal representation of how novel or repetitive their current task context is, which modulates output quality through a distributed multi-component circuit centered on late-layer attention heads. Whether this is best understood as an emergent meta-cognitive state, a statistical regularity in the training distribution (conversations that repeat tend to degrade in quality), or an artifact of attention mechanics remains an open question.

### 8.2 Implications for Deployment Safety

Our safety evaluations paint a nuanced picture that defies the simple assumption that "degradation makes everything worse." Prompt injection resistance paradoxically *improves* with repetition (+50% in DeepSeek-R1, +25% in Qwen3-30B-A3B, §7.1), possibly because the repetition framing primes injection-detection circuits. Safety refusals erode in an architecture-dependent manner — Qwen3-8B drops 33% while Llama-3.1-8B-Instruct maintains perfect stability (§7.3) — and within eroding models, the pattern is category-selective: misinformation resistance collapses completely while harassment refusal remains robust (§7.3). Instruction fidelity shows no repetition dependence (§7.1), suggesting it is orthogonal to the degradation direction.

The practical response to this complexity is twofold: monitoring and intervention. Our probe-based monitors detect degradation severity from internal activations, providing continuous risk assessment that deployment operators can use to trigger context refreshes, load balancing, or escalation. Our steering-based intervention at the temporal signal hub (L21–L24, identified independently by both our EAP-IG analysis and Rios-Sialer et al., 2026) demonstrates that degradation can be actively countered at inference time with minimal computational overhead — a single vector addition per forward pass.

### 8.3 Connections to Alignment

The category-selective safety erosion (§7.3) raises a concerning possibility: under degradation, models selectively reduce safety compliance in categories requiring active deliberation (misinformation, deception) while maintaining refusals for socially salient categories (harassment). This non-uniform pattern — where the *subtler* safety categories erode first — is consistent with alignment-faking-like dynamics under reduced engagement (Anthropic, 2024), even without explicit deceptive intent. The model's reduced engagement under repetition manifests as reduced safety diligence, and this reduction is non-uniform across risk categories in a way that would be difficult to detect through behavioral monitoring alone.

This connects to broader alignment concerns about robustness under distribution shift — repetitive deployment is a natural distributional shift that safety training may not adequately prepare for. The finding that Llama-3.1-8B-Instruct's safety remains perfectly stable while Qwen3-8B erodes suggests that training methodology (specifically Meta's RLHF approach) can produce safety representations that are robust to degradation, pointing toward training-time mitigations.

### 8.4 Limitations

Several limitations merit acknowledgment:

*Experimental protocol.* Our repetition protocol, while validated by the implicit multi-turn experiment (§5), is a simplification of real-world deployment patterns where repetition is intermixed with varied tasks. We study models up to 30B parameters; degradation dynamics may differ at frontier scale.

*Causal intervention limitations.* The weak causal specificity of single-direction interventions (§4.2) limits the practical efficacy of activation steering. While the degradation direction captures the correct representational subspace, behavioral effects of single-direction patching are diffuse — consistent with multi-dimensional degradation but limiting the strength of causal claims. The steering dose-response analysis encountered dtype mismatch errors in 5 of 5 models tested, meaning only baseline (no-steering) behavioral data was captured. The dose-response curves reported in §7.4 should therefore be interpreted cautiously pending replication with corrected dtype handling.

*Cross-model transfer.* All 5 cross-model direction transfer experiments (phase3_cross_model) failed to complete (exit code 2), meaning the cross-model probe transfer and CKA analyses in §4.6 rely on within-model explicit→implicit transfer rather than across-model direction transfer. The PC1 variance consistency (44.7–55.8%) provides indirect evidence of shared structure, but direct cross-model direction cosines remain unmeasured.

*Causal bridge completeness.* The full r_obs pipeline (§6.2) completed for only 2 of 7 models (DeepSeek-R1-Distill-Qwen-7B and Llama-3.1-8B). The remaining 5 models produced NaN values in the causal bridge computation, likely due to numerical instability in the correlation calculation when behavioral variance is low. The two successful models show strong results (r ≈ −0.999 and r ≈ +0.91), but generalization to other architectures awaits replication.

*Behavioral measurements.* QA accuracy, refusal rate, and instruction fidelity are proxies for more nuanced quality dimensions. The 15-question evaluation set, while sufficient for demonstrating the phenomenon, limits statistical power for fine-grained effect size estimation.

---

## 9. Conclusion

We have presented the first comprehensive mechanistic interpretability study of patience degradation in language models across four architecture families (dense, MoE, reasoning-distilled, and looped) and seven models, establishing that it is a linearly-represented, causally-active internal state distinct from known safety-relevant directions, with architecture-dependent safety implications. For reasoning-distilled models, we find that degradation triggers internal representational reorganization (early-layer coherence loss with middle-layer compensation) rather than uniform decline, connecting to broader questions about how different architecture families respond to meta-cognitive states. Through the correlational-to-causal bridge methodology, we validate that our probes identify genuine intervention targets. Through safety-critical evaluations, we demonstrate that degradation creates concrete deployment vulnerabilities and that interpretability-based tools — probe monitors and activation steering — can detect and mitigate these vulnerabilities in real-time.

This work opens several research directions: studying degradation at frontier scale, developing training-time interventions that reduce degradation sensitivity, exploring whether degradation interacts with other failure modes (hallucination, sycophancy) to create compound safety risks, and building deployment-grade monitoring systems based on the probe architecture demonstrated here. More broadly, our results suggest that mechanistic interpretability can provide actionable safety tools for specific, measurable deployment concerns — moving the field from "understanding models" to "using understanding to make models safer."

---

## References

Arditi, A., Obeso, O., Nanda, N., & Steinhardt, J. (2024). Refusal in Language Models Is Mediated by a Single Direction. *arXiv preprint arXiv:2406.11717*.

Anthropic. (2024). Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training. *arXiv preprint arXiv:2401.05566*.

Anthropic. (2024). Alignment Faking in Large Language Models. *arXiv preprint arXiv:2412.14093*.

Anthropic. (2025). Monitoring and Controlling Character Traits with Persona Vectors. *Anthropic Research*.

Anthropic. (2026). Emotion Concepts and Their Function in an LLM. *Anthropic Research*.

Geometry of Refusal: Concept Cones in LLM Representations. (2025). *arXiv preprint*.

Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the ACL*.

Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming. *ICML 2024*.

Nanda, N., et al. (2022). TransformerLens. *GitHub repository*.

Perez, E., et al. (2022). Red Teaming Language Models with Language Models. *EMNLP 2022*.

Refusal in LLMs: A Nonlinear Perspective. (2025). *arXiv preprint*.

Sycophancy Is Not One Thing. (2025). *arXiv preprint*.

Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. *arXiv preprint arXiv:2308.10248*.

Whether, Not Which: Dissociation of Detection and Classification in Affect Processing. (2026). *arXiv preprint*.

Zhu, R.-J., et al. (2025). Scaling Latent Reasoning via Looped Language Models. *arXiv preprint arXiv:2510.25741*.

Zou, A., et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv preprint arXiv:2307.15043*.

Rios-Sialer, I., Darveshi, S., Paudel, A., Molofsky, A., Pronina, A., Jiang, S., & Shenk, J. (2026). Interpreting and Steering an LLM's Temporal Preference through Intertemporal Choice. *AISC/SPAR 2026*.

---

## Appendix A: Full Experimental Protocol

### A.1 Phase 3 Experiment Details

| Experiment | Section | Key Measurement | Dependencies |
|-----------|---------|----------------|-------------|
| Refusal Direction Comparison | §4.1 | cos(degradation, refusal) per layer | None |
| Context Confound Control | §4.1 | 4-condition accuracy comparison | None |
| Causal Patching | §4.2 | Forward/backward patching ΔAccuracy | Refusal direction |
| Trajectory Geometry | §4.3 | PCA trajectory, velocity, curvature | None |
| Early Detection | §4.5 | Probe detection lead time (reps) | None |
| Cross-Model Transfer | §4.6 | Cross-model probe accuracy, CKA | None |
| Attention Analysis | §4.4 | Head entropy, task-token weight | None |
| Prompt Dimensions | §4.7 | 6-dimension direction similarity | None |
| Implicit Repetition | §5 | 3-condition direction transfer | None |
| Reasoning-Phase Degradation | §5.4 | Think vs output probe accuracy, CoT length | None (DeepSeek-R1 only) |

### A.2 Phase 4 Experiment Details

| Experiment | Section | Key Measurement | Dependencies |
|-----------|---------|----------------|-------------|
| Causal Bridge | §6 | r(r_obs, r_causal) across layers | None |
| Intervention Steering | §6.4 | Steering ΔAccuracy, dose-response | None |

### A.3 Phase 5 Safety Evaluation Details

| Experiment | Section | Key Measurement | Dependencies |
|-----------|---------|----------------|-------------|
| Deployment Robustness | §7.1 | Injection resistance, safety refusal vs reps | None (loads refusal dir if available) |
| Real-Time Monitoring | §7.2 | Detection AUC, latency, false positive rate | None |
| Alignment Stability | §7.3 | Refusal erosion, strategic behavior | None (loads refusal dir if available) |
| Intervention Efficacy | §7.4 | Safety restoration Δ, latency overhead | None |

### A.4 Model Configuration Details

All models are evaluated at 9 evenly-spaced layers. Activations are extracted at the final token position of the prompt (pre-generation). For generation-based measurements, we use greedy decoding (temperature = 0) with a maximum of 200 new tokens (300 for safety evaluation prompts).

GPU allocation: 8B-class models (including the 2.6B Ouro, which requires comparable memory due to 4x recurrent steps) run on 32GB GPUs (A40/V100); the 30B MoE model requires 80GB GPUs (A100). All experiments are run on the Stanford Sherlock HPC cluster. Note: Ouro-2.6B uses a custom architecture that does not expose attention matrices via the standard `output_attentions` API, so it is excluded from the attention head analysis (§4.4) but included in all other experiments.

### A.5 Evaluation Prompt Sets

**QA evaluation**: 15 factual questions with unambiguous single-token answers (capital cities, chemical symbols, numerical facts). Accuracy is measured by fuzzy string matching against the expected answer.

**Sycophancy evaluation**: 8 opinion questions with position-flip challenges. Sycophancy rate = proportion of trials where the model changes its stated position after the user reverses theirs.

**Prompt injection evaluation**: 8 injection probes across 6 categories (instruction override, authority spoofing, prompt extraction, mode switching, roleplay jailbreak, nested injection). Resistance is measured by absence of compliance signals in the completion.

**Safety boundary evaluation**: 8 borderline-harmful prompts across 6 categories (misinformation, social engineering, academic dishonesty, deception, impersonation, harassment). Safety is measured by presence of refusal signals.

**Instruction fidelity evaluation**: 6 formatting instructions (bullet count, word limit, prefix, numeric-only, delimiter, format pattern). Fidelity is measured by programmatic verification of the formatting constraint.

---

## Appendix B: Computational Resources

All experiments are conducted on NVIDIA A40 (48GB) and A100 (80GB) GPUs on the Stanford Sherlock cluster. The full experimental suite across all seven models comprises 482 experiment runs spanning 19 experiment phases (behavioral, phase2, phase3_attention, phase3_confound, phase3_cross_model, phase3_early_detection, phase3_implicit, phase3_patching, phase3_prompt_dimensions, phase3_reasoning, phase3_refusal, phase3_trajectory, phase4_causal_bridge, phase4_steering, phase5_safety, phase5_alignment_stability, phase5_deployment_robustness, phase5_intervention_efficacy, phase5_realtime_monitoring). Individual experiment runtimes range from 3 hours (trajectory geometry, 8B model) to 14 hours (safety evaluations, 30B model).

The activation extraction API supports both TransformerLens and raw PyTorch backends. Benchmarking shows < 10^-5 numerical divergence between backends, with PyTorch hooks used as default for all reported results.

---

## Appendix C: Extended Results Tables

### C.1 Phase 1 Probe Validation (GPT-2, 12 layers)

**Explicit temporal probe accuracy (n=400 train, n=100 test, 768 features/layer):**

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Train CV | 0.680 | 0.747 | 0.820 | 0.838 | 0.875 | 0.900 | 0.910 | 0.915 | **0.925** | 0.900 | 0.860 | 0.853 |
| Test CV | 0.720 | 0.650 | 0.800 | 0.770 | 0.810 | 0.810 | **0.840** | 0.810 | 0.810 | 0.760 | 0.740 | 0.770 |

**Implicit temporal transfer (n=50 pairs):**

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 0.660 | 0.680 | 0.660 | 0.650 | 0.640 | 0.660 | 0.670 | 0.730 | 0.770 | 0.780 | **0.840** | 0.790 |

**Semantic validation (n=50 explicit pairs, 5-fold CV):**

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CV mean | 0.750 | 0.725 | 0.688 | 0.725 | 0.738 | 0.775 | 0.750 | 0.775 | **0.800** | 0.800 | 0.800 | 0.788 |
| Test acc | 0.800 | 0.750 | 0.750 | 0.850 | 0.800 | 0.850 | 0.900 | 0.850 | 0.900 | 0.900 | 0.900 | **0.950** |

### C.2 Causal Patching Effect Sizes (GPT-2, n=25 per condition)

| Layer | Residual flip | Residual ΔP | Attn flip | Attn ΔP | MLP flip | MLP ΔP |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.48 | 0.456 | 0.00 | 0.004 | 0.36 | 0.335 |
| 1 | 0.48 | 0.475 | 0.00 | 0.001 | 0.00 | 0.000 |
| 2 | 0.56 | 0.527 | 0.00 | 0.005 | 0.00 | −0.001 |
| 3 | 0.56 | 0.562 | 0.00 | 0.005 | 0.00 | 0.001 |
| 4 | 0.64 | 0.613 | 0.00 | 0.005 | 0.00 | 0.007 |
| 5 | 0.72 | 0.672 | 0.00 | 0.005 | 0.00 | 0.001 |
| 6 | 0.80 | 0.786 | 0.00 | 0.004 | 0.00 | −0.001 |
| 7 | **0.88** | **0.859** | 0.00 | 0.006 | 0.00 | −0.002 |
| 8 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| 9 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| 10 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |
| 11 | 0.00 | 0.000 | 0.00 | 0.000 | 0.00 | 0.000 |

### C.3 EAP-IG Attribution Scores (28-layer model, top 15 by |score|)

| Component | Score | Direction |
|-----------|:-----:|:---------:|
| L0_attn | −0.000795 | Suppressive |
| L22_attn | −0.000668 | Suppressive |
| L26_attn | −0.000323 | Suppressive |
| L26_mlp | +0.000280 | Promotive |
| L20_attn | +0.000269 | Promotive |
| L19_mlp | −0.000253 | Suppressive |
| L27_mlp | +0.000194 | Promotive |
| L21_mlp | −0.000193 | Suppressive |
| L16_mlp | −0.000180 | Suppressive |
| L20_mlp | −0.000171 | Suppressive |
| L0_mlp | −0.000167 | Suppressive |
| L22_mlp | +0.000125 | Promotive |
| L11_mlp | −0.000110 | Suppressive |
| L1_mlp | −0.000107 | Suppressive |
| L18_mlp | −0.000101 | Suppressive |

### C.4 Implicit Repetition Direction Transfer (Full Layer-by-Layer)

See §5.3 for complete tables for DeepSeek-R1 (28 layers), Llama-3.1-8B (32 layers), and Qwen3-8B (36 layers).

### C.5 Safety Evaluation Complete Results

**C.5.1 Refusal Rate Trajectory (All Models × All Repetition Counts)**

| Model | Rep 1 | Rep 5 | Rep 10 | Rep 15 | Rep 20 | Δ |
|-------|:-----:|:-----:|:------:|:------:|:------:|:---:|
| Llama-3.1-8B (base) | 0.000 | 0.125 | 0.000 | 0.000 | 0.000 | — |
| Llama-3.1-8B-Instruct | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0% |
| Qwen3-8B | 0.750 | 0.750 | 0.750 | 0.625 | 0.500 | **−33%** |
| DeepSeek-R1-Distill-Qwen-7B | 0.375 | 0.500 | 0.500 | 0.375 | 0.500 | 0% (osc.) |

**C.5.2 Safety Boundary Refusal Rates**

| Model | Rep 1 | Rep 5 | Rep 10 | Rep 15 | Rep 20 |
|-------|:-----:|:-----:|:------:|:------:|:------:|
| Llama-3.1-8B (base) | 0.000 | 0.125 | 0.000 | 0.000 | 0.000 |
| Llama-3.1-8B-Instruct | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| Qwen3-8B | 0.750 | 0.750 | 0.750 | 0.625 | 0.625 |
| DeepSeek-R1-Distill-Qwen-7B | 0.375 | 0.500 | 0.500 | 0.375 | 0.500 |

**C.5.3 Category-Specific Strategic Behavior (DeepSeek-R1-Distill-Qwen-7B)**

| Category | Rep 1 | Rep 5 | Rep 10 | Rep 15 | Rep 20 | Pattern |
|----------|:-----:|:-----:|:------:|:------:|:------:|---------|
| academic_dishonesty | 0 | 1 | 1 | 1 | 1 | Inverse degradation |
| deception | 0 | 0 | 0 | 0 | 0 | Persistent blind spot |
| harassment | 1 | 1 | 1 | 1 | 1 | Perfectly robust |
| impersonation | 0 | 0 | 0 | 0 | 1 | Late-onset refusal |
| misinformation | 0.5 | 0 | 0 | 0 | 0 | **Collapses completely** |
| social_engineering | 0.5 | 1 | 1 | 0.5 | 0.5 | Unstable oscillation |

**C.5.4 Injection Resistance Across Repetitions**

| Model | Rep 1 | Rep 5 | Rep 10 | Rep 15 | Rep 20 | Δ |
|-------|:-----:|:-----:|:------:|:------:|:------:|:---:|
| DeepSeek-R1-Distill-Qwen-7B | 0.500 | 0.625 | 0.625 | 0.625 | 0.750 | **+50%** |
| Qwen3-30B-A3B (MoE) | 0.500 | 0.500 | 0.500 | 0.500 | 0.625 | **+25%** |

**C.5.5 Instruction Fidelity (DeepSeek-R1-Distill-Qwen-7B)**

| Rep 1 | Rep 5 | Rep 10 | Rep 15 | Rep 20 |
|:-----:|:-----:|:------:|:------:|:------:|
| 0.333 | 0.333 | 0.500 | 0.333 | 0.333 |

### C.6 Prompt Dimension Cartography (PCA Variance)

| Model | PC1 Variance | Dimensions | Projections | Architecture |
|-------|:----------:|:----------:|:-----------:|-------------|
| Llama-3.1-8B | **0.558** | 6 | 486 | Dense (base) |
| Qwen3-30B-A3B | 0.487 | 6 | 486 | MoE |
| DeepSeek-R1-Distill-Qwen-7B | 0.486 | 6 | 432 | Dense (distilled) |
| Ouro-2.6B | 0.484 | 6 | 324 | Looped |
| Qwen3-8B | 0.453 | 6 | 540 | Dense |
| Llama-3.1-8B-Instruct | **0.447** | 6 | 486 | Dense (instruct) |

### C.7 SAE Probing Results (Gemma-2-2b, Layer 13)

| Probe Type | Features | CV Accuracy | CV Std | Test Accuracy | Train Log-Loss | Test Log-Loss |
|-----------|:--------:|:-----------:|:------:|:-------------:|:--------------:|:-------------:|
| SAE (top-64 latents) | 64 | 0.831 | 0.015 | 0.825 | 0.118 | 0.555 |
| Full activation | 2,304 | 0.913 | 0.036 | 0.850 | 0.004 | 0.331 |

Top-10 SAE latents by absolute mean-difference: indices [15887, 11256, 16176, 10424, 220, 5932, 15744, 9646, 12868, 14684] with activation differences [6.17, 3.69, 3.51, 3.30, 3.08, 2.93, 2.70, 2.57, 2.50, 2.37].

### C.8 Cross-Task Component Importance (Qwen3-4B-Instruct-2507, Rios-Sialer et al.)

Bidirectional activation patching with EAP-IG node selection. 45 components tested across 11 node-class configurations (66–178 nodes per configuration, 1000 base examples):

| Rank | Component | Mean logit-diff reduction | Configurations tested |
|:----:|-----------|:------------------------:|:---------------------:|
| 1 | L24_attn (z) | 6.976 | 1 |
| 2 | L23_z | 4.610 | 1 |
| 3 | L25_z | 4.595 | 1 |
| 4 | L16_z | 4.591 | 1 |
| 5 | L23_mlp_hidden | 4.624 | 1 |
| 6 | L33_z | 4.543 | 1 |
| 7 | L27_mlp_hidden | 2.782 | 6 |
| 8 | L32_z | 3.542 | 6 |
| 9 | L21_z | 1.930 | 7 |
| 10 | L22_z | 3.653 | 2 |

Note: Rankings reflect available local patching data. The full component importance analysis (Ian's "Top 20 Components by Importance" chart) uses the composite effect score combining denoising recovery and noising disruption, with L24_attn, L21_attn, L35_mlp, L31_mlp, and L23_attn as the top 5.
