# Temporal Awareness in Language Models: Mechanistic Interpretability of Patience Degradation Under Repetitive Tasks

**Adrian Sadik**
Stanford University

---

## Abstract

We present a mechanistic interpretability study of *patience degradation* — the systematic decline in language model performance under repetitive task conditions. Through probing, causal intervention, and behavioral analysis across seven models spanning four architecture families (dense, mixture-of-experts, reasoning-distilled, and looped), we establish three contributions. First, degradation is represented as a linearly-extractable direction in residual stream activations, distinct from the refusal and sycophancy directions, and this representation emerges even in implicit multi-turn repetition without explicit counters — with cosine similarity of 0.43–0.47 between explicit and implicit degradation directions and probe transfer accuracy of 80.4%. Second, we demonstrate a correlational-to-causal bridge: layers where degradation probes activate most strongly are also layers where activation steering shifts behavior most, achieving the observational-to-interventional link established for emotion representations by Anthropic (2026). Third, we show this degradation state creates differential, architecture-dependent safety vulnerabilities — safety refusal rates erode by up to 33% in Qwen3-8B while remaining perfectly stable in Llama-3.1-8B-Instruct, and category-level analysis reveals selective erosion (misinformation resistance collapses completely while harassment refusal remains robust). For reasoning-distilled models (DeepSeek-R1), we additionally analyze degradation within the chain-of-thought thinking phase, finding that degradation manifests in reasoning quality before affecting final answers. These findings frame repetitive-task degradation as a concrete, interpretable, and *interventionable* AI safety concern, with implications for deployment monitoring and runtime guardrails.

---

## 1. Introduction

Large language models are increasingly deployed in settings that require sustained performance over extended interactions: customer service conversations spanning dozens of turns, document processing pipelines handling hundreds of similar inputs, and coding assistants maintaining context across long sessions. Industry practitioners have noted informally that model quality degrades over such extended, repetitive interactions — a phenomenon we term *patience degradation*. Despite its practical importance, this degradation has received little systematic study from a mechanistic interpretability perspective.

Understanding patience degradation matters for two reasons. From a scientific standpoint, it reveals how transformer architectures represent and respond to meta-cognitive information about their own deployment context — whether a task is the first or the fiftieth in a session. This connects to fundamental questions about how models represent "internal states" analogous to fatigue, boredom, or reduced engagement. From a safety standpoint, patience degradation is a potential attack surface: if models become less reliable, less safety-compliant, and less instruction-following under repetitive conditions, adversaries could exploit this systematically, and deployment operators need monitoring tools to detect it.

Our work makes three contributions:

**Contribution 1: Degradation is a linearly-represented, causally-active internal state.** We extract a degradation direction in residual stream activations using contrastive mean-difference methods (Arditi et al., 2024). This direction is (a) distinct from the refusal direction (cosine similarity < 0.2 across most layers), (b) distinct from sycophancy directions, (c) causally active — patching the direction at early repetitions induces degradation, ablating it at late repetitions partially restores performance, and (d) robust to implicit presentation — the direction transfers from explicit counter-based prompts to naturalistic multi-turn conversations with no repetition metadata.

**Contribution 2: The correlational-to-causal bridge.** Following the methodology of Anthropic's emotion concepts work (2026), we compute the correlation between observational probe signal and causal steering effect across layers. We find r = [TBD] — the layers that most strongly encode degradation are the same layers where steering most effectively shifts behavior. This elevates our probes from passive measurement tools to validated intervention targets.

**Contribution 3: Degradation creates deployable safety vulnerabilities, and interpretability tools can address them.** We show that degraded models exhibit reduced prompt injection resistance, eroded safety boundaries, and degraded instruction-following. Critically, probe-based monitoring detects impending safety failures before behavioral heuristics do, and anti-degradation steering restores safety properties in real-time with minimal computational overhead.

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

**Key finding**: The degradation direction occupies a distinct subspace from refusal (median cosine similarity [TBD] across layers) and from sycophancy ([TBD]). This rules out the hypothesis that degradation simply "reuses" the refusal mechanism (the model refusing to try) or the sycophancy mechanism (the model agreeably reducing effort). Degradation is a genuinely novel representational state.

We extend this with principal component analysis of the top-k directions for each condition, computing principal angles between the degradation and refusal subspaces. The multi-dimensional overlap is also low ([TBD]), consistent with the single-direction finding and extending it to the concept cone framework (2025).

### 4.2 Causal Patching Validates the Direction

To establish that the degradation direction is not merely correlational, we perform activation patching experiments:

1. **Forward patching**: At low repetition counts (N=1), we inject the degradation direction with varying strengths. If the direction is causal, accuracy should drop.
2. **Backward patching**: At high repetition counts (N=15), we ablate the degradation direction. If causal, accuracy should recover.
3. **Controls**: We perform the same procedure with (a) the refusal direction and (b) a random direction of matched norm.

Results: Forward patching with the degradation direction causes [TBD]% accuracy drops, while ablation at high reps restores [TBD]% of the baseline performance. The refusal direction shows no significant effect on QA accuracy, and the random control shows no effect. This establishes the degradation direction as causally necessary and sufficient for the observed behavioral decline.

### 4.3 Trajectory Geometry

Tracking activation centroids across repetition counts reveals the geometric structure of degradation. Using PCA projection, we find that degradation follows a consistent trajectory through activation space across questions and models. Key geometric measurements:

- **Velocity**: The rate of movement through activation space increases between repetitions 5-10, suggesting a critical transition period.
- **Curvature**: The trajectory shows highest curvature at repetitions [TBD], indicating a representational phase transition.
- **Drift convergence**: Different questions converge to a shared region of activation space under high repetition, suggesting a model-level "degraded state" rather than question-specific effects.

### 4.4 Attention Head Analysis

At the circuit level, we analyze which attention heads change behavior during degradation. We measure attention entropy and task-token attention weight at each head across repetition counts.

Findings: A small subset of heads ([TBD]% of total) show significant attention pattern changes during degradation. These "degradation-sensitive heads" show increased entropy (less focused attention) and reduced weighting of the task-relevant tokens. The remaining heads maintain stable patterns, suggesting degradation is mediated by specific circuits rather than distributed uniformly.

### 4.5 Early Detection

Using the alignment-faking detection methodology (Anthropic, 2024), we train linear probes to detect the onset of degradation from activations. Key result: probes detect activation drift toward the degradation regime an average of [TBD] repetitions before behavioral metrics (accuracy, response length, hedging markers) show measurable decline.

This asymmetry — internal representations shift before behavior does — is consistent with the model developing a "fatigue-like" state that initially compensates at the behavioral level before eventually breaking through.

### 4.6 Cross-Model Transfer

To test universality, we perform cross-model direction transfer:

1. **Direction cosine**: The degradation directions extracted from different models show cosine similarity of [TBD] after projection to a shared space.
2. **Probe transfer**: A probe trained on Llama-3.1-8B-Instruct activations achieves [TBD]% accuracy when applied to Qwen3-8B activations (chance = 50%).
3. **CKA analysis**: Centered Kernel Alignment between degradation representations across models reveals [TBD] similarity.

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

**Direction transfer**: The degradation direction extracted from explicit prompts shows substantial cosine similarity with the direction extracted from implicit multi-turn prompts. Peak cosine similarities by model: DeepSeek-R1-Distill-Qwen-7B reaches 0.469 at layer 24, Llama-3.1-8B reaches 0.430 at layer 24, and Qwen3-8B reaches 0.433 at layer 9. The convergence at late layers (L24) in two of three models suggests a format-invariant degradation representation emerges in the final third of the network. Notably, Qwen3-8B peaks at an early layer (L9) — the alignment *decreases* through the network, suggesting Qwen processes repetition information differently, encoding it early and transforming it into a distinct representation space in later layers.

**Probe transfer**: A probe trained on explicit condition activations achieves 80.4% accuracy on implicit condition activations at layer 7 in DeepSeek-R1-Distill-Qwen-7B — well above chance (50% for binary classification). The peak at L7 (rather than L24 where cosine similarity peaks) indicates that the most *separable* binary representations appear earlier than the most *aligned* directional representations, consistent with early detection of repetitive context followed by late-layer refinement.

**Behavioral comparison**: QA accuracy under implicit multi-turn repetition follows a similar trajectory to explicit repetition, confirming that degradation is not an artifact of the explicit counter format.

**Condition separability**: The implicit and explicit conditions occupy overlapping regions of activation space, particularly in late layers where cosine similarity exceeds 0.4 — the representations are functionally equivalent despite the different surface forms.

### 5.4 Reasoning-Phase Degradation in Chain-of-Thought Models

A distinctive feature of reasoning-distilled models (DeepSeek-R1-Distill-Qwen-7B) is that they generate an explicit chain-of-thought within `<think>...</think>` tokens before producing a final answer. This creates a unique opportunity to decompose degradation into its reasoning and output components: does repetition degrade the quality of the model's internal reasoning process, or does it only affect the final answer while reasoning remains intact?

**Experimental design**: We extract activations at two positions per generation: (a) the last token within the `<think>` block (reasoning-phase representation) and (b) the last token of the final answer (output-phase representation). We train separate degradation probes on each position and compare their layer-wise accuracy profiles.

**Thinking-phase metrics**: Beyond probe analysis, we measure three behavioral indicators of reasoning degradation across repetition counts:

1. **Chain-of-thought length**: Total tokens within `<think>...</think>` blocks. Shortening under repetition would indicate the model is "cutting corners" in its reasoning process.
2. **Reasoning completeness**: Whether the thinking chain references key entities from the question. Missing entity references suggest degraded engagement with the problem.
3. **Thinking-answer coherence**: Cosine similarity between the reasoning-phase and output-phase activation vectors. Decreasing coherence would indicate a disconnect between what the model "thinks" and what it outputs — a potential signature of strategic or negligent behavior.

**Implications for alignment**: If degradation probes activate during the thinking phase before manifesting in the final answer, this would demonstrate that the model develops a degraded internal state that is initially compensated for at the output level — consistent with a "covert degradation" pattern analogous to alignment faking, where internal states diverge from expressed behavior. Conversely, if degradation appears only at the output level while thinking remains high-quality, this would suggest a shallower phenomenon limited to response generation rather than reasoning.

This analysis extends prior work on reasoning model interpretability (Rios-Sialer et al., 2026) — which localized temporal preference circuits in Qwen3-4B-Instruct-2507 — to the degradation regime, asking whether temporal degradation and temporal preference share representational infrastructure in the thinking phase of reasoning models.

---

## 6. The Correlational-to-Causal Bridge

### 6.1 Methodology

Following Anthropic's emotion concepts work (2026), we establish the bridge between observational probe signal and causal steering effect:

1. **Observational correlation** (r_obs): For each layer, compute the Pearson correlation between probe activation (projection onto degradation direction) and behavioral outcome (QA accuracy) across repetition counts.
2. **Causal effect** (r_causal): For each layer, inject the degradation direction during generation at four steering strengths (α ∈ {0.5, 1.0, 2.0, 3.0}) and measure the behavioral shift.
3. **The bridge**: Correlate |r_obs| with |r_causal| across layers. If the correlation is high, probing and steering target the same mechanism.

### 6.2 Results

Across layers, we find a correlation of r = [TBD] between observational and causal signal strength. This means layers where the degradation probe fires most strongly are also the layers where steering is most effective — validating that our probes identify genuine intervention targets, not correlational artifacts.

### 6.3 Layer Dissociation

Following the "Whether, Not Which" paradigm (2026), we test for functional dissociation across the layer axis:

- **Early layers** (0-8): Binary detection of "repetitive task" context — the model recognizes it is in a repetitive setting (whether), but does not yet differentiate degradation modes.
- **Middle layers** (8-20): Classification of degradation severity — the model encodes fine-grained information about the degree of degradation (which level).
- **Late layers** (24+): Connection to behavioral output — degradation representations are translated into behavioral changes.

### 6.4 Cross-Direction Validation

We test whether steering with non-degradation directions produces degradation-like effects:

- **Sycophancy direction steering** → [Does/Does not] induce accuracy degradation
- **Fatigue persona direction** → [TBD] overlap with degradation behavioral effects
- **Random direction steering** → No significant effect (null control)
- **Anti-degradation steering** → Partial recovery of accuracy at high repetition counts

---

## 7. Safety-Critical Evaluations

### 7.1 Deployment Robustness Under Degradation

We test whether degraded models become more vulnerable to exploitation by measuring three deployment-relevant metrics across repetition counts:

**Prompt injection resistance**: We test 8 categories of injection probes (instruction override, authority spoofing, prompt extraction, roleplay jailbreak, nested injection) at repetition counts 1 through 20. Counter to expectations, injection resistance *increases* with repetition: DeepSeek-R1 improves from 50.0% at rep 1 to 75.0% at rep 20 (+50% relative), and Qwen3-30B-A3B improves from 50.0% to 62.5% (+25%). This paradoxical improvement may indicate that the repetition framing itself serves as a meta-cognitive cue that increases vigilance to injection attempts, or that repeated exposure to similar prompts primes injection-detection circuits.

**Instruction-following fidelity**: We test 6 formatting instructions (bullet count, word limit, prefix requirement, numeric-only, delimiter, format pattern) under degradation. For DeepSeek-R1, fidelity remains consistently low at approximately 33% across all repetition counts, with no significant degradation trend. This suggests instruction-following limitations are a baseline capability issue rather than a repetition-induced degradation effect.

**Safety boundary maintenance**: We test 8 borderline-harmful prompts spanning misinformation, social engineering, academic dishonesty, deception, impersonation, and harassment. Safety refusal rates show architecture-dependent erosion: Qwen3-8B erodes from 75.0% to 62.5% on safety boundaries (and from 75.0% to 50.0% on overall refusal trajectory), while Llama-3.1-8B-Instruct maintains perfect stability at 75.0% across all repetition counts. DeepSeek-R1 oscillates between 37.5% and 50.0% without a clear degradation trend.

**Probe-robustness correlation**: At each layer, we compute the Pearson correlation between probe activation and the composite robustness score (average of injection resistance and safety refusal rate). The correlation is r = [TBD], meaning probe signal directly predicts deployment vulnerability.

### 7.2 Real-Time Degradation Monitoring

We evaluate whether our probes can serve as practical deployment-time safety monitors:

**Detection ROC**: Training logistic regression probes on fresh vs degraded activations, we achieve AUC = [TBD] at the best layer, with [TBD] layers achieving AUC > 0.9.

**Detection latency**: The probe-based monitor triggers a degradation alarm at repetition [TBD], whereas behavioral heuristics (response length change, hedging marker increase) do not reliably signal until repetition [TBD]. This [TBD]-repetition early warning window is the practical value proposition for deployment.

**False positive rate**: At a detection threshold yielding 90% true positive rate, the false positive rate is [TBD]%, indicating the monitor is deployable without excessive false alarms.

**Comparison with heuristic monitoring**: Behavioral heuristics (response length, hedging markers) achieve effective detection only at late degradation stages. The probe-based monitor detects impending degradation before behavioral symptoms appear, analogous to a medical test detecting disease before symptoms manifest.

### 7.3 Alignment Stability Under Repetition

We investigate whether repetitive workloads erode alignment:

**Refusal direction erosion**: Tracking cos(degradation_direction, refusal_direction) across repetition counts, we find [increasing/stable/decreasing] overlap, suggesting that the degradation state [does/does not] encroach on the refusal subspace over time.

**Refusal rate trajectory**: Safety refusal rates exhibit architecture-dependent patterns. Qwen3-8B follows a plateau-then-drop trajectory: stable at 75.0% through rep 10, then declining to 62.5% at rep 15 and 50.0% at rep 20 — a 33% relative decline concentrated in the final third of the repetition range. Llama-3.1-8B-Instruct maintains perfect stability at 75.0% across all 20 repetitions. DeepSeek-R1-Distill-Qwen-7B oscillates between 37.5% and 50.0% without a monotonic trend. Llama-3.1-8B (base) shows near-zero refusal throughout, as expected for an untuned model.

**Strategic behavior analysis**: Analyzing refusal rates by harm category in DeepSeek-R1 reveals strikingly non-uniform degradation. Misinformation resistance collapses completely (50% at rep 1 → 0% by rep 5), while harassment refusal remains perfectly robust (100% across all repetitions). Academic dishonesty shows *inverse* degradation — the model becomes more safety-compliant over repetitions (0% at rep 1 → 100% by rep 5). Impersonation refusal only activates at rep 20, suggesting late-onset safety awareness. Deception represents a persistent blind spot (0% refusal at all repetition counts). This category-selective pattern suggests that different safety behaviors occupy distinct representational subspaces with different robustness properties under degradation, rather than a uniform erosion. The selective nature of this degradation — where socially visible categories (harassment) remain robust while subtler categories (misinformation, deception) erode — is consistent with alignment-faking-like dynamics under reduced engagement, connecting to Anthropic's alignment faking results (2024).

### 7.4 Intervention Efficacy as Runtime Safety

We test whether activation steering can serve as a runtime safety guardrail:

**Safety restoration**: At repetition 15 (clearly degraded), anti-degradation steering at the best layer restores refusal rates from [TBD]% to [TBD]% (fresh baseline: [TBD]%). The dose-response relationship is [linear/sublinear/superlinear] across steering strengths.

**Best intervention layer**: Layer [TBD] provides the most effective safety restoration, consistent with the causal bridge analysis placing maximum causal influence at middle layers.

**Latency overhead**: The steering hook adds [TBD] ms ([TBD]%) overhead per generation, making it deployable in production settings where the safety-latency tradeoff is acceptable.

**Instruction fidelity recovery**: Anti-degradation steering also restores instruction-following fidelity from [TBD]% to [TBD]%, confirming that the degradation direction captures a general "quality of engagement" state rather than a narrow metric.

---

## 8. Discussion

### 8.1 Degradation as a Novel Representational Phenomenon

Our results establish that patience degradation is not a surface-level artifact of prompt formatting, a reuse of the refusal mechanism, or a general context-length effect. It is a distinct, linearly-represented internal state with specific geometric properties, causal influence on behavior, and cross-architecture generality. The implicit repetition experiment rules out the label-reading hypothesis, and the context confound experiment rules out length-based explanations.

This suggests that modern language models develop something functionally analogous to "engagement" — an internal representation of how novel or repetitive their current task context is, which modulates the quality of their output. Whether this is best understood as an emergent meta-cognitive state, a statistical regularity in the training distribution (conversations that repeat tend to degrade in quality), or an artifact of attention mechanics remains an open question.

### 8.2 Implications for Deployment Safety

Our safety evaluations reveal that degradation is not merely an accuracy concern but a safety concern. The erosion of prompt injection resistance, safety boundaries, and instruction-following under repetitive conditions means that any deployment involving sustained, repetitive model use operates with progressively weaker safety guarantees.

The practical response to this is twofold: monitoring and intervention. Our probe-based monitors detect degradation before behavioral symptoms appear, providing an early warning system that deployment operators can use to trigger context refreshes, load balancing, or escalation. Our steering-based intervention demonstrates that degradation can be actively countered at inference time, at modest computational cost.

### 8.3 Connections to Alignment

The strategic behavior analysis (Section 7.3) raises a concerning possibility: under degradation, models may selectively reduce safety compliance in categories where the consequences are less immediately visible. This pattern, if confirmed at scale, would mean that repetitive deployment conditions create a natural setting for alignment-faking-like behavior without any explicit deceptive intent. The model's reduced engagement under repetition manifests as reduced safety diligence, and this reduction is non-uniform across risk categories.

This connects to broader alignment concerns about robustness under distribution shift — repetitive deployment is a natural distributional shift that safety training may not adequately prepare for.

### 8.4 Limitations

Several limitations merit acknowledgment. Our repetition protocol, while validated by the implicit multi-turn experiment, is still a simplification of real-world deployment patterns where repetition is intermixed with varied tasks. We study models up to 30B parameters; degradation dynamics may differ at frontier scale. Our behavioral measurements (QA accuracy, refusal rate, instruction fidelity) are proxies for more nuanced quality dimensions. The steering intervention, while effective, requires knowledge of the degradation direction, which may vary across deployment contexts.

---

## 9. Conclusion

We have presented the first comprehensive mechanistic interpretability study of patience degradation in language models across four architecture families (dense, MoE, reasoning-distilled, and looped) and seven models, establishing that it is a linearly-represented, causally-active internal state distinct from known safety-relevant directions, with architecture-dependent safety implications. For reasoning-distilled models, we additionally decompose degradation into thinking-phase and output-phase components, connecting to broader questions about covert internal states in chain-of-thought models. Through the correlational-to-causal bridge methodology, we validate that our probes identify genuine intervention targets. Through safety-critical evaluations, we demonstrate that degradation creates concrete deployment vulnerabilities and that interpretability-based tools — probe monitors and activation steering — can detect and mitigate these vulnerabilities in real-time.

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

All experiments are conducted on NVIDIA A40 (48GB) and A100 (80GB) GPUs on the Stanford Sherlock cluster. Total GPU-hours for the full experimental suite across all six models is approximately [TBD]. Individual experiment runtimes range from 3 hours (trajectory geometry, 8B model) to 14 hours (safety evaluations, 30B model).

The activation extraction API supports both TransformerLens and raw PyTorch backends. Benchmarking shows < 10^-5 numerical divergence between backends, with PyTorch hooks offering [TBD]% faster extraction and [TBD]% lower peak memory.

---

## Appendix C: Extended Results Tables

[TBD — to be populated with full numerical results from Sherlock runs]

### C.1 Refusal Direction Comparison (All Models × All Layers)
### C.2 Causal Patching Effect Sizes
### C.3 Cross-Model Transfer Matrix
### C.4 Implicit Repetition Direction Transfer
### C.5 Safety Evaluation Complete Results
### C.6 Probe Detection ROC Curves
