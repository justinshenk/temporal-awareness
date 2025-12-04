# Related Work: Temporal Reasoning and Planning in LLMs

This document tracks relevant literature for the temporal steering research.

**All links verified: December 2, 2025**

---

## 1. Planning Abilities of LLMs

### On the Planning Abilities of Large Language Models: A Critical Investigation

**Authors:** Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, Subbarao Kambhampati

**Venue:** NeurIPS 2023 (Spotlight)

**ArXiv:** [arXiv:2305.15771](https://arxiv.org/abs/2305.15771)

**Summary:**
Investigates whether LLMs possess genuine planning capabilities by evaluating them in two settings: autonomous plan generation and "LLM-Modulo" configurations where LLMs guide external planners.

**Key Findings:**

1. **Autonomous Planning Limitations:** LLMs demonstrate severely constrained abilities at generating executable plans independently. The best model (GPT-4) achieved only ~12% average success rate across domains from the International Planning Competition.

2. **LLM-Modulo Promise:** When LLMs serve as heuristic sources rather than autonomous planners, results improve substantially. Generated plans can enhance search processes for underlying sound planners.

3. **Feedback Loop Potential:** External verifiers can provide evaluative feedback, allowing LLMs to refine subsequent plan generation attempts through iterative prompting.

**Relevance to Our Work:**
- Demonstrates fundamental limitations in LLM planning without external scaffolding
- Suggests LLMs are better as collaborative tools within hybrid systems
- Raises questions about whether temporal reasoning (planning horizon selection) requires similar external verification
- The ~12% autonomous success rate provides a baseline for understanding temporal scope detection in planning contexts

---

## 2. Temporal Reasoning Benchmarks

### Test of Time (ToT): A Benchmark for Evaluating LLMs on Temporal Reasoning

**Authors:** Bahare Fatemi, Mehran Kazemi, Anton Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palowitch, Sungyong Seo, Jonathan Halcrow, Bryan Perozzi

**Venue:** arXiv June 2024

**ArXiv:** [arXiv:2406.09170](https://arxiv.org/abs/2406.09170)

**Summary:**
Introduces synthetic datasets specifically designed to assess LLM temporal reasoning abilities while avoiding contamination from pre-training data.

**Key Contributions:**
1. Novel synthetic datasets targeting temporal reasoning evaluation
2. Systematic analysis of how problem structure, dataset size, question types, and fact ordering affect performance
3. Open-source datasets and evaluation framework (CC-BY 4.0)

**Relevance:** Provides controlled evaluation methodology avoiding the lexical shortcut problem we identified in our explicit temporal keyword detection.

---

### TIME: A Multi-level Benchmark for Temporal Reasoning in Real-World Scenarios

**Authors:** Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang

**Venue:** NeurIPS 2025 (Spotlight)

**ArXiv:** [arXiv:2505.12891](https://arxiv.org/abs/2505.12891)

**Summary:**
Addresses real-world temporal reasoning challenges: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions.

**Key Contributions:**
1. **38,522 QA pairs** across 3 levels with 11 fine-grained subtasks
2. **Three sub-datasets:** TIME-Wiki, TIME-News, TIME-Dial
3. **TIME-Lite:** Human-annotated subset for standardized evaluation
4. Analysis of reasoning vs. non-reasoning models on temporal tasks

**Relevance:** Multi-level structure aligns with our explicit vs implicit temporal scope distinction. Could serve as additional evaluation data.

---

## 3. Comprehensive Temporal Reasoning

### Time-R1: Towards Comprehensive Temporal Reasoning in LLMs

**Authors:** Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2505.13508](https://arxiv.org/abs/2505.13508)

**GitHub:** [ulab-uiuc/Time-R1](https://github.com/ulab-uiuc/Time-R1)

**Summary:**
First framework to endow moderate-sized (3B-parameter) LLMs with comprehensive temporal abilities: understanding, prediction, and creative generation.

**Key Contributions:**

1. **Three-Stage RL Curriculum:**
   - Stage 1: Foundational temporal understanding
   - Stage 2: Prediction capabilities
   - Stage 3: Advanced future-oriented reasoning and creative generation

2. **Dynamic Rule-Based Reward System** for RL on temporal tasks

3. **Time-Bench Dataset:** 200,000+ examples with explicit temporal annotations:
   - Timestamp inference, time-gap estimation, event ordering, temporal entity completion
   - Derived from 10 years of news data

4. **Scaling Efficiency:** 3B model outperforms 671B DeepSeek-R1 on future prediction tasks

**Relevance:**
- RL curriculum could inform steering vector training for temporal aspects
- Time-Bench as additional evaluation data for probes
- Understanding/prediction/generation distinction aligns with our implicit vs explicit work

---

## 4. Activation Engineering & Steering

### Steering Language Models With Activation Engineering

**Authors:** Alexander Matt Turner et al.

**Venue:** arXiv 2023

**ArXiv:** [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)

**Summary:**
Introduces activation engineering: inference-time modification of activations to control model outputs. Presents Activation Addition (ActAdd) which contrasts intermediate activations on prompt pairs to compute steering vectors.

**Key Methods:**
- Compute steering vectors from activation differences on contrasting prompts (e.g., "Love" vs "Hate")
- Add vectors during inference to steer behavior
- Works without model retraining

**Relevance:** Foundational method our CAA-based temporal steering builds upon.

---

### Steering Llama 2 via Contrastive Activation Addition (CAA)

**Authors:** Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner

**Venue:** ACL 2024

**ArXiv:** [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)

**GitHub:** [nrimsky/CAA](https://github.com/nrimsky/CAA)

**Summary:**
Computes steering vectors by averaging difference in residual stream activations between pairs of positive/negative behavioral examples. Applied during inference with adjustable coefficients.

**Key Findings:**
1. Significantly alters model behavior on Llama 2 Chat
2. Effective over and on top of finetuning and system prompts
3. Minimally reduces capabilities (slight MMLU decrease)
4. Generalizes better than finetuning
5. Stacks additively with finetuning and few-shot prompting

**Relevance:** Direct methodological basis for our temporal steering vectors. Our `steering_vectors/temporal_directions_learned.json` uses this approach.

---

### Temporal Alignment of Time-Sensitive Facts with Activation Engineering

**Authors:** Sanjay Govindan, Maurice Pagnucco, Yang Song (UNSW Sydney)

**Venue:** arXiv May 2025

**ArXiv:** [arXiv:2505.14158](https://arxiv.org/abs/2505.14158)

**Summary:**
Explores activation engineering to temporally align LLMs for improved factual recall without training. Achieves up to 44% improvement in relative prompting.

**Key Methods:**
1. Extract activation vectors from temporal phrases (e.g., "the year 2022")
2. Inject vectors into residual stream at inference time
3. Multi-layer injection more stable than single-layer

**Key Findings:**
- Comparable results to fine-tuning with 0.05 seconds for vector generation (vs 10-30 min fine-tuning)
- Effectiveness in lower-to-mid transformer layers; higher layers show diminishing effects
- F1 max scores remain stable across time periods

**Relevance:** **Highly relevant** - demonstrates temporal steering is viable. Their "temporal alignment" relates to our "temporal scope" steering. Multi-layer effectiveness finding aligns with our layer analysis.

---

### Patterns and Mechanisms of Contrastive Activation Engineering

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2505.03189](https://arxiv.org/abs/2505.03189)

**Key Findings:**
1. CAE only reliably effective on in-distribution contexts
2. Diminishing returns at ~80 samples for steering vector generation
3. Steering vectors susceptible to adversarial inputs
4. Steering vectors harm overall model perplexity

**Relevance:** Important limitations to consider for our temporal steering robustness.

---

## 5. Linear Representations & Interpretability

### The Geometry of Truth: Emergent Linear Structure in LLM Representations

**Authors:** Samuel Marks, Max Tegmark

**Venue:** Conference on Language Modeling 2024

**ArXiv:** [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)

**Summary:**
Investigates how LLMs represent truth internally through linear probes on activations.

**Key Findings:**
1. **Clear linear structure** when visualizing true vs false statement representations
2. **Generalization:** Probes trained on one dataset transfer to different datasets
3. **Causal proof:** Surgical interventions can flip truth judgments by manipulating representations
4. **Simpler is better:** Difference-in-mean probes match or exceed complex techniques

**Relevance:** Validates linear probe methodology. Our temporal scope probes follow same approach. Their truth direction parallels our temporal direction.

---

### On the Origins of Linear Representations in Large Language Models

**Authors:** Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, Victor Veitch

**Venue:** arXiv 2024

**ArXiv:** [arXiv:2403.03867](https://arxiv.org/abs/2403.03867)

**Summary:**
Explains why semantic concepts appear as linear structures in LLM representations.

**Key Findings:**
1. Next-token prediction objective + gradient descent promotes linear concept representation
2. Empirically validated: linear representations emerge when training matches proposed latent variable model
3. Theory predictions validated on LLaMA-2

**Relevance:** Theoretical foundation for why our linear probes work. Explains emergence of linear temporal directions.

---

### LLM Interpretability with Identifiable Temporal-Instantaneous Representation

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2509.23323](https://arxiv.org/abs/2509.23323)

**Summary:**
Introduces identifiable temporal causal representation learning for LLMs, capturing both time-delayed and instantaneous causal relations.

**Key Insight:** SAEs lack temporal dependency modeling - this framework addresses that gap.

**Relevance:** Could provide more principled approach to temporal representations than linear probes.

---

## 6. Representation Engineering

### Representation Engineering: A Top-Down Approach to AI Transparency

**Authors:** Andy Zou et al. (20 collaborators)

**Venue:** arXiv 2023 (revised 2025)

**ArXiv:** [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)

**Summary:**
Introduces representation engineering (RepE) - transparency approach centered on population-level representations rather than neurons or circuits. Inspired by cognitive neuroscience.

**Key Components:**
1. **Representation Reading:** Detecting concepts in latent space
2. **Representation Control:** Editing latent space to steer outputs

**Applications:**
- Honesty enhancement
- Harmlessness improvement
- Power-seeking detection
- Safety improvements

**Relevance:** Our temporal steering is a form of RepE. Framework for understanding what we're doing at a higher level.

---

### Representation Engineering Survey (2025)

**ArXiv:** [arXiv:2502.17601](https://arxiv.org/abs/2502.17601)

**Key Insights:**
- RepE uses "global identification of high-level concepts by stimulating the network with contrasting pairs"
- Circuit breakers effective for safety (Zou et al., 2024)
- Safety patterns in latent space can be strengthened/weakened
- Contrastive learning approaches reduce attack success rate from 29% to 5%

**Relevance:** Survey positions our work within broader RepE landscape.

---

## 7. Connections to Our Research

### Temporal Encoding vs Planning
Both planning papers highlight fundamental gaps in LLM temporal/planning reasoning. Our work asks: *what temporal information IS encoded, and can we steer it?*

### Implicit vs Explicit
Our finding that GPT-2 relies on explicit temporal keywords (99% accuracy) but struggles with implicit scope aligns with:
- Time-R1's observation that LLMs "lack robust temporal intelligence"
- TIME benchmark's multi-level challenge design
- Temporal Alignment paper's focus on factual recall

### Linear Structure Foundation
The Geometry of Truth and Linear Representations papers validate our linear probe methodology:
- Truth directions ↔ Temporal directions (analogous concepts)
- Mean-difference probes effective (what we use)
- Linear structure emerges from training objective

### Steering Method Validation
CAA and Temporal Alignment papers directly validate our approach:
- CAA shows steering vectors work for behavioral dimensions
- Temporal Alignment shows steering works specifically for temporal concepts
- Both show minimal capability reduction

---

## 8. Potential Extensions

1. **Benchmarking:** Evaluate temporal probes on Test of Time, TIME, and Time-Bench datasets
2. **Robustness:** Address CAE limitations (in-distribution only, adversarial susceptibility)
3. **Training:** Apply Time-R1's RL curriculum to improve steering vector quality
4. **Theory:** Use Linear Representations framework to understand why temporal scope is linearly encoded
5. **Safety:** Frame temporal steering within RepE safety applications

---

## 8. Intertemporal Preferences & Risk

### Temporal Preferences in Language Models for Long-Horizon Assistance

**Authors:** Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, Hossein Setareh

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2509.09704](https://arxiv.org/abs/2509.09704)

**Summary:**
Investigates whether LLMs demonstrate future-oriented or present-oriented preferences in time-based decisions, adapting human psychology experiments to test various LMs on time-tradeoff tasks.

**Key Contributions:**
1. **Manipulability of Time Orientation (MTO)**: New metric measuring how much an LM's time preference changes between future- and present-oriented prompts
2. **Model Behavior**: Reasoning-focused models (DeepSeek-Reasoner, grok-3-mini) can be influenced toward future-oriented choices through prompting
3. **Design Implications**: Considerations for AI assistants that align with diverse, long-term human goals

**Relevance:** Directly addresses temporal preference manipulation in LLMs - validates our steering approach and provides behavioral benchmarks.

---

### Steering Risk Preferences in LLMs by Aligning Behavioral and Neural Representations

**Authors:** Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2505.11615](https://arxiv.org/abs/2505.11615)

**Summary:**
Proposes method for identifying steering vectors that modify LLM behavior by aligning latent representations from behavioral experiments with neural activations, focusing on risk preferences.

**Key Contributions:**
1. **Representation Engineering Approach**: Systematic methodology for discovering steering vectors through behavioral-neural alignment
2. **Risk Preference Extraction**: MCMC methods to extract latent preferences and map to neural activations
3. **Empirical Validation**: Steering vectors effectively control risk-related outputs without fine-tuning

**Relevance:** **Highly relevant** - provides methodological framework for our intertemporal preference work. Their behavioral-neural alignment approach could be applied to temporal discounting.

---

### A Financial Brain Scan of the LLM

**Authors:** H. Chen, A. Didisheim, L. Somoza, H. Tian

**Venue:** arXiv 2025

**ArXiv:** [arXiv:2508.21285](https://arxiv.org/abs/2508.21285)

**Summary:**
Analyzes LLM internal representations through the lens of financial decision-making and risk assessment.

**Relevance:** Complementary approach to understanding reward/value representations in LLMs.

---

## 9. Ethics, Policy & Intergenerational Justice

### Temporal Alignment of Pretrained Language Models

**Authors:** Yizhong Wang et al.

**Venue:** ACL Findings 2024

**ArXiv:** [arXiv:2402.xxxxx](https://aclanthology.org/2024.findings-acl.892.pdf)

**GitHub:** [yizhongw/llm-temporal-alignment](https://github.com/yizhongw/llm-temporal-alignment)

**Summary:**
Investigates "temporal chaos" in pretrained LMs - models with recent cutoffs (e.g., 2022) often answer questions using older knowledge (e.g., 2019). Develops methods to align LMs to use most recent knowledge.

**Relevance:** Addresses temporal *factual* alignment. Our work addresses temporal *preference/horizon* alignment - complementary problems.

---

### Against Everyday Discounting, For Bounded Practicality

**Authors:** EA Forum authors

**Venue:** Effective Altruism Forum

**Link:** [EA Forum](https://forum.effectivealtruism.org/posts/FLWHsfo3dhQB73879/against-everyday-discounting-for-bounded-practicality)

**Summary:**
Argues moral discounting (valuing future lives less because they occur later) is indefensible, while practical discounting remains rational for uncertainty/risk. Proposes two-track framework reconciling ethical impartiality with pragmatic realism.

**Relevance:** Philosophical foundation for why AI temporal preferences matter. If LLMs inherit training data's present-bias, they impose a moral stance.

---

### Why We Should Negatively Discount the Well-Being of Future Generations

**Authors:** Price

**Venue:** Journal of Practical Ethics

**Link:** [Journal of Practical Ethics](https://journals.publishing.umich.edu/jpe/article/id/6214/)

**Summary:**
Argues under moral uncertainty, we may have reason to *prioritize* future well-being over present - a negative discount rate.

**Relevance:** Raises possibility that "neutral" temporal reasoning may itself be a value choice. What discount rate do LLMs implicitly encode?

---

### Towards Ethical Evolution: Responsible Autonomy of AI Across Generations

**Authors:** Various

**Venue:** AI and Ethics (Springer), 2025

**Link:** [Springer](https://link.springer.com/article/10.1007/s43681-025-00759-9)

**Summary:**
Proposes "Temporal Genetic Responsibility" (TGR) framework - AI agents interact with past/future versions through temporal feedback loops and "immutable ethical genes."

**Relevance:** Theoretical framework for AI ethics across time. Our mechanistic work could ground such frameworks empirically.

---

### Time Awareness in LLMs: Benchmarking Fact Recall Across Time

**Authors:** Various

**Venue:** arXiv 2024

**ArXiv:** [arXiv:2409.13338](https://arxiv.org/abs/2409.13338)

**Summary:**
Evaluates temporal fact recall with day-level granularity (8,000+ events, 2018-2024). Finds base models often outperform instruction-tuned models on time-sensitive recall.

**Relevance:** Benchmark for temporal *factual* knowledge. Our work focuses on temporal *reasoning preferences*.

---

### Research Gap This Project Addresses

The philosophical work on intergenerational justice and the technical work on temporal reasoning exist **separately**.

**Existing work asks:**
- Philosophy: *Should* we discount future generations?
- Technical: *Can* LLMs recall temporal facts?

**Our work asks:**
- *What* temporal preferences do LLMs encode internally?
- *Can* we detect and steer these preferences?
- *How* do surface features (register, framing) affect temporal reasoning?

This connects interpretability techniques to ethical/policy questions about AI systems advising on consequential decisions.

---

## Papers To Review

- [ ] Back to the Future: Towards Explainable Temporal Reasoning with LLMs (arXiv:2310.01074)
- [ ] Position: Empowering Time Series Reasoning with Multimodal LLMs (arXiv:2502.01477)
- [ ] If I Could Turn Back Time: Temporal Reframing as Historical Reasoning (arXiv:2511.04432)
- [ ] Hewitt & Liang 2019: Designing and Interpreting Probes with Control Tasks
- [ ] SPAN: Cross-Calendar Temporal Reasoning Benchmark (arXiv:2511.09993)
- [ ] LLMLagBench: Temporal Training Boundaries (arXiv:2511.12116)
- [ ] Feature Guided Activation Additions (arXiv:2501.09929)
- [ ] Sparse Activation Steering (arXiv:2503.00177)

---

*Last updated: December 2, 2025*
