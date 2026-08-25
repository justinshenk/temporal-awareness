# Related-work citation prep (2026-08-20)

Compiled for the related-work / positioning pass. Every arXiv ID was verified against its
abstract page at compile time. Excludes the 11 entries already in `context_fatigue.tex`'s
bibliography. Each entry ends with a DIFF line — the specific sentence of differentiation the
paper can make against it.

## Tier 1 — must cite AND distinguish (closest prior art)

**camassa2026instructioninduction** — *Do as I Say, Not as I Do: Instruction-Induction Conflict
in LLMs.* Camassa & Shiller, arXiv:2605.20382 (2026).
Pits an explicit instruction against hardcoded assistant turns demonstrating a competing pattern
over up to 50 turns; instruction-following collapses to pattern-following at model-dependent
rates (1–99%), driven mainly by output diversity.
DIFF: their conflicting turns are experimenter-hardcoded and adversarial-by-design, surveyed
across 13 models behaviorally; ours are the model's **own benign replies** (self-reinforcing
precedent), the erosion is shown to be applicability-specific (off-domain filler never erodes at
12× the context), and we add the mechanism: flat system-span attention, plus two independent
causal restorations (attention clamp, policy restatement).

**dongre2026attentioncloses** — *When Attention Closes: How LLMs Lose the Thread in Multi-Turn
Interaction.* Dongre, Hsieh, Lai, et al., arXiv:2605.12922 (2026). [READ IN FULL 2026-08-20]
GAR (attention from response tokens to system-prompt goal tokens, averaged over layers × heads ×
response tokens) declines monotonically over 50 filler turns in 4 families; hard-masking the
instruction out of the attention window collapses recall (→11%) and persona compliance in
Mistral; residual probes decode goal info at AUC ≤0.99 after "closure"; whether behavior
survives vanishing attention is architecture-indexed.
DIFF, three-part: (1) GAR's formula does not normalize by the goal span's token fraction — it is
raw share, which falls by arithmetic as context grows; our enrichment correction removes that
decline entirely, and the corrected quantity is flat in eroding and non-eroding arms alike
(verify formula against their PDF before quoting). (2) Their causal arm is total ablation
(attention necessary = our E5); our clamp is graded and bidirectional — the upclamp *restores*
compliance against 42 counter-exemplars, an arm they lack. (3) Their survival-at-zero-attention
is architecture-indexed with always-inapplicable filler; we show the same model at the same
near-zero share survives (code) or collapses (mmlu) depending on what the filler demonstrates —
content-indexed, invisible to their design. They win on breadth (4 families) and residual
probing; cite as the closest prior art and the phenomenon's independent confirmation.

**davidson2025taskrepresentation** — *Do different prompting methods yield a common task
representation in language models?* Davidson, Gureckis, Lake, Williams, NeurIPS 2025,
arXiv:2505.12075.
Function vectors from demonstration-based vs instruction-based prompts do NOT share a common
task representation — distinct, partly overlapping head sets.
DIFF: closest mechanistic precedent for instruction vs demonstration as separate channels; we
show the two channels **compete** — demonstrations functionally overwrite the instruction's
behavioral force under accumulated context, at constant instruction-span attention, and clamping
either side's weight decides the winner.

**wang2025unabletoforget** — *Unable to Forget: Proactive Interference Reveals Working Memory
Limits in LLMs Beyond Context Length.* Wang & Sun, ICML 2025 Workshop on Long-Context Foundation
Models, arXiv:2506.08184.
PI-LLM paradigm: accuracy declines log-linearly as interfering similar key-value updates
accumulate, independent of raw context length.
DIFF: nearest neighbor to E3's competition result but with no attention instrumentation; we show
near-duplicate interference costs accuracy at **held-constant attention mass**, ruling out
attention capture as its mechanism.

**zhang2023pasta** — *Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs.*
Zhang, Singh, Liu, et al., ICLR 2024, arXiv:2311.02262.
PASTA multiplicatively boosts attention on profiled head subsets toward user-specified spans to
improve instruction compliance.
DIFF: the clamp's closest methods rival — PASTA is a deployable steering method on searched
heads; our clamp is an additive-mask bias uniform over heads, used as a **measurement
instrument** for mediation (bit-identical no-op at scale 1), with the E6 upclamp arm showing the
steering direction and its zero-sum accuracy cost.

**hsieh2024foundinmiddle** — *Found in the Middle: Calibrating Positional Attention Bias
Improves Long Context Utilization.* Hsieh, Chuang, Li, et al., ACL Findings 2024,
arXiv:2406.16008.
U-shaped intrinsic positional attention bias as the root of lost-in-the-middle; global
recalibration recovers up to 15 points.
DIFF: they calibrate a static position-driven bias globally; we clamp specific content spans in
accumulating multi-turn context and establish **when attention is and is not the mediating
variable** (displacement yes, competition and format contamination no).

## Tier 2 — long-context degradation, behavioral

**hsieh2024ruler** — *RULER: What's the Real Context Size of Your Long-Context Language Models?*
Hsieh, Sun, Kriman, et al., COLM 2024, arXiv:2404.06654. Near-perfect needle scores coexist with
sharp degradation on harder long-context tasks. DIFF: purely behavioral; cannot separate the
mechanisms our clamps dissociate.

**modarressi2025nolima** — *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* Modarressi,
Deilamsalehy, Dernoncourt, et al., ICML 2025, arXiv:2502.05167. Removing lexical-match shortcuts
collapses 12 long-context LLMs; attributes difficulty to attention locating semantic matches.
DIFF: correlational attribution across models; we manipulate attention mass causally on one
model and exhibit a constant-mass failure mode their design cannot see.

**kuratov2024babilong** — *BABILong: Testing the Limits of LLMs with Long Context
Reasoning-in-a-Haystack.* Kuratov, Bulatov, Anokhin, et al., NeurIPS 2024 D&B,
arXiv:2406.10149. Models use only 10–20% of claimed context on distributed-fact reasoning.
DIFF: varies distance, distractor load, and reasoning complexity at once; our paired matched-fill
designs vary them independently.

**li2024longiclbench** — *Long-context LLMs Struggle with Long In-context Learning.* Li, Zhang,
Do, et al., arXiv:2404.02060 (2024). Extreme-label ICL collapses as label space and shot count
grow; recency bias toward recent labels. DIFF: content-retrieval failure under many shots; our
E6 is a format-level failure under **three** shots with the instruction's attention intact.

**agarwal2024manyshot** — *Many-Shot In-Context Learning.* Agarwal, Singh, Zhang, et al.,
NeurIPS 2024 Spotlight, arXiv:2404.11018. Hundreds-to-thousands of shots yield large gains.
DIFF: the same lever framed as pure benefit; we show it overriding an explicit system instruction
and identify the causal controls that reverse it.

**guo2024serialposition** — *Serial Position Effects of Large Language Models.* Guo & Vosoughi,
arXiv:2406.15981 (2024). Primacy/recency biases across tasks; prompt-only mitigation unreliable.
DIFF: correlational position survey; our displacement is causally attributed, and competition is
positionless (matched position and fill).

**guo2026stoplistening** — *Stop Listening to Me! How Multi-turn Conversations Can Degrade LLM
Reliability.* Guo, Yan, Baidya, et al., arXiv:2603.11394 (2026). "Conversation tax" up to 30%
on clinical multi-turn benchmarks via suggestion-adoption. DIFF: one behavioral failure mode
measured by outcome only; we decompose the tax into three causally verified mechanisms, two not
explained by suggestion-following.

**chattaraj2026rememberfirst** — *LLMs Remember First, Forget Last: Dual-Process Interference in
Large Language Models.* Chattaraj & Raj, arXiv:2603.00270 (2026). Proactive dominates
retroactive interference across 39 models; the two are uncorrelated. DIFF: temporal-direction
axis for interference; we offer a mechanistic axis (attention-mediated vs not).

**martin2026classifiercontextrot** — *Classifier Context Rot: Monitor Performance Degrades with
Context Length.* Martin & Roger, arXiv:2605.12366 (2026). Safety monitors miss violations 2–30×
more after ~800K benign tokens; periodic reminders partially mitigate. DIFF: outcome-level rot in
a safety setting; our refresh arm is the causal, mechanism-localized version of their reminder
mitigation.

## Tier 3 — instruction following, hierarchy, format control

**wallace2024instructionhierarchy** — *The Instruction Hierarchy: Training LLMs to Prioritize
Privileged Instructions.* Wallace, Xiao, Leike, et al., arXiv:2404.13208 (2024). Trains
system-over-user priority. DIFF: training-time priority does not address our failure — nothing
in the context contradicts the system prompt; precedent, not priority, is what flips.

**zhang2025iheval** — *IHEval: Evaluating Language Models on Following the Instruction
Hierarchy.* Zhang, Li, Zhang, et al., NAACL 2025 oral, arXiv:2502.08745. Models resolve
explicit cross-level conflicts correctly only ~48%. DIFF: instruction-vs-instruction conflict;
ours is instruction-vs-unlabeled-demonstration with no conflicting directive anywhere.

**mu2025systemprompt** — *A Closer Look at System Prompt Robustness.* Mu, Lu, Lavery, Wagner,
arXiv:2502.12197 (2025). Adherence fails under conflicting/adversarial user turns. DIFF: our
eroding pressure is benign and self-generated; total collapse in 3 turns; localized and reversed
mechanistically.

**he2024multiif** — *Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions
Following.* He, Jin, Wang, et al., arXiv:2410.15553 (2024). Turn-over-turn instruction decay
across 14 models. DIFF: documents the trend; we identify a causal driver and a null arm (equal
length, no decay) that isolates precedent from turn count.

**zhou2023ifeval** — *Instruction-Following Evaluation for Large Language Models.* Zhou, Lu,
Mishra, et al., arXiv:2311.07911 (2023). Verifiable-constraint instruction eval. DIFF:
measurement precedent — we apply the checkable-compliance philosophy turn-by-turn inside one
conversation to trace a trajectory.

**sclar2024formatspread** — *Quantifying Language Models' Sensitivity to Spurious Features in
Prompt Design.* Sclar, Choi, Tsvetkov, Suhr, ICLR 2024, arXiv:2310.11324. Up to 76-point
accuracy swings across equivalent formats. DIFF: format as evaluation confound; ours is the
model's own output format drifting live within a conversation, measured and causally reversed.

**wei2023icldifferently** — *Larger Language Models Do In-Context Learning Differently.* Wei,
Wei, Tay, et al., arXiv:2303.03846 (2023). Large models follow flipped-label exemplars over
semantic priors. DIFF: exemplars overriding a **trained prior**; ours override an **explicit
in-context instruction**, and the exemplars are self-generated — the effect feeds itself.

**anil2024manyshot** — *Many-shot Jailbreaking.* Anil, Durmus, Panickssery, et al.,
NeurIPS 2024.
Hundreds of in-context demonstrations of harmful compliance override safety training; attack
effectiveness follows a power law in shot count.
DIFF: adversarial demonstrations overriding *weight-level* safety training at hundreds of
shots; ours are benign, self-generated demonstrations overriding an *in-context* instruction
within three turns, with the contest's causal weight (instruction-span attention) identified
and two full restorations demonstrated.

**hong2025sycophancymultiturn** — *Measuring Sycophancy of Language Models in Multi-turn
Dialogues.* Hong, Byun, Kim, et al., Findings of EMNLP 2025, arXiv:2505.23840. Stance-flipping
under sustained user pressure. DIFF: conformity to the user's pressure vs our conformity to the
assistant's own precedent with zero user pushback.

**chen2024struq** — *StruQ: Defending Against Prompt Injection with Structured Queries.* Chen,
Piet, Sitawarin, Wagner, USENIX Security 2025, arXiv:2402.06363. Channel separation so only the
prompt channel is obeyed. DIFF: our eroding signal contains no instructions at all — channel
separation cannot block behavioral precedent. (Cite at most alongside hines2024spotlighting.)

**hines2024spotlighting** — *Defending Against Indirect Prompt Injection Attacks With
Spotlighting.* Hines, Lopez, Hall, et al., arXiv:2403.14720 (2024). Provenance-marking of
untrusted input. DIFF: our filler is trusted, on-task assistant output; provenance marking does
not address precedent-based erosion.

## Tier 4 — ICL and attention mechanisms

**olsson2022induction** — *In-context Learning and Induction Heads.* Olsson, Elhage, Nanda,
et al., arXiv:2209.11895 (2022). Induction heads as ICL's driving circuit. DIFF: the
circuit-level basis for the format-copying hypothesis we state at span level; we do not verify
it at head level (candidate limitation sentence).

**crosbie2024inductionpattern** — *Induction Heads as an Essential Mechanism for Pattern
Matching in In-context Learning.* Crosbie & Shutova, Findings of NAACL 2025, arXiv:2407.07011.
Ablating induction heads collapses pattern-matching ICL. DIFF: causal circuit evidence that
copying-from-examples exists — the mechanism our mmlu arm's behavior implicates; we implicate it
behaviorally and via span attention, not ablation.

**hendel2023taskvectors** — *In-Context Learning Creates Task Vectors.* Hendel, Geva,
Globerson, Findings of EMNLP 2023, arXiv:2310.15916. Demonstrations compress to a residual
task vector. DIFF: the no-attention-signature alternative mechanism for E6 (mode set upstream);
cite when discussing why flat enrichment does not settle the routing question.

**todd2024functionvectors** — *Function Vectors in Large Language Models.* Todd, Li, Sen Sharma,
et al., ICLR 2024, arXiv:2310.15213. Head-transported compact task representation. DIFF: same
role as hendel2023 — the representational channel our span-level measurement cannot exclude.

**wang2023labelwords** — *Label Words are Anchors: An Information Flow Perspective for
Understanding In-Context Learning.* Wang, Li, Dai, et al., EMNLP 2023, arXiv:2305.14160. Label
words aggregate demonstration semantics. DIFF: explains why demonstrations work; we study the
failure direction — demonstrations displacing the instructed format's behavioral force.

**xiao2023streamingllm** — *Efficient Streaming Language Models with Attention Sinks.* Xiao,
Tian, Chen, et al., ICLR 2024, arXiv:2309.17453. Position-driven sink tokens. DIFF: sinks are
content-independent; our attention effects are content- and structure-driven. Also relevant to
our "other" span accounting.

**gu2024attentionsink** — *When Attention Sink Emerges in Language Models.* Gu, Pang, Du, et al.,
ICLR 2025 Spotlight, arXiv:2410.10781. Sinks as softmax-normalization byproduct of pretraining.
DIFF: training-time account; our redistribution is inference-time and clamp-reversible.

**wu2025positionbias** — *On the Emergence of Position Bias in Transformers.* Wu, Wang, Jegelka,
Jadbabaie, ICML 2025, arXiv:2502.01951. Causal-mask × RoPE-decay theory of position bias.
DIFF: structural account of distance effects; our clamp shows the displacement penalty rides on
attention mass (content-targetable), and competition is position-free entirely.

**zhang2024attentionentropy** — *Attention Entropy is a Key Factor: An Analysis of Parallel
Context Encoding.* Zhang, Wang, Huang, et al., ACL 2025, arXiv:2412.16545. Entropy inflation
from parallel encoding degrades performance. DIFF: encoding-scheme mismatch vs our sequential
accumulation; both treat attention dispersion as diagnostic, ours with per-span causal clamps.

## Tier 4b — added 2026-08-25 for the template-glue localization (E3c′/E6′/E7 Stage-2)

**darcet2023registers** — *Vision Transformers Need Registers.* Darcet, Oquab, Mairal,
Bojanowski, ICLR 2024, arXiv:2309.16588. ViTs spontaneously repurpose low-information
patches as global-computation slots; adding dedicated register tokens absorbs the role.
DIFF: the precedent for "models elect low-information tokens as storage"; our E7 Stage-2
shows the LLM analogue with a causal transplant — the chat template's delimiter tokens
carry 62% of the counterfactual format contrast — and identifies the layer band (≤13).

**sun2024massiveactivations** — *Massive Activations in Large Language Models.* Sun, Chen,
Kolter, Liu, arXiv:2402.17762 (2024). A handful of activations, concentrated on special
and delimiter tokens, are orders of magnitude larger and act as fixed biases; attention
concentrates there.
DIFF: they establish the *existence and bias role* of delimiter-token state observationally;
our patching shows that state is content-bearing and transportable — moving it moves the
format behavior — and our E3c′ closure shows removing access to it disrupts the
demonstrated format at −0.175 net.

**mu2023gist** — *Learning to Compress Prompts with Gist Tokens.* Mu, Li, Goodman,
NeurIPS 2023, arXiv:2304.08467. Instruction content can be trained into dedicated "gist"
tokens that stand in for the full prompt.
DIFF: gisting is engineered at training time into inserted tokens; we find chat-tuned
models do the equivalent natively, electing the template's own delimiters as the
instruction-state carrier — no token was added and no objective asked for it.

## Positioning digest (what we did differently, in one paragraph)

Prior work either (a) documents long-context degradation behaviorally at benchmark level
(RULER, NoLiMa, BABILong, Multi-IF, serial-position and conversation-tax studies), (b) shows
demonstrations can override priors or instructions behaviorally (Wei et al.; Camassa & Shiller),
or (c) measures/steers attention as a global or position-driven quantity (Found-in-the-Middle,
PASTA, sink papers, position-bias theory). What none of them does — and what this paper's
matched-fill paired designs plus the span-targeted attention clamp do — is causally decompose
accumulated-context degradation into mechanisms and say for each whether attention mass is the
mediator: displacement is (clamp-removal reproduces the penalty; restoration recovers),
competition is not (full effect at constant mass), and format contamination is a precedent
contest the instruction loses with its attention intact — yet which either re-weighting or
re-presenting the instruction fully reverses. The instruction-vs-demonstration channel
distinction that Davidson et al. establish representationally, we establish causally and
behaviorally, on the model's own self-generated precedent.

## Entered the tex 2026-08-24 (Dongre bibliography comparison pass)

From this file's tiers: hsieh2024ruler, olsson2022induction, xiao2023streamingllm,
gu2024attentionsink, wu2025positionbias. New (methodological primitives Dongre et al. cite and
we were missing, IDs verified): alain2016probes (arXiv:1610.01644), belinkov2022probing
(Computational Linguistics 48(1)), su2024rope (RoFormer, Neurocomputing 568) — the RoPE-decay
claims and the linear-probe protocol previously had no citations at all. Bibliography 24 → 32.
Structural note: Dongre et al. use a four-theme background appendix (degradation / attention
behavior / probing methodology / architecture); ours stays mechanism-argument-driven — the
methodology citations were folded inline instead.
