# Activation- vs Weight-Manipulable Behaviors — Literature Map

A verified literature survey (deep-research, 26 primary sources, 25/25 claims adversarially
confirmed) of what prior work establishes about **what you can control via activation
interventions vs what requires weight modification** in LLMs — and where this project's own
results land within it. Companion framing for [`lora_icl_safety_synthesis.md`](lora_icl_safety_synthesis.md)
and [`sycophancy/README.md`](sycophancy/README.md).

## The taxonomy

| | **Activation-manipulable** (inference-time, no retraining) | **Weight-manipulable** (durable, written into parameters) |
|---|---|---|
| **What** | High-level / stylistic / "cognitive-state" behaviors: refusal, honesty/truthfulness, sycophancy, sentiment, topic, toxicity, persona | Stored facts/associations; task capabilities; durable alignment changes |
| **Methods** | RepE, ActAdd, CAA, ITI, SAE-feature steering | ROME/MEMIT, task arithmetic, finetuning/LoRA |
| **Mechanism** | Behaviors ride on ~linear directions in the residual stream; one additive vector gives graded, bidirectional control, *on top of* finetuning+prompting | Facts localized to mid-layer MLP "key-value memories"; editing FFN weights is the lever |

**Dividing line:** activation steering changes *how the model currently behaves*; weight editing
changes *what it stores or durably computes*. Steering stacks on top of finetuning rather than being
subsumed (CAA matched/exceeded finetuning for only 3/7 behaviors), and durable alignment erosion is a
*weight* phenomenon steering does not reproduce.

## Activation side — primary sources

- **RepE** — Zou et al. 2023, [`2310.01405`](https://arxiv.org/abs/2310.01405). Population-level
  representation reading + control of honesty, harmlessness, power-seeking.
- **ActAdd** — Turner et al., [`2308.10248`](https://arxiv.org/abs/2308.10248). A single contrastive
  vector ("Love"−"Hate") added in the forward pass, no backprop; steers sentiment/topic/toxicity.
- **CAA** — Rimsky et al., [`2312.06681`](https://arxiv.org/abs/2312.06681). Mean residual-stream
  activation difference over contrastive pairs → one steering vector per behavior; graded,
  bidirectional; complementary to finetuning + prompting.
- **ITI** — Li et al., NeurIPS 2023, [`2306.03341`](https://arxiv.org/abs/2306.03341). Shifts
  activations along learned directions in a few attention heads; TruthfulQA **32.5 → 65.1%** with a
  few hundred examples.
- **SAE-feature steering** — Templeton et al. 2024 (Golden Gate Claude); SAEs decompose the residual
  stream into interpretable, steerable feature directions. *(Medium confidence — primary source
  inferred from adjacent SAE papers, not independently verified in this pass.)*

## Weight side — primary sources

- **Factual localization** — Geva et al. 2021, [`2012.14913`](https://arxiv.org/abs/2012.14913): FFN
  layers act as key-value memories. Meng et al. 2022, [`2202.05262`](https://arxiv.org/abs/2202.05262)
  (ROME): causal tracing localizes facts to mid-layer FFN; a rank-one FFN edit inserts a new fact.
- **Knowledge-editing survey** — Wang et al., [`2310.16218`](https://arxiv.org/abs/2310.16218):
  parameter-modifying methods (ROME, MEMIT, locate-then-edit) as the lever for factual updates.
  *(Contested: ripple effects, sequential-edit degradation; Hase et al. — causal-trace location ≠ best
  edit layer.)*
- **Task arithmetic** — Ilharco et al. ICLR 2023, [`2212.04089`](https://arxiv.org/abs/2212.04089): a
  task vector (finetuned − pretrained weights) is a direction; negating it removes the behavior with
  little control-task change.
- **Finetuning erodes safety** — Qi et al. ICLR 2024, [`2310.03693`](https://arxiv.org/abs/2310.03693):
  10 adversarial examples (<$0.20) strip GPT-3.5 guardrails; even benign finetuning degrades safety.
  A weight-level erosion steering does not write.

## The open debate: is it really *one* direction?

The strong linear-representation / single-direction account of refusal (Arditi et al. 2024) is
contested by 2025–26 work — though partly reconcilable:

- **Pan et al.** ICML 2025, [`2502.09674`](https://arxiv.org/pdf/2502.09674): safety is jointly
  controlled by **multi-dimensional orthogonal** directions — one dominant refusal direction plus
  smaller interpretable sub-features (hypothetical-narrative, role-play).
- **Joad et al.** [`2602.02132`](https://arxiv.org/abs/2602.02132): refusal spans geometrically
  distinct directions across 11 non-compliance categories — **but** steering along any of them gives
  nearly identical refuse/over-refuse trade-offs → **one shared "whether-to-refuse" knob, many
  "how-to-refuse" directions.**
- **Engels et al.** ICLR 2025, [`2405.14860`](https://arxiv.org/abs/2405.14860): some features (days,
  months) are *irreducibly multi-dimensional* — strong "1 concept = 1 line" is false in general.

*(Recency caveat: the multi-dimensional results are new and not yet widely replicated.)*

## Where this project's results land

Our arc is effectively a **two-behavior head-to-head** on the question the survey names as missing —
*which behaviors are steering-sufficient vs weight-required*:

1. **Refusal-harm → weight-manipulable, both halves confirmed.** Our weights-vs-activations result
   (finetuning erodes refusal, ICL of identical content does not) independently reproduces Qi et al.'s
   weight-specific fragility; the low-rank ablatable harm direction reproduces the Arditi handle.

2. **Our context-refusal probe is a data point in the live debate.** The static `r` did **not**
   transfer to long context; a behavior-grounded `d_comply` (mostly orthogonal to `r`) was needed —
   echoing Pan/Joad's "refusal is multiple directions, the static one is incomplete." Yet our
   single-direction ablation still restored refusal, consistent with their shared "whether-to-refuse"
   knob. We land on the reconciliation from a new angle (context-fatigue) the literature has not covered.

3. **Sycophancy → activation/in-context side, showing steering's known limits.** Our `−d_syco` steer
   was modest and over-drove (0.51→0.38, narrow band) — matching documented steering
   reliability/generalization limits. Our many-shot priming (caving 0.00↔0.98, dominating length and
   the static direction) is a textbook instance of the in-context function/task-vector channel and
   many-shot jailbreaking (Anil et al.) — the bridge the survey flagged as under-covered.

**Novel synthesis our experiments support:** behaviors *written into weights* (Qi-style erosion) get
a compact low-rank **activation** handle that even survives long context (our deployable ablation);
behaviors that are **context/policy-driven** (base sycophancy) resist a static direction but are
dominated by the in-context demonstration channel. "Where it is stored" and "how you best control it"
are related but not identical axes — and **long context is the stress test that separates them.**

## Open questions this project is positioned to answer

1. A systematic steering-vs-weight head-to-head: which behaviors are steering-sufficient vs
   weight-required (we have two points; the map is unfilled).
2. The in-context↔weight task-vector bridge: how ICL-induced function/task vectors (Todd, Hendel) and
   many-shot jailbreaking relate to weight-space task vectors (Ilharco) and induction heads — directly
   adjacent to our LoRA-vs-ICL subspace convergence and our priming result. *(Covered in the companion
   [`icl_mechanisms_literature.md`](icl_mechanisms_literature.md): ICL writes a compact task/function
   vector into activations; our subspace study measures the convergence of that vector with the LoRA
   weight shift.)*

## Sources
26 primary sources fetched; the per-claim citations above each carried a 3-0 adversarial verification
vote. Full source list and verification log in the deep-research run output.
