# In-Context Learning Mechanisms — Literature Map

A verified literature survey (deep-research, 23 sources, 25/25 claims adversarially confirmed) of
**what ICL is doing mechanistically and how it bridges to weight-space learning** — the companion to
[`activation_vs_weight_literature.md`](activation_vs_weight_literature.md), and the literature context
for this project's LoRA-vs-ICL subspace study ([`lora_icl/README.md`](lora_icl/README.md)) and
sycophancy many-shot priming ([`sycophancy/README.md`](sycophancy/README.md)).

## The verified core: ICL writes a compact task representation into activations

**Task vectors** (Hendel, Geva & Globerson, EMNLP 2023 Findings, [`2310.15916`](https://arxiv.org/abs/2310.15916)).
ICL decomposes into a *learning* step and an *application* step: `T([S,x]) = f(x; A(S))`. The
demonstrations `S` are compressed into a single query-agnostic vector `θ(S)` — the layer-L hidden
state of the separator token — which, **patched alone** into a forward pass that never sees `S`,
recovers **~80–90% of full ICL accuracy** (no-demo baseline: 10–20%), across LLaMA 7/13/30B, GPT-J,
Pythia, 18 tasks. Best layer is at a *similar intermediate depth* across models. The ~10–20% residual
makes the single vector a strong **approximation**, not a complete account (later work — "One Task
Vector is Not Enough", Adaptive Task Vectors — argues complex tasks need multiple/per-query vectors).

**Function vectors** (Todd et al., ICLR 2024, [`2310.15213`](https://arxiv.org/abs/2310.15213)). A
small set of attention heads, identified by **causal mediation**, transport a compact task
representation; summing their task-conditioned mean outputs gives a **function vector (FV)**. Patching
it raised zero-shot accuracy 5.5%→57.5% (GPT-J), up to 83.8% (Llama-2-70B). FVs are **robust and
transferable** — they trigger the task even in natural-text contexts unlike the ICL prompts they were
extracted from (partial transfer, e.g. 57.5%, not ~100%). **Causal effect peaks in early-middle
layers (~L/3) and drops to near-zero in late layers**, implying FVs don't act purely linearly — they
*trigger late-layer nonlinear computation*.

**Induction heads** (Olsson et al., Anthropic 2022, [transformer-circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html);
[`2209.11895`](https://arxiv.org/abs/2209.11895)). A **prefix-match-and-copy** circuit
(`[A][B]…[A]→[B]`) that forms in an early-training **phase change** — a bump in the loss curve —
coincident with the sharp emergence of ICL. Olsson et al. *hypothesize* induction heads drive the
majority of ICL, but flag the evidence as **causal only for small attention-only models**, merely
correlational once MLPs are present.

**Reconciliation (recent, single-source each — treat as evolving):**
- **FV heads > induction heads** for few-shot ICL in models >1B (Yin & Steinhardt, ICML 2025,
  [`2502.14010`](https://arxiv.org/html/2502.14010v1)): ablating FV heads tanks ICL; ablating
  induction heads (excluding the top-2% FV heads) ≈ random. Induction heads are a **developmental
  precursor** — they emerge ~step 1k, FV heads ~step 16k; many induction heads *become* FV heads, never
  the reverse.
- **But induction heads still matter** for abstract pattern-matching (Crosbie & Shutova, NAACL 2025
  Findings, [`2407.07011`](https://arxiv.org/pdf/2407.07011)): ablating 1% of heads (induction) drops
  ICL up to **31.6%** vs ≤5.8% for random. The two reconcile by task type (abstract pattern-matching
  vs real-task few-shot accuracy), not contradiction.

## Threads fetched but NOT verified in this pass (lower confidence — flagged for honesty)

The workflow fetched primary sources for these but no claim survived into the verified top-25; I
report the headline findings as *literature context*, not verified here:
- **ICL as implicit gradient descent** — von Oswald et al. 2023 ([`2212.07677`](https://arxiv.org/abs/2212.07677)),
  Dai et al. 2022 (ACL 2023 Findings, "GPT learns in-context via implicit meta-gradients"). Construction
  results that a transformer layer *can* emulate a GD step on a linear model; **contested** as a literal
  account of real LLMs (architecture/scale assumptions).
- **In-context vs in-weights trade-off & transience** — Chan et al. NeurIPS 2022
  ([`2205.05055`](https://arxiv.org/abs/2205.05055)): data-distributional properties (burstiness,
  large label spaces) drive whether ICL emerges; Singh et al. 2023: emergent ICL can be **transient**
  over long training.
- **Many-shot ICL** — Agarwal et al. 2024 ([`2404.11018`](https://arxiv.org/abs/2404.11018)):
  hundreds–thousands of shots close much of the gap to finetuning on many tasks; Anil et al. 2024
  (many-shot jailbreaking): many-shot scaling defeats safety training.

## Where this project's results land

Our LoRA-vs-ICL subspace study is, in effect, a **direct measurement of the activation↔weight task-vector
bridge** the survey says is "asserted only by analogy":

1. **ICL forms an activation task vector; finetuning forms a weight task vector (Ilharco); we showed
   they converge.** Our ICL activation shift and LoRA weight shift land in the **same late-layer
   subspace** (cos → 0.81 @ L35, mean-centered overlap ~0.67). That is an empirical instance of the
   bridge between the ICL task/function vector (Hendel/Todd) and weight-space task arithmetic.

2. **A real nuance / follow-up the FV result surfaces.** Todd et al. find the FV's *causal* effect peaks
   **mid-layer (~L/3)** and fades late; our convergence was measured at the **late prediction site
   (L35/42)** — i.e., downstream of the FV trigger, the *applied-task* representation, not the FV
   itself. Not a contradiction, but it means our cosine is measuring the result, not the mechanism.
   **Open follow-up:** extract an actual FV from the DDXPlus ICL prompts and compare it directly to the
   LoRA's mid-layer direction — a sharper test of "same task vector, two delivery routes."

3. **Our "ICL ≈ finetuning on the task, but not its side-effects" answers the survey's open ICL-vs-
   finetuning question.** We found ICL and finetuning **converge on the task subspace** yet **diverge on
   the safety side-effect** (ICL of the same content does not erode refusal — that is weight-specific).
   So: ICL reproduces the task *function vector* but not the weight-written collateral. That is a crisp
   "when does ICL match finetuning, when does it diverge."

4. **Our sycophancy many-shot priming is a behavioral many-shot result** (caving 0.00↔0.98 from cave/hold
   demonstrations) — the in-context channel (Agarwal/Anil) dominating where a single static direction was
   weak, consistent with ICL "writing the task into activations" via demonstrations.

## Open questions (this project is positioned on)
1. Is the ICL task representation one vector or distributed/high-rank for complex (e.g. medical MCQ)
   tasks — and does our DDXPlus ICL admit a single extractable task/function vector? **Partially
   answered (negative):** an FV-extraction attempt on DDXPlus ([`2026-06-02-fv-extraction.md`](2026-06-02-fv-extraction.md))
   was a null — clean 4-shot accuracy (0.71) = shuffled-label accuracy (0.71) ≈ zero-shot (0.67), so
   the model ignores the in-context labels and solves DDXPlus from prior knowledge. **DDXPlus is a
   knowledge task, not an ICL task for this model**, so there is no causal function vector to extract.
   **Followed up on a genuine ICL task** ([`2026-06-02-antonym-fv-vs-lora.md`](2026-06-02-antonym-fv-vs-lora.md)):
   antonyms (zero-shot 0.00 → 10-shot 0.70; a LoRA trained on the mapping generalizes to held-out
   words at 0.67). Result is two-sided: **(a) the coarse (Hendel) ICL task vector and the LoRA
   weight-shift converge to cos +0.766 @ L35 (~46× the random-null std)** — replicating the DDXPlus
   subspace profile (0.81 @ L35) on a *bona fide* ICL task where the LoRA provably generalizes, so the
   "same task direction, two routes (in-context vs in-weights)" claim now holds at the representational
   level; **(b) the sparse head-localized (Todd) FV isolates a *different* sub-direction than the LoRA.**
   With a Todd-faithful **shuffled-label** corruption (an earlier *zero-shot* corruption gave AIE ≈ 0 by
   construction — single heads can't restore a fully-removed signal), the AIE is strongly structured
   (top heads at layers 17/21–25, AIE up to +0.94 nats), so antonym FV heads *do* exist. But the FV
   still doesn't transplant (zero-shot+FV 0.00; cos(FV, task-vector) ≤ 0.21): shuffled-label AIE
   isolates the small **label-mapping** component (Min et al. 2022 — shuffled-label accuracy 0.57 is
   near clean 0.70), whereas the task vector and LoRA are dominated by the large **format/function**
   component (zero-shot 0.00 → 0.70). The "two routes converge (0.77)" result is the *format* direction,
   which single-head causal mediation cannot isolate (removing format = zero-shot = AIE 0). So the
   antonym task vector is two superposed sub-directions; ICL and finetuning share the dominant
   (format/function) one.
2. Does ICL converge to the same activation subspace **and generalization behavior** as weight
   finetuning, or only the former? (We showed subspace convergence; generalization is untested.)
3. The mid-layer FV vs late-layer convergence gap (point 2 above) — where exactly does the ICL/LoRA
   bridge live across depth?

## Sources
23 primary/secondary sources fetched; the verified core claims above each carried a 3-0 adversarial
verification vote. The "not verified in this pass" section is explicitly lower-confidence.
