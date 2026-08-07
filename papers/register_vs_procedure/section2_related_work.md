# §2 Related work — draft 1 (2026-08-07)

Written in the four-part adversarial order from `docs/related_work_register_vs_procedure.md`.
Target ~1 page of the 9. Prose is anonymised throughout ("we", no repo/branch names).

Provenance: every claim about a cited paper below was checked against that paper's abstract,
method section or results table on 2026-08-07 — not from memory. Numbers quoted from
\citet{adila2026weightshifts} are their Table 1 (v2).

---

## 2 Related work

**Steering by a fixed direction.** The dominant paradigm adds a single vector to the residual
stream. CAA \citep{rimsky2024caa} takes the mean activation difference over contrastive pairs;
ActAdd \citep{turner2023actadd} obtains a direction from a pair of natural-language prompts;
ITI \citep{li2023iti} shifts along learned directions at selected attention heads; RepE
\citep{zou2023repe} reads and controls population-level directions for high-level concepts. For
refusal specifically, \citet{arditi2024refusal} show a *one-dimensional* subspace mediates the
behaviour across thirteen chat models. These methods share a commitment we make explicit and then
test: the intervention is a **fixed offset**, identical for every input. Our §4 measures all four
against a conditional map on the same prompts and grids.

**Conditional and matrix-valued steering.** The fixed-offset assumption has been relaxed several
ways, and the resulting family — an input-dependent affine edit of the residual stream — is
occupied ground rather than something we introduce. CAST \citep{lee2025cast} gates a CAA vector
with an explicit logistic classifier, making conditionality a bolted-on component. Conceptors
\citep{postmus2024conceptors} replace the vector with a *matrix*: the steered activation is
$h' = \beta_c C h$, so the shift genuinely depends on the input, and $C = R(R + \alpha^{-2}I)^{-1}$
with $R = X^\top X / n$ is obtained in closed form. CLAS \citep{hsu2026clas} learns an
input-dependent *coefficient*, $h \mapsto h + (c \cdot [h\ 1])\,d$, fit by gradient descent on
next-token loss. ACE \citep{marshall2024ace} decomposes activations affinely, shows prior steering
methods are subsets of terms of that decomposition, and controls refusal across ten models up to
70B. INNSteer \citep{nguyen2026innsteer} goes further, learning invertible nonlinear transforms
that induce input-dependent updates.

Two distinctions matter for what follows. CLAS is **rank-one conditional** — the direction $d$ is
fixed and only its magnitude adapts — where the map we study is full-rank, so its output direction
varies with the input. Conceptors are matrix-valued and closed-form like ours, but differ in
*target*: they take the second moment of a **single** activation set, yielding a shrinkage
projection onto that set's ellipsoid with no regression target, whereas we use the **cross moment**
$X^\top\delta$ of **paired** base and donor activations. That difference — supervision by a second
model — is the only respect in which our instrument is unusual, and we treat it as an instrument
rather than a contribution.

The register half of our result should therefore be read as consistent with this literature rather
than as competing with it. ACE in particular reports the failure mode we also observe, that purely
directional interventions produce incoherent output where affine ones do not, on far more models
than we test. Our §4 exists as a **positive control**: it establishes that the apparatus we then
point at procedures does install and transport a disposition.

**Parameter-efficient finetuning as an activation edit.** LoRA \citep{hu2022lora} learns a low-rank
weight delta; LoReFT \citep{wu2024reft} learns an edit $h + (Wh + b - hR)R^\top$ applied to hidden
representations of a frozen base. The latter is, formally, the same object as a conditional affine
steering map, and the correspondence between the two spaces is not ours to claim:
\citet{adila2026weightshifts} establish a first-order equivalence between activation-space
interventions and weight-space updates, derive the conditions under which steering can replicate
finetuning, and identify the post-block output as the expressive intervention site. Distributed
Alignment Search \citep{geiger2024das} supplies the complementary tool, finding task-relevant
subspaces by gradient descent under a causal objective; it is one rung of our ladder in §6.

**Why near-parity results are not counterexamples.** Two concurrent papers report activation-space
methods approaching finetuning, and both are compatible with our nulls for the same reason.
\citet{nguyen2026innsteer} reach within 3.64\% of LoRA — on alignment traits, refusal and
hallucination, and explicitly not on multi-step reasoning; that is our register side, independently
confirmed, and their silence on procedures is the gap this paper fills.
\citet{adila2026weightshifts} do evaluate GSM8K and AQuA and land within 0.2–0.9\% of
full-parameter tuning, which appears to contradict us directly. It does not, for three reasons that
we make explicit because the surface similarity is otherwise misleading. First, **there is no donor**:
every column of their comparison is trained on the task's own training split, so their claim is
parity with supervised finetuning on the same data, not recovery of a capability that exists in
another model. Ours is a transport question — whether a frozen base can be given a donor's
capability — and no experiment in their paper addresses it. Second, our map receives **no task
loss**; it is fit in closed form to reproduce a donor's activation delta. Third, their intervention
spans **every block**, where the maps we falsify act pointwise at a single layer.

That third difference is the substantive one, and their results argue our case rather than against
it. In their Table 1, ReFT — a single-site subspace edit, the closest published analogue of the map
we study — is the weakest method reported, and degrades furthest exactly where the task is hardest
for the model (GSM8K on Gemma-3-1B: 11.6 against 23.4 for supervised finetuning; ARC-Challenge 27.8
against 48.2; Winogrande on Qwen3-4B 64.9 against 83.0). Their remedy is to abandon the single site
and intervene post-block at every layer. The ordering they measure — single-site subspace edit
$\ll$ distributed post-block edit $\approx$ weight update — is precisely the ordering our §7
density measurements predict from the mechanism, obtained independently on four models we do not
use. Their first-order analysis also isolates a term, $(\Delta W_d)m$, that activation steering
cannot capture alone: a theoretical account of the wall we characterise empirically.

We therefore scope our claim accordingly, and state it in the form the evidence supports: a
procedure does not **transport** through a fitted pointwise map at a layer. Installing one appears
to require distributed, temporally dense intervention trained on the task — which is what
\citet{adila2026weightshifts} supply, and what §7 independently shows to be necessary.

All of \citet{adila2026weightshifts}, \citet{hsu2026clas} and \citet{nguyen2026innsteer} are
concurrent with this work.

---

## Drafting notes (not for the paper)

- **Length**: ~850 words. If §2 must shrink, the first paragraph compresses to two sentences —
  fixed-vector steering is the least contested part. Do **not** cut the fourth paragraph; it is the
  rebuttal and it is load-bearing.
- **`\citet` vs `\citep`** usage assumes `natbib`. Confirm against the NeurIPS 2026 template before
  the first build.
- **Anonymity**: no repo, branch, or username appears above. Keep it that way — the CFP asks for
  GitHub and HuggingFace handles to be scrubbed too.
- **Open**: the GSM8K dataset citation (`cobbe2021gsm8k`) still carries a `% CHECK` on its arXiv id.
  Not referenced in this section; close it before §3.
- **Deliberately omitted**: task vectors / weight-space arithmetic. Adjacent but weight-space, and
  §2 is already the longest non-results section. Revisit only if a reviewer asks.
