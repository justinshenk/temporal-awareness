# §6 + abstract — the rescoped null (2026-08-07)

The headline claim currently generalises past its evidence. This file records the exact wording
change and the reasoning, so the edit can be applied to `register_vs_procedure_abstract.{md,tex}`
without re-deriving it.

## 1. What is wrong with the current sentence

`register_vs_procedure_abstract.md:14–16` reads:

> A primal-ridge conditional map dominates every fixed-vector steering baseline on the harm-refusal
> frontier, yet the identical machinery recovers ≈0 of a reasoning LoRA's gain — and so do MLP,
> on-policy DAgger, per-context, and task-loss (DAS) variants.

Three separate problems, only one of which is about wording:

1. **The MLP clause has no artifact.** Found in the 2026-08-06 audit. DAgger, local-refit and DAS
   each have a committed result; MLP does not. It is either backed by P5 step 5
   (`nonlinear_delta_gsm8k --layer 20`) or it comes out.
2. **The list implies a universal.** Read alongside \citet{adila2026weightshifts} — post-block
   adapters within 0.2–0.9% of full finetuning on GSM8K — the sentence asserts something now false:
   that no activation-space method installs reasoning.
3. **DAS is the exposed rung.** "They had task loss and we didn't" does not cover it: DAS *is*
   task-loss-trained (CE→0.038). What covers it is **placement** — DAS is rank-512 at a *single
   layer*, where their adapter spans *every block*, and placement is the headline contribution of
   their paper. So DAS nulls a site and a parameterisation, not activation space as such.

## 2. Replacement sentence (abstract)

> A primal-ridge conditional map installs and transports a refusal register, reaching the
> harmful-refusal axis where every fixed-vector baseline stays at zero — yet the identical machinery
> recovers ≈0 of a reasoning LoRA's gain, as do on-policy DAgger, per-context, and task-loss (DAS)
> variants.<sup>†</sup> The null is narrower than it first appears and we state it in the form the
> evidence supports: **a procedure does not transport through a fitted pointwise map at a layer.**
> Concurrent work reaches finetuning-level reasoning with adapters trained on the task and applied
> at every block; our mechanism analysis says why that is the shape the problem requires.

<sup>†</sup> "and an MLP variant" is added **only if** P5 step 5 returns ≈0. If it does not, the
clause is removed and the disagreement is reported in §6.

**What changed and why:**

- *"dominates every fixed-vector steering baseline"* → *"reaching the harmful-refusal axis where
  every fixed-vector baseline stays at zero."* The original is literally true (ACE is affine, not a
  fixed vector, so it is not among the baselines beaten) but the framing invited a method claim §4
  no longer makes. The replacement states the measurement, not a ranking.
- **MLP removed** from the running list pending its artifact.
- The scoping sentence is **added, not implied**. A reader who knows the concurrent work must see
  that we know it too, in the abstract, not on page 6.

## 3. §6 scoping paragraph (insert after the ladder table)

> Each rung above shares two properties with the others and with nothing else: the estimator is fit
> to reproduce a donor's activation delta rather than trained on the task, and it acts pointwise at
> a single layer. The null is therefore a null about **transport through a pointwise map**, and it
> should not be read as a claim that reasoning is beyond activation-space intervention in general —
> a reading that concurrent results refute. \citet{adila2026weightshifts} reach within 0.2–0.9% of
> full-parameter finetuning on GSM8K with an adapter trained on the task's own training split and
> applied post-block at every layer. The DAS rung is the one that most nearly bridges the two
> settings, since it is trained under a task objective; what still separates it is placement and
> capacity — rank 512 at one layer against an adapter spanning the whole stack — and placement is
> precisely what their analysis identifies as decisive. We take their result as evidence for the
> mechanism in §7 rather than against the null here: if the recoverable state is distributed across
> layers and dense in time, an intervention that succeeds must be distributed and dense, and theirs
> is. Consistent with this, the weakest method in their own comparison is ReFT — a single-site
> subspace edit, the closest published analogue of the map we falsify — which on the hardest
> settings falls far below supervised finetuning (GSM8K, Gemma-3-1B: 11.6 against 23.4).

## 4. Downstream edits this forces

| file | change |
|---|---|
| `register_vs_procedure_abstract.md` | abstract sentence (§2 above); §6 paragraph (§3 above) |
| `register_vs_procedure_abstract.tex:48` | same sentence, LaTeX form |
| `results/activation_weight_investigation.md:71` | repeats the MLP claim — same treatment |
| `results/attribution/2026-06-16-multihop-generality.md` | verdict table wording, if it inherits the universal |
| `papers/.../numbers.md` | MLP row: pending → artifact or removed |

## 5. Acceptance

- [ ] No sentence in either paper asserts that activation-space methods cannot install reasoning.
- [ ] Every surviving rung in the ladder names a committed artifact.
- [ ] \citet{adila2026weightshifts} is cited in the **abstract-adjacent** text, §2 and §6 — not
      only in related work.
- [ ] If P5 step 5 returns non-zero, the MLP disagreement is reported in §6 as found, and the
      abstract does not quietly keep the clause.
