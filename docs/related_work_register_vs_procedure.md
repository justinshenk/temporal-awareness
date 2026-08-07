# Related work for *Register, Not Procedure* — literature check, 2026-08-07

Run because the paper had **zero citations** and an unexamined novelty claim on the primal-ridge
conditional map. Conclusion up front: the *form* of the map is not novel, the *transport framing* and
the *oracle-anchored nulls* are, and one concurrent paper must be engaged head-on or a reviewer will
read our headline null as already refuted.

## 1. The main contender

**Weight Updates as Activation Shifts: A Principled Framework for Steering.**
Adila, Cooper, Yun, Trost, Sala (UW–Madison). arXiv:2603.00425, Feb 2026.

Overlaps us in four places: it unifies weight-space finetuning with activation-space intervention
(our §1); it defines `δh_oracle = h_FT − h_base` (our lockstep oracle); it evaluates on GSM8K and
AQuA (our task); and it reports post-block steering within **0.2–0.9% of full finetuning**, beating
LoRA and ReFT.

### What it actually does (verified against the paper, not the abstract)

- Base weights **frozen**; adapter parameters trained.
- Trained on **task loss** over the **GSM8K training split (8,790 examples)**, standard SFT objective.
- **No donor model.** Everything is learned from scratch on the task; LoRA and ReFT are baselines,
  not donors.
- The oracle is a **theoretical construct for expressivity analysis** — "a clean learning target for
  understanding expressivity" — **not** an inference-time mechanism. They never patch it in and
  measure what it recovers.
- Key contribution is *placement*: intervene **post-block** (after skip connections), capturing both
  attention and MLP pathways, rather than post-MLP.
- First-order analysis shows activation steering misses a `(ΔW_d)m` term that weight updates capture,
  so the two are related but **not equivalent**.

### Why it does not refute our null

| | Adila et al. | ours |
|---|---|---|
| question | can an activation-space adapter **learn** a task? | can a capability be **transported** from a donor into a frozen base? |
| donor | none | central — the whole object of study |
| supervision | task loss on 8,790 GSM8K problems | none; closed-form ridge onto the donor's `δ` |
| oracle | theoretical, for expressivity analysis | **operationalised** — patched at every decode step, recovers 0.75, then ablated temporally and spectrally |
| intervention site | every block, post-block | single layer (L20), pointwise |
| what varies | the adapter's placement and capacity | the task, apparatus held fixed |

"Can I train a cheap adapter to do math" and "can I copy a donor's math into an untouched model" are
different questions. Our nulls are about **transport through a fitted pointwise map at a layer**.

### The reconciliation — their result is our confirming case

Our mechanism claim is that the procedure state is **distributed across layers and dense in time**,
so recovery needs intervention nearly everywhere, nearly always. Adila et al. intervene at **every
block** at **every token** and succeed. That is the outcome our mechanism predicts. Their finding that
placement must be *post-block* rather than post-MLP is independent evidence for the same claim, and
their missing `(ΔW_d)m` term is a **theoretical account of the wall we measured empirically**.

Cited well, this strengthens §7 rather than threatening it.

### The scoping it forces on us (required, not optional)

The abstract currently generalises past the evidence:

> the identical machinery recovers ≈0 of a reasoning LoRA's gain — and so do MLP, on-policy DAgger,
> per-context, and task-loss (DAS) variants

Read against Adila et al. this implies "no activation-space method installs reasoning," which is now
false. The DAS rung is the exposed one: it *is* task-loss-trained (CE→0.038) and still reads 0.00 —
but it is rank-512 at a **single layer**, where Adila's adapter spans **every block**. So DAS nulls a
particular placement and parameterisation, not activation space as such.

**Required rewording:** the claim is that a procedure does not transport through a fitted pointwise
map at a layer; installing it demands distributed, temporally dense intervention trained on the task
— which is what Adila et al. supply and what our density measurements independently require.

## 2. Occupied ground — the map's form is not novel

- **Contextual Linear Activation Steering (CLAS).** Hsu, Beaglehole, Radhakrishnan, Belkin
  (UCSD/MIT). arXiv:2604.24693, Apr 2026. Input-conditional steering `h + (c·[h 1])d`, coefficient a
  learned linear function of the activation, fit by AdamW on next-token loss.
  **Distinction:** their **direction `d` is fixed** and only its magnitude adapts — rank-1
  conditional. Ours is a full matrix, so `W·a` changes direction per input, and is fit closed-form by
  ridge on paired activations. Ours is strictly more general in conditionality; this must be stated,
  not assumed.
- **Conceptors / compositional affine steering.** Postmus & Abreu, arXiv:2410.16314 and the
  follow-up (OpenReview `0Yu0eNdHyV`). Steering *matrices* as soft projections, with a claimed
  provably-optimal affine steering function. Prior art for "matrix, not vector"; fit as an ellipsoid
  over an activation set rather than regressed onto a donor delta.
- **Beyond Linear Activation Steering (INNSteer).** Nguyen & Le (Indiana). arXiv:2606.08454, Jun
  2026. Invertible nonlinear transforms giving input-dependent updates; within **3.64% of LoRA**.
  **Tests alignment traits, refusal, hallucination — explicitly not GSM8K or any reasoning install.**
  This is a gift: the field's strongest steering result lands on registers and stops at the boundary
  we draw. Use it as independent support for (R), and its silence on procedures as the gap we fill.
- **Activation Space Interventions Can Be Transferred Between LLMs.** arXiv:2503.04429, Mar 2025.
  Autoencoder mappings between base and finetuned counterparts. Nearest neighbour to our (X)
  cross-model transplant; cite there.

## 3. What survives as ours

1. **The transport framing** — fitting closed-form from paired (base, donor) activations to move a
   *specific donor's* capability, rather than learning a task from scratch. CLAS is the closest and is
   rank-1 where we are full-rank.
2. **An operationalised oracle as a positive control.** Adila et al. define the same object and use it
   only analytically. Nobody patches it in, measures 0.75, and then ablates it in time and in spectrum.
   This is what licenses reading our nulls as "not installable" rather than "not tried hard enough."
3. **The boundary itself** — register vs procedure, same apparatus, swept across tasks, replicated on
   a second procedure.

Position the map as the **instrument**, not the claim. A scoped sentence in §3 ("fit closed-form from
paired donor activations, yielding input-conditionality without the explicit gate that prior
conditional-steering methods require") costs one line if prior art turns up, and can be promoted if
none does.

## 4. Consequences for §2

Write it **adversarially**, in this order:

1. Steering as fixed vectors: CAA, Arditi, ActAdd, RepE, ITI — the baselines we beat in §4.
2. Conditional and matrix-valued steering: CAST (explicit gate), conceptors, CLAS — the form is
   occupied; state our distinction.
3. PEFT as activation edit: LoRA, LoReFT/ReFT, and **Adila et al.** — the unification is not ours to
   claim; cite it and inherit it.
4. **The rebuttal paragraph**: why near-parity results on GSM8K (Adila) and on alignment (INNSteer)
   are not in tension with our nulls — different question (learn vs transport), different supervision
   (task loss vs donor regression), different site (every block vs one layer).

All four papers are **concurrent** (Feb–Jun 2026) with work begun before them; say so where it matters.

## 5. Open items

- Verify whether the conceptor steering matrix is genuinely input-conditional (`C·h` suggests yes)
  and how it is fit; the PDF did not extract cleanly.
- Pull Adila et al.'s exact GSM8K numbers per method — the results table did not extract from the PDF.
  Needed before writing the §6 rebuttal.
- Check the RepE / ITI line for any donor-transport variant we have missed.
