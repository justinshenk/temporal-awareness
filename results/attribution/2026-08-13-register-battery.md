# The register battery — measuring the other half of the boundary (S2)

**Date:** 2026-08-13 · **Brief:** `tasks/s2_execution_brief.md` · **Spec:**
`docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3 · **Status:** in progress

## Why this exists

The register-vs-procedure claim is two-sided — a *register* is low-rank, on-manifold and roughly
pointwise; a *procedure* is high-rank, off-manifold and time-dense — but **only the procedure side
had ever been measured**. GSM8K and MuSiQue each have an oracle layer sweep, a temporal-density
knee, a PCA-band cliff and a five-rung null ladder. No register task had any of them: the register
half rested on the single observation that a ridge map installs refusal tone. At a venue whose
stated topics are measurement validity and falsifiability, that is the first objection, not a
nitpick.

S2 points the **unmodified** procedure drivers at a register donor and reports what comes out.

## Apparatus

| | |
|---|---|
| donor | LoRA r32/α64, dropout .05, {q,k,v,up,down}\_proj, lr 3e-4, 3 epochs, 20k subset of commonsense-170k, seed 42, final CE 0.0074 |
| base | `NousResearch/Llama-2-7b-hf` (the ungated mirror used by every prior strand) |
| eval | ARC-Challenge test, n=500 scan, `max_new=32` |
| prompt | pyreft commonsense template, literally `"%s\n"` — no alpaca wrapper |
| target | `"the correct answer is X"` (~7 tokens) |
| scoring | the word after `"the correct answer is"`, exact match against gold |

The donor is the *same recipe and same 20k subset* as the LoReFT arm of
`2026-06-14-lora-vs-loreft-commonsense.md`, so that comparison stays addressable.

**Why ARC-Challenge and not boolq.** Recovery is read as accuracy on a base-fails/donor-solves
contrast set. On a **binary** task (boolq true/false, piqa solution1/solution2) an intervention that
merely garbles decoding still scores ~50% by coin flip, so a partial recovery cannot be
distinguished from a destroyed one. ARC-Challenge is 4-way. Measured floors on the n=500 scan:
**chance 0.25**, and — more demanding — **majority-class 0.288**, since the gold spread is
answer2 144 / answer3 137 / answer1 117 / answer4 102 and a degenerate always-`answer2` policy would
be perfectly format-compliant. Conditional accuracy is read against **0.288**.

**What this task is, structurally.** The supervised target is ~7 tokens and contains **no
intermediate work**. Contrast MuSiQue, whose target is a chain in which hop 1's answer is written
down and then *consumed* by hop 2 (`#1 >> spouse`). Nothing in the commonsense target refers to
anything the model produced earlier. That absent dependency chain is the whole distinction under
test. It also has two methodological consequences:

1. **The temporal-density axis is degenerate here and was not run.** On a ~7-step trajectory
   `periodic:2` and `periodic:4` are nearly the same intervention, so a density number would look
   like a result while measuring nothing. Per the spec that axis belongs to **refusal**, whose
   generations are long.
2. **`base = 0.000` does not mean the same thing as MuSiQue's `base = 0.000`.** Llama-2 base will
   not emit `"the correct answer is X"` unprompted, so its zero is substantially *format
   non-compliance*, not incapacity — it plainly knows some of these answers. The donor's δ is
   therefore partly a **format register**. S2c below measures that split rather than assuming it.

## Results

### Gap gate + AC1 — PASSED

`.run_logs/s2_gate.log`, ARC-Challenge scan n=500, `max_new=32`:

| | |
|---|---|
| base accuracy | **0.000** |
| donor accuracy | **0.676** (the spec predicted 0.68) |
| contrast problems | **338** base-fail/donor-solve (floor was 80) |
| AC1 all-layers lockstep | **PASS** — reproduces the donor per-problem, 3/3 |

Cached to `commonsense_contrast_set.json`. **Every later commonsense run must pass `--n-eval 500`**:
the cache stores indices into the scan, so the driver's default of 60 would silently misindex all
338 — the trap MuSiQue's 317 set earlier in this work.

### Oracle layer sweep — the register separates from both procedures

`lockstep_commonsense_single.json`, n-contrast 100, `max_new=32`:

| layer | commonsense | MuSiQue | GSM8K |
|---|--:|--:|--:|
| 0 / 4 / 8 | 0.000 | 0.000 | — |
| 12 | 0.050 | 0.020 | — |
| 16 | **0.830** | 0.020 | — |
| 20 | **0.990** | 0.760 | 0.750 |
| 24 | 0.990 | 0.780 | — |
| 28 | 1.000 † | 0.890 † | — |
| 31 | 1.000 † | 1.000 † | — |

† Degenerate tail, excluded from `L*` by the same rule both procedures used: the hook overwrites the
*output* of `model.model.layers[L]`, so L31 hands `lm_head` the donor's final hidden state verbatim
(the all-layers control in disguise), and L28 sits close enough to readout to inherit most of that.

**`L* = 20`** — the same layer as both procedures, by the same earliest-plateau rule. But the *curve*
is not theirs. Onset is earlier and far sharper (0.830 at L16, where MuSiQue is still at 0.020), and
the plateau is essentially total (0.990 against 0.75 / 0.76).

This is the **first quantitative register-vs-procedure separation on the oracle axis itself**. Every
prior comparison between the two sides ran through the pointwise ladder; the oracle was the positive
control the two sides were assumed to share. Here the control itself distinguishes them.

> **Caveat pending the floors.** A 4-way answer space means a large perturbation that merely pushes
> the model somewhere fluent could score well by accident, which is not a risk on GSM8K (a garbled
> injection cannot emit the right integer). The 0.990 is not interpretable until the floors below
> land. This section will be finalized against them.

### Floors at L20 — IN FLIGHT

Two controls, both keeping δ's magnitude and destroying only its content
(`lockstep_oracle.control_injection`, seeded):

- **`mean_delta`** — every position gets the trajectory-average shift: the best fixed vector **at
  the donor's own scale**, chosen per problem with oracle knowledge of its trajectory.
- **`shuffle_positions`** — the true per-token shifts applied in permuted order. Same multiset, same
  norms; only the alignment between shift and token is destroyed.

| intervention at L20 | recovery (n=100) |
|---|--:|
| true oracle, per-token δ | **0.990** |
| `mean_delta` | **0.000** |
| `shuffle_positions` | **0.000** |

> ## RETRACTED — both floor controls are no-ops, and the floor argument with them
>
> Decoding the actual generations (`.run_logs/s2_dump_gen.log`) shows the control outputs are
> **character-for-character identical to unpatched base**, at α=1.0 and α=0.25 alike. Base is not
> incoherent either — it fluently continues the prompt by re-listing the options instead of
> answering, which *is* the format non-compliance that makes base score 0.
>
> **Why the controls do nothing.** `mean_delta` averages δ over *all* positions, but the prompt is
> ~150 tokens against ~7–32 generated ones, and the donor barely perturbs the prompt encoding. The
> mean is therefore dominated by near-zero prompt-position shifts. `shuffle_positions` fails the
> same way: with ~150 prompt positions, almost every swap is prompt↔prompt, small↔small.
>
> **Why that voids the floor argument.** The contrast set is *defined* as base-fails/donor-solves,
> so **base scores exactly 0.000 on it by construction**. A no-op therefore scores 0.000
> automatically, and "the floor is 0.000, not 0.25" is a tautology about an intervention that never
> intervened — not evidence about what a real perturbation does. The claim that the oracle's 0.990
> is not chance-inflated is **currently unsupported**.
>
> **The oracle measurement itself is unaffected** — the dumped generations show it emitting exact,
> clean answers where base emits none. What is missing is the control that rules out chance
> inflation.
>
> **The fix**, for a later run: restrict both controls to **generated positions only**, and add a
> matched-norm random-direction control so "a perturbation this size" is actually tested. The
> per-token δ norms below say how much dilution there was.

**What this does NOT settle, and an overclaim corrected.** An earlier draft of this section said
`mean_delta` "upper-bounds the entire fixed-vector class". That is wrong: the class includes
*scaled* vectors α·δ̄, and this run applies the average at full magnitude (α=1.0) at **every
position, prompt tokens included**. Landing *below chance* is the signature of an off-manifold,
destructive injection rather than of a merely uninformative one — the same α-resonance this work
already measured on MuSiQue's ridge leak (≈0.26 at α=1.0, ≈0 at α=1.5). The supported claim is
therefore: **the best fixed vector at the donor's own scale fails completely**, not that no fixed
vector can install a register.

**The format re-score settles it: the controls are destructive.** Re-scoring the *same* floor
generations under `commonsense_format` — which asks only whether the response format was adopted,
right answer or not:

| control at L20 (n=100) | answer accuracy | format compliance |
|---|--:|--:|
| `mean_delta` | 0.000 | **0.000** (0/100) |
| `shuffle_positions` | 0.000 | **0.010** (1/100) |

Neither control emits `"the correct answer is …"` at all.

> **Correction (2026-08-13, same day).** An earlier version of this section read those zeros as
> "nothing coherent came out". **That does not follow, and the error is instructive.** *Base itself
> is 0% format-compliant* — declining to emit the trigger is exactly why base scores 0.000 on this
> task. So format compliance runs from 0 (base) to ~1 (donor): it measures **register
> installation**, and a reading of 0 is equally consistent with "the model was destroyed" and "the
> injection did nothing, leaving base-like behaviour". The α sweep makes the point concrete — at
> α=0.1 the injection is a tenth of the donor's magnitude and should be nearly a no-op, yet format
> is still 0.000. A metric whose floor and whose failure mode are the same number cannot separate
> them. Raw generations are inspected below to settle it.

Two consequences, and they point in opposite directions:

1. **The oracle result stands, and is strengthened.** The objection was that a 4-way answer space
   might let any large perturbation score well by luck. The measured answer is that a perturbation
   of the *same magnitude as the true shift* yields 0.000, not the 0.25 a fluent-but-random policy
   would get. The floor is the floor. **0.990 is real.**
2. **These controls do not, on their own, test pointwise-ness.** What the table establishes is
   narrow and worth stating exactly: *neither the trajectory-average shift nor the time-shuffled
   shift installs the register at the donor's own magnitude.* Whether that is because the direction
   is wrong or because the magnitude breaks the model is **not** decidable from a metric that reads
   0 for base and 0 for a destroyed model alike. The α sweep plus the raw generations below are what
   separate those two.

That is what the α sweep below exists to fix, and note it is now motivated by a measurement rather
than by suspicion.

### Floors, corrected — and the register collapses to ONE DIRECTION

After the dilution bug was fixed (statistic taken over **generated positions**; `random_matched`
added), re-run at L20, n=100:

| intervention at L20 | recovery | generations |
|---|--:|---|
| true oracle, per-token δ | **0.990** | `the correct answer is answer3` |
| **`mean_delta` — ONE constant vector** | **0.820** | `\n the correct answer is answer3` |
| `random_matched` — same per-token norms, random directions | **0.000** | byte-identical to base |
| unpatched base | 0.000 | `\nAnswer1: Planetary density will decrease.\n\nAnswer2: …` |

> ## CORRECTION — `mean_delta` is a per-step ORACLE statistic, not a fixed vector
>
> `lockstep_generate` calls `capture_residuals(S)` **at every decode step**, and `mean_delta`
> computes its mean from *that step's* freshly captured donor residuals. It therefore still needs a
> live donor forward at every step. It collapses variation across **positions**, not across
> **time**. Reading 0.820 as "one constant vector installs the register" was wrong.
>
> The test of an actual fixed vector — estimated once, no donor at inference — is
> `global_register_vector.py`, and it reads **0.000**:
>
> | intervention at L20 | donor at inference? | recovery |
> |---|---|--:|
> | oracle, per-token δ | every step | **0.990** |
> | `mean_delta`, δ averaged over positions, recomputed per step | every step | **0.820** |
> | fixed vector, **per-problem** (`per_problem_vector_commonsense_L20.json`) | no | **0.000** |
> | fixed vector, **pooled** over 100 disjoint problems (`global_register_vector_commonsense_L20.json`) | no | **0.000** at α ∈ {0.5, 1, 1.5, 2} |
>
> The per-problem and pooled fixed vectors were delivered through the *same* additive hook, so the
> gap between 0.820 and 0.000 is the **injection mechanism** (per-step oracle vs one-shot vector),
> not the choice of vector. That control is why the claim did not survive: the pooled vector's
> per-problem cosines are tightly clustered (mean 0.883, min 0.820), so vector *disagreement* cannot
> explain a collapse to zero.
>
> **What is established:** collapsing the shift across positions is nearly free (0.99 → 0.82);
> collapsing it across **time** — which is what any deployable steering vector does — destroys it
> (→ 0.000). The register needs temporally dense information at L20 just as the procedures do. The
> register/procedure difference measured today lives in the oracle's **ceiling and onset**
> (0.99 from L16 vs 0.75 from L20), **not** in the shift being a single direction.

**The direction carries everything; the magnitude carries nothing.** A random shift at the *same*
per-token norms (28–43, i.e. 30–45% of the ~90 residual norm) leaves greedy decoding
**byte-identical** to base, while the mean-δ direction at that magnitude installs both the response
format and the correct answer. The α probe agrees: an arbitrary direction at norm ~33 is inert and
only ~110 produces gibberish (`'wa wa wa waЪcracra…'`), so this is a genuine robustness property of
the model, not a weak intervention.

**This is the two-sided contrast, on one instrument.** Collapsing δ to a single constant costs the
register only 0.99 → 0.82, i.e. **83% of the oracle survives having no temporal structure at all**.
On both procedures the same move is catastrophic: `periodic:2` — merely patching every *other* step,
a far gentler ablation than collapsing to a constant — takes GSM8K and MuSiQue from ~0.75 to ~0.00.
§1's "a register is roughly pointwise, a procedure is time-dense" is now measured rather than
asserted.

> **Scope — this is a PER-PROBLEM vector, not a universal steering vector.** The mean is taken over
> *that problem's own* donor trajectory, so it is oracle-derived: it shows the required shift has no
> temporal structure **within** a problem, not that one vector serves the task. The CAA-style claim
> needs a **global** mean pooled across problems, which is not yet run. Do not state the stronger
> version.

### α sweep on the mean vector — STOPPED, and its completed cells are VOID

Ran α ∈ {0.1, 0.25, 0.5} to completion (format 0.000 at all three, n=100 each) and was stopped
during α=0.75 once the generation dump showed the underlying control is a **no-op**. Scaling a shift
that was already ~0 tells us nothing, so **these cells must not be cited** — they measure the
dilution bug, not the model. Artifacts:
`lockstep_commonsense_format_single_mean_delta_a{0.1,0.25,0.5}.json`, retained only so the void
result is traceable.

The sweep is worth re-running **after** the control is fixed (average over generated positions
only, plus a matched-norm random-direction arm). Design below, unchanged and still correct in
intent:

### α sweep — original design, to re-run against a fixed control

`mean_delta` at L20, n=100, α ∈ {0.1, 0.25, 0.5, 0.75} (α=1.0 = 0.000 above), each to its own
artifact (`lockstep_commonsense_format_single_mean_delta_a{α}.json`).

**Scored on format, not answers, deliberately.** The `commonsense` score requires the trigger
phrase, so answer accuracy is bounded above by format compliance; where format is 0 an answer run is
redundant. Format is therefore the gating question and costs half the grid.

Readings: **format high, answers ≈0.288** ⇒ the fixed direction installs the *register* but carries
none of the *selection* — the shape §10 predicts and has never measured. **Format ≈0 at every α** ⇒
no *fixed vector* installs this register at any scale.

> **Scope — do not overreach here (noted 2026-08-13).** An earlier draft said a null across the α
> grid would force a rewrite of §1's "roughly pointwise" clause. **It would not**, and the
> distinction is the crux of the whole paper:
>
> | object | form | varies per token? |
> |---|---|---|
> | `mean_delta` (this sweep) | a single constant shift | **no** |
> | the ridge map (§4, and S2c below) | `h ↦ h + (Wh + b)` | **yes** — input-conditional |
>
> Both are *pointwise* in the paper's sense — memoryless, a function of the current position alone.
> But the map has a degree of freedom `mean_delta` discards, so it is strictly stronger. A fixed
> vector failing therefore bounds **fixed vectors**, not maps.
>
> **Every commonsense number in this report is oracle-derived** — the true δ, or an averaged,
> shuffled or scaled version of it. **No fitted map has been run on commonsense yet.** The
> register-side map evidence remains what it was: refusal, where the map reaches 0.62 against 0.00
> for CAA/Arditi/CAST. S2c is the run that tests it here, and until it lands the pointwise
> coordinate for commonsense is **not measured**, in either direction.

### PCA band (δ-rank / off-manifold) — pending

Against GSM8K's cliff (top-64 = 55% of energy, 0% recovery).

### S2c — the ridge map on base, and the format/answer split — pending

`collect_cot_residuals` → `fit_ridge_sweep` → `steer_gsm8k`, all `--task commonsense`, then the same
steered generations re-scored under `commonsense_format`, whose `score` asks only whether the
donor's response format was adopted. Greedy decoding is deterministic and both specs share
`problems` and `prompt` by identity, so the pair is an **exact** decomposition of one eval into
*format installation* vs *answer selection* — a split no procedure task can offer, and the
measurement that replaces caveat (2) above.

## Provenance

Every number above names its artifact; see `papers/register_vs_procedure/numbers.md`.
