# The register battery — measuring the other half of the boundary (S2)

**Date:** 2026-08-13 → 2026-08-17 · **Briefs:** `tasks/s2_execution_brief.md`,
`tasks/s2d_execution_brief.md` · **Spec:**
`docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3 · **Status:** measurements
complete (consolidated 2026-08-17; the α re-sweep on the corrected control and the PCA band were
CUT for the deadline per the pushback's priority call — they are listed as cut, not pending)

> **How to read this file.** It is the drafting source for §9 and it went through three same-day
> retractions. Retraction blocks are kept for provenance, but as of the 2026-08-17 consolidation no
> superseded claim stands as live prose outside a marked block — if a sentence here conflicts with
> a `RETRACTED`/`CORRECTION` block, that is a bug in this file, not a judgment call for the reader.

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

> **Caveat, since closed by the corrected floors.** A 4-way answer space means a large perturbation
> that merely pushes the model somewhere fluent could score well by accident, which is not a risk on
> GSM8K (a garbled injection cannot emit the right integer). The corrected floors settle it:
> `random_matched` (independent directions, true per-token norms) and S2d's `random_constant` (one
> coherent direction, matched norm, same plumbing that installs the register when given the mean-δ
> direction) both read **0.000** with base-like generations. The 0.990 is direction-specific, not
> chance-inflated.

### Floors at L20 — first attempt, RETRACTED (kept for provenance)

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

> **SUPERSEDED (consolidation note, 2026-08-17).** Everything from here to "Floors, corrected"
> below is the same-day analysis of the *retracted* first-attempt floors, kept for provenance. Its
> conclusions do not stand: the "controls are destructive" reading and the "the floor is the floor,
> 0.990 is real" argument both rest on controls the RETRACTED block above shows were no-ops. The
> claims that survive do so on the strength of the **corrected** floors ("Floors, corrected") and
> the **S2d** cells, not on anything in this stretch.

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

### Floors, corrected — position-collapse is nearly free, but only with a live donor
*(heading rewritten 2026-08-17; the original said "the register collapses to ONE DIRECTION", which
the CORRECTION block below refutes)*

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

**What this table does and does not compare** *(rewritten 2026-08-17; the original paragraph here
claimed "83% of the oracle survives having no temporal structure at all", which the CORRECTION
above refutes — `mean_delta` re-captures donor residuals every decode step, so it has temporal
structure by construction)*. The supported statement is: **collapsing the shift across positions is
nearly free (0.99 → 0.82, still with a live per-step donor); collapsing it across time destroys it
(fixed vectors → 0.000, three independent ways — see the fixed-vector table above and S2d below).**
Note also that `mean_delta` (a position-collapse) and the procedures' `periodic:2` (a time-ablation)
are **not the same axis** and must not be set side by side as if they were. On the *temporal* axis
the register behaves like the procedures — a vector estimated once installs nothing — and the
measured register/procedure separation lives in the oracle's **onset and ceiling** (L16 0.830 /
plateau 0.990 vs MuSiQue's L16 0.020 / 0.76) and, decisively, in the fitted-map arm (S2c below).

> **Scope.** The mean in `mean_delta` is taken over *that problem's own* donor trajectory, so it is
> oracle-derived: it bounds the shift's positional structure **within** a problem. The CAA-style
> global-vector claim was tested separately and reads **0.000** — pooled over 100 disjoint
> problems, at α ∈ {0.5, 1, 1.5, 2} (`global_register_vector_commonsense_L20.json`).

### S2d — the positional decomposition, and the floor `random_matched` could not supply (2026-08-17)

Three cells, L20, n=100, `--n-eval 500`, max_new 32, run on the replacement box (RTX 5090 32 GB)
after AC1 revalidated 3/3. Every JSON persists its first 8 decoded generations (harness added after
the 08-13 retraction), and the generations were read before any number below was written.

| intervention at L20 | recovery | generations |
|---|--:|---|
| `mean_delta`, **generated positions only** | **0.860** | `\n the correct answer is answer3 …` — register installed |
| `mean_delta`, **prompt positions only** | **0.000** | base-like option re-listing |
| `random_constant` — ONE coherent random direction at ‖mean gen δ‖, all positions | **0.000** | base-like option re-listing |

Artifacts: `lockstep_commonsense_single_{mean_delta_generated,mean_delta_prompt,random_constant}.json`.

**The 0.820 lives entirely in steering the ~7 generated tokens.** Restricting the injection to
generated positions *raises* it slightly (0.860 ≥ 0.820); restricting it to the ~150 prompt
positions yields exactly nothing. Prompt re-encoding contributes nothing to the effect — which also
clears the prompt-length half of the pushback's confound for this statistic (the oracle-ceiling
length confound on 0.99-vs-0.75 remains open; lead with onset).

**`random_constant` is the matched floor `random_matched` was not.** `random_matched` drew an
*independent* random direction per position — partial cancellation downstream is expected —
whereas `mean_delta` injects one *coherent* direction everywhere. `random_constant` closes that
gap: one coherent random direction at the true norm, still 0.000, generations base-like. And
because it shares the exact injection plumbing with the `mean_delta` cells that installed the
register in the same session, this null is **direction-specificity, not a no-op** — the failure
mode the 08-13 retraction taught, this time excluded by construction.

**Fixed vector through the lockstep path (S2d, 2026-08-14): 0.000, with a caveat.** Freezing a
per-problem whole-trajectory vector and delivering it through the *lockstep* path (not the additive
hook) also reads **0.000** (`lockstep_commonsense_single_fixed_vector_per_problem.json`),
generations base-like rather than destroyed — the third independent fixed-vector null. But the
run's own diagnostic puts the frozen vector at cosine **0.544** (min 0.304, max 0.803, 3,023 steps)
to the live running mean, so direction *and* loop both varied; the single-variable "is 0.820 an
early-step artifact" discriminator (re-inject the final-step running mean from step 1) is **not
built and was cut for the deadline**. Until it runs, quote 0.820/0.860 only as "a live per-step
statistic collapsed across positions", never as "no temporal structure". Side finding: successive
running means reaching cosine 0.304 means the required shift **rotates substantially within a
7-token generation** — measured evidence against reading this register as one direction, consistent
with the centred-R² table below.

### α sweep on the mean vector — STOPPED, and its completed cells are VOID

Ran α ∈ {0.1, 0.25, 0.5} to completion (format 0.000 at all three, n=100 each) and was stopped
during α=0.75 once the generation dump showed the underlying control is a **no-op**. Scaling a shift
that was already ~0 tells us nothing, so **these cells must not be cited** — they measure the
dilution bug, not the model. Artifacts:
`lockstep_commonsense_format_single_mean_delta_a{0.1,0.25,0.5}.json`, retained only so the void
result is traceable.

**Status 2026-08-17: the re-run is CUT for the deadline.** The question it was built to answer has
since been answered by stronger cells: the S2d `random_constant` and generated/prompt decomposition
supply the corrected floor, and the fixed-vector class reads 0.000 through three independent
constructions. The design is retained below only so a post-deadline run can pick it up.

### α sweep — original design (CUT, retained for a post-deadline run)

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

### PCA band (δ-rank / off-manifold) — CUT for the deadline

Against GSM8K's cliff (top-64 = 55% of energy, 0% recovery). Cut per the pushback's priority call
(item 7: S2c and the GSM8K sweep are load-bearing; the PCA band and the corrected
`shuffle_positions` are the first cuts). §9/§10 must state the commonsense δ-rank coordinate as
**not measured**, not as a small number.

### S2c — the ridge map on base: the register installs, and it carries nothing else

`collect_cot_residuals` → `fit_ridge_sweep` → `steer_gsm8k`, all `--task commonsense`, then the same
steer re-scored under `commonsense_format`, whose `score` asks only whether the donor's response
format was adopted. Greedy decoding is deterministic and both specs share `problems` and `prompt` by
identity, so the pair is an **exact** decomposition of one eval into *format installation* vs
*answer selection* — a split no procedure task can offer.

#### The first attempt was VOID, and the accuracy could not have told us

The 2026-08-13 run had died leaving no artifact at all. Relaunched 2026-08-14, it produced
`steer_commonsense.json`: **0.000 at every layer** (8/12/16/20/24/28/31, α=1.0, n=500). Reading the
generations — per the rule this project keeps re-learning — shows that is a **destruction** result,
not a transport null:

| condition @L20 | generation |
|---|---|
| unpatched base | `\nAnswer1: Planetary density will decrease.\n\nAnswer2: …` |
| steered α=0.5 | **byte-identical to base** — a no-op |
| steered α=1.0 | `\n\end​​​​​​…` — degenerate zero-width-space repetition, every problem |

**Cause: the fit window did not match the application window.** `collect_cot_residuals` used
`cot_token_slice`, which keeps *generated positions only* — **6 tokens per problem** here — while
`LinearPrimalSteerHook` applies the map at **every** position. So ~94% of the positions the map was
applied to lay off its fit distribution, and it extrapolated there to roughly double the correct
magnitude: `mean‖Wa‖ = 53.6` against `mean‖a‖ = 97.2` (ratio **0.551**) where the true δ ratio is
~0.3–0.45. On GSM8K the same mismatch is mild — the chain is ~250 of ~400 positions — which is why
it never surfaced in three prior strands. Fixed by `--fit-positions all`.

#### The all-positions arm

Fit on all ~41 positions per problem: **8,046 train / 2,393 held-out tokens** against the CoT
window's 1,200 / 360. `sweep_commonsense_allpos.json`, maps in `maps_commonsense_allpos`.

| layer | R²_te | constant baseline | R²_te centred |
|---|--:|--:|--:|
| 12 | 0.9785 | 0.0345 | 0.9777 |
| 16 | 0.9489 | 0.0599 | 0.9457 |
| **20** | **0.9293** | **0.1064** | **0.9209** |
| 24 | 0.9176 | 0.1431 | 0.9038 |
| 28 | 0.9136 | 0.1771 | 0.8950 |
| 31 | 0.9311 | 0.1216 | 0.9215 |

`r2_te` divides by the **uncentred** Σ‖δ‖², so it credits a map for merely reproducing δ's constant
component — and how constant δ is happens to be the exact property this paper contrasts. The
constant baseline (`GramAccumulator.constant_r2`, streamed as a first moment) removes that
confound, and `R²_te_centred = (R²_te − const)/(1 − const)` is an identity requiring no refit.

**The constant explains only 0.106 at L20**, so over the full sequence the required shift is
strongly input-conditional. That is *consistent with* `mean_delta`'s 0.820 rather than in tension
with it: over the ~6 **generated** positions the shift is close to one direction (‖mean gen δ‖ 29
against per-token 42), while over the ~97 **prompt** positions it is diverse and cancels. Every
"the register is one direction" sentence must therefore name the window it holds on — the earlier
controls looked only at the generated window, which is the window where it is true.

#### Installation is real, narrow, and format-only

Generation dump at L20 (`.run_logs/s2c_gens_allpos.log`), `‖Wa‖/‖a‖ = 0.224`:

| α | steered generation | reading |
|---|---|---|
| 0.25 | byte-identical to base | no-op |
| 0.5 | byte-identical to base | no-op |
| **0.75** | **`the correct answer is answer1`** | **the donor's format, installed** |
| donor | `the correct answer is answer3` | correct per problem |

At α=0.75 the map emits the register format and answers **`answer1` almost every time**. Over
n=40 contrast problems (`.run_logs/s2c_answer_hist.log`, L20, α=0.75):

| | |
|---|--:|
| format compliance | **34/40 = 0.850** (base 0.004) |
| accuracy | **0.200** |
| predicted `answer1` | **31/40** |
| other predictions | one each of `answer2`, `answer3`, `answer4`, and six junk strings (`0`, `13`, `41°`, `43°f`, `respiratory`, `true`) |
| gold distribution | `answer1` 8 / `answer2` 8 / `answer3` 19 / `answer4` 5 |

**Every one of the 8 correct answers is a problem whose gold happened to be `answer1`.** The
accuracy is exactly the base rate of the constant the map emits, and it sits *below* both the 0.25
chance floor and the 0.288 majority-class floor. So this is a degenerate constant policy, not weak
selection — the discriminator being the *distribution*, which a 5-problem dump could not have
supplied.

**Replicated at n=60 on a disjoint-sized sample** (`.run_logs/s2_alpha_selection.log`, same layer
and α): accuracy **0.267**, format 0.850, `answer1` on **45/60**, and gold `answer1` is **16/60 =
0.267**. The accuracy equals the constant's base rate to three decimals a second time, which is the
signature the reading predicts and chance would not reproduce. This is row 1 of the S2c prediction table: *the map installs the register and nothing
else.* It also retro-explains the procedure leak as a pure register push, and it is the cleanest
available statement of the paper's thesis.

#### The register installs almost completely — the ladder's register arm

`commonsense_format` sweep, n=500 scan, α=0.75, `maps_commonsense_allpos`, references measured on
the same 500 problems: **base 0.004 / donor 1.000**.

| layer | 8 | 12 | 16 | **20** | **24** |
|---|--:|--:|--:|--:|--:|
| format compliance | 0.010 | 0.016 | 0.140 | **0.972** | **0.998** |

A fitted pointwise map installs the donor's register **essentially completely** from L20 onward,
against base's 0.004. Set beside the same instrument on the procedures — under **matched
all-positions fits** (2026-08-17): GSM8K 0.09 @L20 / 0.12 @L24, MuSiQue 0.07 @L20 / 0.26 @L24;
the CoT-window values quoted here before that date (0.03/0.12, 0.26/0.45) are unmatched-window —
this is the two-sided contrast the paper has so far asserted from one side only:

> **a register transports through a fitted pointwise map at a single layer; a procedure does not.**

Note the populations differ and must not be conflated: the 0.972 is over the **500-problem scan**,
while the 0.850 compliance quoted with the answer histogram is n=40 over the **contrast set**
(base-fails/donor-solves). The contrast-set figure is the one that pairs with the 0.200 accuracy.

#### There is no α at which the map both installs the register and selects

The geometry (below) says the map's shift points near the donor's but is only ~62% of its norm at
α=0.75, so the obvious objection is that selection was lost to **under-scaling** rather than absent
from the map. Tested directly (`.run_logs/s2_alpha_selection{,2}.log`, L20, n=60 contrast; the
magnitude-matching α computed from ‖δ_donor‖=45.60 against ‖W·a‖=36.88 is **1.24**):

| α | accuracy | format | `answer1` | reading |
|---|--:|--:|--:|---|
| 0.75 | 0.267 | 0.850 | 45/60 | format installed, constant answer |
| 0.90 | 0.283 | 0.850 | 45/60 | same |
| 1.00 | 0.233 | 0.767 | 41/60 | same, degrading |
| **1.24 — magnitude-matched to the donor** | **0.033** | **0.083** | 4/60 | **format collapses too** |
| 1.50 | 0.000 | 0.000 | 0/60 | fully destroyed |

Scaling to the donor's own magnitude does **not** recover selection — it loses the format as well.
The installation window is roughly α ∈ [0.75, 1.0] and the policy is a degenerate constant
throughout it. So the map works only where it is *deliberately undersized*: at 60–80% of the
donor's norm it installs the surface form, and at the donor's actual norm it breaks the model.
Whatever the donor does with the remaining ~40% of that shift, this map cannot reproduce. The
objection is answered and the claim is now magnitude-controlled.

This also retro-explains the CoT-window disaster: that map sat at ratio **0.551** against this one's
**0.224**, i.e. permanently above the top of the window — which is why its α=0.5 was a no-op and its
α=1.0 produced `\end​​​​` garbage, with no usable cell in between. Same α-resonance already measured
on MuSiQue's ridge leak, but here the band's location is set by the fit window.

### Base knew the answers — what the contrast set actually measures

Judgment call (c) of the brief asserted that base scores 0.000 "by format non-compliance, not
incapacity — the base model plainly knows some of these answers", and said §9 must *disclose* it.
It was never measured. It is now (`.run_logs/s2_format_rescue.log`, n=100 contrast problems), and
the assertion is correct and larger than expected:

| arm | accuracy | format compliance |
|---|--:|--:|
| base, zero-shot (the contrast protocol's own prompt) | **0.000** | 0/100 |
| **base, 4-shot — format supplied by demonstration** | **0.630** | **100/100** |
| **base, 4-shot with WRONG exemplar labels** | **0.630** | **100/100** |
| donor LoRA, zero-shot | 1.000 | 100/100 |

The scrambled-label arm is the control that makes this readable: 4-shot demonstrates the *task* as
well as the format, so a high few-shot score could have meant either. Relabelling the exemplars
wrongly (3 of the 4 actually changed) leaves accuracy at **exactly 0.630** and the prediction spread
essentially unchanged (23/17/23/37 against 23/16/28/33). The demonstrations therefore supply
**format and nothing else** — a replication of the standard random-label ICL finding, cited here as
a control rather than a result.

Base scores **0.630 on problems the contrast set defines as base-failures**, with predictions spread
over all four options (23/16/28/33) rather than pinned to one. Note the zero-shot prompt *already
states* `Answer format: answer1/answer2/answer3/answer4` and base still complies 0/100 — so the
deficit is not ignorance of the format but the ordinary fact that a base model follows a
demonstration and not an instruction.

**Consequence for everything upstream.** The commonsense base→donor gap is roughly two-thirds
*format*, so the oracle's 0.990, the L16 0.830 onset and `mean_delta`'s 0.820 are substantially
measuring **format installation** rather than capability transfer. This does not void them — a
format register is a register, and installing one is exactly the claim — but §9 cannot describe the
0.990 as recovering a capability base lacks.

**And it sharpens what the map does.** Placing the four interventions side by side:

| intervention | format | selection | accuracy |
|---|---|---|--:|
| fixed vector (CAA-style, pooled or per-problem) | ✗ | — | 0.000 |
| ridge map @L20, α=0.75 | ✓ 0.85 | **✗ constant `answer1`** | 0.200 |
| 4-shot prompt | ✓ 1.00 | **✓ base's own** | **0.630** |
| donor LoRA | ✓ 1.00 | ✓ donor's | 1.000 |

The map does not merely fail to *add* selection — it **destroys selection base already had**, while
a prompt that installs the same format leaves it intact. That is a stronger and more interesting
statement of the register/procedure boundary than "installs the register and nothing else".

**Caveat, stated not buried.** The few-shot arm changes the prompt, so it is not the contrast
protocol's "base"; it answers "does base know?", which is the question, but it is a separate control
rather than a cell of the same table. (The in-context-learning caveat — that exemplars demonstrate
the task as well as the format — is closed by the scrambled-label arm above.)

#### Do demonstration and the map move the model the same way?

Both reach ~full format compliance but differ completely in what survives, so the geometry is worth
asking directly. Measured at the **last prompt position** — the generation site, where both prompts
end with the same question text — as three shifts from one common reference state `a_zero`
(`.run_logs/s2_geometry.log`, L20, α=0.75, n=30):

| pair | mean cosine | sd |
|---|--:|--:|
| **map ~ donor** | **+0.788** | 0.013 |
| few-shot ~ donor | +0.380 | 0.021 |
| **few-shot ~ map** | **+0.332** | 0.016 |

| shift | mean norm |
|---|--:|
| `‖a_zero‖` (the state itself) | 48.68 |
| `‖δ_few‖` | 53.50 |
| `‖δ_donor‖` | 44.75 |
| `‖δ_map‖` | 27.92 |

**The map tracks the donor's direction but undershoots its magnitude** — cosine 0.788, remarkably
tight, at 62% of the donor's norm. **In-context demonstration reaches the same behaviour by a
substantially different route**: ~0.38 from the donor, ~0.33 from the map. In 4096 dimensions random
pairs sit near 0, so these are far from orthogonal and real shared structure exists — but nothing
like the map's 0.788. So "install the format" is not a single direction: the weight-edit route and
the ICL route are roughly 70° apart, and only the ICL route leaves selection intact.

> **Confound, open.** The few-shot prompt is 324 tokens longer, so `δ_few` contains whatever a
> longer context does to the residual generically, not only the format demonstration. Until a
> length-matched preamble that demonstrates *no* format is subtracted, `δ_few` names a "few-shot
> context effect", not a "format direction". **Do not quote the cosines as a format-direction claim
> before that control runs.**

> **Do not mix two different norms.** `‖Wa‖/‖a‖ = 0.224` earlier is averaged over *all* prompt
> positions (where `‖a‖ ≈ 97`); `‖a_zero‖ = 48.68` here is the single last position. Both are
> correct and they are different quantities.

## Consolidated verdict — what §9 may draft from (2026-08-17)

**Quotable, with artifacts (rows in `numbers.md`):**

1. **Oracle axis separates register from procedure.** Same driver, same rule: L\*=20 for all three
   tasks, but onset L16 **0.830** (MuSiQue 0.020) and plateau **0.990** (0.76 / 0.75). Lead with
   the *onset* — it is within-task-across-layer and immune to the generation-length confound; the
   0.99-vs-0.75 *ceiling* comparison carries that confound and must be caveated (~7-token target vs
   256–512-token chains).
2. **The fitted-map arm is the paper's two-sided contrast on one instrument.** Format compliance
   @α=0.75: L20 **0.972** / L24 0.998 (base 0.004), against the procedures under **matched
   all-positions fits** (2026-08-17): GSM8K 0.09/0.12, MuSiQue 0.07/0.26 at L20/L24
   (`steer_results_allpos.json`, `steer_multihop_allpos.json`; the earlier CoT-window 0.03/0.12 and
   0.26/0.45 are unmatched-window values — see numbers.md).
   A register transports through a fitted pointwise map at a single layer; a procedure does not.
3. **The map installs the register and destroys selection.** Accuracy under the map equals the
   constant policy's base rate twice (0.200 vs 8/40; 0.267 vs 16/60); no α installs and selects
   (magnitude-matched α=1.24 collapses format to 0.083). Beside the 4-shot arm (format 1.00,
   accuracy 0.630, base's own selection intact) this is: *the weight-route register install costs
   selection base already had; the prompt-route install does not.*
4. **The contrast set is ~two-thirds format.** Base 4-shot 0.630 = scrambled-label 0.630. §9 may
   not describe the oracle's 0.990 as recovering a capability base lacks.
5. **Fixed vectors install nothing** — 0.000 three independent ways (per-problem additive hook,
   pooled-over-100-problems at four α, per-problem through the lockstep path). On the temporal
   axis the register matches the procedures.
6. **The 0.820/0.860 is generated-token steering with a live per-step donor.** Positional
   decomposition: generated-only 0.860, prompt-only 0.000, `random_constant` floor 0.000.
   Direction-specific by construction (same plumbing installs the register).

**Not quotable / not measured:**

- "The register is one direction" — refuted at trajectory level (fixed vectors 0.000; running-mean
  rotation to cos 0.304; centred R² 0.921 means the map's shift is strongly input-conditional).
- "0.820 survives having no temporal structure" — the statistic is recomputed per step; the
  early-step discriminator was cut.
- The few-shot geometry cosines as a "format direction" claim — length-matched preamble control
  not run.
- Commonsense δ-rank / PCA band — **not measured** (cut). The `shuffle_positions` corrected re-run —
  not run; only the retracted no-op version exists.
- The `constant_r2` value as a register/procedure coordinate — retracted (0.096 GSM8K vs 0.106
  commonsense; the discriminator is the *centred* R²: 0.653 vs 0.921).

## Provenance

Every number above names its artifact; see `papers/register_vs_procedure/numbers.md`.
