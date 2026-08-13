# Multi-hop generality: does the procedure thesis survive a non-arithmetic procedure?

**Question.** Every "procedure does not install" result so far is GSM8K arithmetic. The register-vs-
procedure thesis predicts the same signature on *any* multi-step procedure: full-δ lockstep oracle
recovers, pointwise-map ladder ≈0, recovery temporally dense. **H_general**: all three axes
replicate on multi-hop QA. **H_arith**: some axis diverges — the wall was arithmetic-specific.
Either is publishable; this run adjudicates.

**Task.** MuSiQue (`dgslibisey/MuSiQue`) *open-book*: the gold supporting passages are inlined into
the instruction, so the donor learns multi-hop *composition over given facts* — the analogue of
GSM8K's in-problem numbers — not parametric recall. Answerable-only, seeded shuffle (42). Scoring is
SQuAD-normalized exact match against gold + aliases (`multihop_prompts.answer_match`).

**Apparatus.** Byte-identical to the GSM8K arm, via a task registry
(`attribution_common.TASKS`: problems / prompt / score / format_gold) that the five drivers consume
under `--task multihop`; GSM8K stays the default and its committed results are unchanged. Base
`NousResearch/Llama-2-7b-hf`; donor = LoRA r32/α64 on ~20k open-book chains
(`train_lora_multihop.py`, LLM-Adapters recipe, matches the commonsense arm).

## P0 — the recoverable budget exists (gate PASSED)

500-problem validation scan, greedy, identical prompts for base and donor
(`multihop_gap.json`):

| | exact match |
|---|---|
| base (open-book, zero-shot) | **0.000** |
| donor LoRA | **0.634** |

317 base-fails/donor-solves contrast problems (gate needed ≥80), cached with the GSM8K schema
(`multihop_contrast_set.json`) and reused verbatim by every later phase. Base 0.000 is the same
degenerate-format failure mode as GSM8K's base: the capability budget is entirely the donor's.

## P1 — full-δ lockstep oracle

AC1 (wiring): all-layers lockstep reproduces the donor's greedy decoding exactly
(3/3 per-problem match). Positive control, 100 contrast problems
(`lockstep_multihop_control.json`):

| injection | acc | recovery |
|---|---|---|
| all 32 layers | 1.000 | **+1.000** |

Single-layer sweep L ∈ {0,4,8,12,16,20,24,28,31}, 100 contrast problems
(`lockstep_multihop_single.json`), with the GSM8K curve (`2026-06-09` run, 20 contrast) alongside:

| L | multihop | GSM8K |
|--:|--:|--:|
| 0 | +0.000 | 0 |
| 4 | +0.000 | 0 |
| 8 | +0.070 | 0 |
| 12 | +0.020 | 0 |
| 16 | +0.020 | 0.20 |
| 20 | **+0.760** | **0.75** |
| 24 | +0.780 | 0.75 |
| 28 | +0.890 † | 0.95 † |
| 31 | +1.000 † | 0.95 † |

† Degenerate tail, excluded from L\* by the same rule as GSM8K: the hook overwrites the *output* of
`model.model.layers[L]`, so L31 hands lm_head the LoRA's final hidden state verbatim (the all-layers
control in disguise) and L28+ sit close enough to readout to inherit most of that triviality.

**The curve replicates in full**: zero through the early stack, a sharp onset into a plateau at
exactly **L20 with recovery +0.760 vs GSM8K's 0.75**, and the same degenerate rise at the readout
end. Multihop's onset is even sharper (L16 = +0.02 vs GSM8K's 0.20). **L\* = 20** — the same layer,
selected by the same earliest-plateau rule, so the P2/P3 GSM8K configs port unchanged.

## P2 — pointwise-map ladder @L20

Residuals collected on the donor's CoT (200 fit / 60 held-out problems; 11,901 / 3,444 CoT tokens —
multihop chains are ~60 tokens, shorter than GSM8K's ~250). Ridge maps fit per layer over
λ ∈ logspace(−1, 7, 17) (`sweep_multihop.json`): at L20, λ\* = 3.16e3, **held-out R²_te = +0.71**.
The GSM8K comparison here was unsourced when first written (corrected 2026-08-06): an earlier draft
cited "≈0.61" with no artifact; the only committed sweep, `sweep_smoke.json`, gave a floor of 0.367.
**The P5 refit (2026-08-10, `sweep.json`, full n_fit 200 / n_te 60) measured it: GSM8K L20
R²_te = 0.610 at λ\* = 3.16e3** — the unsourced figure was accurate, and now has an artifact behind
it; GSM8K L24 fits slightly better still (0.636). Multihop L20 remains the stronger fit at 0.714.
As in GSM8K, the *open-loop geometry is good* — the ladder question is whether it survives
closed-loop decoding.

| rung | recovery | GSM8K analogue |
|---|--:|--:|
| ridge steer @L20 | **+0.26** scan / **+0.21** contrast at n=100 (the first-pass n=20 contrast read of +0.35 was small-n inflation; `nonlinear_delta_multihop_L20_n100.json`) | **+0.03 [0.01, 0.08]** measured 2026-08-10 (P5, n=200) ‡ |
| nonlinear MLP @L20 | **+0.00** (val cos +0.822 / R² +0.675 vs ridge +0.636 / +0.270 — better fit, zero closed-loop; +0.01 at n=100 contrast) | **+0.00 [0, 0.17]** measured 2026-08-10 (n=20; cos 0.806 / R² 0.651 vs ridge 0.631 / 0.330 — **same paradox**) |
| on-policy DAgger (joint all-layer) | **+0.00 / +0.00 / +0.00** (rounds 0–2, `dagger_refit_multihop.json`) | 0.00 all rounds |
| full-δ oracle @L20 | **+0.760** | 0.75 |

### Provenance of the GSM8K comparison column (audit, 2026-08-06)

Two cited GSM8K numbers turned out to have no artifact behind them, so every cross-task cell in this
report was traced to a file. Result:

| comparison cell | GSM8K value cited | artifact | verdict |
|---|---|---|---|
| full-δ oracle @L20 | 0.75 | `temporal_oracle_L20.json` (`periodic_1`) | **sourced** |
| temporal-density column (all 9 gates) | 0.750 / 0.000 ×5 / 0.000 @2.9% / 0.700 @94.4% / 0.000 @9.4% | `temporal_oracle_L20.json` — exact match on every cell | **sourced** |
| DAgger, all rounds | 0.00 | `dagger_refit_gsm8k.json` (rounds 0–2, joint) | **sourced** |
| plan-vs-execute (P4) | 0.968 / 0.895 / 0.835 | `gold_token_lens_L20.json`, re-run and reproduced exactly today | **sourced** |
| ridge steer @L20 | "≈0.05" | none at audit; **P5 measured +0.03 [0.01, 0.08]** (`steer_results_layers.json`, n=200) | **closed 2026-08-10** — the unsourced figure was nearly right ‡ |
| ridge "≈0 at every layer" | ≈0 | per-layer only L0/L1/L14/L16/L31 at audit; **P5 measured the full curve — false**: L24 +0.12, L28 +0.13, L31 +0.12, intervals exclude 0 | **refuted by measurement** ‡ |
| held-out R²_te @L20 | "≈0.61" | `sweep_smoke.json` gave 0.367; **P5 full refit: 0.610** (`sweep.json`) | **closed 2026-08-10** — figure vindicated |
| nonlinear MLP @L20 | 0.00, "same paradox" | **P5 measured 0.00 [0, 0.17] with better geometry than ridge** (`nonlinear_delta_gsm8k_L20.json`) | **closed 2026-08-10** — paradox confirmed (n=20 power caveat) |

The three structural axes (oracle, temporal density, plan-vs-execute) are fully backed; **every
unsourced cell was in the P2 ladder row** — the one axis this report called a divergence. The P5 run
(2026-08-10, `2026-08-10-gsm8k-ridge-layer-probe.md`) closed all four: R²_te and the MLP paradox
were vindicated, the "≈0.05" point value was nearly right, and the "≈0 at every layer" claim was
refuted — the leak the audit's five layers happened to miss lives at L24–L31.

‡ **Closed by measurement 2026-08-10** (P5 run, `steer_results_layers.json`; the history of the gap
is kept for the record). At audit time (2026-08-06) every committed GSM8K ridge-steering measurement
read 0.00, but the *per-layer* ones covered only L0/L1/L14/L16/L31 at α ≤ 1.0
(`steer_results_smoke.json`, n=12/50); the rest were all-layer **joint** injections
(`short_arithmetic.json`, `local_refit_gsm8k.json`, `dagger_refit_gsm8k.json`), and L20/L24 had
never been probed. The P5 refit + per-layer sweep (n_eval=200, α=1.0, max_new=512, base 0.000 /
LoRA 0.650 measured under the same protocol) filled the curve — see the P2b table. Headline:
**GSM8K is 0.03 [0.01, 0.08] at L20 but 0.12 [0.07, 0.19] at L24** — small, late, and *non-zero*.
The audit-era correction "every GSM8K ridge number really is 0.00" (600b5f7) is itself refuted; the
original unsourced "≈0.05" was closer to the truth than the correction that replaced it. Note the
historical joint-injection nulls coexist with the single-layer L24 leak only if mid-stack injections
actively corrupt the trajectory and swamp the late-layer gain — itself informative about the map's
off-manifold behaviour.

**Reading (rewritten 2026-08-10, matched comparison).** Two rungs replicate exactly (MLP pending its
first GSM8K artifact; DAgger flat at 0 with on-policy data). The ridge rung is now measured on both
tasks at every layer, same n, α, and generation budget, and the honest statement is: **both tasks
leak, with the same shape and different amplitude.** Both curves are ≈0 through mid-stack and
develop their leak late (GSM8K: 0/0/.05/.03/.12/.13/.12 over L8…L31; multihop: 0/0/0/.26/.45/.38/.24).
At the matched layers the separation is clean — L24 0.12 [0.07, 0.19] vs 0.45 [0.35, 0.56],
intervals disjoint — so the *amplitude* is task-dependent by ~3.75× at the peak. The wall exists on
both tasks (oracle beats the best map by ≥0.5 of budget on GSM8K, ≥0.3 on multihop), but "the
GSM8K rung is 0.00" is no longer available as a statement of the divergence; the correct statement
is a shape replication with task-dependent leak size, consistent with the register-vs-procedure
split: the late-stack, output-adjacent (register-like) component of the delta transports through a
pointwise map on both tasks, and its share of the budget is what differs — small on arithmetic
(computation-limited), large on open-book hop composition (scaffold-limited). One reversal to keep
honest: at L16 GSM8K reads 0.05 [0.02, 0.10] where multihop is 0.00 [0.00, 0.03], so neither task
dominates the other everywhere.

### P2b — characterizing the ridge divergence (follow-up)

The first-pass caveat was that the divergence sat at one layer and one α. Both dimensions are now
swept on the same 200-problem scan (base 0.000 / donor 0.630 supplied as fixed references).

**α sweep @L20** (`steer_multihop_alpha_L20.json`): recovery is a *narrow resonance at α = 1.0*,
collapsing on both sides —

| α | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| recovery | +0.01 | +0.01 | +0.02 | **+0.26** | +0.05 | 0.00 | 0.00 |

Under-driving the map is as fatal as over-driving it: the transportable component only functions
when the injected shift matches the true δ in scale, echoing the Goldilocks bands seen throughout
the steering strand — but far sharper (a ±25% mis-scale forfeits ~all of the leak).

**Layer sweep @α = 1.0** (`steer_multihop_layers.json`): the leak is not L20-specific — it follows
the oracle plateau and peaks *later* than the oracle onset —

| L | 8 | 12 | 16 | 20 | 24 | 28 | 31 |
|---|--:|--:|--:|--:|--:|--:|--:|
| multihop ridge steer | 0.00 | 0.00 | 0.00 | +0.26 | **+0.45** | +0.38 | +0.24 |
| multihop oracle | +0.07 | +0.02 | +0.02 | +0.76 | +0.78 | +0.89† | +1.00† |
| **GSM8K ridge steer** (P5, 2026-08-10) | 0.00 | 0.00 | +0.05 | +0.03 | **+0.12** | +0.13 | +0.12 |

GSM8K row: `steer_results_layers.json`, n_eval=200, α=1.0, max_new=512, base 0.000 / LoRA 0.650
measured under the same protocol; 95% intervals in the P5 report
(`2026-08-10-gsm8k-ridge-layer-probe.md`) — the non-zero cells exclude 0, the zero cells bound at
[0, 0.03].

So the leak is a *curve* on **both** tasks, not a point and not one task's property: on multihop the
linearly transportable fraction grows through the plateau to nearly half the budget at L24 before
decaying toward the readout; on GSM8K the same late-onset shape appears at roughly a quarter of the
amplitude (peak +0.13 at L28) and without the late decay. (No † on the steering rows — steering adds
`W·a` to base's own state, so nothing is degenerate about late layers here.) At the matched layers
the amplitude separation is clean (L24 and L28 intervals disjoint); at L16 the ordering reverses
(GSM8K +0.05, multihop 0.00). The former reading — that the curve is multihop-specific — was an
artifact of GSM8K never having been probed where its own leak lives.

**Contrast-set check at n=100** (`nonlinear_delta_multihop_L20_n100.json`): ridge +0.21 / MLP +0.01
— the scan and contrast estimates of the L20 leak now agree at ~0.2–0.26, and the
better-geometry/zero-recovery MLP paradox survives the larger n.

## P3 — temporal density @L20

Gated lockstep on 20 contrast problems (`temporal_oracle_multihop_L20.json`), GSM8K reference
(`2026-06-14-temporal-oracle-L20.md`) alongside:

| gate | frac patched | recovery | GSM8K |
|---|--:|--:|--:|
| periodic_1 (full oracle) | 1.000 | **0.750** | **0.750** |
| periodic_2 | 0.501 | 0.050 | 0.000 |
| periodic_3 | 0.336 | 0.100 | 0.000 |
| periodic_4 | 0.250 | 0.000 | 0.000 |
| periodic_6 | 0.168 | 0.000 | 0.000 |
| periodic_8 | 0.125 | 0.000 | 0.000 |
| answer_only | 0.000 † | 0.000 † | 0.000 (result_only, 2.9%) |
| reasoning_only | 1.000 | **0.750** | **0.700** (planning_only, 94.4%) |
| step_boundary | 0.070 | 0.050 | 0.000 (9.4%) |

† **Vacuous, not evidence**: `answer_only` patches steps after "The answer is:" appears, but with
nothing patched beforehand the degenerate base trajectory never emits the marker, so the gate fired
on 0% of steps and the run trivially equals base. GSM8K's `result_only` escaped this only because
base emits `=`-digits mid-CoT. The informative structural gate is `reasoning_only`.

**Reading.** The density axis replicates in full: full-rate patching recovers 0.750, half-rate
collapses to 0.050, every sparser periodic gate is ≈0 (marginally softer shoulder at k=2–3 than
GSM8K's hard 0.000), the thin structural gate (`step_boundary`, 7%) is ≈0, and `reasoning_only`
equals the full oracle while patching ~100% of steps — skipping the answer span is free, exactly as
skipping GSM8K's result digits was. The trajectory state is temporally dense here too: no sparse
subset — periodic or structural — installs the capability.

**n=100 confirmation** (`temporal_oracle_multihop_L20_n100.json`): the two decisive gates rerun on
the full 100-problem contrast eval reproduce the n=20 reads — `periodic_2` 0.060 (vs 0.050) and
`reasoning_only` 0.760 (vs 0.750, = the single-layer oracle's +0.760 at the same n). The knee and
the structural-complement signature are not small-n artifacts.

## P4 — plan vs execute (the E1b analogue)

P3 says the trajectory state is temporally dense, but not *what* the dense thing is doing. E1b
answered that on GSM8K by teacher-forcing base on a correct chain and lensing the **gold** next
token by its role: base predicted genuinely computed results at 0.968 — better than the chain at
large (0.835) — so its deficit was trajectory control, not per-step arithmetic. P4 runs the same
lens on MuSiQue (`gold_token_lens_multihop_L20.json`).

Two differences from the GSM8K arm, both in multihop's favour. The chain is the **gold** chain,
teacher-forced verbatim: MuSiQue's donor was trained on `format_multihop_solution`, so the
supervised target is already in-format (GSM8K's dataset CoT is not in MetaMath format, so E1b had
to generate the donor's own CoT and verify it). And the roles are built **by construction** —
`chain_token_roles.multihop_chain_spans` renders the chain and its role spans in one pass, and the
same function produces the training target, so labels cannot drift from the text; token roles come
from fast-tokenizer character offsets, with the teacher-forced ids asserted to round-trip. There is
no anchoring step and no drop rate, and the "hop answer repeated inside its own sub-question" case
is right by design rather than by search. This is also why P4 is a lens and not a causal gate:
unlike GSM8K's `=`, MuSiQue's `Step i: <sub-question> <answer>.` has no delimiter, so "am I inside
the answer span" is not decidable online — which is exactly what made P3's `answer_only` gate
vacuous. Teacher-forcing removes that failure mode by construction; every class below is non-empty.

n = 317 contrast problems (all of them), 19,970 scored tokens, **LoRA-TF sanity 0.950**. That is
lower than GSM8K's 0.997 by design, not by defect: GSM8K forces the donor's *own greedy* CoT, whose
TF-accuracy is ≈1 by construction, whereas here the donor is forced on the gold target it was
trained to approximate but does not reproduce greedily. 0.950 confirms the prompt and chain join
are right.

| role | n | TF-acc | final rank | lens rank L20→L31 |
|---|--:|--:|--:|---|
| all | 19970 | 0.725 | 0 | 2 1 1 0 0 0 0 |
| sub_question (plan) | 8334 | **0.671** | 0 | 2 1 0 0 0 0 0 |
| hop_answer (execute) | 3004 | **0.725** | 0 | 3 1 0 0 0 0 0 |
| — hop 1 | 956 | 0.637 | 0 | 4 1 0 0 0 0 0 |
| — hop ≥ 2 | 2048 | 0.767 | 0 | 3 1 0 0 0 0 0 |
| final_answer (copy) | 1586 | **0.933** | 0 | 0 0 0 0 0 0 0 |
| scaffold (format) | 7046 | 0.742 | 0 | 4 3 2 3 1 0 0 |

Tokens are not independent within a chain, so the decisive differences carry a 95% bootstrap
interval resampling **problems**, not tokens (317 clusters):

| contrast | Δ TF-acc [95%] |
|---|---|
| execute − plan | **+0.055 [+0.040, +0.069]** |
| execute − all | +0.001 [−0.010, +0.011] (spans 0) |
| hop ≥ 2 − hop 1 | **+0.130 [+0.106, +0.154]** |
| copy − execute | **+0.207 [+0.193, +0.222]** |

**Reading — the ordering replicates, the elevation does not.** Base agrees with execution tokens
more than with planning tokens, and the gap survives problem-level clustering (+0.055, interval
clear of 0). Directionally that is E1b's result: given the working, the harder part for base is
deciding *what to ask next*, not resolving the hop. But the effect here is that **planning tokens
are the worst class, not that execution tokens are exceptional** — `execute − all` is +0.001 with
an interval spanning 0, against GSM8K's **+0.133 [+0.096, +0.173]** elevation of computed results
over the chain at large. So the replication is of the *sign*, not the magnitude or the shape.

The lens columns say why the shape differs, and this is the cleaner structural finding. GSM8K's
computed digits were rank 18 at L20 and crystallized to 0 only by L24 — the signature of a result
being *computed* across the upper stack. **No multihop class shows that.** Every role starts within
a few ranks of 0 at L20 and is resolved by L24; `final_answer` is rank 0 at every layer including
L20, the pure-copy signature, and its 0.933 is the highest class by +0.207 over execution. Under
open-book framing the hop answer is present verbatim in the prompt, so multihop "execution" is
retrieval-under-a-pointer rather than computation — there is nothing here that is late-computed the
way an arithmetic result is. The procedure thesis survives, but the per-step work it leaves to base
is of a different kind on this task.

The hop-index split is the one place where the naive prediction inverts, and it should not be
over-read. Composition-deficit reasoning predicts hop ≥ 2 (which must consume hop 1's answer) to be
*harder*; measured, it is **easier** by +0.130 [+0.106, +0.154]. Teacher-forcing is the reason: it
hands base every earlier hop for free, which is precisely what base cannot produce on its own, and
by hop ≥ 2 the format, the entities in play and the prior answer are all fixed in context, while
hop 1 is the least constrained token in the chain. So this contrast does **not** test composition —
under teacher forcing it cannot. What it does show is that supplying the trajectory converts the
nominally hard part into the easy part, which is what a trajectory-control deficit predicts and a
per-step-composition deficit does not.

**Verdict on this axis: H_plan, weakly.** Base's multihop deficit is not per-step hop resolution;
plan tokens are its worst class and given-context makes later hops easier. But "weakly" is load-
bearing: the plan/execute gap is 5.5 points where GSM8K's compute elevation was 13.3, and the
sub-question class is the one most deflated by *surface-form entropy* — a sub-question is free-form
natural language with many acceptable paraphrases, so low TF-agreement there partly measures
wording choice rather than planning failure. That artifact pushes in the same direction as H_plan
and cannot be separated from it with this design.

## Verdict

| axis | GSM8K | multihop | replicates? |
|---|---|---|---|
| oracle recovers | 0.75 @L20 (all-layers ≈1) | **+0.760 @L20** (all-layers +1.000) | **YES** |
| pointwise ladder | mid-stack ≈0; **late-stack leak +0.12–0.13 @L24–L31** (P5, n=200, intervals exclude 0); MLP pending; DAgger/DAS 0 | MLP +0.00, DAgger +0.00 — ridge **same shape**, larger: +0.21–0.26 @L20, peaking +0.45 @L24 (α=1.0-resonant) | **SHAPE YES, amplitude task-dependent (~3.75× @L24, disjoint CIs)** |
| temporal density | sharp knee; planning-heavy | knee at k=2 (0.05); reasoning_only = oracle @100% | **YES** |
| plan vs execute (P4) | execute (computed) 0.968 ≫ all 0.835 (**+0.133 [+0.096,+0.173]**); computed crystallizes L20→L24 | execute 0.725 > plan 0.671 (**+0.055 [+0.040,+0.069]**) but = all (+0.001, spans 0); nothing crystallizes | **SIGN ONLY** |

**H_general holds on two of three structural axes exactly** — the full-δ oracle concentrates at the same layer
with the same magnitude, and the trajectory state is temporally dense with the same
structural-complement signature. The ladder axis, now measured on both tasks at matched layers and n
(P5, 2026-08-10), lands as a **shape replication with task-dependent amplitude** rather than the
divergence earlier drafts claimed: both tasks are ≈0 through mid-stack and leak late, GSM8K peaking
at +0.13 [0.08, 0.20] where multihop reaches +0.45 [0.35, 0.56], intervals disjoint. The pointwise
wall exists on both (oracle beats the best map by ≥0.5 of budget on GSM8K, ≥0.3 on multihop; on-policy
and joint estimators still collapse to 0), but its height is a task property, not a constant. The
follow-up sweeps (P2b) sharpen the character of the multihop leak: α = 1.0-resonant (±25% mis-scale
forfeits it) and layer-humped over the oracle plateau; the matching GSM8K resonance test has not yet
run. Reading: the late-stack, output-adjacent (register-like) component of the delta transports on
both tasks; its share of the budget is what differs — hop composition over passages given in-context
is scaffold-limited, arithmetic is computation-limited. This sharpens the claim the same way as
before, minus the overstated null: "procedures do not install" should be "the *procedure core* does
not install; the transportable register share is task-dependent, and it is nowhere zero in the late
stack."

P4 adds a fourth axis and lands softer than the other three. The plan-before-execute ordering
replicates in sign — planning tokens are base's worst class, and handing base the trajectory makes
the later, nominally-composed hops *easier* — so "the deficit is trajectory control" survives the
task change. What does not replicate is E1b's positive elevation of execution above the chain at
large, and with it the layer-wise crystallization that made GSM8K's claim causal-looking: nothing
in the multihop chain is computed late in the stack, because open-book hop answers are copies from
context. The sharper way to state the thesis after two procedures: **the trajectory-control deficit
is general; the per-step work that base retains is task-specific, and only on arithmetic is it
computation rather than retrieval.**

Honest caveats: n = 2 procedures; the GSM8K/multihop maps were fit on different days (identical
driver, seed, protocol) and on corpora of different token counts (GSM8K CoTs ~175 tokens vs
multihop's ~60 at equal problem counts); the GSM8K recovery denominator is 0.650 (max_new=512
protocol) vs multihop's 0.630, so equal accuracy gains convert to slightly smaller GSM8K recoveries;
GSM8K's 4–6-hit cells at L16/L20 have not been transcript-checked against numeric-match coincidence;
the GSM8K α-resonance test at the leak layers has not run, so α=1.0 may not be GSM8K's peak;
`answer_only` gate vacuous on this contrast set (see P3 †). P4-specific: open-book framing means
hop answers are verbatim in the prompt, so "execute" here is retrieval, not computation;
teacher-forcing isolates execution from planning *by construction*, so P4 cannot show base could
plan the chain, and the hop-index split cannot test composition (it supplies the earlier hops);
sub-question TF-agreement is deflated by paraphrase entropy in a direction that flatters H_plan;
and P4 is a lens, not an intervention — unlike the oracle and density axes it carries no causal
claim.

## Repro

```bash
uv run python -m scripts.attribution.train_lora_multihop --config configs/attribution/multihop_llama2.yaml
uv run python -m scripts.attribution.multihop_gap --config configs/attribution/multihop_llama2.yaml --n-eval 500
# NOTE: --n-eval 500 everywhere below — the cached contrast indices index a 500-problem scan.
uv run python -m scripts.attribution.lockstep_patch_gsm8k --config configs/attribution/multihop_llama2.yaml --n-eval 500 --n-contrast 3 --validate
uv run python -m scripts.attribution.lockstep_patch_gsm8k --config configs/attribution/multihop_llama2.yaml --mode control --n-eval 500 --n-contrast 100
uv run python -m scripts.attribution.lockstep_patch_gsm8k --config configs/attribution/multihop_llama2.yaml --mode single --layers 0,4,8,12,16,20,24,28,31 --n-eval 500 --n-contrast 100
# P2b sweeps (rename steer_multihop.json between runs — the output name ignores --layers/--alphas):
uv run python -m scripts.attribution.steer_gsm8k --config configs/attribution/multihop_llama2.yaml --layers 20 --alphas 0.25,0.5,0.75,1.25,1.5,2.0 --base-acc 0.000 --lora-acc 0.630
uv run python -m scripts.attribution.steer_gsm8k --config configs/attribution/multihop_llama2.yaml --layers 8,12,16,24,28,31 --alphas 1.0 --base-acc 0.000 --lora-acc 0.630
uv run python -m scripts.attribution.nonlinear_delta_gsm8k --config configs/attribution/multihop_llama2.yaml --layer 20 --n-contrast 100 --out results/attribution/nonlinear_delta_multihop_L20_n100.json
uv run python -m scripts.attribution.temporal_oracle_gsm8k --config configs/attribution/multihop_llama2.yaml --layer 20 --n-contrast 100 --gates periodic:2 reasoning_only --out results/attribution/temporal_oracle_multihop_L20_n100.json
# P4 gold-token lens (multihop teacher-forces the gold chain; GSM8K generates + verifies the donor's own CoT):
uv run python -m scripts.attribution.gold_token_lens_gsm8k --config configs/attribution/multihop_llama2.yaml --task multihop --layer 20 --n-contrast 317
uv run python -m scripts.attribution.gold_token_lens_gsm8k --config configs/attribution/metamath_llama2_gsm8k.yaml --task gsm8k --layer 20 --n-contrast 20
```

All seeded (42); CPU tests: `tests/test_multihop_{data,prompts}.py`, `tests/test_attribution_tasks.py`,
`tests/test_chain_token_roles.py` (role construction + offset mapping), `tests/common/test_bootstrap_stats.py`
(the problem-clustered intervals). The GSM8K arm is also the refactor's regression check: it reproduces
the committed E1b table exactly (0.968 / 0.895 / 0.906 / 0.835, LoRA-TF 0.997, every lens-rank median).
