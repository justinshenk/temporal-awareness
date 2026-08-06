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
λ ∈ logspace(−1, 7, 17) (`sweep_multihop.json`): at L20, λ\* = 3.16e3, **held-out R²_te = +0.71**
(GSM8K L20: ≈0.61). As in GSM8K, the *open-loop geometry is good* — the ladder question is whether
it survives closed-loop decoding.

| rung | recovery | GSM8K analogue |
|---|--:|--:|
| ridge steer @L20 | **+0.26** scan / **+0.21** contrast at n=100 (the first-pass n=20 contrast read of +0.35 was small-n inflation; `nonlinear_delta_multihop_L20_n100.json`) | 0.00 — but **never probed at L20** ‡ |
| nonlinear MLP @L20 | **+0.00** (val cos +0.822 / R² +0.675 vs ridge +0.636 / +0.270 — better fit, zero closed-loop; +0.01 at n=100 contrast) | 0.00 (same paradox) |
| on-policy DAgger (joint all-layer) | **+0.00 / +0.00 / +0.00** (rounds 0–2, `dagger_refit_multihop.json`) | 0.00 all rounds |
| full-δ oracle @L20 | **+0.760** | 0.75 |

‡ **The GSM8K ridge column is weaker than it looks, and weaker than this report first claimed**
(corrected 2026-08-06). Every committed GSM8K ridge-steering measurement reads 0.00, but the
*per-layer* ones cover only L0/L1/L14/L16/L31 at α ≤ 1.0 (`steer_results_smoke.json`, n=12/50);
the rest are all-layer **joint** injections (`short_arithmetic.json`, `local_refit_gsm8k.json`,
`dagger_refit_gsm8k.json`). **L20 and L24 — the two layers where the multihop leak appears — were
never probed per-layer on GSM8K.** The full-run output the config names
(`results/attribution/steer_results.json`) was never committed and no longer exists on disk. An
earlier draft of this report cited "≈0.05" for this cell; no artifact backs that number (the 0.05s
in the GSM8K corpus belong to the PCA-band oracle and the lesion control, different experiments).
The honest statement is below.

**Reading.** Two of three rungs replicate exactly (MLP's better-geometry/zero-recovery paradox; DAgger
flat at 0 with on-policy data). The ridge rung *appears to diverge*: multihop's L20 edit is partially
linearly transportable (~¼ of the budget) where every GSM8K ridge measurement is 0.00. The wall
exists — the oracle still beats every pointwise map by ≥0.5 of the budget — but it looks *lower* for
multihop: hop composition over given passages plausibly has a larger register-like (linearly
mappable) component than arithmetic. **This comparison is not yet matched**: multihop's leak is
measured at L20/L24, GSM8K's null is measured at other layers and under joint injection, so a
same-layer GSM8K probe at L20/L24 (α grid) is required before the divergence can be asserted as a
fact rather than an inference.

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
| ridge steer | 0.00 | 0.00 | 0.00 | +0.26 | **+0.45** | +0.38 | +0.24 |
| oracle | +0.07 | +0.02 | +0.02 | +0.76 | +0.78 | +0.89† | +1.00† |

So on the multihop side the leak is a *curve*, not a point: the linearly transportable fraction of
the edit grows through the plateau to nearly half the budget at L24 before decaying toward the
readout. (No † on the steering rows — steering adds `W·a` to base's own state, so nothing is
degenerate about late layers here.) The matching GSM8K curve **does not exist** (see ‡): GSM8K
ridge steering was probed per-layer only at L0/L1/L14/L16/L31, all 0.00, so the claim that this
curve is multihop-specific rests on those five layers plus the joint-injection nulls, none of them
at L20 or L24.

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
| pointwise ladder | ≈0 (ridge/MLP/DAgger/DAS) | MLP +0.00, DAgger +0.00 — but **ridge +0.21–0.26 @L20, peaking +0.45 @L24** (α=1.0-resonant) | **PARTIAL** |
| temporal density | sharp knee; planning-heavy | knee at k=2 (0.05); reasoning_only = oracle @100% | **YES** |
| plan vs execute (P4) | execute (computed) 0.968 ≫ all 0.835 (**+0.133 [+0.096,+0.173]**); computed crystallizes L20→L24 | execute 0.725 > plan 0.671 (**+0.055 [+0.040,+0.069]**) but = all (+0.001, spans 0); nothing crystallizes | **SIGN ONLY** |

**H_general holds on two of three structural axes exactly** — the full-δ oracle concentrates at the same layer
with the same magnitude, and the trajectory state is temporally dense with the same
structural-complement signature. The one divergence is *quantitative, not qualitative*: the pointwise
wall exists (oracle beats every map by ≥0.3 of the budget at every layer; better-fitting nonlinear
and on-policy estimators still collapse to 0) but the *linear* rung leaks ~¼ of the budget at L20
and nearly half at L24, where every GSM8K ridge measurement is 0.00 — with the important caveat (‡)
that GSM8K was never probed per-layer at L20 or L24, so this axis is an *unmatched* comparison
pending that run. The follow-up sweeps (P2b) sharpen the
character of that leak: it is α = 1.0-resonant (±25% mis-scale forfeits it) and layer-humped over
the oracle plateau rather than tied to the oracle's onset layer. Reading: hop composition over
passages given in-context has a larger register-like (linearly transportable) component than
arithmetic — consistent with the thesis's own register-vs-procedure split rather than against it,
but it sharpens the claim: "procedures do not install" should be "the *procedure core* does not
install; its size is task-dependent."

P4 adds a fourth axis and lands softer than the other three. The plan-before-execute ordering
replicates in sign — planning tokens are base's worst class, and handing base the trajectory makes
the later, nominally-composed hops *easier* — so "the deficit is trajectory control" survives the
task change. What does not replicate is E1b's positive elevation of execution above the chain at
large, and with it the layer-wise crystallization that made GSM8K's claim causal-looking: nothing
in the multihop chain is computed late in the stack, because open-book hop answers are copies from
context. The sharper way to state the thesis after two procedures: **the trajectory-control deficit
is general; the per-step work that base retains is task-specific, and only on arithmetic is it
computation rather than retrieval.**

Honest caveats: n = 2 procedures; **the GSM8K per-layer steering reference does not cover L20/L24**
(‡ — the ridge-divergence axis is an unmatched comparison until GSM8K is probed at those layers,
which needs its maps refit since they are no longer cached locally);
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
