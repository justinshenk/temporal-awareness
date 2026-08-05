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
| ridge steer @L20 | **+0.26** scan / **+0.21** contrast at n=100 (the first-pass n=20 contrast read of +0.35 was small-n inflation; `nonlinear_delta_multihop_L20_n100.json`) | ≈0.05 |
| nonlinear MLP @L20 | **+0.00** (val cos +0.822 / R² +0.675 vs ridge +0.636 / +0.270 — better fit, zero closed-loop; +0.01 at n=100 contrast) | 0.00 (same paradox) |
| on-policy DAgger (joint all-layer) | **+0.00 / +0.00 / +0.00** (rounds 0–2, `dagger_refit_multihop.json`) | 0.00 all rounds |
| full-δ oracle @L20 | **+0.760** | 0.75 |

**Reading.** Two of three rungs replicate exactly (MLP's better-geometry/zero-recovery paradox; DAgger
flat at 0 with on-policy data). The ridge rung *diverges*: multihop's L20 edit is partially linearly
transportable (~¼ of the budget vs GSM8K's ≈0.05). The wall exists — the oracle still beats every
pointwise map by ≥0.5 of the budget — but it is *lower* for multihop: hop composition over given
passages appears to have a larger register-like (linearly mappable) component than arithmetic.

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

GSM8K's phase-3 sweep put ridge steering at ≈0 at every layer, so the divergence is a *curve*, not
a point: the linearly transportable fraction of the multihop edit grows through the plateau to
nearly half the budget at L24 before decaying toward the readout. (No † on the steering rows —
steering adds `W·a` to base's own state, so nothing is degenerate about late layers here; the L31
map genuinely moves a quarter of the budget where GSM8K's moved nothing.)

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

## Verdict

| axis | GSM8K | multihop | replicates? |
|---|---|---|---|
| oracle recovers | 0.75 @L20 (all-layers ≈1) | **+0.760 @L20** (all-layers +1.000) | **YES** |
| pointwise ladder | ≈0 (ridge/MLP/DAgger/DAS) | MLP +0.00, DAgger +0.00 — but **ridge +0.21–0.26 @L20, peaking +0.45 @L24** (α=1.0-resonant) | **PARTIAL** |
| temporal density | sharp knee; planning-heavy | knee at k=2 (0.05); reasoning_only = oracle @100% | **YES** |

**H_general holds on two of three axes exactly** — the full-δ oracle concentrates at the same layer
with the same magnitude, and the trajectory state is temporally dense with the same
structural-complement signature. The one divergence is *quantitative, not qualitative*: the pointwise
wall exists (oracle beats every map by ≥0.3 of the budget at every layer; better-fitting nonlinear
and on-policy estimators still collapse to 0) but the *linear* rung leaks ~¼ of the budget at L20
and nearly half at L24, vs ~5% anywhere on arithmetic. The follow-up sweeps (P2b) sharpen the
character of that leak: it is α = 1.0-resonant (±25% mis-scale forfeits it) and layer-humped over
the oracle plateau rather than tied to the oracle's onset layer. Reading: hop composition over
passages given in-context has a larger register-like (linearly transportable) component than
arithmetic — consistent with the thesis's own register-vs-procedure split rather than against it,
but it sharpens the claim: "procedures do not install" should be "the *procedure core* does not
install; its size is task-dependent."

Honest caveats: n = 2 procedures; the GSM8K per-layer steering reference is the committed phase-3
result (its maps are no longer cached locally, so the comparison is cross-run, not same-day);
`answer_only` gate vacuous on this contrast set (see P3 †).

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
```

All seeded (42); CPU tests: `tests/test_multihop_{data,prompts}.py`, `tests/test_attribution_tasks.py`.
