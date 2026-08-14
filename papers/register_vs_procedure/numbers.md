# `numbers.md` — every figure in the paper, mapped to the artifact it came from

**Rule (from the design spec, after `600b5f7` / `c3e2c62` / `089534e` found four unsourced numbers,
one of them in the abstract): no number enters the tex without a row here.** A cell whose artifact
cannot be opened reads "not measured", not a number. Before asserting "X diverges from Y", open Y's
artifact and confirm it was measured at the same layer, α and injection mode.

Paths are relative to the repo root. `results/` is gitignored, so "artifact" means the file on the
GPU box; the reports that quote them are committed.

---

## Figures

| # | figure | file | source artifact(s) | status |
|---|---|---|---|---|
| F1 | temporal-density knee, GSM8K + MuSiQue overlaid | `figures/f1_temporal_density.png` | `temporal_oracle_L20.json`, `temporal_oracle_multihop_L20.json` | **built 2026-08-13** |
| F2 | oracle layer sweep, L20 peak | — | `lockstep_multihop_single.json` (n=100) — **GSM8K arm has no artifact, see below** | **blocked** |
| F3 | null ladder vs oracle, with S1 intervals | — | `short_arithmetic.json`, `local_refit_gsm8k.json`, `dagger_refit_gsm8k.json`, `das_subspace_L20.json` | pending |
| F4 | variance-band cliff | — | `lockstep_pca_band_L20.json` | pending |
| F5 | gold-token lens by role | — | `gold_token_lens_L20.json`, `gold_token_lens_multihop_L20.json` | pending |
| F6 | refusal Pareto frontier | — | `refusal_frontier.json` | pending |
| F7 | α-resonance / layer hump | — | `steer_multihop_alpha_L20.json`, `steer_multihop_layers.json`, `steer_results_layers.json` | pending |

### F1 — notes that must survive into the caption

- Both series are **n=20 contrast problems**, matched. MuSiQue also has an n=100 re-run
  (`temporal_oracle_multihop_L20_n100.json`: periodic_2 0.060, reasoning_only 0.760) which confirms
  the n=20 reads; the figure uses n=20 for both so the comparison is like-for-like.
- Error bars are percentile-bootstrap 95% intervals over **problems** (`src/common/bootstrap_stats`),
  10,000 draws, seed 42. Problems are the independent unit.
- **The segment between frac 0.5 and frac 1.0 is an interpolation across a region with no
  measurements, and the caption must say so.** `periodic(k)` takes integer k, so the reachable
  fractions are 1, 1/2, 1/3, 1/4 … — there is no gate between k=1 and k=2. The structural gates
  (planning/reasoning only, at 94% and 100%) are the only evidence inside that span. The claim the
  figure supports is "sparse gating recovers nothing", not a measured curve shape near the knee.
- MuSiQue's `answer_only` sits at frac **0.000** and is **vacuous, not a null**: the unpatched base
  never emits "The answer is:", so the gate never fires. Do not read it as evidence.

---

## Numbers quoted in prose

| claim | value | artifact | verified |
|---|---|---|---|
| GSM8K oracle @L20 | 0.75 | `temporal_oracle_L20.json` (periodic_1) | 2026-08-13 |
| MuSiQue oracle @L20 | 0.75 / +0.76 sweep | `temporal_oracle_multihop_L20.json`, `lockstep_multihop_single.json` | 2026-08-13 |
| GSM8K ridge steer @L20 | +0.03 [0.01, 0.08], n=200 | `steer_results_layers.json` | P5, 2026-08-10 |
| GSM8K ridge steer @L24 | +0.12 [0.07, 0.19], n=200 | `steer_results_layers.json` | P5, 2026-08-10 |
| MuSiQue ridge steer @L24 | +0.45 [0.35, 0.56] | `steer_multihop_layers.json` | P2b |
| GSM8K L20 held-out R²_te | 0.610 @ λ*=3.16e3 | `sweep.json` | P5, 2026-08-10 |
| MuSiQue L20 held-out R²_te | 0.714 | `sweep_multihop.json` | P2 |
| GSM8K MLP rung @L24, n=100 | 0.00 [0, 0.04] vs ridge 0.10 [0.05, 0.18] | `nonlinear_delta_gsm8k_L24_n100.json` | P5b |
| cross-task transplant @L28 | 0.13 (= native exactly) | `steer_transplant_multihop_maps_on_gsm8k.json` | P5b |
| ARC-Challenge chance / majority floor | 0.25 / **0.288** | `data/commonsense/ARC-Challenge_test.json`, n=500 scan | 2026-08-13 |

### S2 — the register battery (commonsense / ARC-Challenge, Llama-2-7b + r32 LoRA donor)

| claim | value | artifact | verified |
|---|---|---|---|
| gap gate: base / donor on the n=500 scan | 0.000 / 0.676, **338** contrast problems | `commonsense_contrast_set.json` | 2026-08-13 |
| commonsense oracle @L16 | 0.830 | `lockstep_commonsense_single.json` (n-contrast 100) | 2026-08-13 |
| commonsense oracle @L20 | **0.990** | `lockstep_commonsense_single.json` (n-contrast 100) | 2026-08-13 |
| `random_matched` floor @L20 | **0.000**, byte-identical to base | `lockstep_commonsense_single_random_matched.json` | 2026-08-13 |
| fixed vector, per-problem, additive hook @L20 | 0.000 | `per_problem_vector_commonsense_L20.json` | 2026-08-13 |
| fixed vector, per-problem, **through the lockstep path** @L20, n=100 | **0.000** | `lockstep_commonsense_single_fixed_vector_per_problem.json` | S2d, 2026-08-14 |
| fixed vector, pooled over 100 disjoint problems @L20 | 0.000 at α ∈ {0.5, 1, 1.5, 2} | `global_register_vector_commonsense_L20.json` | 2026-08-13 |
| commonsense L20 held-out R²_te (**CoT window — see VOID below**) | 0.8934 @ λ*=1.00e2 | `sweep_commonsense.json` (superseded) | S2c, 2026-08-14 |

#### S2c, all-positions fit (`--fit-positions all`) — the arm that supersedes the void one

Fit window now matches the application window (`LinearPrimalSteerHook` steers every position).
8,046 train / 2,393 held-out tokens over 200/60 problems, `accumulators_commonsense_allpos`,
maps in `maps_commonsense_allpos`, sweep in `sweep_commonsense_allpos.json`. The CoT-window arm is
intact and separate in `sweep_commonsense.json` (`r2_te*`=0.8934, `r2_const_te`=null — that tree
predates first moments), so the two are directly comparable.

| layer | R²_te | `r2_const_te` | R²_te_centred |
|---|--:|--:|--:|
| 12 | 0.9785 | 0.0345 | 0.9777 |
| 16 | 0.9489 | 0.0599 | 0.9457 |
| **20** | **0.9293** | **0.1064** | **0.9209** |
| 24 | 0.9176 | 0.1431 | 0.9038 |
| 28 | 0.9136 | 0.1771 | 0.8950 |
| 31 | 0.9311 | 0.1216 | 0.9215 |

**The constant baseline is only 0.11 at L20**, so over the full sequence commonsense δ is *not*
constant-dominated and essentially all of the map's fit is genuinely input-conditional. This is
consistent with, not contrary to, `mean_delta`'s 0.820: over the ~6 **generated** positions the
required shift really is close to one direction (‖mean gen δ‖ 29 against per-token 42), while over
the ~97 prompt positions it is diverse and cancels. Any "the register is one direction" sentence
must name the window it holds on.

**Installation is real and narrow, and it carries format only.** Generation dump at L20
(`.run_logs/s2c_gens_allpos.log`), `‖Wa‖/‖a‖ = 0.224` at prompt positions:

| α | steered generation |
|---|---|
| 0.25, 0.5 | **byte-identical to unpatched base** |
| **0.75** | **`the correct answer is answer1`** — the donor's format, installed |
| donor | `the correct answer is answer3` (correct per problem) |

At α=0.75 the map emits the register format but answers **`answer1` on 31 of 40** — a degenerate
constant, not selection. Over n=40 (`.run_logs/s2c_answer_hist.log`): format compliance **0.850**,
accuracy **0.200**, and **all 8 correct answers are problems whose gold was `answer1`**, i.e. the
accuracy is exactly the base rate of the constant emitted, below both the 0.25 chance and 0.288
majority floors.

**`commonsense_format` sweep — the register arm of the ladder** (`steer_commonsense_allpos.json`,
n=500 scan, α=0.75, `maps_commonsense_allpos`, refs measured on the same 500: base **0.004** /
donor **1.000**):

| layer | 8 | 12 | 16 | **20** | 24 | 28 | 31 |
|---|--:|--:|--:|--:|--:|--:|--:|
| format compliance | 0.010 | 0.016 | 0.140 | **0.972** | 0.998 | 1.000 | 1.000 |

Against the same instrument on the procedures (GSM8K 0.03 @L20 / 0.12 @L24; MuSiQue 0.26 / 0.45),
this is the two-sided contrast measured on one axis: **a register transports through a fitted
pointwise map at a single layer; a procedure does not.** L28/L31 are the least informative cells —
a map that close to readout shapes the logits almost directly — but they are *not* degenerate in the
way the oracle sweep's L28/L31 were, since this hook is additive (`h + αWh`) rather than an
overwrite of the layer output.

#### The contrast set is ~two-thirds format, not capability (2026-08-14)

`.run_logs/s2_format_rescue.log`, n=100 contrast problems. Tests the brief's judgment call (c),
which asserted this and never measured it:

| arm | accuracy | format compliance |
|---|--:|--:|
| base, zero-shot (contrast protocol's prompt) | 0.000 | 0/100 |
| **base, 4-shot (format by demonstration)** | **0.630** | 100/100 |
| **base, 4-shot with WRONG exemplar labels** | **0.630** | 100/100 |
| donor LoRA | 1.000 | 100/100 |

The scrambled-label arm (`.run_logs/s2_format_rescue2.log`) is the control that licenses the
reading: 4-shot demonstrates the task as well as the format, but relabelling the exemplars wrongly
leaves accuracy at **exactly 0.630** with an essentially unchanged prediction spread. The
demonstrations supply **format only** — the standard random-label ICL finding, used here as a
control, not claimed as a result.

Base scores **0.630 on problems defined as base-failures**, predictions spread 23/16/28/33 across
the four options. The zero-shot prompt already *states* the format and base still complies 0/100 —
a base model follows a demonstration, not an instruction.

**So the commonsense oracle 0.990, the L16 0.830 onset and `mean_delta`'s 0.820 substantially
measure format installation, not capability transfer.** They remain valid as register measurements;
they may not be described as recovering a capability base lacks. Four-way comparison:

| intervention | format | selection | accuracy |
|---|---|---|--:|
| fixed vector (CAA-style) | ✗ | — | 0.000 |
| ridge map @L20, α=0.75 | ✓ 0.85 | ✗ constant | 0.200 |
| 4-shot prompt | ✓ 1.00 | ✓ base's own | 0.630 |
| donor LoRA | ✓ 1.00 | ✓ donor's | 1.000 |

The map **destroys selection base already had**; a prompt installing the same format preserves it.

#### The destruction is magnitude-controlled — no α installs format *and* selects

`.run_logs/s2_alpha_selection{,2}.log`, L20, n=60 contrast. Magnitude-matching α = 1.24, from
‖δ_donor‖ = 45.60 against ‖W·a‖ = 36.88 at the generation site:

| α | accuracy | format | `answer1` |
|---|--:|--:|--:|
| 0.75 | 0.267 | 0.850 | 45/60 |
| 0.90 | 0.283 | 0.850 | 45/60 |
| 1.00 | 0.233 | 0.767 | 41/60 |
| **1.24 (magnitude-matched)** | **0.033** | **0.083** | 4/60 |
| 1.50 | 0.000 | 0.000 | 0/60 |

Scaling the map to the donor's own magnitude loses the **format** too, so the lost selection is not
an under-scaling artifact. The map installs the register only where deliberately undersized
(60–80% of the donor's norm). At α=0.75 and 0.90 the accuracy (0.267 / 0.283) again equals the base
rate of `answer1` in the sample (16/60 = 0.267).

Geometry at the same site (`.run_logs/s2_geometry.log`, n=30): cos(map, donor) **+0.788** (sd 0.013),
cos(few-shot, donor) +0.380, cos(few-shot, map) +0.332; norms ‖a_zero‖ 48.68, ‖δ_few‖ 53.50,
‖δ_donor‖ 44.75, ‖δ_map‖ 27.92. **Not quotable as a "format direction" claim yet** — the few-shot
prompt is 324 tokens longer, so `δ_few` includes a generic long-context effect that a length-matched
no-format preamble has not yet been subtracted.

**Status of PUSHBACK item 5 (is 0.820 an early-step artifact?) — PARTIALLY answered, still not
clean.** Freezing a per-problem whole-trajectory vector and pushing it down the *lockstep* path
reads **0.000** against `mean_delta`'s 0.820. But the run's own diagnostic shows the frozen vector
sits at cosine **0.544** (min 0.304, max 0.803, over 3,023 decode steps) to the live running mean
`mean_delta` injects — so vector *direction* and the *loop* both changed, and the contrast is not
single-variable. The deployable claim is solid (a vector estimated once, no donor at inference,
installs nothing — now shown three independent ways). The narrow question needs the pushback's own
construction: record the vector `mean_delta` has converged to at its **final** step, then re-inject
that constant from step 1.

Incidentally, cosine as low as 0.304 between successive running means means the required shift
**rotates substantially within a 7-token generation** — measured evidence against reading this
register as one direction.

**`mean_delta` @L20 = 0.820 (`lockstep_commonsense_single_mean_delta.json`) is CONTESTED — do not
quote it, and do not quote any sentence resting on it** ("collapsing across positions is nearly
free", "83% of the oracle survives having no temporal structure"). It is not a fixed vector:
`lockstep_generate` re-runs `capture_residuals` every decode step, so the statistic is recomputed
from a live donor forward each step, and the early steps are near-oracle **by construction** —
at step 1 there are no generated rows at all, so `generated_rows` falls back to the whole sequence;
at step 2 the "mean" **is** the true δ of the first generated token; at step 3 it is the mean of two
true δs. Those early tokens are the trigger phrase, i.e. the span that decides the score. The
discriminator is `--control fixed_vector` (a frozen whole-trajectory mean through the identical
delivery path); until it lands, 0.820 has no interpretation.

**The `random_matched` 0.000 does not floor a constant-vector claim either.** It draws an
*independent* direction per position, where `mean_delta` injects one *coherent* direction at every
position; independent draws partially cancel downstream where a coherent shift accumulates. The
matched-by-construction floor is `--control random_constant` (one random direction at
‖mean generated δ‖), not yet run. The per-token δ norms this rests on (prompt ~28–30, generated
~41–43, base residual ~90 at L20) currently exist **only in `.run_logs/s2_delta_norms.log`** — no
JSON, so they are an uncitable cell until one exists.

**On the commonsense R²_te = 0.8934.** It is *not* like-for-like with GSM8K's 0.610: the fit used
**1,200** CoT tokens against GSM8K's **34,893**, and the held-out target is a near-constant 6-token
phrase (`"the correct answer is answerX"`) rather than a diverse arithmetic chain, so the regression
problem is much easier. What it does establish, and what it was run for, is that the commonsense map
is **not data-starved**: a steering null from it is attributable to transport, not to an
underdetermined fit.

### F2 — a fifth unsourced comparison, found 2026-08-13

`2026-06-16-multihop-generality.md:50-66` prints a GSM8K oracle **layer sweep** column — L16 0.20,
L20 0.75, L24 0.75, L28 0.95, L31 0.95 — and rests the claim "**L\* = 20, the same layer**, selected
by the same earliest-plateau rule" on it. That column has **no artifact in the tree**:

- `lockstep_single.json` is `task=gsm8k` with **`n_contrast=1` and a single layer, 20** — a smoke run.
- No file anywhere references `lockstep_single`, and no other report contains the L16=0.20 figure.
- The likely cause is the **same output-name hazard that cost the α grid its JSON**:
  `lockstep_patch_gsm8k` names GSM8K output `lockstep_{mode}.json` regardless of `--layers` or
  `--n-contrast`, so a later 1-problem smoke overwrote the real sweep in place.

What is safe and what is not:

- **Safe** — "the GSM8K L20 oracle recovers 0.75 (n=20)". Doubly sourced: `temporal_oracle_L20.json`
  `periodic_1` = 0.750 and `downstream_lesion_L20.json` level 0 `recovery_patch` = 0.75.
- **Not safe** — "L20 is GSM8K's earliest plateau" / "the same layer" / any GSM8K value at
  L16/L24/L28/L31 in an oracle sweep. Those need a re-run of
  `lockstep_patch_gsm8k --mode single --layers 0,4,…,31` at real n, written to a **non-colliding
  filename**. Until then F2 is a MuSiQue-only figure and the cross-task "same layer" sentence must
  be softened to the L20 point comparison, which is sourced.

### VOID — the S2c commonsense ridge arm, first attempt (2026-08-14)

`steer_commonsense.json` reads **0.000 at every layer** (8/12/16/20/24/28/31, α=1.0, n=500, refs
base 0.000 / donor 0.676) and **must not be reported as a transport null.** The decoded generations
say it is a *destruction* result:

| condition @L20 | generation |
|---|---|
| unpatched base | `\nAnswer1: Planetary density will decrease.\n\nAnswer2: …` |
| steered α=0.5 | **byte-identical to base** — the map is a no-op |
| steered α=1.0 | `\n\end​​​​​​…` — degenerate zero-width-space repetition, every problem |
| donor | `the correct answer is answer3` |

**Cause: the fit window did not match the application window.** `collect_cot_residuals` used
`cot_token_slice`, which keeps *generated positions only* — 6 tokens per problem here — while
`LinearPrimalSteerHook` applies the map at **every** position. So ~94% of the positions the map is
applied to were off its fit distribution, and it extrapolates to roughly double the correct
magnitude there: measured at L20, `mean‖Wa‖ = 53.6` against `mean‖a‖ = 97.2` (ratio **0.551**),
where the true δ has per-token norm 27.6–43 (ratio ~0.3–0.45). At α=0.5 the shift scales into the
inert band and does nothing; at α=1.0 it destroys. There is no α in between that installs anything.

This was invisible in the accuracy alone — on a base-fails/donor-solves contrast set, "no-op",
"destroyed" and "complied but chose wrong" all read 0.000. It is the same failure mode as the
2026-08-13 no-op floors, in the opposite direction.

**Do not quote `sweep_commonsense.json` or `steer_commonsense.json`.** The R²_te = 0.8934 in the
former is a real fit statistic *on the CoT window*, but it describes a map that cannot be deployed
at the positions it is deployed at. Superseded by the `_allpos` arm (`--fit-positions all`), which
fits on all ~103 positions (~20,600 tokens rather than 1,200) so that the fit and application
windows agree. On GSM8K the same mismatch is mild — the chain is ~250 of ~400 positions — which is
why it never surfaced before.

### VOID — measured, but measuring a bug (2026-08-13)

The S2 floor controls (`mean_delta`, `shuffle_positions`) at L20, and the α ∈ {0.1, 0.25, 0.5} grid
built on `mean_delta`, all read 0.000 — because the controls are **no-ops**. `mean_delta` averaged δ
over *all* positions, and with ~150 prompt tokens against ~7–32 generated ones the mean is dominated
by near-zero prompt shifts; decoded generations are character-for-character identical to unpatched
base. Since the contrast set is defined as base-fails/donor-solves, base scores 0.000 on it by
construction, so a no-op scores 0.000 automatically.

**Nothing from these runs may enter the paper**, including the appealing sentence "the floor is
0.000, not chance". The oracle numbers they were meant to defend are unaffected, but the defence
itself has to be re-earned with controls restricted to generated positions plus a matched-norm
random direction.

### Known gaps — do not quote until closed

- **α grid @L24/28**: 9 of 12 cells exist **only in `.run_logs/p5b_alpha_grid.log`**; the run died
  before writing JSON. Log reads suggest the L24 peak is at **α=0.75 (0.095)**, above the α=1.0
  used in the headline sweep — so F7 and any α claim need the re-run first.
- **S2 register battery**: in flight 2026-08-13. §9 and §10 have no numbers yet.
- The GSM8K and MuSiQue **Gram accumulators were deleted 2026-08-13** to clear a disk quota that had
  already truncated a trained adapter. The derived `maps/` and `maps_multihop/` are intact, so every
  steering number above remains reproducible; refitting the ridge sweep from scratch would require
  re-running `collect_cot_residuals` (~24 min GPU for GSM8K).
