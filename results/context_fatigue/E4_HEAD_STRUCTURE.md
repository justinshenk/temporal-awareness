# E4 — per-head and per-layer structure of the two mechanisms

**Verdict: the dissociation between displacement and competition survives, but two claims made
from a single layer were wrong. Evidence-tracking heads exist (255 of 1,024 attend the evidence
more than its size warrants; L16H17 puts 0.626 of its attention on a span that is 0.102 of the
context) — layer 24 simply has none. And competition does drain the evidence's attention mass, by
0.0186 at layer 17, against 0.0003 at layer 24 where the paper measured. Averaged over all 32
layers displacement removes 0.0455 of evidence mass and competition removes 0.0022, while costing
0.188 and 0.085 accuracy respectively — so competition still costs far more accuracy per unit of
mass than displacement, but the separation is nothing like the figure a single layer implied.**

Run 2026-08-19 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · **all 32 layers × 32 heads = 1,024
heads** · no GQA (`num_key_value_heads = 32`) · artifacts `e1_heads_all/heads.csv`,
`e3_heads_all/heads.csv`, `head_structure.json` · drivers `run_distance_sweep.py --per-head`,
`run_competition_sweep.py --per-head`, analysis `analyze_head_structure.py`.

## Why this needed new runs

`span_share()` reduces the captured `[n_heads, seq]` attention to a scalar with `.mean()` before
anything reaches disk, so no committed CSV holds per-head data. Both re-runs reproduce their parent
panel exactly — E1 at n = 192 per arm, fill 0.69, 0 overflow skips; E3 at 365 paired probes, 15
starved, 4 skipped, shared options 0.00 / 0.80 / 3.75 — and the head-averaged contrasts come back
identical to `E3_COMPETITION.md`, so this is the same measurement unreduced.

**A first pass captured layer 24 only, and every conclusion drawn from it about head identity was
wrong.** Layer 24 is recorded below alongside the rest rather than removed, because the paper's
existing attention claims are indexed there.

## 1. Which heads read the evidence

`enrichment` is a head's evidence share divided by the evidence span's share of context tokens
(0.102). At 1.0 a head attends to the vignette exactly in proportion to its size; below 1.0 it
under-weights it. Without this control a long span looks like a head specialty.

| head | evidence share | question share | enrichment |
|---|---:|---:|---:|
| L16H17 | 0.6257 | 0.1333 | 6.13 |
| L0H14 | 0.5487 | 0.0717 | 5.37 |
| L17H6 | 0.4864 | 0.1937 | 4.76 |
| L3H17 | 0.4484 | 0.2469 | 4.39 |
| L5H15 | 0.4084 | 0.1698 | 4.00 |
| L16H25 | 0.3884 | 0.3867 | 3.80 |

**255 of 1,024 heads have enrichment above 1.** Layer 16 holds three of the top ten; the rest
cluster in layers 0–5. At layer 24, **zero of 32** heads clear 1.0 — the best is L24H19 at 0.91.

Mean evidence share by layer puts layer 24 near the bottom of the model: 0.0408, against 0.183 at
layer 3, 0.155 at layer 0 and 0.143 at layer 16. The paper's attention claims are anchored to a
layer where the evidence receives unusually little attention.

## 2. Displacement: a uniform drain

At layer 24, **32 of 32 heads lose evidence mass** from `local` to `back_20`, every one
significant at Bonferroni (α = 0.05/32), so nothing is hidden by averaging.

Ranked by mass actually lost:

| head | local | back_20 | mass lost |
|---|---:|---:|---:|
| L24H19 | 0.0925 | 0.0178 | 0.0747 |
| L24H10 | 0.0851 | 0.0230 | 0.0621 |
| L24H24 | 0.0834 | 0.0224 | 0.0610 |
| L24H3 | 0.0712 | 0.0154 | 0.0558 |
| L24H28 | 0.0198 | 0.0001 | 0.0197 |

An earlier version of this report ranked these by *fractional* drain and led with L24H28 at
"99.3%". It goes 0.0198 → 0.0001: the largest percentage and nearly the smallest movement in the
set, while L24H19 loses forty times as much mass and ranked seventh. **Ratios on small bases do
not order effects.** The fractional drain averages 0.689 with sd 0.147 and is uncorrelated with
how much mass a head held (r = +0.08) — displacement scales heads down by a similar fraction
regardless of what they carried.

**One uniform odds-scale reproduces 57.6% of the per-head pattern** (best single bias −1.376
nats; per-head implied bias sd 0.796, range −4.95 to −0.39). This bears directly on E1d: the paper
excuses its partial necessity by saying a uniform clamp "cannot reconstruct which heads should
carry the restored mass", and the clamp's shape is in fact more than half right. The excuse is
weakened, not eliminated.

## 3. Competition: layer-dependent, and not absent

| layer | displacement drain | competition drain | mean per-head \|Δ\| |
|---|---:|---:|---:|
| 16 | 0.0100 | −0.0110 | 0.0136 |
| **17** | 0.0654 | **−0.0186** | 0.0223 |
| 18 | 0.0324 | −0.0093 | 0.0106 |
| **24** | 0.0284 | **−0.0003** | 0.0026 |

At layer 24 competition moves the evidence's mass by 0.0003 — indistinguishable from zero. At
layer 17 it moves it by 0.0186, sixty times more, and in the same direction displacement moves it.
Averaged over all 32 layers: displacement 0.0455, competition 0.0022.

The two drain profiles are **negatively correlated across layers** (r = −0.32): displacement bites
hardest at layers 0–3 and 16, competition at 17–18. That is independent support for two mechanisms
which the single-layer measurement could not have provided.

**The "50× larger share change needed" figure in the paper is a layer-24 artifact** and should not
survive. It also rested on an invalid conversion: the 6.29-accuracy-per-unit-share slope was
measured with a clamp that biases *all* layers at once, indexed by the layer-24 reading, so it
cannot convert a layer-24-only delta from an experiment whose layers move independently. The
mass-versus-accuracy comparison above makes the same point without the slope.

## 4. Per-head redistribution at layer 24

| contrast | mean Δ | mean per-head \|Δ\| | sign-flip null | heads ≠ 0 (Bonferroni) |
|---|---:|---:|---:|---:|
| `random` − `near_dup` | −0.00027 | 0.00257 | 0.00043 | 25 (19) / 32 |
| `disjoint` − `near_dup` | +0.00152 | 0.00357 | 0.00048 | 26 (23) / 32 |
| `random` − `disjoint` | −0.00179 | 0.00222 | 0.00033 | 28 (25) / 32 |

`mean |Δ|` is positive under any noise, so it is quoted against a **paired sign-flip null**:
flipping the sign of a probe's whole 32-head vector is valid under exchangeability and preserves
the cross-head structure the statistic is about (2,000 permutations, all p ≤ 0.0005).

At layer 24 the heads move 0.0026 while the net moves 0.0003 — against a local evidence share of
0.041, so roughly 6% of the evidence's mass shifts between heads. Real and above the noise floor,
but small in absolute terms, and an earlier version of this report overstated it by leading with
the ratio.

## What this changes in the paper

- "Competition leaves the evidence's attention mass untouched" is false at layers 14–19 and true
  at 24. The claim must be stated with its layer, or restated on the all-layer average.
- "50× larger share change" comes out.
- The dissociation stands: 0.0455 vs 0.0022 of mass for 0.188 vs 0.085 of accuracy.
- Whether the layer-17 reallocation is cause or consequence of the wrong answer is untested. The
  per-head clamp would settle it, and it is the same instrument E1d needs.
