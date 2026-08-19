# E4 — per-head structure of the two mechanisms

**Verdict: the two mechanisms have different per-head signatures, and one of the paper's
sentences was wrong. Displacement drains every head (32/32 lose mass, all significant at
Bonferroni, nothing cancels). Competition does not change the evidence's *total* attention mass
but does measurably *reallocate* it across heads — mean per-head |Δ| is 0.00257 against a
sign-flip null of 0.00043 (6×, p ≤ 0.0005), with 19/32 heads individually significant at
Bonferroni. "Competition leaves the evidence's attention mass untouched" is true of the head
average and false of the heads.**

Run 2026-08-19 · `allenai/OLMo-2-1124-7B-Instruct` · L24 · seed 42 · 32 heads, no GQA
(`num_key_value_heads = 32`) · artifacts `results/context_fatigue/e1_heads/heads.csv`,
`e3_heads/heads.csv`, `head_structure.json` · drivers `run_distance_sweep.py --per-head`,
`run_competition_sweep.py --per-head`, analysis `scripts/context_fatigue/analyze_head_structure.py`.

## Why this needed new runs

`span_share()` reduces the captured `[n_heads, seq]` attention to a scalar with `.mean()` before
anything reaches disk, so no committed CSV holds per-head data — it had to be re-measured, not
re-analysed. Both re-runs reproduce their parent panel exactly: E1 at n = 192 per arm, fill 0.69,
0 overflow skips; E3 at 365 paired probes, 15 starved, 4 skipped, shared options 0.00 / 0.80 /
3.75. Attention-only, so no generation and no accuracy column.

## 1. Where the evidence's attention lives

| arm | effective heads (of 32) | top-4 fraction |
|---|---:|---:|
| `local` | 27.28 | 0.261 |
| `back_2` | 26.32 | 0.250 |
| `back_5` | 25.27 | 0.274 |
| `back_10` | 25.37 | 0.264 |
| `back_20` | 25.82 | 0.254 |

`effective heads` is the exponential of the entropy of the normalized per-head shares: 32 if the
mass were spread perfectly evenly, 1 if a single head held all of it. At 25–27 the evidence's
attention is **broadly distributed**, mildly concentrated (top-4 carries 26% against 12.5% for
uniform). It barely changes with distance, so **displacement drains without concentrating**.

This answers the blunt-instrument worry: the head-averaged share is not hiding a two-head circuit.

## 2. Displacement: a uniform drain

- **32 of 32 heads lose evidence mass** from `local` to `back_20`, every one significant at
  Bonferroni (α = 0.05/32).
- `corr(local share, drain) = +0.964` — the heads holding the most evidence mass lose the most in
  absolute terms.
- But not proportionally: the *fractional* drain spans **31.8% to 99.3%**, so one head keeps two
  thirds of its evidence attention while another loses essentially all of it.
- Redistribution ratio **1.00** — since every head moves the same way, mean |Δ| equals |mean Δ| by
  construction. There is nothing for a head average to hide.

## 3. Competition: reallocation at constant total

| contrast | mean Δ (head-averaged) | mean per-head \|Δ\| | sign-flip null | ratio | heads ≠ 0 (Bonferroni) |
|---|---:|---:|---:|---:|---:|
| `random` − `near_dup` | **−0.00027** | 0.00257 | 0.00043 | **9.63** | 25 (**19**) / 32 |
| `disjoint` − `near_dup` | +0.00152 | 0.00357 | 0.00048 | 2.35 | 26 (23) / 32 |
| `random` − `disjoint` | −0.00179 | 0.00222 | 0.00033 | 1.24 | 28 (25) / 32 |

The `mean Δ` column reproduces `E3_COMPETITION.md` exactly (−0.00027 / +0.00152 / −0.00179), so
this is the same measurement, unreduced.

`mean |Δ|` is positive under any noise, so it is quoted against a **paired sign-flip null**: flip
the sign of a probe's whole 32-head difference vector (a valid relabelling under exchangeability,
and one that preserves the cross-head structure the statistic is about), 2,000 permutations. All
three contrasts clear their null at p ≤ 0.0005, the minimum resolvable at that permutation count.

**The headline contrast is the extreme case.** `random` − `near_dup` has the *smallest* net change
in mass and the *largest* ratio of per-head movement to net movement: heads move nearly ten times
as much as their average, in opposing directions that cancel almost exactly.

## 4. What this changes in the paper

The dissociation survives and is sharper than before — the two mechanisms now differ in their
attention signature, not merely in whether one is present:

- **Displacement** removes evidence mass from every head at once.
- **Competition** holds the total fixed and changes *which heads* carry it.

What does **not** survive is the wording. The paper said competition costs accuracy "while leaving
the evidence's attention mass untouched" and "while leaving it intact". Untouched is false: 19 of
32 heads move at Bonferroni. The defensible claim is that the *total* is unchanged, which is what
the dose-response of §3.2 is denominated in — 6.29 accuracy per unit **head-averaged** share — and
so the arithmetic ruling out dilution is unaffected: the quantity that predicts accuracy did not
move, and 2% of the observed effect is still all that mass can account for.

But "competition has no attention signature" was never established and is now refuted. Whether the
reallocation is *causal* for the accuracy cost is untested: a per-head clamp that reproduces the
`near_dup` head pattern under `random` context is the experiment, and it is exactly the
pattern-matched clamp already named as open for E1d.

## Against §6 of the brief

- *Confirms:* the head-averaged null is not an artifact of averaging in the sense that mattered
  most — the quantity the dose-response is built on genuinely does not move.
- *Falsifies:* the stronger reading, that competition leaves attention alone. It does not.
- *New open question:* is the reallocation cause or consequence of the wrong answer?
