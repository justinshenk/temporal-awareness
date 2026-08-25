# E1d′ — pattern-matched per-head restoration: the 0.28 was not the instrument

**Verdict: restoring `back_20`'s evidence mass to the `local` arm's per-layer, per-head
pattern recovers no more of the displacement penalty than the uniform clamp — 0.235
[−0.158, 0.588] vs 0.375 [0.167, 0.654] of the penalty, per-head − uniform = −0.021
[−0.083, +0.042] ns — despite the pattern being installed faithfully (mean per-head share
error 0.004 against target shares ≈ 0.077; installed bias SD 1.22 across heads, i.e. the
intervention was genuinely non-uniform). The conclusion's "test we would run next" is run,
and it eliminates head-uniformity as the reason mass restoration is partial: what
restoration cannot rebuild is not *which heads* carry the mass. The remaining candidates
are the within-span token-level pattern (one bias per head still spreads uniformly over the
span's tokens) and a genuinely second, non-mass positional channel.**

Run 2026-08-25 · `allenai/OLMo-2-1124-7B-Instruct` · driver `run_evidence_clamp.py
--clamp-arm back_20 --donor-arm local --reference-layer 0..31 --per-head-pattern` ·
n = 192 paired items (6 sessions × 4 depths × 8 probes), forced-choice scoring · artifacts
`results/context_fatigue/e1d_head_pattern/` · brief `tasks/per_token_capture_brief.md`
Stage 2. Instrument: `PerHeadSpanAttentionClamp` (mask expanded to `[b, H, q, k]`, one
bias per query head per layer) + `solve_per_head_pattern` (closed-form per-head logit
identity, 3 refinement passes under the full all-layer clamp).

## Conditions (all four measured on the same 192 items)

| condition | evidence share | accuracy |
|---|---|---|
| `back_20` (natural) | 0.0301 | 0.359 |
| `back_20_clamped` (uniform, per-item local target) | 0.0759 | 0.422 |
| `back_20_headpattern` (per-layer×head local pattern) | 0.0774 | 0.401 |
| `local` (donor) | 0.0759 | 0.531 |

## Paired gaps (10,000 draws over items)

- displacement penalty (local − back_20): **+0.172 [+0.104, +0.245] SIG**
- uniform restoration − back_20: **+0.063 [+0.026, +0.104] SIG**
- per-head restoration − back_20: +0.042 [−0.021, +0.109] ns
- **per-head − uniform: −0.021 [−0.083, +0.042] ns**
- residual local − per-head: +0.130 [+0.057, +0.203] SIG
- recovered fraction, paired bootstrap: uniform **0.375 [0.167, 0.654]**, per-head
  **0.235 [−0.158, 0.588]**

The uniform arm's recovered fraction is consistent with the committed all-layer E1d
(0.28 [0.07, 0.61]), so the session anchors.

## Reading

The brief's decision grid was: recovered fraction ≫ 0.28 ⇒ the uniform clamp was the
limitation; ≈ 0.28 ⇒ evidence for a second positional channel. The outcome is the second
branch, with the stronger-than-planned instrument: the donor's head pattern was measured
per item (not a panel mean), installed at every layer, and verified in place to 0.004 mean
share error. Head identity is not the missing ingredient.

Two candidates remain for the unrecovered ~0.6–0.75 of the penalty:

1. **Within-span structure.** The clamp scales each head's odds uniformly across the
   evidence span's ~90 tokens, so the *distribution over tokens within the span* stays
   back_20's, only its total (per head) becomes local's. If retrieval reads specific
   token positions (the E3c′ result says content, not mass, does the work on the
   competitor side), a token-level pattern clamp — the stored-row instrument exists as of
   Stage 0 — would be the next refinement.
2. **A non-attention positional channel** (e.g. positional encoding of the evidence's
   location surviving in the residual stream regardless of what attention is paid to it),
   which no attention-side clamp can restore.

## Caveats

- The per-head arm's CI includes both 0 and the uniform arm's estimate; the claim is not
  "per-head is worse," only that it is not better, with the paired contrast's CI
  half-width ±0.06 at n=192.
- `solve_per_head_pattern` hit its targets after 3 refinement passes; the residual 0.004
  mean error is 5% of the mean target share and unbiased in sign.
