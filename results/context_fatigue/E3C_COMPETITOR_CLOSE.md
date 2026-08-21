# E3c — competition's cost is carried by *reading* the competitor instances

**Verdict: RESCUE — the brief's first outcome fired, and the paper's title claim must be
revised.** Closing every context occurrence of the probe's option names at generation time
recovers **59%** of the competition penalty (+0.055 [+0.006, +0.104], paired n=365), with a
size-matched random-closure control at exactly zero (−0.006 [−0.036, +0.025]) and a residual
gap to `random` that no longer excludes zero (+0.038 [−0.011, +0.088]). Competition IS
attention-mediated — **competitor-side, not evidence-side**. The E3 evidence-mass null stands
untouched; what falls is the inference from it that competition does not act through attention.

Run 2026-08-21 · `allenai/OLMo-2-1124-7B-Instruct` · eager attention · seed 42 · paired n=365
(15 starved + 4 overflow skips out of 384 — the identical panel construction as the committed
`e3_competition/`) · gold leaks 0 · artifacts `e3c_competitor_close/` · driver
`run_competition_sweep.py --close-arms` · brief `tasks/e3c_competitor_close_brief.md`.

## Why this run existed

The user raised the hole directly: E3's dissociation showed the *evidence's* attention mass
unchanged under competition, but the filler receives the same total mass in every arm, and in
`near_dup` that mass lands on instances of the probe's own answer candidates. At constant
evidence mass, the pool of attention over answer-relevant content gains misleading members.
E3's claim ruled out evidence starvation; it never tested the competitor-reading route.

## Arms and results

Four arms per probe, all generated eager, `near_dup`/`random` contexts identical to the
committed run (same seeds, same `select_by_option_overlap`):

| arm | intervention | accuracy | parse |
|---|---|---:|---:|
| `near_dup` | none | 0.4192 | 0.929 |
| `near_dup_comp_close` | scale-0 closure of every context occurrence of the probe's option names | **0.4740** | 0.945 |
| `near_dup_rand_close` | closure of size-matched random spans in the same region | 0.4137 | 0.899 |
| `random` | none | 0.5123 | 0.940 |

Competitor spans: 30.0 per probe, 127.9 tokens, union attention share (all-layer mean) at the
final position **0.0077** — under 1% of the distribution carries a 5.5-point effect when
removed.

Paired bootstrap (10,000 draws, `paired_accuracy_gap`):

| contrast | estimate [95% CI] | |
|---|---|---|
| rescue: `comp_close` − `near_dup` | **+0.0548 [+0.0055, +0.1041]** | sig |
| control: `rand_close` − `near_dup` | −0.0055 [−0.0356, +0.0247] | clean null |
| net: `comp_close` − `rand_close` | **+0.0603 [+0.0082, +0.1123]** | sig |
| competition gap: `random` − `near_dup` | +0.0932 [+0.0384, +0.1479] | replicates committed +0.085 |
| residual: `random` − `comp_close` | +0.0384 [−0.0110, +0.0877] | n.s. |

Recovered fraction 0.59; closure lands statistically indistinguishable from the `random`
stream, though the point estimate leaves ~40% of the gap possibly prefill-borne — the claim is
"substantially mediated", not "fully".

**Robustness.** Parsed-only (both arms parsed): rescue +0.0581 [+0.0031, +0.1131] (n=327)
survives; net vs control +0.0568 [+0.0000, +0.1136] (n=317) sits exactly at the boundary —
reported as such. Harness anchors: in-run `near_dup` 0.419 vs committed 0.427 and `random`
0.512 vs committed 0.512 (this run is eager, the committed run sdpa — no drift). None of the
brief's void conditions fired: the control is null, parse rates hold (0.90–0.95), the natural
arm matches, and `random`-arm contexts contain fewer shared-option mentions by construction.

## What this changes

1. **The paper's title claim.** "Only displacement acts through attention mass" is false as
   stated. The correct dissociation is *which* attention: displacement is mediated by the
   **evidence's** mass (starvation — E1c/E1f), competition by attention to the **competitors**
   (misdirection — this run), and the two are dissociated on the evidence-mass measurement that
   E3 did correctly.
2. **The mechanism count survives; the taxonomy sharpens.** Both accuracy mechanisms ride on
   attention, in opposite directions: too little on the evidence, any at all on impostors.
   Precedent remains the odd one out (below).
3. **The E6 contrast is now a designed dissociation.** The *same* scale-0 closure that restores
   nothing for format erosion (`e6_exemplar_close/`: privileged reading epiphenomenal, mode
   prefill-installed) rescues accuracy here. Competition is generation-time *reading*;
   precedent is prefill *installation*. One intervention, opposite outcomes, two mechanisms.
4. **Needle benchmarks** now decompose fully: burying a needle starves its evidence AND supplies
   readable impostors, and both routes are attentional — which is why "attention dilution"
   explanations of long-context failure keep almost-working: they name the right substrate and
   the wrong span.

## Open

- Whether the residual ~40% is prefill interference or instrument slack (the closure only
  covers verbatim option-name mentions; paraphrases and the vignettes' symptom overlap remain
  readable).
- Per-layer localization of the competitor-reading route (the committed all-layer profile put
  competition's drain at L17–18; whether closure's effect concentrates there is one capture
  sweep away).
