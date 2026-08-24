# E2a — causal attention-mass dose-response

**Verdict: the headroom claim is NOT supported — degradation begins at the ≈0.15 share that
accumulation reaches, with no margin below it. The "cliff" framing is retired: the levels below
0.15 turn out to be near-ablation of the query, not points on a dose-response (see *intervention
magnitudes*).**

Run 2026-08-18 · `allenai/OLMo-2-1124-7B-Instruct` · reference layer **L24** · seed 42 · n = 110
per level · artifacts `results/context_fatigue/e2a_mass_clamp/` · driver
`scripts/context_fatigue/run_mass_clamp.py`.

## What was measured

The current query's post-softmax attention share was **set**, not inferred: a constant bias on the
span's key columns scales its odds, and softmax renormalizes. Cold-start contexts (system prompt +
the case, no accumulation) have a natural share of **0.258** at L24. Each item was scored at its
natural share and at six clamped levels. Scoring is forced-choice over the option letters, so there
are no parse failures. Overflow guard skipped 0 of 110.

| level | achieved share | n | accuracy | vs natural (95% CI) |
|---|---|---|---|---|
| natural | 0.258 | 110 | **0.545** | — |
| 0.30 | 0.300 | 110 | 0.518 | +0.027 [−0.100, +0.164] |
| 0.20 | 0.200 | 110 | 0.509 | +0.036 [−0.091, +0.173] |
| 0.15 | 0.150 | 110 | 0.382 | **+0.164 [+0.036, +0.291]** |
| 0.10 | 0.100 | 110 | 0.164 | **+0.382 [+0.264, +0.500]** |
| 0.05 | 0.050 | 110 | 0.164 | **+0.382 [+0.264, +0.500]** |
| 0.02 | 0.020 | 110 | 0.164 | **+0.382 [+0.264, +0.500]** |

**Adjacent steps** (read alongside *intervention magnitudes* below):

| step | drop | 95% CI | significant |
|---|---|---|---|
| natural → 0.30 | +0.027 | [−0.100, +0.164] | no |
| 0.30 → 0.20 | +0.009 | [−0.118, +0.145] | no |
| 0.20 → 0.15 | +0.127 | [+0.000, +0.264] | borderline (lo = 0.000) |
| **0.15 → 0.10** | **+0.218** | **[+0.100, +0.327]** | **yes** |
| 0.10 → 0.05 | 0.000 | [−0.100, +0.100] | no |
| 0.05 → 0.02 | 0.000 | [−0.100, +0.100] | no |

## Reading

**The plateau is real but it stops at 0.20.** Natural (0.258), 0.30 and 0.20 are mutually
indistinguishable — mass can be removed down to a fifth of the distribution at no cost.

**0.15 is already on the downslope.** It costs 16.4 points against natural with a CI excluding
zero. This is the number that matters: 0.15 is where the paper says accumulation actually lands.
So accumulation does not stop "far short" of the floor — it stops **at the shoulder**.

**Below 0.15 the model does not degrade, it collapses.** At 0.05 and 0.02 it answers "A" for
**110 of 110** items; at 0.10, for 103 of 110. The 0.164 "accuracy" at those levels is just the
base rate of gold being "A" — the model has stopped reading the question, not merely got worse at
it. That is why the three deepest levels are identical to three decimal places. Those levels reach
their targets only at biases of −4.7 to −6.1 nats, i.e. by deleting the query from what the model
can attend to; they are excluded from interpretation below.

## Against §6

§6's *confirming* branch asked for "a plateau across levels at or above the ≈0.15 accumulation
reaches, then a cliff below it". The plateau exists but its lower edge is **0.20, not 0.15**.
§6's *falsifying* branch asked whether "the cliff sits at or above 0.15 → mass **is** near-binding,
and the 'headroom' language must go". The cliff *onset* is at 0.15 and the collapse completes by
0.10.

The honest verdict is the falsifying one, in its weaker form: **the shape the paper predicts is
there, and the margin it implies is not.** "Enough, not maximal" survives; "there is headroom"
does not. The paper should quote the measured plateau edge (0.20) and the 16.4-point cost at 0.15
rather than describing accumulation as stopping short of a floor.

## The intervention magnitudes — read the levels before reading the curve

The six levels are not six equal steps down. The bias needed to hit each target says what kind of
intervention it actually is:

| level | median scale | median bias | what it is |
|---|---|---|---|
| 0.30 | 1.4635 | +0.38 nats | a clamp **up** — for 82.7% of items (natural median 0.247) |
| 0.20 | 0.5525 | −0.59 nats | genuine mild down-clamp (up for 2.7%) |
| 0.15 | 0.1534 | −1.88 nats | genuine down-clamp |
| 0.10 | 0.0089 | −4.72 nats | **near-ablation** |
| 0.05 | 0.0060 | −5.11 nats | **near-ablation** |
| 0.02 | 0.0023 | −6.09 nats | **near-ablation** |

Scale falls **17-fold between targets 0.15 and 0.10** for a 0.05 change in share. The share
saturates: attention sinks hold enough mass that pushing the query span below ~0.13 requires
suppressing it almost entirely. So the "cliff" between 0.15 and 0.10 coincides exactly with the
transition from graded clamping to effective ablation, and **must not be read as the model's mass
requirement**. What the three deepest levels show is what happens when the query is removed from
what the model can attend to — the collapse to a constant "A" is the expected consequence of that,
not a discovery about dilution.

The defensible reading is therefore narrower than a dose-response with a cliff:

- Share can be cut from 0.258 to **0.20 at no measurable cost** (a genuine, plausible-magnitude
  intervention: median scale 0.55).
- At **0.15** — median scale 0.15, still a plausible intervention — accuracy falls **16.4 points**
  with a CI excluding zero.
- Below 0.15 the manipulation is off-manifold and those points are excluded from interpretation.

That leaves the headroom conclusion intact and the cliff framing retired: degradation begins at the
share accumulation reaches, and there is no comfortable margin below it.

## Limitations, stated plainly

1. **Clamping to 0.15 is not the same as accumulating to 0.15.** Here a *short* context has its
   query starved; under accumulation the model reaches 0.15 while also holding a great deal of
   other context, some of it usable. The dose-response isolates mass, which is what E2 was for,
   but the two conditions are not interchangeable and the comparison is suggestive rather than
   exact. E2b addresses the accumulated case directly.
2. **The floor is degenerate, and so is the step into it.** Below 0.15 accuracy is not a graded
   measure of anything — it is the base rate of a constant response, produced by an intervention
   that effectively removes the query. Neither the location nor the depth of the drop between 0.15
   and 0.10 should be quoted as a mass threshold; see *intervention magnitudes*.
3. **The 0.20 → 0.15 step is borderline** (CI lower bound exactly +0.000). The claim "0.15 is
   degraded" rests on the comparison against natural, which is comfortably significant, rather
   than on that single adjacent step.
4. **Share is not comparable across harnesses with different query lengths.** The E2b preflight
   measures the random-subject MMLU stream at 0.137 cold and **0.095 at deep fill** — below this
   experiment's 0.15 — while scoring 0.75, nowhere near collapse. A ~60-token MMLU question simply
   draws less share than a ~300-token DDXPlus case. The 0.15 figure here is specific to
   DDXPlus-length queries and must not be quoted as a general floor.
5. **Natural cold-start share is 0.258 here, not the ≈0.35 the brief assumed.** The paper's own
   committed data (`results/olmo_attention_instruct/attention_stats.csv`, L24) gives
   `frac_current_query` of 0.3995 in the lowest fill bin, falling to 0.1591 in the top bin. This
   harness reproduces the *trajectory* (0.253 cold → 0.149 at 69% fill) but starts lower, most
   likely a difference in how the query span is delimited. Levels were chosen against the measured
   value, not the assumed one.

## Mechanics worth recording

- Under **sdpa** a purely causal mask is optimized away to `None` before reaching `self_attn`, so
  the clamp has nothing to bias. A single **masked left-pad token** forces an explicit mask.
  `attn_implementation="eager"` also works but materializes `[1, H, N, N]` per layer and **OOMs the
  32 GB card at 4k** — the box is an RTX 5090, not the 96 GB card the older notes assume.
- sdpa hands down a **boolean** keep-mask, not an additive float one; the clamp converts it, in the
  query's dtype (sdpa rejects a mismatched bias dtype).
- The padded input is used for *every* condition including the unclamped baseline, so within-E2
  comparisons are exact. Versus an unpadded forward the sdpa mask path moves logits by ~2% of
  scale (argmax agreement 97.7%, last-token argmax unchanged).

## E2a addendum (2026-08-24): the ladder re-denominates fractionally — and accumulation passes the edge

All-layer re-run (`e2a_alllayer/`, n=110, `--reference-layer 0..31`, levels at the committed
run's fractions of natural). Natural all-layer query share 0.463, accuracy 0.536. Paired
contrasts vs natural: 0.54 (above natural) +0.000 [−0.036, +0.036]; 0.36 (0.775×) +0.045
[−0.018, +0.118] n.s.; **0.27 (0.581×) +0.118 [+0.027, +0.209]** significant. Levels 0.18 and
below are excluded by the committed degeneracy criterion (−3.63 to −5.98 nats; modal answer
"A" on 99–110 of 110). The fractional structure is identical to the committed L24 ladder:
flat through 0.775× natural, significant cost at 0.581×.

Accumulated trajectory in the same units (20 items/point, the driver's own construction):
0.464 (cold) → 0.295 (1 case) → 0.245 (2) → 0.218 (4) → 0.208 (6) → **0.202 (8 cases)**.
Where the L24 readout said accumulation stops *at* the cost edge (0.15 both), the all-layer
readout says it passes it: 0.202 sits below the first significantly costly clamp level
(0.27). Accuracy nonetheless stays flat under accumulation (§4.1's null), so the
ICL-compensation reading strengthens — accumulation does not stop short of the harmful
region; it enters it and in-context learning pays the difference.
