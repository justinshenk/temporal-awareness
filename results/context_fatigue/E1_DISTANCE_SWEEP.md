# E1 — within-harness distance sweep

**Verdict: CONFIRMED on every §6 criterion.** Distance carries the coefficient; fill does not.

Run 2026-08-18 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · artifacts in
`results/context_fatigue/e1_distance_sweep/` (`turns.csv`, `summary.json`) · driver
`scripts/context_fatigue/run_distance_sweep.py`.

## What was varied

One thing: **where the answer-bearing evidence sits**. Each session accumulates short MMLU filler
turns; at four depths the transcript is snapshotted and every DDXPlus probe is asked once per arm,
with its vignette placed *k* user turns before its question.

At a given snapshot the arms share the *same* filler, the question text is byte-identical across
arms (including the referent "For the patient described earlier"), and mean fill is **0.688 in
every arm** — so position is the only thing that moves.

## Results

n = 192 per arm (§6 asks ≥150). Chance = 0.200. Overflow guard skipped **0 of 192** probes.

| arm | distance | n | accuracy | vs `local` (95% CI) | unparsed |
|---|---|---|---|---|---|
| `local` | 0 | 192 | **0.464** | — | 11.5% |
| `back_2` | 2 | 192 | 0.359 | +0.104 [+0.005, +0.203] | 13.0% |
| `back_5` | 5 | 192 | 0.292 | +0.172 [+0.073, +0.266] | 26.6% |
| `back_10` | 10 | 192 | 0.250 | +0.214 [+0.120, +0.307] | 27.1% |
| `back_20` | 20 | 192 | 0.276 | +0.188 [+0.094, +0.281] | 18.2% |

**Joint fit, accuracy ~ fill + distance** (case-resampled bootstrap, 4,000 draws):

| predictor | β | 95% CI | significant |
|---|---|---|---|
| distance | **−0.00761** | [−0.01173, −0.00346] | **yes** |
| fill | −0.00725 | [−0.21006, +0.18658] | no |

**`local` is flat with fill**: β_fill = −0.294 [−0.767, +0.184], not significant — the paper's
null survives inside the arm that reproduces its design.

## Against §6

- *Confirms:* `local` flat with fill ✔; every `back_k` gap's 95% CI excludes zero, the k=20 arm
  included ✔; in the joint fit distance is significant and fill is not ✔.
- *Falsifies:* would have required `back_20` flat. It is not.

One honest qualification: the decline is monotone through k=10 (0.464 → 0.359 → 0.292 → 0.250)
and then **ticks up at k=20** (0.276). `back_10` and `back_20` are not distinguishable from each
other; the effect looks like a decline that saturates by ~10 turns, not one that keeps deepening.

## Robustness

**Parsed responses only** (19.3% of responses never emit a letter): the effect survives and is
*fully* monotone — `local` 0.524, `back_2` 0.413, `back_5` 0.397, `back_10` 0.343, `back_20`
0.338, all four gaps excluding zero, distance β = −0.00748 [−0.01213, −0.00283].

**The unparsed rate is not a distance effect.** Regressed the same way: distance β = +0.00318
[−0.00015, +0.00654], *not* significant; fill β = −0.290 [−0.452, −0.118], significant. So the
arms' differing parse rates do not confound the distance result. Unparsed responses are reasoning
preambles ("Given the symptoms described for the patient, such as pain that is worse when lying
down…") truncated by `max_new=32`, not refusals — a larger budget would reduce them further.

## Design deviations from the brief, and why

1. **Filler is MMLU, not DDXPlus.** A full DDXPlus case averages 309 tokens against OLMo-2's 4,096
   window, so ~13 turns fit: `back_20` could not exist and `back_10` would only occur above ~85%
   fill, making distance collinear with fill. §6 asks for a joint fit separating them, which
   collinear predictors cannot deliver. Short filler (~74 tokens) puts 21 turns at 43% fill. The
   probe remains a full DDXPlus case.
2. **Every arm gives the evidence its own turn**, `local` included (so `local` is distance 0
   rather than the vignette inlined into the question turn). Otherwise the arms would differ in
   turn structure as well as distance.
3. **An explicit referent** was added to the question in *all* arms, so the deep arms measure
   retrieval at distance rather than whether the model noticed a patient was mentioned.

## Prior run voided

`e1_distance_sweep_VOID_option_bias/` — see its `VOID.md`. Options were unshuffled, so gold was
"A" in 71.4% of probes while arms differed in letter bias; `max_new=8` truncated before the letter
with an unparsed rate that tracked distance; and 26.7% of cases carried fewer than five options,
286 of them a single option answerable without the vignette. All three are fixed here: options
shuffled (gold uniform to within 0.6pp), `max_new=32`, five-way choices only.

## What this does for the paper

The central null is now bounded from the other side inside our own harness. The same model, items,
prompt and metric that show **no accuracy cost across accumulation** show a **−0.19 to −0.21 cost**
when the same evidence is moved 10–20 turns back at identical fill. The task is not too easy to
degrade; it degrades when the evidence stops being local. That is the localization claim, measured
rather than argued, and it closes gap **G1**.
