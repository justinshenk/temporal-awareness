# E2b — does restoring attention mass rescue the deep-fill dip?

**Verdict: the committed dip is a single-bin small-sample artifact of where the original run's fill
cap fell.** The harness reproduces the committed result *exactly, item for item*; the dip survives
neither extending those same sessions past 0.88 nor fourteen fresh sessions. The rescue arm is null
and uninformative, because there is nothing to rescue.

Run 2026-08-18 · `allenai/OLMo-2-1124-7B-Instruct` · L24 · seed 42 · 26 sessions · artifacts
`results/context_fatigue/e2b_dip_rescue/` and `results/context_fatigue/e2b_scoring_control/` ·
driver `scripts/context_fatigue/run_mass_clamp.py --mode e2b`.

## The replication

Random-subject MMLU stream, accumulating to 93% fill, overflow-guarded, forced-choice scored.
n = 780 turns over 26 sessions; n = 108 in the top bin (the committed pool had 91).

| fill bin | n | accuracy | query share @L24 |
|---|---|---|---|
| 0–20% | 184 | 0.538 | 0.122 |
| 20–40% | 164 | 0.573 | 0.105 |
| 40–60% | 163 | 0.620 | 0.098 |
| 60–80% | 161 | 0.634 | 0.100 |
| **≥80%** | 108 | **0.583** | 0.096 |

- **Top bin minus rest: −0.006 [−0.105, +0.092]** — covers zero.
- **Accuracy ~ fill slope: +0.096 [−0.039, +0.219]** — mildly *positive*, not significant.
- The mass drain is present and matches expectation: query share falls **0.122 → 0.096**.

The committed interval [−0.249, −0.031] and this one [−0.105, +0.092] overlap on [−0.105, −0.031],
so at the level of independent samples the two runs are not formally incompatible. The exact-item
analysis in the next section settles it more sharply than any interval comparison can.

## The exact-item replication — this is the decisive analysis

My e2b driver turned out to use the committed driver's seed formula, so its sessions 0–11 present
the **identical question sequence** as `results/random_context/turns.csv`. That converts a
sample-vs-sample argument into an item-level one.

| check | result |
|---|---|
| matched items | **344** across all 12 sessions |
| subject-sequence match | **344/344 = 100%** |
| gold agreement | 1.000 |
| accuracy | committed 0.6221 vs mine 0.6221 |
| **per-item agreement** | **1.000** — every item scored identically |
| context fill agreement | max abs difference 0.0003 |
| dip on those items | **−0.1874 [−0.3710, −0.0034] in both, significant in both** |

**There is no implementation difference.** The harness is bit-identical to the committed one, and it
reproduces the committed dip precisely on the committed items. Whatever explains the disagreement is
sample composition, not method.

### Where the dip actually lives

| fill bin | n | accuracy |
|---|---|---|
| 0.70–0.80 | 79 | 0.633 |
| 0.80–0.85 | 40 | 0.625 |
| **0.85–0.88** | **31** | **0.419** |
| 0.88–0.93 | 37 | **0.703** |

The dip is a **narrow trough at 0.85–0.88 with n = 31**, with normal accuracy on both sides. The
committed run's maximum fill is **0.8784** — so its entire top bin *is* that trough, and its
`fill_target = 0.88` cap is what made the trough the end of the data.

| sample | n deep | dip |
|---|---|---|
| committed 12 sessions, capped 0.88 | 31 | **−0.1874 [−0.3710, −0.0034]** sig |
| *the same 12 sessions* extended to 0.93 | 48 | −0.0973 [−0.2479, +0.0533] n.s. |
| 14 fresh sessions (unseen seeds) | 60 | **+0.0902** [−0.0405, +0.2235] n.s. |
| all 26 sessions | 108 | +0.0048 [−0.0931, +0.1015] n.s. |

Running the *same sessions* four percentage points further in fill halves the dip and removes its
significance. Fresh sessions give the **opposite sign**.

### Two artifacts ruled out

- **Overflow-guard selection**: the guard skipped **1 of 781** items (0.13%), so the deep bins are
  not a filtered subsample.
- **Item length**: the 0.85–0.88 and 0.88–0.93 bins have near-identical median item lengths (86 vs
  90 tokens), so the recovery above 0.88 is not explained by easier short items surviving.

## Scoring was tested and excluded

The obvious suspect was the scoring rule: the committed run generated a response and extracted a
letter, counting unparseable output as wrong, while this run scores forced-choice over the option
letters. E1 had shown parse failures running at 19.3% and varying by condition, so a
fill-dependent parse rate could manufacture a dip.

A control run scored **both ways on the same forwards, same items** (`e2b_scoring_control/`):

| rule | top bin − rest |
|---|---|
| forced-choice | −0.0060 [−0.1052, +0.0919] |
| generation + letter extraction | +0.0048 [−0.0931, +0.1015] |

**Parse failures: 0 of 780.** The two rules agree bin for bin. MMLU items prompted with "reply with
only the letter" produce clean single-letter output; the 19.3% unparsed rate in E1 came from
DDXPlus five-way diagnosis, which invites reasoning. Scoring is therefore **not** the explanation
for the non-replication, and the hypothesis that motivated this control is rejected.

## The rescue arm

At ≥80% fill each turn was clamped back up to **its own session's** mean cold-start share, a
within-session control rather than a target imported from elsewhere.

- paired n = 97; share restored **0.092 → 0.123**
- rescued − natural = **+0.021 [−0.124, +0.165]**, covers zero
- flips: 2 items wrong→right, 0 right→wrong

This is a null, but a weak instrument in both directions: the restoration is only ~3 points of
share, and E2a established that accuracy is flat across that range anyway. Even a real dilution
effect of the size the paper describes would not necessarily show up under so small a restoration.
**Do not quote this as evidence that mass is not the mechanism** — quote E2b's main result instead.

## What this means for the paper

The top-bin dip is the paper's **one positive result** — `context_fatigue.tex:164`, the "final 20%
is not flat" section, and commit `b3eb335` ("The top-bin dip is real, and it proves the paper's
mechanism"). It does not survive an independent replication with larger n, and the most plausible
methodological explanation has been tested and ruled out.

Stated as precisely as the evidence allows: the committed number is **correctly computed from its
data**, and this work reproduces it exactly on that data. What it is not is a property of the
model — it is a property of one 31-item bin that the original protocol's fill cap made terminal.
Gap **G3** is closed in the least convenient direction: the explanandum does not exist.

What follows for the paper:
1. **The "final 20% is not flat" section needs rewriting.** The claim that the last fifth of the
   window shows a genuine cost is not supported once the same sessions run four points further.
2. **The abstract's scoping should widen.** The null was restricted to "the first ~80%" because of
   this dip; with the dip gone, the flat result extends across the measured range to ~93%.
3. **E4 is moot.** A window-position control for a dip that is not there has nothing to test, and
   the GPU time should go elsewhere.
4. **Keep the committed artifacts.** They are not wrong, and the fine-bin table above is a better
   illustration of small-n fragility than anything synthetic.

## Limitations

1. **Share is not comparable to E2a.** This stream sits at 0.096 query share at deep fill with no
   accuracy cost, while E2a found a 16.4-point cost at 0.15 on DDXPlus. The reconciliation is query
   length — a ~300-token case draws more share than a ~60-token question — but that reconciliation
   is an assumption here, not something measured.
2. **The drain is small.** 0.122 → 0.096 is not the halving (0.35 → 0.15) the paper reports for the
   DDXPlus stream, so this stream is a weaker test bed for a mass effect than the brief assumed.
3. **Assistant turns are the model's own answers**, as in the committed protocol, but recorded as a
   single letter rather than a truncated generation.
