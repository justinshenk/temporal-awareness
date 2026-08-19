# VOID — do not quote these numbers

First E1 run (2026-08-18). Two driver bugs invalidate it:

1. **Options were not shuffled.** DDXPlus lists the differential in rank order and the true
   pathology is usually first, so gold was "A" in **71.4%** of probes. The arms differ in letter
   bias (`local` answered "A" 35% of the time, `back_10` only 10%), so the apparent accuracy
   decline is partly a letter-preference difference, not an evidence-distance effect.
2. **`max_new=8` truncated responses** before the letter in many cases. The unparsed rate tracked
   distance (local 25% → back_10 48%), so unparsed-as-wrong scoring inflated the gap in exactly
   the direction of the hypothesis.

A third contamination was found while fixing these: 26.7% of cases had fewer than 5 differential
options, including 286 with a **single** option where "A" is correct without reading the vignette.

Superseded by `e1_distance_sweep/` (shuffled options, uniform gold, max_new=32, 5-way choices
only, raw responses recorded).
