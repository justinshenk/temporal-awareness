# E3b/E3c — is competition's cost carried by *reading* the competitor instances?

**Date:** 2026-08-21. **Motivation (user, in chat):** E3's headline — competition costs accuracy
at constant *evidence* attention mass — leaves one attention-level route untested. The filler
receives the same total mass in every arm, but in `near_dup` that mass lands on instances of the
probe's own answer candidates (shared option names attached to other patients and other golds).
At constant evidence mass, the "pool of attention over answer-relevant content" gains misleading
members. If the cost routes through reading those instances at generation time, competition IS
attention-mediated after all — competitor-side, not evidence-side — and the paper's title claim
("only displacement acts through attention mass") needs revising. If closure is null, the
interference is installed at prefill (the E6 exemplar-close pattern) and the second-mechanism
claim gets its strongest form.

## Design (E6 fa_close transplanted to E3)

Paired over the same probe pool, seeds, and context selection as `e3_competition` (seed 42,
n_context 8, min_overlap 3). Four arms per probe, all generated eager:

| arm | context | intervention |
|---|---|---|
| `near_dup` | near-duplicate | none (in-run natural baseline) |
| `near_dup_comp_close` | near-duplicate | scale-0 closure of every occurrence of the probe's option names in the context region |
| `near_dup_rand_close` | near-duplicate | closure of size-matched random spans in the same region (geometry control) |
| `random` | natural stream | none (reference for the gap) |

Competitor spans = all occurrences of the probe's 5 option strings **strictly before the
evidence turn** (the probe's own option list and vignette are never touched). Spans are merged
if overlapping. Measurement rider (E3b): one capture forward per probe records the competitor
spans' union share (all-layer mean) before closure, plus span count and token total.

## Outcomes and what they mean

- **Rescue** (`comp_close` − `near_dup` positive, CI excluding 0, rand control null, landing at
  or near `random`): competition is attention-mediated via competitor spans → §4.3 and the title
  claim must be revised before submission.
- **Null** (`comp_close` ≈ `near_dup`, rand control also null): reading the competitors at
  generation time is not the route; interference is prefill-installed → add the result to §4.3
  as the strongest form of the dissociation.
- **Void conditions:** rand control moves accuracy (instrument too blunt); parse rate drops
  under closure (broken generation); `near_dup` in-run baseline misses the committed 0.427 by
  more than noise (harness drift); competitor spans found in `random` arm exceed near_dup's
  (selection broken).

## Files

- `src/probes/context_fatigue/attention_clamp.py`: `locate_phrase_spans` (new, test-first).
- `scripts/context_fatigue/run_competition_sweep.py`: `--close-arms` mode.
- Artifacts → `results/context_fatigue/e3c_competitor_close/`, report `E3C_COMPETITOR_CLOSE.md`.

## Acceptance

Preflight (2 probes) prints per-arm rows with closed-span counts > 0 in near_dup arms and 0
touched tokens outside the context region. Full run n=365 paired; paired bootstrap on
comp_close−natural, rand−natural, natural−random. Report quotes artifact filenames, n, skip
counts, and the explicit verdict including which outcome fired.
