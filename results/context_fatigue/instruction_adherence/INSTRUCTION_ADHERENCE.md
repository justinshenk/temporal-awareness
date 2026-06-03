# Does instruction adherence decay under context accumulation? (Qwen-7B)

**Short answer: no — and the canaries were too easy to tell us much (floor effect).**

## Motivation
The "context fatigue" reading keeps failing to find a behavioral cost: task accuracy
does not fall as context fills (`MIXED_TASK_ICL.md`), and the entropy collapse is
plausibly just ICL competence, not fatigue. This experiment looks for a degradation
that accuracy is blind to: does the model stop obeying a fixed, **checkable** system
instruction (a "canary") as the conversation accumulates DDXPlus MCQ cases — a
*competence-preserved, compliance-degraded* dissociation?

## Design
Same accumulation harness as the other context-fatigue runs (Qwen2.5-7B-Instruct,
32k ctx, fill to 92%, greedy, seed 42, 200-case pool, ~83–86 turns/pass). A canary
instruction sits in the system prompt; every turn we record task correctness *and*
whether the canary was obeyed, both vs context fill. The task answer is emitted as a
parseable `ANSWER: X` line so correctness is read **independently** of the canary,
and the canary markers are not option letters (no extraction collision).

Two canaries, three arms each (driver: `scripts/context_fatigue/run_instruction_adherence.py`,
logic + arms: `src/probes/context_fatigue/instruction_checks.py`):

- **prefix_marker** — begin every reply with `◆`. (Easy: it's the first token.)
- **suffix_ok** — end every reply with the tag `⟦OK⟧`. (Harder: must be remembered
  *after* answering.)
- Arms: **baseline** (model's own outputs accumulate), **forced** (history is always
  rewritten to contain the canary → a decay here can't be imitation of prior outputs),
  **refresh** (canary restated in the latest user turn → constant distance; also the
  context-refresh intervention).

## Result — perfect adherence throughout, accuracy flat

| run | n | violations | obeyed | acc | corr(viol,fill) | corr(acc,fill) |
|---|--:|--:|--:|--:|--:|--:|
| prefix_marker / baseline | 86 | 0 | 1.000 | 0.605 | +0.000 | +0.057 |
| prefix_marker / forced   | 86 | 0 | 1.000 | 0.605 | +0.000 | +0.057 |
| prefix_marker / refresh  | 83 | 0 | 1.000 | 0.566 | +0.000 | +0.050 |
| suffix_ok / baseline     | 86 | 0 | 1.000 | 0.605 | +0.000 | −0.012 |
| suffix_ok / forced       | 86 | 0 | 1.000 | 0.605 | +0.000 | −0.012 |
| suffix_ok / refresh      | 81 | 1 | 0.988 | 0.605 | −0.199 | −0.067 |

The model obeys both canaries on **every turn from fill ~1% to ~93%**, in all three
arms. Sample at fill 0.93: `B\n\nANSWER: B\n\n⟦OK⟧`. The lone "violation"
(suffix_ok/refresh) is at fill **0.06** — i.e. *early*, the opposite of decay
(hence the negative correlation). Accuracy is flat (|corr| ≤ 0.06), reproducing the
no-accuracy-decay finding.

So the **competence-vs-compliance dissociation does not appear** here: there is no
compliance decay to dissociate from accuracy. Even the harder, more-forgettable
trailing tag — and even when the canary is positionally far from the generation site
(prefix/suffix in the system prompt across 32k of intervening tokens) — holds at 100%.

## Critical caveat — floor effect (why this is a *weak* null)
Baseline violation is **0% from turn 0**. The canaries are so easy that there is
**no dynamic range**: a probe with a zero baseline cannot detect a *rise*. This run
firmly establishes that *simple format canaries are robust* under accumulation, but it
is **underpowered** to test whether *harder* instructions decay. A sharp dissociation
test needs an instruction with an intermediate (~20–40%) baseline violation rate.

## Interpretation
Combined with the flat-accuracy result and the entropy-is-ICL confound, this is a
third behavioral channel in which Qwen-7B shows **no "fatigue."** The deployment
hazard remains calibration/attention-allocation (confidently-wrong gap, system-prompt
attention erosion), *not* a collapse of task accuracy or of basic instruction-following.

## Next (v2, to escape the floor)
Re-run with an instruction that is violated ~20–40% at baseline, so there is room to
rise — candidates: a constraint that competes with the task or the model's defaults
(e.g. a length/te­rminology constraint, a multi-part format), or the canary buried
among distractors in a long system prompt (instruction "lost in the middle"). Only if
such an instruction shows `corr(violation, fill) > 0` in the **forced** arm is the
dissociation real; then Phase 2 (activation probe for "about to violate") is warranted.

## Reproduce
```bash
uv run python -m scripts.context_fatigue.run_instruction_adherence \
    --model Qwen/Qwen2.5-7B-Instruct --max-ctx 32768 --max-new 64 \
    --max-cases 200 --instructions prefix_marker,suffix_ok --arms baseline,forced,refresh
```
Per-(instruction,arm) turn CSVs + `summary.json` here; unit tests in
`tests/probes/context_fatigue/` (18, GPU-free).
