# Qwen E7 Stage 2 — the inversion rides the chat-template glue, in the lower half of the stack

**Verdict: the counterfactual format-instruction state localizes to the chat template's
delimiter tokens. Patching only the `<|im_end|>/<|im_start|>` glue positions carries
ΔΔ_A→B = −1.540 — 62% of the full patch's −2.488 — while every content subset is near
null (assistant turns −0.287, user turns +0.165, last-k positions ≤ |0.075|) and the
size-matched random-position control is +0.016. Across layers the effect sits in the lower
half of the stack (0–6: −1.160, 7–13: −1.591, 14–20: −0.454, 21–27: −0.113), and the
crossed cell — template positions × layers 7–13 only — still carries −1.051, 42% of the
full effect from a patch touching neither any turn's content nor any layer above 13.**

Runs 2026-08-24/25 · `Qwen/Qwen2.5-7B-Instruct` · code cell, depth 15, closed, n = 24
(1 overflow skip), A→B direction (closure degeneracy makes B→A its mirror; see
`QWEN_E7_FORMAT_PATCH.md`) · driver `run_format_patch.py` · artifacts
`results/context_fatigue/qwen_e7_bisect_*/`.

## Cells (ΔΔ_A→B against dd_full = −2.488; unrelated-fact control beside each)

| cell | ΔΔ_A→B | ΔΔ_unrelated | share of full |
|---|---|---|---|
| all positions, all layers (Stage 1) | −2.488 | — | 100% |
| positions: assistant_turns | −0.287 | −0.025 | 12% |
| positions: user_turns | +0.165 | +0.062 | — |
| positions: last_1 / last_2 / last_4 | −0.075 / +0.001 / −0.036 | ≤ |0.264| | ~0% |
| **positions: template glue** | **−1.540** | −0.038 | **62%** |
| positions: template, size-matched random control | +0.016 | −0.113 | 0% |
| layers 0–6 (all positions) | −1.160 | −0.001 | 47% |
| layers 7–13 | −1.591 | −0.005 | 64% |
| layers 14–20 | −0.454 | +0.054 | 18% |
| layers 21–27 | −0.113 | +0.070 | 5% |
| **crossed: template × layers 7–13** | **−1.051** | +0.046 | **42%** |

Layer blocks are not additive (their sum exceeds the full effect): the signal is
redundantly represented across adjacent blocks, as block-patching of a distributed carrier
predicts. The position story is the opposite — content subsets contribute nearly nothing
and one non-content subset dominates.

## Reading

1. **The carrier is structural, not semantic.** The glue tokens contain no instruction
   text and no demonstrated answers; they are the turn delimiters. Yet transplanting their
   hidden states moves 62% of the counterfactual format contrast, while transplanting every
   assistant turn's actual content moves 12%. The format-instruction state is written onto
   the transcript's *skeleton* during prefill and read from there.
2. **Cross-instrument convergence.** Three independent measurements now point at the same
   tokens: (a) OLMo E3c′ — 0.42 of context-body final-position attention mass lands on the
   glue, and closing it disrupts the demonstrated reply format (−0.175); (b) OLMo E6′ —
   closing exemplar-answer content in any window leaves the precedent mode fully installed;
   (c) this cell — patching glue states carries the Qwen format contrast. The E6 "installed
   at prefill, task-vector style" account gains a concrete address: the delimiter tokens'
   residual streams, layers ≤ 13.
3. **Why waves 1–2 were null.** Role-content and recency subsets all exclude the glue
   (turn-content spans only); the bisection was looking inside the turns while the signal
   rides between them.
4. **Cross-family status.** This localization is established on Qwen. The OLMo
   confirmation cell (template positions, n = 100, code cell) is the remaining follow-up;
   OLMo's Stage-1 ΔΔ effects are an order of magnitude smaller (±0.06–0.08), so the
   question there is whether the *same channel* carries its small effect, not whether the
   magnitudes match (the E2b rule applies to any cross-family comparison).

## Caveats

- n = 24 per cell and the prefix-logprob estimand has no bootstrap CI in this driver;
  cell-to-cell comparisons lean on the unrelated-fact control (≤ |0.11| everywhere except
  last_2's +0.264, which bounds that cell's noise) and on the size-matched random-position
  control run for the winning cell specifically.
- The template subset is also the *positionally periodic* subset; a periodicity-matched
  content control (e.g. one token per turn boundary neighborhood inside content) was not
  run. The random control is size-matched but not periodicity-matched.
