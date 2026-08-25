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

## Conclusions

Each claim states its evidence from this run's cells, then its position against prior work
(citation keys resolve in `context_fatigue_paper/related_work_citations.md`; all arXiv IDs
verified against their abstract pages).

**C1 — The carrier of the format-instruction state is the transcript's skeleton, not any
turn's content.** *Evidence:* patching only the `<|im_end|>/<|im_start|>` delimiter
positions moves −1.540 of the −2.488 contrast (62%); every content subset moves ≤ |0.287|
(≤ 12%); the size-matched random-position control moves +0.016. *Prior work:* this is the
instruction-state analogue of `wang2023labelwords` — where label words anchor the
*semantics* of demonstrations, the delimiters anchor the conversation's *format regime* —
and of `darcet2023registers`, where models spontaneously repurpose low-information tokens
as computation slots. `sun2024massiveactivations` reports that delimiter and special
tokens host massive activations functioning as fixed biases; C1 sharpens that from
correlate to carrier: transplanting these tokens' states transports the behavioral
contrast.

**C2 — The state is formed and consumed in the lower half of the stack.** *Evidence:*
layer blocks 0–6 / 7–13 / 14–20 / 21–27 carry −1.160 / −1.591 / −0.454 / −0.113; the
crossed cell (glue positions × layers 7–13 only) still carries −1.051 (42%) while touching
no content token and no layer above 13. *Prior work:* the depth profile matches where
compact task representations are reported to form (`hendel2023taskvectors`,
`todd2024functionvectors`), and the position×depth address makes concrete the
"instruction vs demonstration as distinct channels" separation that
`davidson2025taskrepresentation` establishes representationally.

**C3 — Within that band the state is redundantly distributed, not point-localized.**
*Evidence:* the four layer blocks sum to −3.32 against a full effect of −2.488 —
overlapping carriers, not additive components. *Consistency:* E6's four failed erasure
strategies (no single estimated linear direction removes the mode) said the same thing
from the representation side.

**C4 — Three instruments, two families, one address.** *Evidence:* (a) OLMo E3c′ —
0.42 of context-body final-position attention mass lands on the glue, and closing it
disrupts the demonstrated reply format (−0.175 net); (b) OLMo E6′ — closing
exemplar-answer content in every attention window leaves the precedent mode fully
expressed; (c) this run — patching glue states carries 62% of the Qwen format contrast.
*Prior work:* `mu2023gist` shows instruction state can be *trained* into dedicated
compression tokens; C4 says chat-tuned models do this natively, electing the template's
delimiters. Against the sink literature (`xiao2023streamingllm`, `gu2024attentionsink`):
sinks absorb mass content-independently as normalization slack, whereas the glue tokens
here carry content-bearing, transplantable causal state — attention concentration on
special positions and functional storage at special positions are different claims, and
this run supplies the interventional evidence for the second. It also refines
`dongre2026attentioncloses`: the system prompt's *span* is not where the live format
state resides once context accumulates — the contest is settled on the skeleton.

**Why Waves 1–2 had to be null:** role-content and recency subsets are built from
turn-content spans, which exclude the glue by construction. The bisection was searching
inside the turns while the signal rides between them.

## What this does not yet establish

- **Periodicity is a confound candidate.** The glue subset is also the positionally
  periodic subset; the random control is size-matched but not periodicity-matched. A
  one-token-per-turn-boundary *content* control would separate "delimiter tokens" from
  "periodic positions".
- **No per-cell CI.** n = 24 and the prefix-logprob estimand carries no bootstrap in this
  driver; comparisons lean on the unrelated-fact control (≤ |0.11| everywhere except
  last_2's +0.264) and the winning cell's dedicated random control.
- **Cross-family confirmation is pending.** The OLMo template cell (n = 100) is running;
  OLMo's Stage-1 ΔΔ is an order of magnitude smaller (±0.06–0.08), so the readable
  question there is whether the same channel carries its small effect — not whether the
  magnitudes match (the E2b rule: cross-family differences need the
  difference-in-differences, not CI-vs-CI).
- **Who writes the glue state is not yet separated:** the system instruction at prefill,
  the accumulated demonstrations, or both. E6′'s closure nulls point at the
  demonstration side, but a write-time bisection (patch glue states captured at different
  accumulation depths) would settle it.
