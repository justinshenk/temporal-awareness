# E6′ — prefill-only vs decode-only exemplar closure: installation does not route through the answer keys

**Verdict: closing the filler-answer channel blocks the precedent mode in NO window —
compliance stays 0.000 under prefill-only, decode-only, and all-window closure alike, so the
mode's installation does not require reading the exemplar answers at any point of the probe's
forward. The window split instead localizes a different channel perfectly: the exemplar
answers' contribution to *accuracy* is entirely prefill-borne (−0.175 [−0.300, −0.075] SIG
prefill-only; exactly 0.000 item-for-item decode-only; all-window closure ≡ prefill-only
closure item-for-item). Reading the demonstrated answers helps solve the case while the
prompt is being encoded, and contributes nothing at generation time — while the format mode
rides on neither.**

Run 2026-08-25 · `allenai/OLMo-2-1124-7B-Instruct` · mmlu filler, depth 42 (fill 0.877),
n = 40/arm, 0 overflow skips · driver `run_format_erosion.py --filler mmlu --depths 0 42
--close-windows` · artifacts `results/context_fatigue/e6_close_windows/` · brief
`tasks/per_token_capture_brief.md` Stage 3. `SpanAttentionClamp` gained
`window={all,prefill,decode}`; prefill-only releases the mask bias on cached decode steps
(bit-identity pinned by test), decode-only is the complement.

## Anchors reproduce the committed run

Depth 0: compliance 0.875, accuracy 0.450, system share 0.1903 (committed: 0.875 / 0.19).
Depth 42 natural: compliance 0.000, accuracy 0.650, system share 0.0196. In-session
all-window arms: fa_close 0.000 compliant (committed 0.000), fq_close 0.100 (committed
0.132), rand1_close 0.000, fa_matched 0.000.

## The window split (all at depth 42)

| arm | window | compliance | accuracy | mean chars |
|---|---|---|---|---|
| natural | — | 0.000 | 0.650 | 107 |
| fa_close | all | 0.000 | 0.475 | 47 |
| fa_close_prefill | prefill | 0.000 | 0.475 | 15 |
| fa_close_decode | decode | 0.000 | 0.650 | 105 |
| rand1_close_prefill | prefill | 0.000 | 0.650 | 162 |
| rand1_close_decode | decode | 0.000 | 0.650 | 105 |

Paired accuracy contrasts (10,000 draws over probes):

- fa_close_prefill − rand1_close_prefill: **−0.175 [−0.300, −0.075] SIG**
- fa_close_decode − rand1_close_decode: **+0.000 [+0.000, +0.000]** — item-for-item
  identical correctness
- fa_close − fa_close_prefill: +0.000 [+0.000, +0.000] — the all-window arm adds nothing
  beyond its prefill component, item-for-item
- fa_close_prefill − natural: −0.175 [−0.300, −0.075]

## Reading

1. **Installation localization (the brief's causal test): null in both windows.** The
   hypothesis grid was "prefill-only closure blocks the mode where decode-only does not ⇒
   installation is prefill-attention to the exemplars." Neither blocks it. With the
   committed generation-time null re-anchored in the same session, no window of
   attention-to-exemplar-answers is necessary for the precedent mode. The mode must be
   carried by context structure the fa spans do not cover — consistent with E3c′'s finding
   (same day, different instrument) that 0.42 of context-body final-position mass sits on
   template glue/turn boundaries whose closure disrupts the demonstrated format, and with
   E7's bisection pointing away from role-content positions. prefill-closed replies are
   15-char bare letters: the mode is not merely surviving, it is fully expressed.
2. **The accuracy channel splits perfectly by window.** All of the exemplar-answer
   channel's accuracy value is consumed during prefill (the ICL benefit of seeing worked
   answers is encoded into the context representations); decode-time access to the same
   spans is worthless (closing them at decode changes nothing, item-for-item). This is the
   cleanest window-localization the program has produced, and it is the same pattern E1's
   evidence-mass story predicts: what matters is what the prompt encoding absorbed, not
   what generation re-reads.
3. **fq_close remains the only compliance mover** (0.100 here, 0.132 committed) — closing
   filler *questions* releases a sliver of compliance, again pointing at the questions'
   structural demonstration, not the answers, as closer to the mode's carrier.

## Caveats

- n = 40 per arm, one depth, one filler arm (mmlu), per the brief's scope.
- Decode-window closure of fa/rand1 produced identical per-item correctness; with 32-token
  spans out of ~3,600 the decode-side perturbation is small, so the decode null is
  "no detectable effect at this dose," bounded by the CI of the prefill contrast it mirrors.
- Compliance is floor (0.000) in every closure arm, so the compliance nulls are bounded by
  n=40 at floor — a rescue as small as ~0.07 would have been visible (fq_close's 0.100 was).
