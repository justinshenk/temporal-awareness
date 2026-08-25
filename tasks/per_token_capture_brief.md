# Per-token attention capture — targeting data for three follow-ups

**Date:** 2026-08-25. **Motivation (user, in chat):** per-token attention weights add little as
*headline evidence* — §4.2's observational-correlations caution (within-arm share↔accuracy has
the opposite sign to the causal effect, β = −11.2) applies at any granularity — but they are
the targeting data three existing open questions need. The capture is cheap: Appendix B's
reconstruction already computes the full final-position attention row per layer/head and
aggregates it to span shares before storing; the change is to (optionally) store the row.

## 1. Problem statement

Three open threads are blocked on not knowing *where* attention actually lands at token
granularity:

1. **The ~40% competitor-closure residual (§4.3 / Limitations).** E3c's closure covers
   verbatim option-name mentions only (30.0 spans, 127.9 tokens/probe). Whether the residual
   is instrument slack (mass on paraphrases, shared symptom phrases, context vignette bodies)
   or prefill-borne interference is explicitly open.
2. **The pattern-matched clamp (§4.2 / Conclusion's "test we would run next").** Mass
   restoration recovers only 0.28 because the uniform clamp cannot reconstruct *which* heads
   (and which tokens) should carry the restored mass. The `local` arm's per-token, per-head
   pattern is the template that clamp needs.
3. **Prefill installation of the precedent mode (§4.5, Refine #2).** Generation-time closure
   shows decode-time reading does not *sustain* the mode; nothing yet localizes installation.
   The causal test is a prefill-only closure; per-token prefill capture documents the route.

Plus a presentation win: one real token-level heatmap of a displacement pair (local vs
back_10) — the honest version of the span-tinted mock in the attention-explainer artifact —
as a talk / camera-ready panel.

## 2. Solution approach

**Stage 0 — capture extension (CPU-cheap, one small code change).**
Extend the attention-capture path to optionally store the final pre-generation position's
attention row: per layer, per head, the post-mask post-softmax weights over all context
positions, plus the token strings and span boundaries already computed for share
aggregation. Storage: one `(layers × heads × ctx)` float16 array per probe (~4k ctx ×
32 × 32 ≈ 8 MB fp16; store all-layer/head-mean row by default, full tensor behind a flag).
A `--capture-prefill-rows` variant stores the same for every prompt position at the layers
of interest (larger; restrict to a probe subsample).

**Stage 1 — targeted-closure re-run (E3c′).** One capture pass over the committed `near_dup`
panel ranks context tokens by received mass. Build an expanded closure arm from the measured
hot set (cap the token budget at k× the verbatim closure's 127.9 tokens for a size-matched
comparison), with the same size-matched random-closure control. Rescue beyond +0.055 that
survives the control converts "substantial, not total" into a localized residual.

**Stage 2 — pattern-matched restoration (E1d′).** From the `local` arm's captured rows,
extract the per-head evidence-mass pattern; re-run restoration clamping `back_20`'s evidence
up *per head* to the local pattern instead of uniformly. Recovered fraction >> 0.28 ⇒
instrument limitation confirmed; ≈ 0.28 ⇒ evidence for a second positional channel within
displacement.

**Stage 3 — prefill-only closure (E6′).** Close exemplar-answer keys during prefill only
(release at decode) and decode-only (existing result) on the mmlu erosion arm at one depth.
Prefill-only closure blocking the mode where decode-only did not = causal localization of
installation; per-token prefill capture (which prompt positions read the exemplars) then
illustrates the route.

**Stage 4 — figure.** Token heatmap of one displacement pair from the stored rows
(all-layer mean), rendered with the existing figure style; candidate camera-ready panel.

## 3. Files likely to be modified

- `src/probes/context_fatigue/attention_capture.py` — row storage option
- `scripts/context_fatigue/run_competition_sweep.py` (`--close-arms` path) — measured-span
  closure input
- `scripts/context_fatigue/run_evidence_clamp.py` — per-head clamp target pattern
- `scripts/context_fatigue/run_format_erosion.py` / clamp plumbing — prefill-only closure
  window
- `src/probes/context_fatigue/paper_figures.py` — heatmap panel builder
- new driver only if the capture pass doesn't fit an existing one; multi-word filename

## 4. Non-goals

- No per-token *observational* claims in the paper's main results (the §4.2 caution stands;
  captured rows are targeting/illustration, interventions carry claims).
- No retraining, no SAE work, no new datasets, no OLMo re-runs beyond the three targeted
  arms (artifacts for the committed OLMo panels are gone; Stages 1–3 run on whichever box
  is available — Qwen replicates 1–2 fully; Stage 1's OLMo version needs the e3 panel
  regenerated from seed).
- No change to committed results or numbers.md rows.

## 5. Operational constraints

- GPU required for all stages (7B bf16, eager attention for capture).
- Same seeds/panel construction as the committed runs; paired analyses via
  `paired_accuracy_gap` (10,000 draws); overflow guard semantics unchanged.
- Full-tensor row storage behind a flag; default all-layer mean row to keep artifacts small.
- Closure arms must keep the size-matched random control and parse-rate check (void
  conditions as in e3c brief).

## 6. Acceptance criteria

- Stage 0: capture row for a smoke probe matches the framework's `output_attentions`
  reconstruction (max |Δ| at the E3-attention tolerance); span shares recomputed from the
  stored row equal the stored aggregates.
- Stage 1: expanded-closure arm reports rescue, control, and residual-vs-random with CIs;
  outcome interpreted per the e3c brief's rescue/null/void grid.
- Stage 2: per-head restoration reports recovered fraction with CI against the 0.28 baseline.
- Stage 3: prefill-only vs decode-only closure compliance ladder at matched depth, n≥40.
- Stage 4: heatmap panel builds from stored rows via `--only appendix` pipeline.

## 7. TDD (test-forward)

Write tests first:
- row-vs-aggregate consistency (stored row → span shares == stored shares) on a fixture;
- hot-set span construction (merging, evidence-turn exclusion, token budget cap) on synthetic
  token streams;
- prefill-only closure window: clamp active during prefill positions, exactly identity at
  decode steps (bit-identical logits on a no-op scale);
- per-head clamp target: solver hits per-head targets within tolerance on a tiny model stub.

Expected: existing tests untouched and passing; new tests fail before implementation, pass
after; no test depends on gitignored results/.
