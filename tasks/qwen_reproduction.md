# Qwen reproduction — headline experiments only (user-scoped 2026-08-24)

**Goal:** reproduce the paper's headline results on a second model family,
`Qwen/Qwen2.5-7B-Instruct` (cached). Explicitly out of scope (user): post-training
dose-response (needs a public SFT/DPO chain), F90871 SAE clamp (Gemma-Scope-specific),
WildChat signatures (already run on Qwen-2.5 — free), non-headline appendix analyses
(per-head structure, E1d/E1e/E2a) unless a headline result disagrees and needs them.

**Conventions:** all-layer pooled share readout (`--reference-layer 0..27`; Qwen2.5-7B has 28
layers) — no per-family reference-layer choice, consistent with the OLMo all-layer
re-denomination (2026-08-24). `max_ctx` 4096 (same budget as OLMo, so fill is comparable).
Seed 42, same panel constructions, paired bootstrap over cases. HF_HUB_OFFLINE=1. Preflight
every run; validate every grader against real Qwen generations before trusting a ladder
(this program's five voided runs were all format-drift artifacts).

## Queue (strict order; each ~step gated on the previous)

- [ ] **Q0 gate:** OLMo all-layer re-runs finish first (e1f_alllayer RUNNING, e2a_alllayer
      queued), paper robustness edit lands.
- [ ] **Q1 E1 distance sweep** + `--measure-attention`: headline ladder + fill-β null.
      `run_distance_sweep.py --model Qwen/Qwen2.5-7B-Instruct --reference-layer 0..27`.
      Preflight must include answer-extraction check on real replies.
- [ ] **Q2 E1c mass removal** (sufficiency): donor back_20, clamp local. Records Qwen's
      natural all-layer shares → calibrates Q3's ladder.
- [ ] **Q3 E1f dose-response**: 6 levels from Q2's natural down through its back_20 share.
- [ ] **Q4 E3 competition** (paired n=365 panel) + all-layer attention addendum + **E3c
      closure** (eager attention).
- [ ] **Q5 E5 system clamp**: share profile, then the neutral-context clamp arm
      (compliance collapse at graded dose).
- [ ] **Q6 E6 format erosion**: mmlu/gsm8k/code ladders, then recovery arms at the deepest
      depth of whichever arm erodes (do not assume mmlu erodes on Qwen — check first).
- [ ] **Q7 accumulation null**: random-subject MMLU stream, bounded null + adherence canaries.

## Runtime budget (32 GB RTX PRO 4500, bf16)

Q1 ~2h · Q2 ~1h · Q3 ~2h · Q4 ~4–6h · Q5 ~2–3h · Q6 ~3–4h · Q7 ~3–5h → **~17–23 GPU-hours**,
sequential. Wall-clock ~2 days with preflights and grader iteration.

## Family adaptation notes

- Capture already validated on Qwen2 GQA exactly 0.0 (`attention_capture.py` tests).
- Clamp biases the additive mask — family-generic; sdpa needs the padded-token trick
  (already in drivers); closure arms need eager attention.
- Qwen chat template injects a default system prompt when none is supplied — E5/E6 pass
  explicit system prompts, so verify the rendered transcript in preflight.
- Recheck `max_new` truncation per experiment: Qwen replies are longer-winded than OLMo's.
- Artifacts under `results/context_fatigue/qwen_*/`; one report per experiment quoting
  artifact filenames, n per cell, skip counts, verdict (brief §9 conventions).
