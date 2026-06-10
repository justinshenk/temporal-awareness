# Lockstep dual-forward residual oracle (Step 0 + Step 1, the linchpin)

## 1. Problem statement
Steering (`LinearPrimalSteerHook`, inject `α·Wᵀa` at layer L every decode step) recovered
**0%** of the LoRA's GSM8K capability. We don't yet know if that null is **fundamental**
(residual *values* at a site can't carry the multi-step procedure) or **estimator-driven**
(the fitted `W` is just a bad map). Step 1 settles it with the *oracle* version of the same
intervention: inject the LoRA's **true** residual (not a fitted `W·a`) at layer L, every
decode step, and measure capability recovery. If even the oracle recovers nothing at any L,
the steering null was fundamental. That retroactively licenses Steps 2–5.

## 2. Agreed solution approach — lockstep dual-forward
At each decode step `t`, context `S_t = [prompt, b_1…b_{t-1}]` is **base's own** emitted tokens.
1. Forward **LoRA** (adapter on) on `S_t`; capture residual at the target layer(s), all positions.
2. Forward **base** (adapter off) on `S_t`; **overwrite** the target layer(s) residual with LoRA's.
3. Base emits `b_t` from the patched stream; append; repeat.

Both passes run on the *same base-generated tokens* → positions aligned; context is base's own
→ no in-context-correctness leak; re-applied every step → exact oracle of all-step steering.
Faithful + efficient: the steering hook patches each position **once when first computed**, which
is KV-cache compatible (two caches: one LoRA, one base), so this is **O(T)** per problem.

**Confirmed design call:** copying **all layers** is degenerate-to-LoRA (final-layer residual
overwrite ⇒ base logits = LoRA logits ⇒ trajectory = LoRA greedy). So:

| Run | Inject layers | Role |
|---|---|---|
| Positive control | all (0..N-1) | MUST recover ≈ lora_acc → validates wiring/alignment |
| **Headline** | single L, sweep L | recovery-vs-L curve = the real signal |
| Cumulative | ≤ L, sweep L | sharp vs smeared depth transition (Step 2 preview) |
| Cheap bracket | single-input, answer-pos, teacher-forced | fast per-layer logprob screen to narrow L |

## 3. Metric (Step 0)
- Contrast set: GSM8K, `NousResearch/Llama-2-7b-hf` + `metamath-lora-rank-16-alpha-32`.
  Restrict to **base-fails / LoRA-solves** problems (the recoverable budget). base 0.0 → LoRA 0.6
  on the 50-problem eval (`results/attribution/icl_gsm8k.json`); denominator = the ~30 solved.
- Readout: exact-match accuracy (`extract_pred_number` + `numeric_match`).
- Continuous companion: log-prob of the gold-answer tokens at the answer position.
- recovery(acc) = (acc − base_acc) / (lora_acc − base_acc).

## 4. Files
- NEW `src/probes/attribution/lockstep_oracle.py` — `OverwriteResidualHook`, `lockstep_generate`,
  `run_lockstep_recovery` (control / single / cumulative via a layer-set arg).
- NEW `scripts/attribution/lockstep_patch_gsm8k.py` — CLI: load, build contrast set, run control +
  sweep, write JSON + short report.
- NEW `tests/attribution/test_lockstep_oracle.py` — pure-logic tests on a CPU fake model.
- REUSE `attribution_common.py` (load/data/accuracy), `extraction.PerTokenResidualCapture`,
  `gsm8k_prompts` (templates/scoring). No edits to existing modules expected.

## 5. Non-goals (do NOT change)
- No edits to `steering_hook.py`, `gram_accumulator.py`, the existing steer/refit scripts, or
  the main experiment pipeline. No new deps. Not implementing Steps 2–5 yet (gated on Step 1).
- Not re-deriving `W`; the oracle uses captured residuals directly.

## 6. Operational constraints
- Single GPU, 7B bf16; two KV caches + adapter toggle per step. Start no-cache (obviously correct)
  to pass the control test, then add KV cache for the sweep. Greedy/deterministic throughout.
- max_new bounded (≤256; GSM8K CoT is short). Layer sweep may stage via the cheap bracket if the
  full 32-layer lockstep is too slow — log any narrowing, never silently cap.

## 7. Acceptance criteria
- **AC1 (linchpin):** positive control (all-layers lockstep) reproduces LoRA greedy tokens
  exactly on a sample → control_acc == lora_acc. If not, the apparatus is wrong; stop.
- **AC2:** single-layer + cumulative sweeps produce a recovery curve over L on the contrast set.
- **AC3:** result JSON has base_acc, lora_acc, per-L {single, cumulative} acc + recovery, control.
- **AC4:** unit tests (fake CPU model) green: hook overwrite fires, all-layers ⇒ identical logits
  to the reference forward, decode loop + two-cache bookkeeping correct.

## RESULTS (2026-06-09) — complete; writeup at results/attribution/2026-06-09-lockstep-oracle.md
- AC1 PASS: all-layers control reproduces LoRA exactly.
- Single-layer recovery (contrast acc): L0-12=0, L16=.20, L20=.75, L24=.75, L28/31=.95 (28/31 ~degenerate).
  → null is ESTIMATOR-DRIVEN, residual values carry the procedure (oracle works through 11 base layers @L20).
- Geometry @L20 (base-traj): cos(W·a,δ_true)=0.61, ‖W·a‖/‖δ_true‖=0.80, R²=0.31 (vs R²_te=0.61 on LoRA-CoT).
- Fidelity sweep @L20: recovery(t=0/.5/.8/.9/1)=.05/.40/.70/.70/.75 → graceful, saturates by t≈0.8;
  procedure TOLERATES shift error; ridge map sits below the recovery onset (under-fit, not perfection-needed).
- Nonlinear MLP estimator @L20: val cos 0.63→0.80, R² 0.33→0.64 (better fit) but recovery STILL 0% (ridge 0.05).
  Diagnosis (diagnose_nonlinear_steer.py): generations COHERENT (not off-manifold) but loop/restate w/o computing
  → MSE captures format not computation; closed-loop compounding/distribution-shift. Lever does NOT work feed-forward.
  Strong claim: LoRA capability is patchable by exact residual (oracle) but NOT reproducible by any layer-local
  feed-forward map (carrying signal low-variance + closed-loop-fragile). Next levers: on-policy nonlinear DAgger,
  variance-whitened objective, or multi-layer injection.
- Files: src/probes/attribution/{lockstep_oracle,shift_geometry}.py;
  scripts/attribution/{lockstep_patch_gsm8k,compare_map_vs_oracle,lockstep_fidelity_sweep}.py;
  tests/test_{lockstep_oracle,shift_geometry}.py (14 passing). Note: on branch context-fatigue-datasets (unrelated) — not committed.

## 8. TDD / test expectations (test-forward)
1. Fake model with `.model.layers` (ModuleList) + `lm_head`; deterministic. Test: `OverwriteResidualHook`
   overwrites layer output at given positions; all-layers injection makes the "base" forward's logits
   equal the "lora" forward's logits (the degeneracy = positive control, asserted in miniature).
2. Test `lockstep_generate` decode loop on the fake model: all-layers ⇒ token-for-token == reference
   greedy; single-layer ⇒ runs, returns ids of expected length, stops on eos.
3. GPU validation (script, not pytest): AC1 on 3 real problems before trusting any sweep number.
