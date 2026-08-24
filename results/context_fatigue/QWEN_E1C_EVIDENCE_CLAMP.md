# Q2 — E1c evidence-mass clamp on Qwen2.5-7B-Instruct

**Verdict: CONFIRMED — mass is sufficient for the whole distance effect. Clamping `local`
evidence to `back_20`'s share reproduces 107% of the distance gap; the clamped arm is
statistically indistinguishable from `back_20`.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · max_ctx 4096 · all-layer pooled
share readout (`--reference-layer 0..27`) · artifacts in
`results/context_fatigue/qwen_e1c_evidence_clamp/` (`turns.csv`, `summary.json`) · driver
`scripts/context_fatigue/run_evidence_clamp.py --clamp-arm local --donor-arm back_20`.

## Results

192 probes, each scored under all three conditions (576 rows, 0 skips). Forced-choice
letter-logit scoring — no free-text grader in this design.

| condition | n | accuracy | evidence share (all-layer) |
|---|---|---|---|
| `local` | 192 | 0.630 | 0.0390 |
| `back_20` | 192 | 0.474 | 0.0088 |
| `local_clamped` | 192 | 0.464 | 0.0087 |

Mean clamp scale 0.119; solver hit the per-item donor target (achieved 0.0087 vs donor
0.0088).

**Paired gaps** (`paired_accuracy_gap`, 10,000 draws, items shared exactly):

| contrast | Δacc | 95% CI |
|---|---|---|
| `local` − `back_20` (the distance gap) | +0.156 | [+0.089, +0.224] |
| `local` − `local_clamped` (mass alone, position fixed) | +0.167 | [+0.104, +0.229] |
| `local_clamped` − `back_20` (residual) | −0.010 | [−0.057, +0.037] |

Sufficiency = 106.7% of the distance gap from mass alone; the residual CI is centred on
zero. On OLMo the all-layer re-denomination gave ~91% with a CI including full
reproduction — Qwen lands at full reproduction directly.

## Calibration output for Q3 (E1f dose-response)

Qwen natural all-layer shares measured here: **local 0.0390**, **back_20 0.0088**.
Ladder per the task spec (≈0.86× natural, geometric through the back_20 share, one level
below): `--levels 0.0335 0.0240 0.0172 0.0123 0.0088 0.0070`.
