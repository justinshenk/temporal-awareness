# Q3 — E1f evidence-share dose-response on Qwen2.5-7B-Instruct

**Verdict: CONFIRMED — a smooth, monotone share→accuracy dose-response with no knee,
continuing below the back_20 share. The slope independently reproduces Q1's
attention→accuracy coefficient.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · max_ctx 4096 · all-layer pooled
readout (`--reference-layer 0..27`) · 192 probes at `local` position, 35 filler turns ·
levels calibrated from Q2's measured naturals (local 0.0390, back_20 0.0088) · artifacts
`results/context_fatigue/qwen_e1f_share_sweep/` (`turns.csv`, `summary.json`) · driver
`run_evidence_clamp.py --levels 0.0335 0.0240 0.0172 0.0123 0.0088 0.0070`.

## Results

Forced-choice letter-logit scoring; solver hit every target (achieved = target to ≤1e-4).
11/192 items had natural share below the top level (skipped upward, per design), hence
n=181 there.

| level (achieved share) | n | accuracy | natural − level, paired (95% CI) |
|---|---|---|---|
| natural (0.0388) | 192 | 0.667 | — |
| 0.0335 | 181 | 0.646 | +0.011 [−0.017, +0.039] ns |
| 0.0240 | 192 | 0.641 | +0.026 [−0.010, +0.068] ns |
| 0.0172 | 192 | 0.573 | **+0.094 [+0.042, +0.146]** |
| 0.0123 | 192 | 0.474 | **+0.193 [+0.125, +0.255]** |
| 0.0088 (= back_20) | 192 | 0.432 | **+0.234 [+0.161, +0.307]** |
| 0.0070 | 192 | 0.380 | **+0.286 [+0.214, +0.359]** |

At the back_20-share level the clamp costs +0.234 — bracketing Q2's independent full-clamp
estimate (+0.167 with donor-targeted per-item shares), and the drop at 0.0070 shows the
curve is still falling **below** the deepest natural share the distance arms produce: no
floor, no knee.

**Slope** (accuracy ~ achieved share over clamped rows, case-resampled 4,000 draws):
+10.44 [+7.59, +13.57] — Q1's independent `correct ~ evidence_share` fit gave +10.7
[+3.5, +17.6]. Two designs (natural variation vs causal clamp), one coefficient.
Piecewise slopes above vs below 0.0088 are +7.6 [+3.8, +11.6] vs +4.2 [−61, +69] (only
1.5 levels below — underdetermined, but nothing suggests a flat regime).

## Notes

- Preflight (`qwen_e1f_share_sweep_preflight/`) verified per-item natural measurement and
  solver convergence before the run.
- Skips: 0 probes; 11 upward-clamp levels skipped by design (`level >= natural`).
- Consistent with the OLMo all-layer re-denomination: dose-response is smooth (no knee),
  reconciling E1c sufficiency with E1e's flat C2 arm via curve position, not curve shape.
