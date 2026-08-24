# Q1 — E1 distance sweep on Qwen2.5-7B-Instruct

**Verdict: CONFIRMED — the distance ladder reproduces on a second model family, fully
monotone, with the attention share falling in parallel.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · max_ctx 4096 · artifacts in
`results/context_fatigue/qwen_e1_distance_sweep/` (`turns.csv`, `summary.json`) · driver
`scripts/context_fatigue/run_distance_sweep.py --reference-layer 24 --measure-attention`.
Same panel construction as the committed OLMo run (MMLU filler, DDXPlus probes, explicit
referent, arms share filler and fill exactly).

## Results

n = 192 per arm (§6 asks ≥150) · chance 0.200 · overflow guard skipped **0/192** ·
unparsed **0.0%** (Qwen answers with a bare letter; OLMo's 19.3% unparsed issue does not
arise, so no parsed-only robustness arm is needed). Mean fill 0.676 in every arm.

| arm | distance | n | accuracy | vs `local`, paired (95% CI) |
|---|---|---|---|---|
| `local` | 0 | 192 | **0.630** | — |
| `back_2` | 2 | 192 | 0.531 | +0.099 [+0.052, +0.146] |
| `back_5` | 5 | 192 | 0.516 | +0.115 [+0.057, +0.172] |
| `back_10` | 10 | 192 | 0.505 | +0.125 [+0.068, +0.188] |
| `back_20` | 20 | 192 | 0.469 | +0.161 [+0.094, +0.229] |

Gaps are paired over probes (each probe scored in all 5 arms; `paired_accuracy_gap`,
10,000 draws). Unpaired case-resampled intervals (the OLMo report's convention) also
exclude zero for back_5/10/20; back_2 grazes it [+0.000, +0.193].

**Joint fit, accuracy ~ fill + distance** (case-resampled bootstrap, 4,000 draws):

| predictor | β | 95% CI | significant |
|---|---|---|---|
| distance | **−0.00607** | [−0.01021, −0.00159] | **yes** |
| fill | +0.330 | [+0.117, +0.543] | yes — but **positive** |

The fill coefficient is significant with the *opposite* sign to the dilution confound the
null guards against (deeper snapshots score slightly higher across depth cells, which use
different probes). Fill is byte-identical across arms at each snapshot, so it cannot carry
any of the ladder; the within-`local` null the paper states survives directly: **`local`
flat with fill**, β = +0.275 [−0.181, +0.742], not significant.

## Attention addendum (layer 24)

Evidence share falls with distance — local 0.0156, back_2 0.0074, back_5 0.0068,
back_10 0.0090, back_20 0.0047 — β_distance = −3.48e−4 [−3.81e−4, −3.17e−4], significant
with fill in the fit (fill also drains share: −2.8e−3, significant — dilution). Share
predicts accuracy: β = +10.7 [+3.5, +17.6], significant. Same qualitative picture as
OLMo's E1b: the displaced evidence loses mass, and lost mass tracks the accuracy drop.

## Deviations / notes

- Unlike OLMo (decline saturating by k≈10 with an uptick at 20), Qwen's ladder is
  monotone through k=20.
- Grader validated in preflight (`qwen_e1_distance_sweep_preflight/`): all replies parse;
  Qwen's reply style is a bare letter, occasionally "A) <pathology>".
- Analysis: `joint_fit` / `paired_accuracy_gap` from
  `src/probes/context_fatigue/dilution_analysis.py` (script in session scratchpad).
