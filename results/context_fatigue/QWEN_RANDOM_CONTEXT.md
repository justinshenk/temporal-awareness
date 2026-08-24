# Q7 — accumulation null (random-subject stream) on Qwen2.5-7B-Instruct

**Verdict: CONFIRMED — accumulation alone costs nothing. A random-subject MMLU stream
shows no accuracy decay with fill (slope −0.057 [−0.238, +0.135], ns), flat to a full
window; the committed Qwen adherence-canary run already showed no compliance decay in the
same regime.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · max_ctx 4096, fill target 0.88 ·
12 sessions per mode · artifacts `results/context_fatigue/qwen_random_context/`
(`turns.csv`, `accuracy_by_fill.csv`, `summary.json`; smoke run in
`qwen_random_context_preflight/` — the driver has no `--preflight`, so a 1-session smoke
substituted, grader inspected on real replies) · driver `run_random_context.py`.

## Results (836 turns; parse rate 0.998 under `--max-new 8`)

| fill bin | random acc (n) | coherent acc (n) |
|---|---|---|
| 0–20% | 0.75 (79) | 0.91 (106) |
| 20–40% | 0.75 (83) | 0.96 (114) |
| 40–60% | 0.68 (77) | 0.92 (113) |
| 60–80% | 0.73 (73) | 0.91 (111) |
| 80–100% | 0.75 (32) | 0.79 (48) |

Case-resampled fill slopes (4,000 draws): random **−0.057 [−0.238, +0.135] ns**; coherent
−0.090 [−0.200, +0.022] ns. Position effect: <5 vs ≥5 turns Δ = +0.01 (random). Overall:
random 0.727, coherent 0.911 — the coherent stream's same-subject ICL advantage (+0.18)
is the masking effect the random arm removes, and with it removed there is still nothing
for "context rot" to explain: the random stream is flat through 100% fill.

## Adherence canaries

Covered by the committed Qwen run `instruction_adherence/INSTRUCTION_ADHERENCE.md`
(same accumulation harness, 32k ctx, fill to 0.92): no adherence decay in any arm, with
the report's own caveat that the prefix canary sat at a floor of ease. Q5's clamp
supplies the causal complement on this box: compliance moves when the instruction's
attention mass is *forced* down, not when context merely accumulates.

## Queue closure

This completes Q1–Q7. Cross-family summary of the reproduction lives in the per-experiment
reports (`QWEN_E1_DISTANCE_SWEEP`, `QWEN_E1C_EVIDENCE_CLAMP`, `QWEN_E1F_SHARE_SWEEP`,
`QWEN_E3_COMPETITION`, `QWEN_E5_SYSTEM_CLAMP`, `QWEN_E6_FORMAT_EROSION`, this file), plus
the interleaved E7 reports (`E7_FORMAT_PATCH`, `QWEN_E7_FORMAT_PATCH`).
