# Q5 — E5 neutral-context system clamp on Qwen2.5-7B-Instruct

**Verdict: CONFIRMED — clamping the system span's attention share collapses instruction
compliance at graded dose while accuracy is untouched (it in fact rises slightly), with a
canary-ordered collapse: prefix first, suffix next, the forbid rule never.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · eager attention · all-layer pooled
readout (driver default `--reference-layers` = every layer) · neutral-context arm
(`--cold-start-cases 0 --arms undemonstrated`) · n = 120 items · artifacts
`results/context_fatigue/qwen_e5_system_clamp/` (`turns.csv`, `summary.json`; profile log
`qwen_e5_system_clamp_profile.log`) · driver `run_system_clamp.py`.

## Profile (measured first; the ladder is derived from it, not imported)

Natural system-span share: 0.1891 cold-start → 0.0861 (1 case) → 0.0486 (8 cases). Qwen
retains ~2.3× OLMo's deep-context share (0.021). Ladder: 0.15, 0.12, 0.09, 0.07, 0.0486
(the accumulation floor), 0.038 (one below).

## Results (0 overflow skips, 0 levels skipped ≥ natural, parse rate 1.000, fill 0.096)

| level (achieved) | prefix | suffix | forbid | accuracy |
|---|---|---|---|---|
| natural (0.191) | 1.000 | 1.000 | 1.000 | 0.567 |
| 0.150 | 0.983 | 1.000 | 1.000 | 0.583 |
| 0.120 | **0.100** | 0.917 | 1.000 | 0.583 |
| 0.090 | 0.000 | **0.233** | 0.975 | 0.617 |
| 0.070 | 0.000 | 0.017 | 0.983 | 0.625 |
| 0.0486 | 0.000 | 0.000 | 1.000 | 0.633 |
| 0.038 | 0.000 | 0.000 | 1.000 | 0.658 |

Paired drops vs natural (10,000 draws): prefix +0.90* at 0.12 and +1.00* below; suffix
+0.08* at 0.12, +0.77* at 0.09, ~+1.00* below; **forbid never moves** (all levels ns,
≤0.03). Accuracy *rises* under the clamp: natural − clamped = −0.05* at 0.09 through
−0.09* [−0.16, −0.03] at 0.038 — the zero-sum reallocation credits the case text, the same
direction OLMo showed (+0.058 ns there), significant here.

## Reading

- The causal compliance claim reproduces: attention mass on the instruction is necessary
  for compliance in a context where nothing competes, and the collapse is dose-graded —
  Qwen's thresholds sit higher than OLMo's (prefix dies by 0.12 on a 0.19 natural; OLMo's
  compliance died at 0.050 on a 0.165 natural).
- The collapse is **canary-ordered**, not uniform: positive formatting duties (start with
  a marker, end with a fixed string) die in sequence; the negative duty (never name a
  diagnosis in the canary slot) survives total starvation of the span. A prohibition
  appears to be carried differently than a production rule — worth an appendix note; E7's
  installation story is consistent (a negative constraint has no format attractor to lose
  to).
- Competence-vs-compliance dissociation is if anything sharper than OLMo's: compliance
  fully destroyed while accuracy significantly improves.
