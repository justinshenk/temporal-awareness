# The conditionality index's computation pole: constant self-steer recovers 0.000 on GSM8K

**Verdict: the register-vs-procedure prediction lands exactly.** The same constant mean-shift
estimator that recovers 0.66–0.71 of a DDXPlus adapter's effect in two Qwen chat models
recovers **0.000** of the MetaMath LoRA's GSM8K capability at every layer tried — with the
striking rider that GSM8K's per-case shifts are *also* mostly constant in direction
(mean cos 0.910 with their mean, vs 0.99 on DDXPlus), so the shift's constancy does not
predict what a constant can carry. The conditional component is small in norm and entirely
load-bearing.

Run 2026-08-21 · Llama-2-7b (NousResearch mirror) + `LoRA-TMLR-2024/metamath-lora-rank-16-
alpha-32` · 100 GSM8K test problems, MetaMath-faithful prompt and answer parse, max_new 256,
greedy · driver `scripts/attribution/selfsteer_gsm8k.py` · artifacts
`results/attribution/selfsteer_gsm8k/`.

| arm | accuracy | parse |
|---|---:|---:|
| floor (base) | 0.000 | 0.00 |
| ceiling (base + MetaMath LoRA) | 0.620 | 0.85 |
| self-steer L8 / L16 / L20 / L24 (mean shift, decode-time, α=1) | 0.000 each | 0.00 each |

**Recovered fraction: 0.000** (best layer), against 0.71 / 0.68 (DDXPlus, Qwen-7B/1.5B, same
estimator, same protocol) and ~1.0 (Paper B's E6 format mode). The parse rate is the tell for
*how* it fails: on DDXPlus the constant vector installed both the answer format and the
discrimination; here it installs neither — the base never reaches "The answer is:" with or
without the vector, consistent with the program's prior finding that static injection washes
out or compounds under self-generated CoT.

## The estimator ladder on GSM8K (all against the 0.62 ceiling)

| estimator | input-conditionality | recovery |
|---|---|---|
| constant mean shift (this run) | none | **0.000** |
| ridge maps / local refit / DAgger / DAS (committed rungs) | per-token linear | +0.03..+0.12 (bounded ≤0.23 @95%) |
| the adapter itself | full | 1.0 by definition |

On DDXPlus the *bottom* rung already carries 0.71. That is the register-vs-procedure
distinction as a measurement: register components (mode, format, readout policy over features
the base already computes) live in the constant term; procedural components (multi-step
computation the base cannot do) are invisible to it, and mostly invisible even to conditional
linear maps.

## Conditionality index, current table

| task / substrate | index (constant-recoverable fraction) |
|---|---:|
| format instruction / OLMo-2-7B-Instruct (Paper B E6) | ~1.0 |
| DDXPlus MCQ / Qwen2.5-7B-Instruct | 0.71 |
| DDXPlus MCQ / Qwen2.5-1.5B-Instruct | 0.68 |
| GSM8K / Llama-2-7b + MetaMath | **0.000** |

Caveats: floors differ in kind (Qwen floors are constant-letter collapses at their own chance;
the Llama base floor is genuine incapacity at parse 0.00); one α per arm here (the DDXPlus arm
showed α-sensitivity at deep layers); index measured at the best single layer, and the
multi-layer probe on DDXPlus showed stacking constants only hurts.
