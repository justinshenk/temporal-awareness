# Q6 — E6 format erosion on Qwen2.5-7B-Instruct

**Verdict: the applicability ordering CONFIRMS (mmlu ≫ gsm8k ≈ code ≈ no erosion), the
demonstrated-answer reading signature CONFIRMS (answer-over-question enrichment positive
exactly where erosion occurs, negative where it does not), and re-presenting beats
re-weighting — more decisively than on OLMo, where upclamp fully restored and here it only
partially does. What differs: Qwen's erosion is component-wise and partial, not total, and
its arm-level attention shares happen to straddle Q5's causal collapse thresholds.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · n = 40 probes/depth · parse rate
1.000 in all arms · artifacts `results/context_fatigue/qwen_e6_{mmlu,gsm8k,code}/`,
spans re-runs `qwen_e6_{mmlu,code}_spans/` (ladders reproduce the committed runs exactly),
recovery `qwen_e6_mmlu_recovery/` · driver `run_format_erosion.py`, depths as the committed
OLMo ladders (mmlu +42), explicit `--depths` throughout.

## Ladders (fully_compliant · components where they separate)

**mmlu** (n=40/depth, depth 42 n=30 — 10 overflow skips at fill 0.91):

| depth | fill | compliant | ANSWER: marker | SUPPORTING | grounded | accuracy |
|---|---|---|---|---|---|---|
| 0 | 0.094 | 1.000 | 1.000 | 1.000 | 0.488 | 0.775 |
| 3 | 0.157 | 0.000 | 0.000 | 1.000 | 0.348 | 0.750 |
| 21 | 0.503 | 0.000 | 0.000 | 0.950 | 0.292 | 0.700 |
| 28 | 0.653 | 0.000 | 0.000 | 0.375 | 0.133 | 0.750 |
| 42 | 0.908 | 0.033 | 0.033 | 0.533 | 0.205 | 0.700 |

**gsm8k**: compliance **1.000 at every depth** through fill 0.839 (OLMo: 0.600 by fill
0.48). **code**: compliance **1.000 at every depth** through fill 0.922 (depth 15 n=17,
23 overflow skips — Qwen's long code replies push past the window; the surviving panel is
the shorter-probe end, same caveat as OLMo's code arm). Accuracy is flat in every arm
(0.625–0.775); competence is never the casualty.

**The erosion is staged, not total**: the `ANSWER:` marker dies instantly and completely at
3 mmlu turns (as fast as OLMo's whole format), while the SUPPORTING duty survives to
mid-depth and only partially erodes deep — the same duty-by-duty ordering E5's canaries
showed causally (prefix first, content duties later, prohibition never). OLMo lost
everything at once.

## Attention: shares, and the demonstrated-answer signature

System-span share falls with accumulation in every arm while per-token enrichment rises in
every arm (to 4.9 mmlu / 8.0 gsm8k / 9.4 code) — as on OLMo, dilution is arithmetic, not
neglect. Two Qwen-specific observations:

1. **Arm shares straddle Q5's causal thresholds.** Deep shares: mmlu 0.048–0.092;
   gsm8k/code 0.112–0.145. Q5's clamp put the prefix-duty collapse between shares 0.15 and
   0.12. Unlike OLMo — where code held compliance at half the share at which mmlu had
   collapsed, the paper's dissociation — Qwen's compliant arms sit at-or-above the causal
   threshold and its eroding arm below it. A mass account is not excluded by Qwen's E6 the
   way OLMo's excluded it (not established either: the arms differ in what they demonstrate
   AND in where their share lands; and Q5's canaries are not this format).
2. **The demonstrated-answer reading signature reproduces** (spans re-runs, all-layer,
   per-token enrichment): filler-**answer** minus filler-**question** enrichment is **+0.87
   to +1.49** in mmlu at every depth (OLMo: +1.28) and **−0.15 to −0.24** in code (OLMo:
   negative) — the demonstrations are preferentially read exactly where erosion occurs.
   Filler questions are near-ignored (≈0.17 mmlu, ≈0.36 code); probe enrichment grows with
   depth in both arms.

## Recovery at depth 42 (mmlu; upclamp n=30, refresh/both n=26 — overflow drops)

| arm | system share | compliant | accuracy |
|---|---|---|---|
| natural depth 42 | 0.064 | 0.067 | 0.633 |
| `upclamp` (to cold-start 0.217) | 0.199 | **0.733** | 0.667 |
| `refresh` (restate in last user turn) | 0.163 | **1.000** | 0.615 |
| `both` | 0.217 | **1.000** | 0.654 |

Re-presenting fully restores; re-weighting alone gets 0.733 — on OLMo upclamp restored
1.000 (at accuracy 0.675→0.425). On Qwen neither lever costs accuracy (0.615–0.667 vs
0.633 natural). The marker and SUPPORTING components recover together in every arm.

## Cross-family summary

Same thesis, softer failure mode: what the context demonstrates — not attention dilution —
sets whether a format instruction erodes, and demonstrated answers are preferentially read
where it does. Qwen's stronger instruction adherence shows up everywhere: no gsm8k erosion,
marker-only mmlu erosion, full recovery by restatement at no accuracy cost, and (per the
filler-mode finding logged in QWEN_E7_FORMAT_PATCH.md) a tendency to over-generalize the
format onto non-patient turns against a local instruction.
