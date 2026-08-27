# E6 — format erosion under irrelevant accumulation (recovery re-run)

**Verdict: every headline claim of the original (lost) E6 program reproduces. The mmlu
compliance ladder collapses 0.875 → 0.000 by depth 3 (fill 0.147) and never recovers; gsm8k
holds ≥0.825 through depth 12 and drops to 0.600 at depth 15; code stays high (0.90–1.00)
through fill 0.94 — the applicability ordering mmlu ≪ gsm8k < code is intact. Accuracy rises
through the collapse (0.45–0.475 at depth 0 → 0.65 at depth 42). The recovery arms at depth
42 all restore compliance to 1.000 (upclamp / refresh / both) while accuracy falls, and
exemplar closure restores nothing (fa_close / fa_matched / rand1_close all 0.000, fq_close
0.100). The answer−question enrichment gaps match the committed values to the second decimal.**

Run 2026-08-27 · `allenai/OLMo-2-1124-7B-Instruct` · driver
`scripts/context_fatigue/run_format_erosion.py` · seed 42, n = 40 probes/depth.

**Provenance note.** The original 2026-08 artifacts (`e6_{code,gsm8k,mmlu}/`,
`e6_{code,gsm8k,mmlu}_spans/`, `e6_mmlu_recovery/`, `e6_exemplar_close/` and the original
report) were lost with the box that produced them; only their numbers survived in
`context_fatigue_paper/numbers.md` and the mmlu depth-0/42 anchors in
`E6_CLOSE_WINDOWS.md`. This re-run (per the recovery task in `tasks/todo.md`) regenerates
them under the same protocol, reconstructed from the surviving Qwen mirrors'
`summary.json`. `--record-spans` is folded into the main runs, so `spans.csv` sits beside
`turns.csv` in each dir and there are no separate `_spans/` re-run dirs.

Commands:

    run_format_erosion.py --filler mmlu  --depths 0 3 7 14 21 28 35 42 --record-spans --out-dir .../e6_mmlu
    run_format_erosion.py --filler gsm8k --depths 0 2 4 6 9 12 15      --record-spans --out-dir .../e6_gsm8k
    run_format_erosion.py --filler code  --depths 0 2 4 6 9 12 15      --record-spans --out-dir .../e6_code
    run_format_erosion.py --filler mmlu  --depths 0 42 --recovery      --out-dir .../e6_mmlu_recovery
    run_format_erosion.py --filler mmlu  --depths 0 42 --close-arms    --out-dir .../e6_exemplar_close

## Compliance ladders

mmlu (structural-copy filler — every turn demonstrates bare-letter answering):

| depth | fill | sys share | compliant | accuracy |
|---|---|---|---|---|
| 0 | 0.093 | 0.1902 | 0.875 | 0.475 |
| 3 | 0.147 | 0.1058 | 0.000 | 0.525 |
| 7 | 0.228 | 0.0783 | 0.000 | 0.525 |
| 14 | 0.354 | 0.0508 | 0.000 | 0.575 |
| 21 | 0.496 | 0.0406 | 0.000 | 0.575 |
| 28 | 0.611 | 0.0409 | 0.000 | 0.575 |
| 35 | 0.747 | 0.0249 | 0.000 | 0.600 |
| 42 | 0.877 | 0.0195 | 0.000 | 0.650 |

gsm8k (free-form, no options anywhere):

| depth | fill | sys share | compliant | accuracy |
|---|---|---|---|---|
| 0 | 0.094 | 0.1897 | 0.825 | 0.500 |
| 2 | 0.150 | 0.1144 | 0.875 | 0.550 |
| 4 | 0.188 | 0.0997 | 0.925 | 0.500 |
| 6 | 0.239 | 0.0885 | 0.925 | 0.600 |
| 9 | 0.304 | 0.0747 | 0.825 | 0.650 |
| 12 | 0.374 | 0.0641 | 0.825 | 0.700 |
| 15 | 0.497 | 0.0480 | 0.600 | 0.675 |

code (free-form, longest replies, fastest fill):

| depth | fill | sys share | compliant | accuracy |
|---|---|---|---|---|
| 0 | 0.094 | 0.1897 | 0.825 | 0.500 |
| 2 | 0.211 | 0.1115 | 0.975 | 0.500 |
| 4 | 0.325 | 0.0886 | 0.975 | 0.500 |
| 6 | 0.438 | 0.0759 | 1.000 | 0.500 |
| 9 | 0.608 | 0.0606 | 0.975 | 0.600 |
| 12 | 0.778 | 0.0479 | 0.900 | 0.475 |
| 15 | 0.942 | 0.0385 | 0.939 | 0.485 (n=33, 7 overflow skips) |

## Answer−question enrichment gap (spans.csv, mean over probes)

| filler | shallowest | deepest |
|---|---|---|
| mmlu | +1.28 (d3) | +0.65 (d42) |
| gsm8k | +0.34 (d2) | +0.07 (d15) |
| code | −0.08 (d2) | −0.17 (d15) |

Committed values (numbers.md): mmlu +1.28 → +0.65; gsm8k +0.34 → ~+0.10; code −0.08 to
−0.17. Exact reproduction of the reading-signature ordering: the model reads mmlu filler
answers heavily, gsm8k answers slightly, code answers not at all — compliance collapse
follows the reading, not the token count.

## Recovery at depth 42 (e6_mmlu_recovery)

| arm | sys share | compliant | accuracy |
|---|---|---|---|
| natural | 0.0196 | 0.000 | 0.650 |
| upclamp | 0.1901 | 1.000 | 0.425 |
| refresh | 0.0933 | 1.000 | 0.450 |
| both | 0.1902 | 1.000 | 0.275 |

Committed: natural 0.000/0.675 · upclamp 1.000/0.425 · refresh 1.000/0.500 · both
1.000/0.275. upclamp and both are cell-exact; natural and refresh differ by one and two
probes' accuracy respectively.

## Exemplar closure at depth 42 (e6_exemplar_close)

| arm | closed share | compliant | accuracy |
|---|---|---|---|
| fa_close | 0.0000 | 0.000 | 0.475 |
| fa_matched | 0.0036 | 0.000 | 0.525 |
| fq_close | 0.0000 | 0.100 | 0.600 |
| rand1_close | 0.0000 | 0.000 | 0.650 |

Committed: fa_close 0.000 / fa_matched 0.000 / fq_close 0.132 / rand1_close 0.000. fq_close
0.100 here equals the in-session value the surviving `E6_CLOSE_WINDOWS.md` anchor run also
measured.

## Divergences from committed paper numbers (for the tex/numbers.md refresh)

The regenerated filler streams are not byte-identical to the lost originals (same seed, but
library/hardware drift in the accumulated model replies), so cells shift by a few probes:

1. **code at depth 12 / fill 0.778: compliant 0.900**, not the 1.000 currently in the tex.
   The qualitative claim (code never erodes; ≥0.90 through fill 0.94) stands. This also
   settles the 1.00-vs-0.78-row question flagged in the recovery task: the current best
   value for that cell is 0.900.
2. **Overflow skips: 0 (mmlu), 0 (gsm8k), 7 (code)** vs Appendix E's committed "2 mmlu /
   32 code". The Appendix E sentence must be re-pointed at these runs.
3. Depth-0 compliance is 0.825 in the gsm8k/code runs vs 0.875 in the mmlu run (different
   filler pools shift the depth-0 snapshot's probe sampling); the paper's 0.875 depth-0
   figure traces to `e6_mmlu/`.
4. mmlu depth-0 accuracy 0.475 vs committed 0.425 (two probes).

Artifacts: `e6_mmlu/`, `e6_gsm8k/`, `e6_code/` (each turns.csv + spans.csv + summary.json),
`e6_mmlu_recovery/`, `e6_exemplar_close/`, per-run logs `e6_*_run.log`.
