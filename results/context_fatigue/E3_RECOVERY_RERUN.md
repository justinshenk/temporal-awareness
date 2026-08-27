# E3 / per-head artifact recovery re-run — 2026-08-27

The original OLMo competition and per-head artifacts (`e3_competition/`, `e3_attention/`,
`e3c_competitor_close/`, `e1_heads_all/`, `e3_heads_all/`, `head_structure.json`) were lost
with their box; the committed reports (`E3_COMPETITION.md`, `E3C_COMPETITOR_CLOSE.md`,
`E4_HEAD_STRUCTURE.md`) carried the numbers with no raw artifacts behind them. This re-run
(per the recovery task in `tasks/todo.md`) regenerates all six under the reports' own
invocations: `run_competition_sweep.py` (defaults / `--attention-only` / `--close-arms` /
`--per-head`), `run_distance_sweep.py --per-head`, `analyze_head_structure.py`. Seed 42,
`allenai/OLMo-2-1124-7B-Instruct`.

## Anchor reproduction

| claim | committed | this re-run |
|---|---|---|
| panel construction | 365 used, 15 starved, 4 skips, 0 leaks | identical |
| disjoint arm accuracy | 0.485 | **0.485** (exact) |
| near_dup / random accuracy | 0.425 / 0.512 | 0.422 / 0.515 (sweep); 0.425 / 0.512 (closure run, exact) |
| evidence share @L24, disjoint | 0.03409 / question 0.11512 | 0.0341 / 0.1152 |
| closure row | near_dup 0.425 → comp_close ~0.476, 59% of penalty | 0.425 → 0.474, 56% (comp_close matches the surviving `e3c_hot_close` 0.474 exactly) |
| random-closure control | ~0.406 | 0.405 |
| displacement drain, all-layer mean | 0.0455 | **0.0455** (exact) |
| competition drain, all-layer mean | 0.0022 | **0.0022** (exact) |
| cross-family DiD (OLMo−Qwen penalty) | ns | +0.063 [−0.006, +0.134], ns (4,000 draws, vs surviving `qwen_e3_competition/`) |

The joint fit and remaining Appendix F cells are recomputable from the committed
`turns.csv`/`heads.csv` in these dirs; `head_structure.json` is the analyzer's output on the
new CSVs.

Artifacts: `e3_competition/`, `e3_attention/`, `e3c_competitor_close/`, `e1_heads_all/`,
`e3_heads_all/`, `head_structure.json`, per-run logs `e3_*_run.log`, `e1_heads_all_run.log`,
`head_structure_analysis.log`, preflight `e3_competition_preflight/`.
