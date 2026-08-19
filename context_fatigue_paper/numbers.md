# `numbers.md` — every number in the context-fatigue abstract, mapped to its artifact

**Rule (inherited from `papers/register_vs_procedure/numbers.md`): no number enters the tex
without a row here.** A cell whose artifact cannot be opened reads "not measured", not a number.

`results/` is gitignored, so "artifact" means the file on the GPU box; the committed report that
quotes it is the durable record. All causal runs: `allenai/OLMo-2-1124-7B-Instruct`, layer 24,
seed 42, bootstrap 10,000 draws resampling **cases**.

## Figures

| # | figure | builder | source artifact(s) |
|---|---|---|---|
| 1a | distance ladder, accuracy + evidence share | `fig_distance_ladder` | `e1_distance_sweep/turns.csv`, `e1_with_attention/turns.csv` |
| 1b | share→accuracy dose-response | `fig_mass_dose` | `e1f_share_knee/turns.csv` (balanced panel), `e1c_evidence_clamp/` for the diamond |
| 1c | competition arms | `fig_competition` | `e3_competition/turns.csv` |
| 2a | attention reallocation vs fill | `fig_attention` | `olmo_attention/` (see `CONTEXT_ROT_ATTENTION.md`) |
| 2b | WildChat homogeneity | `fig_wildchat` | `wildchat_homogeneity/WILDCHAT_HOMOGENEITY.md` (documented constants) |
| 2c | calibration gap | pre-existing `calibration_gap.pdf` | pooled DDXPlus MCQ streams, n=154 |

Regenerate: `uv run python scripts/context_fatigue/make_paper_figures.py`.

## §4 — the distance sweep

| claim | value | artifact / report |
|---|---|---|
| `local`→`back_20` accuracies | 0.464 / 0.359 / 0.292 / 0.250 / 0.276, n=192/arm | `e1_distance_sweep/`, `E1_DISTANCE_SWEEP.md` |
| mean fill identical across arms | 0.688 | same |
| overflow skips | 0 of 192 | same |
| distance β | −0.00761 [−0.01173, −0.00346] | same |
| fill β | −0.00725 [−0.21006, +0.18658] | same |
| `local` flat with fill | β = −0.294 [−0.767, +0.184] | same |
| parsed-only ladder | 0.524 / 0.413 / 0.397 / 0.343 / 0.338 | same |

## §5 — mass mediation

**All contrasts PAIRED** (`paired_accuracy_gap`), recomputed 2026-08-19 into
`results/context_fatigue/dilution_paired.json` by `scripts/context_fatigue/analyze_dilution_paired.py`.
The unpaired intervals previously reported are retained in that JSON for audit.

| claim | value | artifact / report |
|---|---|---|
| evidence share falls with distance | 0.0408 → 0.0124, r = −0.83 | `e1_with_attention/`, `E1_MECHANISM.md` |
| within-arm share↔accuracy trap | β = −11.2 [−20.0, −2.6] | same |
| E1c sufficiency | +0.2021 [+0.1379, +0.2672], paired n=174 | `e1c_evidence_clamp/` |
| E1c lands on `back_20` | −0.0249 [−0.0833, +0.0345] | same |
| **114%** of the displacement penalty | 0.2021 / 0.1772 | same |
| E1d necessity (partial) | +0.0546 [+0.0172, +0.0977] | `e1d_evidence_rescue/` |
| E1d residual | +0.1226 [+0.0536, +0.1925] | same |
| E1f endpoint | +0.1985 [+0.1145, +0.2824], n=131 balanced | `e1f_share_knee/` |
| E1f largest adjacent step | +0.053 (natural → 0.036) | same |
| E1f slope | 6.29 accuracy per unit share, R²=0.966 | derived from same |
| E1e matched-token shares | 0.0104 vs 0.0108 | `e1e_dissociation/` |
| E1e C2 (64% mass cut) | +0.0260 [−0.0417, +0.0938] | same |
| E2a natural share / accuracy | 0.258 / 0.545, n=110/level | `e2a_mass_clamp/`, `E2A_MASS_CLAMP.md` |
| E2a cost at 0.15 | +0.164 [+0.036, +0.291] | same |

## §6 — competition

| claim | value | artifact / report |
|---|---|---|
| arm accuracies | random 0.512 / disjoint 0.485 / near_dup 0.427, paired n=365 | `e3_competition/`, `E3_COMPETITION.md` |
| shared options per arm | 0.80 / 0.00 / 3.75 of 5 | same |
| `random` − `near_dup` | +0.0849 [+0.0301, +0.1397] | same |
| `disjoint` − `near_dup` | +0.0575 [+0.0055, +0.1123] | same |
| `random` − `disjoint` | +0.0274 [−0.0192, +0.0740] n.s. | same |
| shared-options β / fill β | −0.0208 [−0.0392, −0.0029] / −0.2844 [−0.6396, +0.0751] | same |
| control agreement with E1 `local` | +0.049 [−0.037, +0.134]; +0.021 [−0.065, +0.107] | same |
| parsed-only | +0.0818 [+0.0252, +0.1384], n=318 | same |
| gold leaks / starved / overflow | 0 / 15 / 4 | `e3_competition/summary.json` |
| **evidence share unchanged** | −0.00027 [−0.00088, +0.00035] | `e3_attention/` |
| mass-predicted share of the effect | 2.0% (6.5% at CI bound); 50× share change needed | derived from E1f slope |

## §7 — the withdrawn result

| claim | value | artifact / report |
|---|---|---|
| the withdrawn dip | −0.141 [−0.249, −0.031], n=91 | `random_context_topbin/turns_pooled.csv`, **withdrawn** in `NULL_STATISTICS.md` §2 |
| per-item agreement with the committed run | 1.000 over 344 items | `E2B_DIP_RESCUE.md` |
| fine bins | 0.625 (n=40) / **0.419 (n=31)** / 0.703 (n=37) | same |
| committed run's max fill | 0.8784 | same |
| pooled 26 sessions | +0.005 [−0.105, +0.092] | same |

## §8 — signatures (unchanged from the previous version)

| claim | value | artifact / report |
|---|---|---|
| flat accuracy, random / coherent | corr −0.02 [−0.10, +0.05] n=699 / +0.01 n=1001 | `NULL_STATISTICS.md` §1 |
| equivalence bounds | 4.1 / 9.4 points | same |
| WildChat late/early | median 0.99, corr −0.02, 52% down | `WILDCHAT_DYNAMICS.md` |
| homogeneity partial r | −0.151 | `WILDCHAT_HOMOGENEITY.md` |
| homogeneous / heterogeneous | 0.897 / 1.001 | same |
| OLMo post-training entropy | 1.06 → 0.58 → 0.33 → 0.20; ratio 1.64 → 0.47 | `olmo_gradient/gradient.json`, `RLHF_DOSE_RESPONSE.md` |
| L24 correlations with fill | −0.93 / +0.89 / −0.71 / +0.95 | `CONTEXT_ROT_ATTENTION.md` |
| per-case inversion | +0.045 [−0.003, +0.097], n=115 | `NULL_STATISTICS.md` §3 |
| F90871 clamp | 0.442 → 0.465 vs clean 0.293 | `results/f90871_steering/` |
| calibration gap | 0.85→0.95, r=+0.72 [+0.64,+0.79]; wrong 0.86→0.95, r=+0.69 | pooled DDXPlus, n=154 |
| instruction adherence | corr(violation, fill) = 0 | `instruction_adherence/` |
