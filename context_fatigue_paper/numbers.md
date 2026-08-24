# `numbers.md` — every number in the context-fatigue abstract, mapped to its artifact

**Rule (inherited from `papers/register_vs_procedure/numbers.md`): no number enters the tex
without a row here.** A cell whose artifact cannot be opened reads "not measured", not a number.

`results/` is gitignored, so "artifact" means the file on the GPU box; the committed report that
quotes it is the durable record. All causal runs: `allenai/OLMo-2-1124-7B-Instruct`, layer 24,
seed 42, bootstrap 10,000 draws resampling **cases**.

## Figures

| # | figure | builder | source artifact(s) |
|---|---|---|---|
| 1 | framework diagram (four knobs) | TikZ inline in the tex | none (schematic) |
| 2a | distance ladder, accuracy + evidence share | `fig_distance_ladder` | `e1_distance_sweep/turns.csv`, `e1_with_attention/turns.csv` |
| 2b | share→accuracy dose-response | `fig_mass_dose` | `e1f_share_knee/turns.csv` (balanced panel), `e1c_evidence_clamp/` for the diamond |
| 2c | competition arms | `fig_competition` | `e3_competition/turns.csv` |
| 3a | E6 compliance ladders + mmlu accuracy | `fig_format_erosion` | `e6_{code,gsm8k,mmlu}/turns.csv`; accuracy re-graded from stored replies per `E6_FORMAT_EROSION.md` (lead-line fix; depths 3/7 = 0.500/0.525) |
| 3b | E6 system enrichment by fill | `fig_format_enrichment` | same ladders |
| 3c | E6 recovery arms at depth 42 | `fig_format_recovery` | `e6_mmlu_recovery/turns.csv` (natural = same run's depth-42 NaN-arm rows) |
| 4a | attention reallocation vs fill | `fig_attention` | `olmo_attention/` (see `CONTEXT_ROT_ATTENTION.md`) |
| 4b | WildChat homogeneity | `fig_wildchat` | `wildchat_homogeneity/WILDCHAT_HOMOGENEITY.md` (documented constants) |
| 4c | calibration gap | pre-existing `calibration_gap.pdf` | pooled DDXPlus MCQ streams, n=154 |
| A5a | post-training dose-response | `fig_dose_response` | `olmo_gradient/gradient.json` |
| A5b | accuracy by fill, random vs coherent | `fig_random_context` | `random_context/accuracy_by_fill.csv` |

Regenerate: `uv run python scripts/context_fatigue/make_paper_figures.py`. Section numbers
refer to the 2026-08-21 reconstruction (methodology/experimental-design/results-by-finding, with
detailed analyses in Appendices A–H).

## §4.2 — displacement: the distance sweep

| claim | value | artifact / report |
|---|---|---|
| `local`→`back_20` accuracies | 0.464 / 0.359 / 0.292 / 0.250 / 0.276, n=192/arm | `e1_distance_sweep/`, `E1_DISTANCE_SWEEP.md` |
| mean fill identical across arms | 0.688 | same |
| overflow skips | 0 of 192 | same |
| distance β | −0.00761 [−0.01173, −0.00346] | same |
| fill β | −0.00725 [−0.21006, +0.18658] | same |
| `local` flat with fill | β = −0.294 [−0.767, +0.184] | same |
| parsed-only ladder | 0.524 / 0.413 / 0.397 / 0.343 / 0.338 | same |

## §4.2 — mass mediation (same subsection, the four interventions; tokens-vs-turns and the query floor now in Appendix G)

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
| E1d necessity (partial, all-layer share-matched) | +0.047 [+0.010, +0.083] | `e1d_alllayer/`, `E1_MECHANISM.md` addendum |
| E1d penalty (same paired panel) | +0.167 [+0.099, +0.240] | same |
| E1d recovered fraction | 0.28 [0.07, 0.61] | same |
| (superseded: L24-indexed rescue +0.0546 [+0.0172, +0.0977], residual +0.1226 [+0.0536, +0.1925] — `e1d_evidence_rescue/`; not in the tex) | | |
| E1f endpoint | +0.1985 [+0.1145, +0.2824], n=131 balanced | `e1f_share_knee/` |
| E1f largest adjacent step | +0.053 (natural → 0.036) | same |
| E1f slope | 6.29 accuracy per unit share, R²=0.966 | derived from same |
| E1e matched-token shares | 0.0104 vs 0.0108 | `e1e_dissociation/` |
| E1e C2 (64% mass cut) | +0.0260 [−0.0417, +0.0938] | same |
| E2a natural share / accuracy | 0.258 / 0.545, n=110/level | `e2a_mass_clamp/`, `E2A_MASS_CLAMP.md` |
| E2a cost at 0.15 | +0.164 [+0.036, +0.291] | same |

## §4.3 — competition

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

## §4.3 — the competitor-closure test (E3c)

Run 2026-08-21, eager, paired n=365 (same panel as `e3_competition/`: 15 starved + 4 skips),
artifacts `e3c_competitor_close/`, report `E3C_COMPETITOR_CLOSE.md`, driver
`run_competition_sweep.py --close-arms`, brief `tasks/e3c_competitor_close_brief.md`.

| claim | value | artifact / report |
|---|---|---|
| arm accuracies | near_dup 0.4192 / comp_close 0.4740 / rand_close 0.4137 / random 0.5123 | `e3c_competitor_close/turns.csv` |
| rescue: comp_close − natural | +0.0548 [+0.0055, +0.1041] sig | same, `paired_accuracy_gap` |
| control: rand_close − natural | −0.0055 [−0.0356, +0.0247] n.s. | same |
| net vs control | +0.0603 [+0.0082, +0.1123] sig | same |
| competition gap (in-run replication) | +0.0932 [+0.0384, +0.1479] (committed +0.085) | same |
| residual: random − comp_close | +0.0384 [−0.0110, +0.0877] n.s. | same |
| recovered fraction | 0.59 | same |
| competitor spans / tokens / union share | 30.0 / 127.9 / 0.0077 (all-layer) | same |
| parsed-only rescue | +0.0581 [+0.0031, +0.1131], n=327 | same |
| parsed-only net vs control | +0.0568 [+0.0000, +0.1136], n=317 (boundary) | same |
| harness anchors | near_dup 0.419 vs committed 0.427; random 0.512 vs 0.512 | same |

## Appendix F — per-layer and per-head structure

All 32 layers x 32 heads = 1,024 heads. An earlier pass measured layer 24 only; every head-identity
conclusion from it was wrong and is superseded here.

| claim | value | artifact / report |
|---|---|---|
| displacement drain, all-layer mean | 0.0455 | `e1_heads_all/heads.csv`, `E4_HEAD_STRUCTURE.md` |
| competition drain, all-layer mean | 0.0022 | `e3_heads_all/heads.csv`, same |
| competition drain at L17 / L16 / L18 | −0.0186 / −0.0110 / −0.0093 | same |
| competition drain at L24 | −0.00027 [−0.00088, +0.00035] | matches `E3_COMPETITION.md` |
| corr of the two drain profiles across layers | −0.32 | same |
| heads enriched on the evidence | 255 of 1,024 (0 of 32 at L24) | same |
| most evidence-concentrated head | L16H17, share 0.626, enrichment 6.13 | same |
| evidence span as a share of context | 0.102 | `e1_heads_all/turns.csv` |
| L24 mean evidence share | 0.0408 (vs 0.183 at L3, 0.143 at L16) | same |
| displacement: heads losing mass at L24 | 32/32, all significant at Bonferroni | same |
| fractional drain | 0.689 (sd 0.147), r with level = +0.08 | same |
| uniform odds-scale fit to per-head drain | R² = 0.576 | same |
| competition per-head \|Δ\| at L24 | 0.00257 vs sign-flip null 0.00043, p ≤ 0.0005 | same |
| heads significant at Bonferroni, L24 | 19 of 32 | same |

**Withdrawn:** the "50x larger share change needed" figure. It was read off layer 24 alone, and
its arithmetic used the 6.29-per-unit-share slope — measured with an all-layer clamp indexed at
layer 24 — to convert a layer-24-only delta from an experiment whose layers move independently.

## Not claimed: the late-window dip

The top-fill-bin dip (−0.141 [−0.249, −0.031], n=91) appeared in an earlier draft and **is not in
the paper** — no number from it enters the tex, so it has no rows here. It was withdrawn on the
evidence in `E2B_DIP_RESCUE.md` and `NULL_STATISTICS.md` §2, which remain the record. Do not
reintroduce it.

## §4.4–4.5 + Appendix H — precedent (E5 + E6)

E5: `e5_neutral/`, `e5_system_clamp/`, `e5_profile`, report `E5_SYSTEM_CLAMP.md`. Share pooled
over all 32 layers. E6: `e6_{code,gsm8k,mmlu}/` (+ `_spans/` re-runs, exact replication,
max |Δ| = 0.000), `e6_mmlu_recovery/`, `e6_exemplar_close/`, `e6_format_probes/`,
`e6_mode_steering{,_r2,_r3}/`, `e6_probe_dir_erase_{mmlu,gsm8k}/`, report `E6_FORMAT_EROSION.md`.

| claim | value | artifact / report |
|---|---|---|
| natural system share, 0 → 8 prior cases | 0.1661 → 0.0210 (8×) | `e5_profile` |
| neutral-context clamp | share 0.1652 → 0.0500 (−2.57 nats) | `e5_neutral/` |
| compliance collapse | prefix canary 0.992 → 0.025 (+0.967 [+0.933, +0.992]); suffix 1.000 → 0.000, 120/120 flips | same |
| accuracy under clamp | 0.525 → 0.467, +0.058 [−0.017, +0.133]; parse 0.967 | same |
| demonstrated / undemonstrated arms | 3.00/3 vs 1.00/3 canaries at every clamp level; reply length 8 vs 1 chars, zero variance, 720/arm | `e5_system_clamp/` |
| compliance ladders | code 0.875 → 1.000 (interpret to depth 12, fill 0.778); gsm8k ≥0.825 to depth 12, 0.600 at depth 15 (fill 0.480); mmlu 0.875 → 0.000 at depth 3 (fill 0.147) | `e6_{code,gsm8k,mmlu}/` |
| matched fill ≈ 0.5 ordering | code 1.00 / gsm8k 0.60 / mmlu 0.00 | same |
| mmlu accuracy through the collapse | 0.425 at depth 0; 0.500–0.684 at depths 3–42 | `e6_mmlu/` |
| system enrichment (flat-to-rising) | code 1.51 → 3.07; gsm8k 1.51 → 2.04; mmlu 1.50 → 2.14 → 1.47 | same |
| code compliant at half mmlu's collapsed share | code full compliance at raw share 0.061; mmlu collapsed by 0.11 | `E6_FORMAT_EROSION.md` |
| answer−question enrichment gap | mmlu +1.28 [+1.22, +1.34] → +0.65; gsm8k +0.34 [+0.33, +0.36] → ~+0.10; code −0.08 to −0.17 (CIs exclude 0) | `e6_*_spans/` |
| exemplar closure restores nothing | fa_close 0.000 / fa_matched 0.000 / fq_close 0.132 / rand1_close 0.000 compliant | `e6_exemplar_close/` |
| probe 1: instruction presence | transfer AUC 1.000 at every depth 0–42, perm p = 0.000; layer-mean 0.985 → 0.791 | `e6_format_probes/` |
| probe 2: upcoming compliance | LOO-AUC 0.822 at stack L21 (null 0.485, p = 0.000), n = 80 | same |
| shared-axis cosine (mmlu, gsm8k mean-diffs) | median +0.746 (+0.747 at L21) | same |
| decode-time install | real 0.000 vs matched-norm random 0.525 compliant (natural 0.875); acc 0.175 vs 0.375 | `e6_mode_steering_r3/` |
| erase attempts | 4 strategies (1-layer, 11-layer, probe direction ×2 contexts) all null with clean controls | `e6_mode_steering*/`, `e6_probe_dir_erase_*/` |
| L21 linear code rank | AUC 0.822 → 0.619 → 0.505 under iterative projection (rank ≈ 2) | CPU re-probe on `e6_format_probes/` captures |
| recovery at depth 42 | natural 0.000/0.675 · upclamp 1.000/0.425 · refresh 1.000/0.500 · both 1.000/0.275 (compliance/accuracy) | `e6_mmlu_recovery/` |

## §4.6 + Appendix I — signatures (unchanged from the previous version)

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

## Appendix D — example transcripts

Both transcripts are reconstructed byte-exactly from the committed seeds by
`scripts/context_fatigue/dump_example_transcripts.py`, which asserts the reconstruction against
the stored artifacts (E1 local 2049 tokens, E3 near_dup 3343 / random 2984 tokens, pathologies,
golds) and writes `results/context_fatigue/example_transcripts/`. Generations are quoted from
the runs' own `turns.csv` (row set: E1 session 0 probe 0, all arms; E3 probe 0, all arms).

| claim | value | artifact |
|---|---|---|
| E1 example: gold / fill | D (Acute pulmonary edema) / 0.5002 every arm | `e1_distance_sweep/turns.csv` rows 0–4 |
| E1 example generations | local "D" correct; back_2/5/10/20 unparsed prose | same |
| E3 example: gold | E (Viral pharyngitis) | `e3_competition/turns.csv` rows 0–2 |
| E3 example generations | disjoint "E", random "E", near_dup "D" (URTI) | same |
| near_dup context overlap | 4 of 5 options shared in each of 8 cases; URTI in 6 of 8 | `example_transcripts/metadata.json` |

## §4.2 + Appendix G — all-layer re-denomination (2026-08-24)

Re-runs of the three L24-dose experiments with the clamp target share-matched on the
all-32-layer mean (`--reference-layer 0..31`). Reports: `E1_MECHANISM.md` (E1c/E1f addenda),
`E2A_MASS_CLAMP.md` (E2a addendum).

| claim | value | artifact |
|---|---|---|
| E1c all-layer removal cost | +0.151 [+0.099, +0.208], paired n=192 | `e1c_alllayer/` |
| E1c all-layer penalty / recovered fraction | +0.167 [+0.094, +0.240] / 0.91 [0.60, 1.46] | same |
| E1c all-layer clamped − back_20 | +0.016 [−0.052, +0.083] | same |
| E1f all-layer sweep | 0.497 → 0.305, balanced n=167, largest step 0.060 | `e1f_alllayer/` |
| E1f↔E1c all-layer agreement | +0.168 [+0.102, +0.234] vs +0.151 [+0.099, +0.208] (0.017) | both |
| E2a all-layer ladder | natural 0.463/0.536; 0.775× +0.045 n.s.; 0.581× +0.118 [+0.027, +0.209]; ≤0.18 degenerate (−3.63 nats, modal-A 99/110) | `e2a_alllayer/` |
| all-layer accumulated query trajectory | 0.464 → 0.202 over 8 cases (20 items/point) | `E2A_MASS_CLAMP.md` addendum |
