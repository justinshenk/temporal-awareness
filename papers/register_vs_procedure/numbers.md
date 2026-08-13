# `numbers.md` — every figure in the paper, mapped to the artifact it came from

**Rule (from the design spec, after `600b5f7` / `c3e2c62` / `089534e` found four unsourced numbers,
one of them in the abstract): no number enters the tex without a row here.** A cell whose artifact
cannot be opened reads "not measured", not a number. Before asserting "X diverges from Y", open Y's
artifact and confirm it was measured at the same layer, α and injection mode.

Paths are relative to the repo root. `results/` is gitignored, so "artifact" means the file on the
GPU box; the reports that quote them are committed.

---

## Figures

| # | figure | file | source artifact(s) | status |
|---|---|---|---|---|
| F1 | temporal-density knee, GSM8K + MuSiQue overlaid | `figures/f1_temporal_density.png` | `temporal_oracle_L20.json`, `temporal_oracle_multihop_L20.json` | **built 2026-08-13** |
| F2 | oracle layer sweep, L20 peak | — | `lockstep_multihop_single.json` (n=100) — **GSM8K arm has no artifact, see below** | **blocked** |
| F3 | null ladder vs oracle, with S1 intervals | — | `short_arithmetic.json`, `local_refit_gsm8k.json`, `dagger_refit_gsm8k.json`, `das_subspace_L20.json` | pending |
| F4 | variance-band cliff | — | `lockstep_pca_band_L20.json` | pending |
| F5 | gold-token lens by role | — | `gold_token_lens_L20.json`, `gold_token_lens_multihop_L20.json` | pending |
| F6 | refusal Pareto frontier | — | `refusal_frontier.json` | pending |
| F7 | α-resonance / layer hump | — | `steer_multihop_alpha_L20.json`, `steer_multihop_layers.json`, `steer_results_layers.json` | pending |

### F1 — notes that must survive into the caption

- Both series are **n=20 contrast problems**, matched. MuSiQue also has an n=100 re-run
  (`temporal_oracle_multihop_L20_n100.json`: periodic_2 0.060, reasoning_only 0.760) which confirms
  the n=20 reads; the figure uses n=20 for both so the comparison is like-for-like.
- Error bars are percentile-bootstrap 95% intervals over **problems** (`src/common/bootstrap_stats`),
  10,000 draws, seed 42. Problems are the independent unit.
- **The segment between frac 0.5 and frac 1.0 is an interpolation across a region with no
  measurements, and the caption must say so.** `periodic(k)` takes integer k, so the reachable
  fractions are 1, 1/2, 1/3, 1/4 … — there is no gate between k=1 and k=2. The structural gates
  (planning/reasoning only, at 94% and 100%) are the only evidence inside that span. The claim the
  figure supports is "sparse gating recovers nothing", not a measured curve shape near the knee.
- MuSiQue's `answer_only` sits at frac **0.000** and is **vacuous, not a null**: the unpatched base
  never emits "The answer is:", so the gate never fires. Do not read it as evidence.

---

## Numbers quoted in prose

| claim | value | artifact | verified |
|---|---|---|---|
| GSM8K oracle @L20 | 0.75 | `temporal_oracle_L20.json` (periodic_1) | 2026-08-13 |
| MuSiQue oracle @L20 | 0.75 / +0.76 sweep | `temporal_oracle_multihop_L20.json`, `lockstep_multihop_single.json` | 2026-08-13 |
| GSM8K ridge steer @L20 | +0.03 [0.01, 0.08], n=200 | `steer_results_layers.json` | P5, 2026-08-10 |
| GSM8K ridge steer @L24 | +0.12 [0.07, 0.19], n=200 | `steer_results_layers.json` | P5, 2026-08-10 |
| MuSiQue ridge steer @L24 | +0.45 [0.35, 0.56] | `steer_multihop_layers.json` | P2b |
| GSM8K L20 held-out R²_te | 0.610 @ λ*=3.16e3 | `sweep.json` | P5, 2026-08-10 |
| MuSiQue L20 held-out R²_te | 0.714 | `sweep_multihop.json` | P2 |
| GSM8K MLP rung @L24, n=100 | 0.00 [0, 0.04] vs ridge 0.10 [0.05, 0.18] | `nonlinear_delta_gsm8k_L24_n100.json` | P5b |
| cross-task transplant @L28 | 0.13 (= native exactly) | `steer_transplant_multihop_maps_on_gsm8k.json` | P5b |
| ARC-Challenge chance / majority floor | 0.25 / **0.288** | `data/commonsense/ARC-Challenge_test.json`, n=500 scan | 2026-08-13 |

### F2 — a fifth unsourced comparison, found 2026-08-13

`2026-06-16-multihop-generality.md:50-66` prints a GSM8K oracle **layer sweep** column — L16 0.20,
L20 0.75, L24 0.75, L28 0.95, L31 0.95 — and rests the claim "**L\* = 20, the same layer**, selected
by the same earliest-plateau rule" on it. That column has **no artifact in the tree**:

- `lockstep_single.json` is `task=gsm8k` with **`n_contrast=1` and a single layer, 20** — a smoke run.
- No file anywhere references `lockstep_single`, and no other report contains the L16=0.20 figure.
- The likely cause is the **same output-name hazard that cost the α grid its JSON**:
  `lockstep_patch_gsm8k` names GSM8K output `lockstep_{mode}.json` regardless of `--layers` or
  `--n-contrast`, so a later 1-problem smoke overwrote the real sweep in place.

What is safe and what is not:

- **Safe** — "the GSM8K L20 oracle recovers 0.75 (n=20)". Doubly sourced: `temporal_oracle_L20.json`
  `periodic_1` = 0.750 and `downstream_lesion_L20.json` level 0 `recovery_patch` = 0.75.
- **Not safe** — "L20 is GSM8K's earliest plateau" / "the same layer" / any GSM8K value at
  L16/L24/L28/L31 in an oracle sweep. Those need a re-run of
  `lockstep_patch_gsm8k --mode single --layers 0,4,…,31` at real n, written to a **non-colliding
  filename**. Until then F2 is a MuSiQue-only figure and the cross-task "same layer" sentence must
  be softened to the L20 point comparison, which is sourced.

### Known gaps — do not quote until closed

- **α grid @L24/28**: 9 of 12 cells exist **only in `.run_logs/p5b_alpha_grid.log`**; the run died
  before writing JSON. Log reads suggest the L24 peak is at **α=0.75 (0.095)**, above the α=1.0
  used in the headline sweep — so F7 and any α claim need the re-run first.
- **S2 register battery**: in flight 2026-08-13. §9 and §10 have no numbers yet.
- The GSM8K and MuSiQue **Gram accumulators were deleted 2026-08-13** to clear a disk quota that had
  already truncated a trained adapter. The derived `maps/` and `maps_multihop/` are intact, so every
  steering number above remains reproducible; refitting the ridge sweep from scratch would require
  re-running `collect_cot_residuals` (~24 min GPU for GSM8K).
