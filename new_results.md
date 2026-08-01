# version: 1

# New results for the rebuttal — live document

Updated by the experiment-running session as results land and are VERIFIED.
The paper session should consume rows marked **VERIFIED** only; **PENDING**
rows name what is coming so sections can be drafted ahead. Raw artifacts:
HF dataset `unrulyabstractions/temporal-awareness`. Verification detail:
`VERIFICATION_LOG.md` (entries 11+).

## Campaign design

Five models, five domains, no model/domain pair repeated — model and domain
effects never confound:

| Model | Domain | Status |
|---|---|---|
| Qwen3-4B-Instruct-2507 | investment | paper baseline (existing) |
| Llama-3.1-8B-Instruct | health (QALYs, patient/doctor) | geometry VERIFIED |
| gemma-2-9b-it | climate (tCO2 prevented, policy maker) | geometry VERIFIED |
| Mistral-7B-Instruct-v0.3 | education ($k lifetime earnings, student) | running |
| Qwen3.5-4B | startup ($k revenue, founder) | running |

All runs: 3,000 samples, turn-transition positions only (`chat_suffix` +
tail), resid_post + attn_out, fp16 storage, bf16 inference, TransformerLens.

## 1. Geometry — turn-transition PCA (paper Fig. 7 / Appendix M)

**VERIFIED — Gemma-2-9B-it, climate.** 2,943 valid samples. The paper's
story replicates on a different family and a non-financial domain: at
`<end_of_turn>` preference classes overlap and no-horizon prompts sit off
the horizon manifold; horizon forms a clean seconds-to-millennia ordinal
gradient; by the role token `model` preference splits into two separated
clusters. Selected panel: **L33** (of 42 → 0.79 fractional depth; full
progression begins ~L21 ≈ 0.5). Figure + paper-style caption:
`geometry/fig7_final/gemma2_9b_climate/` (L33 resid_post + attn_out).
All 30 per-layer figures: `geometry/gemma2_9b_climate_plots/`. Raw
activations: `geometry/gemma2_9b_climate.tar.gz` (3.10 GB).

**VERIFIED — Llama-3.1-8B-Instruct, health.** 2,517 valid of 3,000 (the
0.5-vs-5 reward-string ambiguity skips ~16%, logged, non-biasing: skips are
a formatting collision, not a choice-dependent filter). Same progression at
Llama's own turn tokens: overlap at `<|eot_id|>`, short-cluster detaches at
`assistant`, full split by `<|end_header_id|>`. Selected panel: **L21**
(of 32 → 0.66 fractional depth). `geometry/fig7_final/llama31_8b_health/`,
plots `geometry/llama31_8b_health_plots/`, archive
`geometry/llama31_8b_health.tar.gz`.

**Cross-model regularity worth stating in the rebuttal**: the collapse site
sits at ~0.6–0.8 fractional depth in all three models measured so far
(Qwen L31/36 = 0.86 readout, Llama L21/32, Gemma L33/42), echoing the
fractional-depth invariance the paper found for patching (Appendix Q).

**PENDING**: Mistral-7B/education geometry; Qwen3.5-4B/startup geometry
(each auto-produces plots + a visually-selected fig7_final panel).

## 2. Causal localization — coarse activation patching (paper §5.1 / App. J)

Method note for the rebuttal: these sweeps cover **all layers 0..N-1**.
The paper's own sweep hardcoded a 0.45-depth floor (layers 16-35 on Qwen),
so its "L17-35" claim could not be distinguished from the sweep bound; the
new runs close that gap. 24 contrastive pairs, resid_post + attn_out +
mlp_out, denoising + noising.

**PENDING**: loc_llama_health, loc_gemma_climate, loc_mistral_education,
loc_qwen35_startup → `localization/loc_*.tar.gz` (each includes
`aggregated/analysis/processed_results.json` with the component ranking;
note the JSON ranks by recovery alone while the paper's figure ranked by
recovery+disruption — use recovery+disruption for comparability).

## 3. Probing (paper App. G)

Per-layer logistic probes at the last prompt token, implicit 300-pair set,
pair-aware 80/20 split, 10x shuffled-label control, zero-shot transfer to
the explicit set. Output: per-model CSV + one accuracy-vs-fractional-depth
figure overlaying all four models (directly comparable to Qwen's 99.2% @
L26). **PENDING** → `probing/turn_preference/`.

## 4. Steering — CAA with improvements (paper §5.3 / App. S)

Improvements over the paper's protocol, worth naming in the rebuttal:
(a) **random-direction control** at matched norm (the paper had none);
(b) label-order counterbalanced forced-choice scoring; (c) sweep at
fractional depths {0.50-0.65} so layers are comparable across models.
Output: per-model CSV (layer_frac, alpha, S, S_ctrl, lift) + heatmaps.
**PENDING** → `steering/extreme_sweep/`.

## 5. Behavioral: extreme/inconsistent discounting (paper App. O)

New from-scratch probe focused on the paper's headline pathology: titrated
indifference points (binary search, 20x cap), hyperbolic k at boundary,
flags for no-boundary/extreme-k, magnitude-effect reversals,
non-monotonicity, label-swap preference reversals. Four models.
**PENDING** → `behavioral/extreme_discount/`.

## 6. Behavioral: coherence / Fig-8 rows (paper App. P)

Full 960-prompt investment instrument (identical grid to the paper's
30-model panel — deliberately NOT subsampled so the new rows are exactly
comparable). Per model: **temporal reasoning** (1-5y zone deliverable-option
rate), **order stability**, **label stability** — formatted as drop-in rows
for Figure 8. Note the published figure already has Mistral-7B and
Qwen3.5-4B rows = sanity anchors for our re-runs.
**PENDING** → `behavioral/coherence/`.

## Known caveats to carry into any text

- Llama-3.1's chat template injects `Today Date: 26 Jul 2024` into every
  prompt (constant across samples, so it cannot create horizon geometry,
  but should be disclosed).
- Health-domain skip rate ~16% from the 0.5-vs-5 rendering ambiguity.
- Turn-token rel_pos indices are NOT comparable across families (assistant
  is r3 on Qwen, r2 on Llama; Gemma's role word is `model`); figures label
  actual tokens.
- Localization n_pairs=24 (paper used 71) — adequate for layer-band
  localization, underpowered for fine component ranking.

## How to update this file

Bump `version:` at the top, move items PENDING → VERIFIED only after the
artifact was opened/viewed from HF (not from an agent's claim), append
one-line provenance (HF path + what was checked). Keep prose in the user's
voice: short sentences, active, no em dashes.
