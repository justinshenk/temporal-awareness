# version: 5

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
| ~~Qwen3.5-4B~~ Qwen3-4B-Inst-2507 | startup ($k revenue, founder) | geometry VERIFIED (see note) |

All runs: 3,000 samples, turn-transition positions only (`chat_suffix` +
tail), resid_post + attn_out, fp16 storage, bf16 inference, TransformerLens.

## IMPORTANT NAMING NOTE (v4)

Qwen3.5-4B is architecturally incompatible with TransformerLens (hybrid
linear attention, Qwen3_5ForConditionalGeneration; not in the TL registry).
The HF artifacts named `qwen35_4b_startup` are therefore
**Qwen3-4B-Instruct-2507 — the paper's own model — on the STARTUP domain**;
the substitution is stated in their caption.md. Cite them as a
DOMAIN-generalization result (investment -> startup) for the target model,
never as a newer-Qwen result. A true Qwen3.5 run would need the HF backend
end to end.

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

**VERIFIED — Mistral-7B/education**: 2,984/3,000 valid (16 skips, 0.5% —
no ambiguity epidemic in education), archive 1.2 GB byte-verified, fig7
winner **L19 (0.59)** viewed: full ordinal manifold, clean split, no-horizon
off-manifold. Note Mistral's template yields two turn tokens ([/INST], 'I').
`geometry/mistral7b_education.tar.gz`, `_plots/`, `fig7_final/`.

**VERIFIED — Qwen3-4B-Inst-2507/startup** (see naming note): 2,992/3,000
valid, archive 1.86 GB byte-verified, fig7 winner **L19 resid_post** viewed
(monotonic seconds->millennia gradient at <|im_end|>, No-Horizon separate).
With investment (paper) + startup (new), the geometry story now holds for
the SAME model on two domains and for four model families overall.
`geometry/qwen35_4b_startup.tar.gz` (name retained; content is Qwen3-4B).

Fig-7 final panels on HF: Qwen3-4B/startup L19, Llama L21, Gemma L33,
Mistral L19 — all selected by visual inspection with image tokens.

## 2. Causal localization — coarse activation patching (paper §5.1 / App. J)

Method note for the rebuttal: these sweeps cover **all layers 0..N-1**.
The paper's own sweep hardcoded a 0.45-depth floor (layers 16-35 on Qwen),
so its "L17-35" claim could not be distinguished from the sweep bound; the
new runs close that gap. 24 contrastive pairs, resid_post + attn_out +
mlp_out, denoising + noising.

**VERIFIED — ALL THREE: Mistral-7B/education, Gemma-2-9B/climate,
Llama-3.1-8B/health.** n=24 pairs (Llama 23: one skipped by the ambiguity
guard), every layer 0..N-1, three components, zero errors, archives
byte-verified on HF (`localization/loc_mistral_education.tar.gz`,
`localization/loc_gemma_climate.tar.gz`). Scores below are mean recovery +
disruption (the paper's figure ordering).

| Model | Top attention | Attention band (frac) | MLP | Early layers |
|---|---|---|---|---|
| Mistral-7B (final) | L16 (0.52) +1.03 | 0.48-0.61 | late L28-L31 | -0.006 silent |
| Gemma-2-9B (final) | L25 (0.61) +0.58 | 0.56-0.68 | L26 + late L32-L39 | -0.026 silent |
| Llama-3.1-8B (final) | L17 (0.55) +0.98 | 0.45-0.58 | late L26-L29 | -0.010 silent |
| Qwen3-4B (paper) | L24 (0.67) | 0.58-0.67 | L31/L35 | never swept |

Three cross-family regularities: (1) a causal attention band at ~0.45-0.68
fractional depth in every model; (2) late-layer MLP accumulation; (3) layers
below ~0.2 depth causally silent — MEASURED in the three new models, whereas
the paper's own sweep hardcoded a 0.45-depth floor and never tested them.
In every model the causal attention band contains or abuts that model's
steering optimum (Mistral L16-19 vs steer L19; Gemma L23-28 vs steer L21;
Llama L14-18 vs steer L18), closing the localization-intervention loop
across architectures. All three archives:
`localization/loc_{mistral_education,gemma_climate,llama_health}.tar.gz`.

## 3. Probing (paper App. G) — VERIFIED, all four models

Per-layer logistic probes, implicit 300-pair set, pair-aware 80/20 split,
10x shuffled-label control, zero-shot transfer to the explicit set.

| Model | Best layer (frac) | Acc | Shuffled | Transfer |
|---|---|---|---|---|
| Qwen3-4B | L17 (0.47) | 95.0% | 52.8% | 74.2% (L20 ties, 84.0%) |
| Llama-3.1-8B | L29 (0.91) | 95.8% | 47.4% | 70.1% |
| Gemma-2-9B | L23 (0.55) | 95.8% | 49.8% | 85.8% |
| Mistral-7B | L14 (0.44) | 96.7% | 53.0% | 85.8% |

Temporal preference is >=95% linearly decodable in every family; controls
at chance. `probing/turn_preference/` (9 files, sizes verified twice).
Caveat: Gemma and Mistral ran on the HF-hook backend (TL OOM / no registry
entry), identical resid_post hook points.

## 4. Steering — CAA with improvements (paper §5.3 / App. S)

VERIFIED, all four models. Improvements over the paper's protocol:
(a) random-direction control at matched norm (paper had none); (b)
label-order counterbalanced scoring; (c) fractional-depth sweep
{0.50-0.65} for cross-model comparability.

| Model | Best | S steer | S control | Baseline | Beats ctrl |
|---|---|---|---|---|---|
| Llama-3.1-8B | L18 a35 | 17.38 | 6.37 | 2.25 | 18/20 |
| Mistral-7B | L19 a20 | 12.12 | 2.59 | -2.17 | 19/20 |
| Qwen3-4B | L18 a20 | 5.87 | 2.85 | 0.86 | 20/20 |
| Gemma-2-9B | L21 a50 | 3.86 | 1.44 | 1.42 | 20/20 |

Steering beats the random-vector control in 77/80 configs; the steerable
band sits at 0.50-0.59 fractional depth in every family, matching the
paper's L19-22/36 sweet spot. `steering/extreme_sweep/` (26 files, md5 +
size verified; heatmaps viewed). Caveats: process_weights=False loading
(TL-advised at reduced precision; CAA is invariant); Mistral-v0.3 via a
corrected v0.1 TL mapping (rope_theta=1e6, no sliding window).

## 5. Behavioral: extreme/inconsistent discounting (paper App. O)

ON HF, all four models (JSONs 29-143 KB + summary CSV + k-vs-delay
figure at `behavioral/extreme_discount/`). From-scratch probe: titrated
indifference points (binary search, 20x cap), hyperbolic k, extreme-k
flags, magnitude reversals, non-monotonicity, label-swap reversals.
Numbers not yet extracted into this file; open
`extreme_discount_summary.csv` for per-cell k values. UNVERIFIED at the
number level (files listed on HF, contents not yet read).

## 6. Behavioral: coherence / Fig-8 rows (paper App. P)

VERIFIED — five runs on the full 960-prompt grid (not subsampled), zero
unparseable responses anywhere. Zone coherence (%ST in the 1-5y reasoning
zone, paired denominator n=288):

| Model | Zone coherence | %LT overall |
|---|---|---|
| gpt-4o-mini (API row) | 100.0% | 31% |
| Gemma-2-9B | 95.1% | 40% |
| Llama-3.1-8B | 52.4% | 52% |
| Mistral-7B | 51.0% | 54% |
| Qwen3-4B-Inst-2507 | 50.3% | 59% |

**Anchor check passed**: our Qwen re-run gives 50.3% vs 50.0% recomputed
from the paper's own reference run — the instrument reproduces. Notable:
Gemma-2-9B is near-coherent (95.1%) while the other three local models sit
at chance, refining the paper's "only frontier API models are coherent"
claim. Order/label stability per model are in each run's heatmap figures
(`behavioral/coherence/coh_*/`); Fig-8-format rows still to be extracted
into one table.

## Reproducibility flag (discount probe)

Two independent Qwen3-4B discount runs disagree: the A6000 run terminated
every cell at the probe caps (no_boundary/always_delayed, 42 queries) while
the eval-box rerun found boundaries with 19 label-swap reversals (21 cells,
31.5 s). Same script, different box/backend. Do not quote Qwen discount
numbers until this is reconciled; the other three models have single runs.

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
