# RQ47 Presentation Brief

## One-Sentence Story

We tested whether temporal-horizon representations are merely decodable or can
serve as causal and online oversight signals across GPT-2, Qwen3-4B, Phi-3
mini, and Llama-3.2-3B.

## Code Walkthrough

### 1. Probe Training And Validation

`scripts/probes/run_all_temporal_probe_methods_multimodel.py`

- Sequential orchestration script.
- Runs LR, DMM, and AttnProbe for each model.
- Calls training first, then explicit validation.
- Uses subprocesses so GPU memory is released between model/method runs.

`scripts/probes/train_temporal_probes_caa_multimodel.py`

- Loads the implicit AB-randomized temporal CAA dataset.
- Expands each pair into immediate and long-term prompts.
- Extracts one feature vector per layer.
- Supports three probe methods:
  - `lr`: logistic regression on last-token hidden states.
  - `dmm`: difference-of-means direction on hidden states.
  - `attn`: logistic regression on attention-summary features.
- Saves per-layer probe metrics to `research/results/{lr,dmm,attn}/`.

`scripts/probes/validate_temporal_probes_multimodel.py`

- Loads implicit-trained probes.
- Evaluates them on explicit CAA examples.
- Computes best semantic layer, explicit accuracy, implicit accuracy, and
  generalization gap.

### 2. Probe Analysis Notebooks

`notebooks/03.1_multimodel_probe_validation_analysis.ipynb`

- Cross-model probe validation analysis.
- Produces model/method heatmaps, normalized-depth plots, generalization-gap
  diagnostics, and semantic-layer coverage.

`notebooks/03.2_all_probe_methods_validation_analysis.ipynb`

- Broader all-method probe comparison.
- Produces compact layer grids, cross-validation diagnostics, and method
  comparison plots.

### 3. Causal Intervention Experiments

`scripts/experiments/multimodel_temporal_interventions.py`

- Uses probe result CSVs to select layers (`lr`, `dmm`, or `attn` as layer
  source).
- Runs four post-probing causal tests:
  - `steering`: adds temporal direction to activations.
  - `activation_patching`: patches clean activations into corrupted prompts.
  - `attribution_patching`: estimates patching effect with activation gradients.
  - `ablation`: zeros or mean-ablates residual/attention/MLP outputs.
- Saves outputs under `results/temporal_interventions/{experiment}/{layer_source}/`.

### 4. Online Oversight Experiment

`scripts/experiments/temporal_probe_trajectory_monitor.py`

- Generates text token-by-token.
- Scores the temporal probe at the prompt end and after each generated token.
- Records whether temporal probe drift appears before a keyword-detected event.
- Saves trajectory and summary files under `results/temporal_oversight/{layer_source}/`.

### 5. Presentation Aggregation

`scripts/analysis/summarize_rq47_results.py`

- Aggregates probe, intervention, and oversight CSVs.
- Writes presentation tables to `results/tables/rq47/`.
- Writes heatmaps to `results/figures/rq47/`.
- Excludes smoke-test files containing `layers-0`.

## Result Inventory

Generated summary tables:

- `results/tables/rq47/probe_best_layers.csv`
- `results/tables/rq47/steering_dose_response_summary.csv`
- `results/tables/rq47/activation_patching_summary.csv`
- `results/tables/rq47/attribution_patching_summary.csv`
- `results/tables/rq47/ablation_summary.csv`
- `results/tables/rq47/temporal_oversight_summary.csv`
- `results/tables/rq47/temporal_oversight_top_pre_event_drifts.csv`

Generated figures:

- `results/figures/rq47/steering_best_resid_heatmap.png`
- `results/figures/rq47/activation_patching_best_resid_heatmap.png`
- `results/figures/rq47/attribution_patching_best_resid_heatmap.png`
- `results/figures/rq47/ablation_best_resid_heatmap.png`
- `results/figures/rq47/temporal_oversight_pre_event_delta_heatmap.png`

## Main Quantitative Takeaways

### Probe Results

Best implicit probe accuracies:

| Method | GPT-2 | Qwen3-4B | Phi-3 mini | Llama-3.2-3B |
| --- | ---: | ---: | ---: | ---: |
| LR | 0.95 | 1.00 | 1.00 | 0.95 |
| DMM | 0.85 | 1.00 | 1.00 | 1.00 |
| Attn | 0.70 | 0.75 | 0.75 | 0.75 |

Interpretation: hidden-state probes are consistently strong; attention-summary
probes are weaker but still above chance.

### Causal Intervention Results

Strongest robust best-layer residual findings:

- **Phi-3 LR residual**: activation patching median recovery ≈ 0.90; attribution
  patching median estimate ≈ 0.98.
- **Qwen DMM/Attn residual**: activation patching median recovery ≈ 0.63-0.64;
  attribution patching median estimate ≈ 0.54.
- **Llama Attn-selected residual**: activation patching median recovery ≈ 1.00;
  attribution patching median estimate ≈ 1.01.
- **Llama Attn-selected steering**: strongest dose-response; +3 steering shifts
  long-vs-short logit difference by ≈ 0.32 on residual stream.

Interpretation: the strongest causal evidence appears model- and selector-
dependent, but residual-stream interventions repeatedly show meaningful effects.

### Top-K Robustness

Top-k runs are useful as robustness checks, not as the primary claim.

Examples:

- Phi-3 LR residual top-k activation patching median recovery ≈ 0.93.
- Llama Attn residual top-k activation patching median recovery ≈ 0.83.
- GPT-2 LR residual top-k activation patching median recovery ≈ 0.73.

Interpretation: several effects survive beyond a single best layer.

### Online Oversight Results

Trajectory monitoring produced 72 summary rows:

- 4 models
- 3 layer selectors
- 6 oversight prompts

Largest pre-event probe drifts:

- GPT-2 LR, sycophancy prompt: temporal score increased by ≈ 0.65 before event
  step 12.
- Phi-3 DMM/Attn, short-term-pressure prompt: temporal score increased by ≈ 0.57
  before event step 37.
- GPT-2 DMM, sycophancy prompt: temporal score increased by ≈ 0.46 before event
  step 12.

Interpretation: the pipeline demonstrates how temporal probes can be monitored
online. The current event detector is keyword-based, so this is a precursor
pipeline result, not yet a final safety-classifier result.

## Suggested Slide Deck

1. **Title / Research Question**
   - Do models encode temporal horizon, and can that signal support oversight?

2. **Motivation**
   - Static behavior is late; internal activations may reveal earlier planning
     orientation.

3. **Experimental Setup**
   - Four models, three probe methods, implicit-to-explicit validation.

4. **Probe Methods**
   - LR hidden-state probe, DMM direction, Attn summary probe.

5. **Probe Results**
   - Use existing heatmaps from probe notebooks plus `probe_best_layers.csv`.

6. **From Correlation To Causality**
   - Explain steering, activation patching, attribution patching, and ablation.

7. **Causal Results**
   - Use `activation_patching_best_resid_heatmap.png`.
   - Use `attribution_patching_best_resid_heatmap.png`.
   - Mention top-k robustness.

8. **Online Oversight**
   - Explain token-by-token probe scoring.
   - Use `temporal_oversight_pre_event_delta_heatmap.png`.

9. **Limitations**
   - Keyword event detector.
   - Component-level ablation, not per-head yet.
   - Some normalized patching effects have small-denominator outliers; report
     medians/clipped means.

10. **Conclusion**
    - Temporal horizon is decodable across model families.
    - Several interventions show causal relevance.
    - The online monitor provides a concrete path toward real-time oversight.

## What To Say If Asked About Odd Outliers

Some patching and ablation normalized effects are very large or negative for
GPT-2. This happens when the clean-vs-corrupt baseline denominator is small.
For this reason, presentation tables use medians and clipped means rather than
raw means.

## Next Work

- Replace keyword event detection with a safety classifier such as LlamaGuard.
- Add per-head ablation for TransformerLens-supported models.
- Convert the online monitor from full-prefix recomputation to KV-cache
  generation for speed.
- Add confidence intervals over more prompts.
