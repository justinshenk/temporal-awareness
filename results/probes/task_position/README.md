# Task-Position Probes

Research log for the task-position probe investigation on Gemma 2 9B IT and Qwen 2.5 7B Instruct, using DDXPlus medical-diagnosis traces from the context-fatigue experiments. The goal was to train probes that decode the model's *belief of how far it is through its current task* and then test whether that representation is causally implicated in the calibration-gap pattern identified in the existing context-fatigue writeup.

**Headline finding:** both models encode `within_task_fraction` (R² ≈ 0.95) and `tokens_until_boundary` (R² ≈ 0.91 in log1p space) as linearly-decodable directions in the residual stream, already at layer 0, orthogonal to raw token position. The signal is real, reproducible across model families, and correlates with a calibration gap that widens at higher subjective lateness. **However, seven independent causal interventions (steering, multi-layer steering, head ablation, two flavors of activation patching) all produced clean nulls**: the probe direction is a bystander readout of an earlier attention-pattern computation, not a lever that downstream layers use to set confidence. The final story is a correlational finding pinned down by systematic intervention nulls.

**Bonus finding:** Qwen 2.5 7B IT and Gemma 2 9B IT differ by ≈0.35 AUC on upcoming-failure prediction at matched data (Gemma 0.96, Qwen 0.61). This matches the original context-fatigue writeup's 0.67 Qwen number and is a clean cross-family effect that deserves its own investigation.

## Spec and plan

- Spec: [`docs/superpowers/specs/2026-04-13-task-position-probes-design.md`](../../../docs/superpowers/specs/2026-04-13-task-position-probes-design.md)
- Plan: [`docs/superpowers/plans/2026-04-13-task-position-probes.md`](../../../docs/superpowers/plans/2026-04-13-task-position-probes.md)

## Results by experiment

| Version | Experiment | Key finding | Writeup |
|---|---|---|---|
| v1 | Train probes + A1/A2/A4 analyses on Gemma-9B-IT | R² 0.95 @ L10 for `within_task_fraction`; A1 orthogonality binary (target probes orthogonal to raw position); A2 baseline AUC 0.96 (no task-position lift); A4 calibration gap rises with subjective lateness | [`2026-04-13-v1-results.md`](2026-04-13-v1-results.md) |
| v2 | Causal steering @ L10 (target 0.1 / 0.9) | Null: 0–1 flips, <0.2pp confidence shift | [`2026-04-13-v2-steering.md`](2026-04-13-v2-steering.md) |
| v2 | F90871 SAE feature correlation with probe readout | Weak negative correlation (r = −0.042), directionally consistent with "boundary detector" but mechanistically load-weak | [`2026-04-13-v2-f90871-correlation.md`](2026-04-13-v2-f90871-correlation.md) |
| v3-1 | Multi-layer steering {L10, L20, L30, L41} | Null: effects indistinguishable from noise, not monotone in number of steered layers — single-layer-compensation hypothesis rejected | [`2026-04-13-v3-multilayer-steering.md`](2026-04-13-v3-multilayer-steering.md) |
| v3-5 | Attention-feature correlation hunt | `att_current_case` at L20 is strongest correlate (mean r = +0.42), consistent across all 4 test traces; BOS sink / system-prompt erosion / attention entropy all rejected (|r| < 0.1) | [`2026-04-13-v3-attention-correlation.md`](2026-04-13-v3-attention-correlation.md) |
| v4 | L20 head ablation (top-1 / top-3 / top-5 / bottom-3 / random-3) | Null: all 16 L20 heads positively correlate with probe readout (no dichotomy); ablating top heads does not flatten the A4 gap-vs-bin slope and is indistinguishable from random controls | [`2026-04-13-v4-head-ablation.md`](2026-04-13-v4-head-ablation.md) |
| v5 | Dense early-layer probe sweep | **Signal is present at L0** (R² 0.90); peak at L6 (R² 0.957); gradual degradation toward output layer. Explains why v2–v4 interventions at L10+ were all downstream of the causal computation | [`2026-04-13-v5-layer-sweep.md`](2026-04-13-v5-layer-sweep.md) |
| Qwen | Full pipeline on Qwen 2.5 7B Instruct | Same R² curve, L0 localization, A1 orthogonality as Gemma. Failure-prediction AUC 0.607 vs Gemma 0.957 — ~0.35 architectural gap | [`2026-04-14-qwen-v1-results.md`](2026-04-14-qwen-v1-results.md), [`2026-04-14-qwen-v5-layer-sweep.md`](2026-04-14-qwen-v5-layer-sweep.md), [`2026-04-14-qwen-vs-gemma-comparison.md`](2026-04-14-qwen-vs-gemma-comparison.md) |
| v6 | Clean → fatigued activation patching (direction-projected at L0/L10 + full-residual at L0) | Null (zero prediction flips across 270 patched cases); v6-2 sanity slice confirms no RoPE-mismatch degeneration. The prediction-site residual is a readout position, not a decision position | [`2026-04-14-v6-patch-experiment.md`](2026-04-14-v6-patch-experiment.md) |

## Artifact directory layout

```
results/probes/task_position/
├── README.md                             (this file)
├── 2026-04-13-v1-results.md              v1: probes + A1/A2/A4
├── 2026-04-13-v2-steering.md             v2: steering null
├── 2026-04-13-v2-f90871-correlation.md   v2: F90871 SAE correlation
├── 2026-04-13-v3-multilayer-steering.md  v3-1: multi-layer steering
├── 2026-04-13-v3-attention-correlation.md v3-5: attention correlates
├── 2026-04-13-v4-head-ablation.md        v4: L20 head ablation
├── 2026-04-13-v5-layer-sweep.md          v5: dense layer sweep (signal @ L0)
├── 2026-04-14-qwen-v1-results.md         Qwen v1
├── 2026-04-14-qwen-v5-layer-sweep.md     Qwen v5
├── 2026-04-14-qwen-vs-gemma-comparison.md Qwen vs Gemma comparison
├── 2026-04-14-v6-patch-experiment.md     v6: clean→fatigued patching
├── 2026-04-13-v*-records.csv             Per-case records for each intervention
├── gemma-9b-it/                          v1 Gemma artifacts
│   ├── correctness.json                  Per-case eval: gold/pred/correct/option_probs/prediction_site
│   ├── activations.pt                    (gitignored — 11 GB)
│   └── probes/                           Trained ridge probes + metrics
├── gemma-9b-it-v5/                       v5 Gemma artifacts (dense layer set)
│   ├── activations.pt                    (gitignored — 25 GB)
│   └── probes/                           Trained ridge probes + metrics + a1_orthogonality
└── qwen-7b-it/                           Qwen artifacts
    ├── correctness.json
    ├── activations.pt                    (gitignored — 23 GB)
    └── probes/                           Trained ridge probes + metrics + a1_orthogonality
```

Activation files (`activations.pt`) are NOT committed to git — they are 11–25 GB each. Re-generate them from the `scripts/probes/task_position/extract_activations.py` driver using the same seeds. Extraction is deterministic.

## Reproducing

Everything uses `uv run`, seed 42, and the DDXPlus test split.

### Gemma v1 pipeline

```bash
# Extract (30-40 min on H200)
HF_TOKEN=... uv run python -m scripts.probes.task_position.extract_activations \
    --model google/gemma-2-9b-it \
    --layers 0,10,20,30,41 \
    --out-dir results/probes/task_position/gemma-9b-it \
    --eval-correctness \
    --n-traces 20

# Train probes
uv run python -m scripts.probes.task_position.train_probes \
    --activations results/probes/task_position/gemma-9b-it/activations.pt \
    --out-dir results/probes/task_position/gemma-9b-it/probes

# Run A1/A2/A4 + writeup
uv run python -m scripts.probes.task_position.run_analyses \
    --activations results/probes/task_position/gemma-9b-it/activations.pt \
    --correctness results/probes/task_position/gemma-9b-it/correctness.json \
    --split results/probes/task_position/gemma-9b-it/probes/split.json \
    --metrics results/probes/task_position/gemma-9b-it/probes/metrics.csv \
    --out results/probes/task_position/2026-04-13-v1-results.md
```

### Gemma v5 (dense early-layer sweep)

```bash
HF_TOKEN=... uv run python -m scripts.probes.task_position.extract_activations \
    --model google/gemma-2-9b-it \
    --layers 0,2,4,6,8,10,15,20,25,30,35,41 \
    --out-dir results/probes/task_position/gemma-9b-it-v5 \
    --n-traces 20

uv run python -m scripts.probes.task_position.run_v5_layer_sweep \
    --activations results/probes/task_position/gemma-9b-it-v5/activations.pt \
    --v1-split results/probes/task_position/gemma-9b-it/probes/split.json \
    --out-dir results/probes/task_position/gemma-9b-it-v5/probes \
    --out-md results/probes/task_position/2026-04-13-v5-layer-sweep.md
```

### Qwen pipeline

```bash
HF_TOKEN=... uv run python -m scripts.probes.task_position.extract_activations \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-ctx 8192 \
    --layers 0,2,4,6,8,10,12,14,18,22,27 \
    --out-dir results/probes/task_position/qwen-7b-it \
    --eval-correctness \
    --n-traces 20

uv run python -m scripts.probes.task_position.train_probes \
    --activations results/probes/task_position/qwen-7b-it/activations.pt \
    --out-dir results/probes/task_position/qwen-7b-it/probes

uv run python -m scripts.probes.task_position.run_analyses \
    --activations results/probes/task_position/qwen-7b-it/activations.pt \
    --correctness results/probes/task_position/qwen-7b-it/correctness.json \
    --split results/probes/task_position/qwen-7b-it/probes/split.json \
    --metrics results/probes/task_position/qwen-7b-it/probes/metrics.csv \
    --out results/probes/task_position/2026-04-14-qwen-v1-results.md
```

### Causal intervention experiments

All require Gemma activations + correctness.json + trained probes:

```bash
# v2 single-layer steering
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_steering_experiment

# v2 F90871 SAE correlation
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_f90871_correlation

# v3-1 multi-layer steering
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_multilayer_steering

# v3-5 attention correlation
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_attention_correlation

# v4 head ablation
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_head_ablation

# v6 clean→fatigued patching
HF_TOKEN=... uv run python -m scripts.probes.task_position.run_patch_experiment
```

## The one-paragraph summary of the v1→v6 story

Gemma 2 9B IT and Qwen 2.5 7B Instruct both encode a linearly-decodable "fraction through current task" signal in their residual stream at the output of their first transformer block, with R² ≈ 0.95 and near-perfect orthogonality to raw positional encoding, decoded at matched depth fractions (~7–14% of the stack), replicated cleanly across model families. The signal covaries with a calibration-gap pattern — overconfidence grows with subjective-lateness quintile — and supports the original "temporal activation monitoring" framing of the research program. Seven independent causal interventions (single-layer steering at L10; multi-layer steering at L10/20/30/41; L20 head ablation of the top attention-to-current-case heads; direction-projected activation patching at L0 and L10; full-residual L0 activation patching at the prediction site) all produce clean nulls — zero prediction flips, sub-0.2 percentage point confidence shifts, A4 bin structure preserved. The cleanest explanation is that the within_task_fraction direction is a *bystander readout* of an earlier attention-pattern computation that the prediction-site residual records as a side effect, and downstream layers do not read from the prediction site's value of that direction when computing the output. The calibration gap is driven by the case-content tokens preceding the prediction site, not by the prediction-site residual itself — a conclusion that is consistent with the v6-2 sanity slice (no downstream-generation degeneration from a single-token L0 replacement) and with the v4 finding that ablating the top L20 att_current_case heads does not reduce the gap. The cross-architectural A2 gap (Gemma 0.957 vs Qwen 0.607 failure-prediction AUC on matched data) is a separate and unexplained finding worth pursuing independently.

## Source code

| Module | Purpose |
|---|---|
| [`src/probes/task_position/labels.py`](../../../src/probes/task_position/labels.py) | Pure-function labelers for per-token targets |
| [`src/probes/task_position/probes.py`](../../../src/probes/task_position/probes.py) | `RidgeProbe` class |
| [`src/probes/task_position/analysis.py`](../../../src/probes/task_position/analysis.py) | A1 / A2 / A4 analysis functions |
| [`src/probes/task_position/steering.py`](../../../src/probes/task_position/steering.py) | `ProbeSteeringHook` for v2 / v3 experiments |
| [`src/probes/task_position/patching.py`](../../../src/probes/task_position/patching.py) | `DirectionPatchHook` / `FullResidualPatchHook` for v6 |
| [`src/probes/task_position/sae_features.py`](../../../src/probes/task_position/sae_features.py) | Gemma-Scope SAE feature encoding |
| [`src/probes/extraction.py`](../../../src/probes/extraction.py) | `PerTokenResidualCapture` forward hook |
| [`src/probes/ddxplus.py`](../../../src/probes/ddxplus.py) | DDXPlus dataset helpers (copied from the legacy script) |
| [`tests/probes/task_position/`](../../../tests/probes/task_position/) | Unit tests for labelers and probes |
