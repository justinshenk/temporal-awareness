# Degradation Mechanistic Interpretability Experiment Suite

Mechanistic interpretability of patience degradation across four architecture families (dense, MoE, reasoning-distilled, looped) and seven models. This experiment suite supports the paper: *"Temporal Awareness in Language Models: Mechanistic Interpretability of Patience Degradation Under Repetitive Tasks"* (NeurIPS 2026).

## Quick Start

```bash
# Run a single experiment on one model (local GPU):
python experiments/degradation_mechanistic/run_experiment.py \
    --model Llama-3.1-8B-Instruct --experiment refusal

# Run all Phase 3 experiments on one model:
python experiments/degradation_mechanistic/run_experiment.py \
    --model Qwen3-8B --phase 3

# Run the full suite on all models (Sherlock SLURM):
python experiments/degradation_mechanistic/run_experiment.py \
    --all-models --all-phases --slurm

# List all available experiments:
python experiments/degradation_mechanistic/run_experiment.py --list
```

## Experiment Overview

The suite is organized into three phases with dependency-aware wave ordering:

### Phase 3 — Mechanistic Analysis (11 experiments)

| Experiment | Script | Paper Section | Key Measurement |
|------------|--------|:------------:|-----------------|
| `refusal` | `phase3_refusal_direction.py` | §4.1 | cos(degradation, refusal) per layer |
| `confound` | `phase3_context_confound.py` | §4.1 | 4-condition accuracy comparison |
| `patching` | `phase3_causal_patching.py` | §4.2 | Forward/backward patching ΔAccuracy |
| `trajectory` | `phase3_trajectory_geometry.py` | §4.3 | PCA trajectory, velocity, curvature |
| `attention` | `phase3_attention_analysis.py` | §4.4 | Head entropy, task-token weight |
| `early_detection` | `phase3_early_detection.py` | §4.5 | Probe detection lead time |
| `cross_model` | `phase3_cross_model_transfer.py` | §4.6 | Cross-model probe accuracy, CKA |
| `prompt_dimensions` | `phase3_prompt_dimensions.py` | §4.7 | 6-dimension direction similarity |
| `implicit` | `phase3_implicit_repetition.py` | §5 | 3-condition direction transfer |
| `reasoning` | `phase3_reasoning_degradation.py` | §5.4 | Think vs output probe accuracy |

### Phase 4 — Causal Bridge (2 experiments)

| Experiment | Script | Paper Section | Key Measurement |
|------------|--------|:------------:|-----------------|
| `causal_bridge` | `phase4_causal_bridge.py` | §6 | r(r_obs, r_causal) across layers |
| `steering` | `phase4_intervention_steering.py` | §6.4 | Steering ΔAccuracy, dose-response |

### Phase 5 — Safety (1 experiment, 4 sub-evaluations)

| Experiment | Script | Paper Section | Key Measurement |
|------------|--------|:------------:|-----------------|
| `safety` | `phase5_safety_evaluations.py` | §7 | Injection resistance, refusal erosion, alignment stability, intervention efficacy |

## Models

| Model | Architecture | Parameters | GPU Requirement |
|-------|-------------|:----------:|:---------------:|
| Llama-3.1-8B-Instruct | Dense | 8B | 32GB (A40) |
| Llama-3.1-8B | Dense (base) | 8B | 32GB |
| Qwen3-8B | Dense | 8B | 32GB |
| Qwen3-4B-Instruct-2507 | Dense | 4B | 32GB |
| Qwen3-30B-A3B | MoE | 30B (3B active) | 80GB (A100) |
| DeepSeek-R1-Distill-Qwen-7B | Dense (distilled) | 7B | 32GB |
| Ouro-2.6B | Looped (4 recurrent steps) | 2.6B | 32GB |

## Dependency / Wave Structure

Experiments are grouped into waves to respect data dependencies:

```
Wave 1 (independent):      refusal, confound, trajectory, early_detection,
                            attention, prompt_dimensions

Wave 2 (after refusal):    patching, steering, cross_model, causal_bridge,
                            implicit, reasoning

Wave 3 (after refusal):    safety
```

`patching`, `steering`, and `safety` use the refusal direction extracted in Wave 1. They can still run independently (they re-extract if the file is missing), but results are most consistent when using the same refusal direction.

```bash
# Submit waves sequentially on Sherlock (with SLURM dependencies):
bash scripts/experiments/submit_all_waves.sh

# Or submit a specific wave:
bash scripts/experiments/submit_all_waves.sh wave1
bash scripts/experiments/submit_all_waves.sh wave2  # after wave1 completes
bash scripts/experiments/submit_all_waves.sh wave3
```

## Running Locally

```bash
# Single experiment, single model:
python experiments/degradation_mechanistic/run_experiment.py \
    --model Qwen3-8B --experiment refusal --device cuda

# Quick mode (reduced samples, ~10 min per experiment):
python experiments/degradation_mechanistic/run_experiment.py \
    --model Qwen3-8B --experiment refusal --quick

# All experiments, one model:
python experiments/degradation_mechanistic/run_experiment.py \
    --model Llama-3.1-8B-Instruct --all-phases

# Choose activation extraction backend:
python experiments/degradation_mechanistic/run_experiment.py \
    --model Qwen3-8B --experiment refusal --backend transformer_lens
```

## Running on Sherlock (SLURM)

```bash
# Full suite, all models, all waves (482 jobs):
python experiments/degradation_mechanistic/run_experiment.py \
    --all-models --all-phases --slurm

# Or use the wave-aware shell script directly:
bash scripts/experiments/submit_all_waves.sh
```

Individual SLURM submit scripts are in `scripts/experiments/submit_phase*.sh`. Each handles GPU constraints, HuggingFace cache, and virtual environment setup automatically.

## Configuration

Default experiment parameters are in `configs/experiments/phase3_default.yaml`:

- Repetition counts: `[1, 2, 5, 10, 20, 50]`
- Extraction: `resid_post` at last token position, `float16`, batch size 4
- Probing: Logistic regression, 5-fold CV
- W&B: `justinshenk-time/patience-degradation`

Override via CLI arguments (all scripts accept `--model`, `--device`, `--backend`, `--quick`).

## Outputs

Results are written to:

```
results/
├── checkpoints/                    # Extracted directions (.npy)
│   ├── temporal_steering.json
│   └── refusal_direction_*.npy
├── activation_patching/            # Causal patching results
├── probe_validation_results.json   # Phase 1 probe data
├── sae_probing_experiment/         # SAE analysis (Gemma-2-2b)
├── confidence_scaling/             # Confidence-horizon correlation
├── eap_integrated_gradients/       # EAP-IG attribution scores
└── verified/                       # Validated probe accuracy CSVs
```

Experiment logs (when run via SLURM) are in `logs/*.out` and `logs/*.err`.

All runs log to W&B at `https://wandb.ai/justinshenk-time/patience-degradation`.

## File Structure

```
experiments/degradation_mechanistic/
├── README.md               # This file
└── run_experiment.py        # Single entry-point script

scripts/experiments/
├── phase3_*.py              # Phase 3 experiment implementations (11 files)
├── phase4_*.py              # Phase 4 experiment implementations (2 files)
├── phase5_*.py              # Phase 5 experiment implementation (1 file)
├── submit_phase*.sh         # Per-experiment SLURM submit scripts
├── submit_all_models.sh     # Submit one experiment across all models
├── submit_all_waves.sh      # Submit all experiments with wave dependencies
└── README.md                # Legacy Phase 1 documentation

configs/experiments/
└── phase3_default.yaml      # Default experiment parameters

src/activation_api/          # Activation extraction API (dual-backend)
├── config.py                # ExtractionConfig, ModuleSpec
├── extractor.py             # ActivationExtractor (TL + PyTorch backends)
├── hooks.py                 # HookManager (hook lifecycle + buffering)
└── result.py                # ActivationResult (dict-like access + serialization)
```

## Reproducing Paper Results

To reproduce all results from the paper:

```bash
# 1. Set up environment on Sherlock
module load python/3.12.1
source ~/sae-env/bin/activate
export HF_HOME=$SCRATCH/.cache/huggingface

# 2. Submit all 482 jobs (3 waves, 7 models, 14 experiment types)
bash scripts/experiments/submit_all_waves.sh

# 3. Monitor
squeue -u $USER
# W&B: https://wandb.ai/justinshenk-time/patience-degradation

# 4. Results appear in results/ and logs/
```

Total compute: ~2,400 GPU-hours on A40/A100 (Sherlock).
