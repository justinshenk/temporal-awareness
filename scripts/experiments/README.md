# Experiments

Three experiments for the temporal-awareness project, run on Stanford's Sherlock GPU cluster.

| # | Experiment | Script | RQ | W&B Project |
|---|-----------|--------|-----|-------------|
| 1 | SAE Feature Stability | `sae_feature_stability.py` | RQ1 | `sae-feature-stability` |
| 2 | Sequential Activation Tracking | `sequential_activation_tracking.py` | RQ1 | `sequential-tracking` |
| 3 | Patience Degradation | `patience_degradation.py` | RQ3 | `patience-degradation` |

## Models

| Model | Experiments | SAE | Notes |
|-------|-----------|-----|-------|
| `gemma-2-2b` | 1, 2, 3 | gemma-scope-2b-pt-res-canonical | Modern arch, high-quality SAEs |
| `gpt2` | 1, 2, 3 | gpt2-small-res-jb | Standard mech interp baseline |
| `pythia-70m` | 1, 2, 3 | pythia-70m-deduped-res-sm | Minimal scale test |
| `Qwen/Qwen2.5-3B-Instruct` | 3 | None (activation probes only) | Instruction-tuned, RLHF'd |
| `meta-llama/Llama-3.1-8B-Instruct` | 3 | None (activation probes only) | Largest model, needs 48GB GPU |

## Sherlock Setup

### 1. Environment

```bash
module load python/3.12.1
module load py-pyarrow/18.1.0_py312
python -m venv ~/sae-env
source ~/sae-env/bin/activate
pip install -r requirements-sherlock.txt
```

### 2. HuggingFace Auth

Some models are gated (Gemma-2-2b, Llama-3.1-8B). Accept licenses on HuggingFace, then:

```bash
huggingface-cli login
```

The token is stored at `~/.cache/huggingface/token`. SLURM scripts export it automatically.

### 3. HF Cache on Scratch

Home quota is 15GB — models won't fit. Use scratch:

```bash
mkdir -p $SCRATCH/.cache/huggingface
export HF_HOME=$SCRATCH/.cache/huggingface
```

SLURM scripts set this automatically.

## Running Experiments

### Quick Test (single layer, 3 repetition counts)

```bash
# Single model, single experiment
sbatch scripts/experiments/train_sae_stability.sh gemma-2-2b quick
sbatch scripts/experiments/train_seq_tracking.sh gpt2 quick
sbatch scripts/experiments/train_patience_deg.sh pythia-70m quick

# Instruction-tuned models (needs more memory)
sbatch scripts/experiments/train_patience_deg_large.sh "Qwen/Qwen2.5-3B-Instruct" quick
sbatch scripts/experiments/train_patience_deg_large.sh "meta-llama/Llama-3.1-8B-Instruct" quick
```

### Full Run (multiple layers, 7 repetition counts)

```bash
# All experiments, all models
bash scripts/experiments/submit_all_experiments.sh full

# Or individually
sbatch scripts/experiments/train_sae_stability.sh gemma-2-2b full
sbatch scripts/experiments/train_patience_deg.sh gpt2 full
```

### Script Arguments

All SLURM scripts follow the same pattern:

```
sbatch <script>.sh MODEL MODE [EXTRA_ARGS]
```

- `MODEL`: model name from MODEL_CONFIGS (e.g., `gemma-2-2b`, `gpt2`, `pythia-70m`)
- `MODE`: `quick` or `full` (default: `full`)
- `EXTRA_ARGS`: passed through to the Python script (e.g., `"--layers 5 10"`)

### Interactive Testing

```bash
sh_dev -g 1
source ~/sae-env/bin/activate
export HF_HOME=$SCRATCH/.cache/huggingface
python scripts/experiments/patience_degradation.py --model gemma-2-2b --device cuda --quick
```

## Monitoring

### Job Queue

```bash
squeue -u $USER
```

### Job Output

```bash
cat slurm-<JOBID>.out
```

### W&B

Results are logged to the `justinshenk-time` W&B team:

- https://wandb.ai/justinshenk-time/sae-feature-stability
- https://wandb.ai/justinshenk-time/sequential-tracking
- https://wandb.ai/justinshenk-time/patience-degradation

### Results Files

```
results/
  sae_feature_stability/    # Experiment 1
  sequential_tracking/      # Experiment 2
  patience_degradation/     # Experiment 3
```

Each directory contains timestamped JSON results and PNG plots.

## GPU Requirements

| Model | VRAM | SLURM Script | GPU Constraint |
|-------|------|-------------|----------------|
| gemma-2-2b | ~16GB | `train_*.sh` | `GPU_BRD:TESLA` (V100) |
| gpt2 | ~2GB | `train_*.sh` | `GPU_BRD:TESLA` (V100) |
| pythia-70m | ~1GB | `train_*.sh` | `GPU_BRD:TESLA` (V100) |
| Qwen2.5-3B-Instruct | ~8GB (fp16) | `train_patience_deg_large.sh` | Any |
| Llama-3.1-8B-Instruct | ~24GB (fp16) | `train_patience_deg_large.sh` | Check `node_feat -p gpu \| grep GPU_` |

## Troubleshooting

**403 Gated Repo Error**: Accept the model license on HuggingFace and run `huggingface-cli login`.

**Home Quota Exceeded**: Move HF cache to scratch: `mv ~/.cache/huggingface/hub $SCRATCH/.cache/huggingface/`

**SafetensorError (corrupted download)**: Delete and re-download: `rm -rf $SCRATCH/.cache/huggingface/hub/models--<org>--<model>`

**Module not found (sae_lens, etc.)**: Activate venv first: `source ~/sae-env/bin/activate`
