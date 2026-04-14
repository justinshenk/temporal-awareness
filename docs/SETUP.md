# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU with >= 32 GB VRAM (for 8B models) or >= 80 GB (for Qwen3-30B-A3B)
- HuggingFace account with accepted model licenses (Llama 3.1, DeepSeek-R1)

## Installation

```bash
# Clone and install
git clone https://github.com/justinshenk/temporal-awareness.git
cd temporal-awareness
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env to add: HF_TOKEN, WANDB_API_KEY
```

## HPC Setup (Stanford Sherlock)

The experiments are designed to run on SLURM-managed GPU clusters. On Sherlock:

```bash
# Load required modules
module load python/3.12.1
module load py-pyarrow/18.1.0_py312

# Activate the shared virtualenv
source /home/groups/barbarae/molofsky/ml-env/bin/activate

# For Ouro-2.6B (requires older transformers):
source /home/groups/barbarae/molofsky/ouro-env/bin/activate

# Set cache directories
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

## Running Experiments

### Single experiment, single model
```bash
sbatch scripts/experiments/submit_phase3_refusal.sh Llama-3.1-8B-Instruct
```

### Single experiment, all 7 models
```bash
bash scripts/experiments/submit_all_models.sh refusal
```

### Full experiment suite (Wave 1 — no dependencies)
```bash
for exp in refusal confound trajectory early_detection attention implicit prompt_dimensions reasoning; do
    bash scripts/experiments/submit_all_models.sh $exp
done
```

### Wave 2 (requires refusal results)
```bash
for exp in patching steering cross_model causal_bridge safety; do
    bash scripts/experiments/submit_all_models.sh $exp
done
```

## Verification

After experiments complete, verify claims from the paper:

```bash
make verify-quick    # Uses cached results
make verify          # Full re-run (requires GPU)
```

## W&B Dashboard

All experiments log to the `patience-degradation` project:
https://wandb.ai/justinshenk-time/patience-degradation
