#!/usr/bin/bash
# ── SLURM submission for Phase 3: Reasoning-Phase Degradation Analysis ──
#
# Analyzes how chain-of-thought quality degrades under repetition.
# Targets reasoning models only (DeepSeek-R1, Qwen3-8B, Qwen3-4B).
#
# Usage:
#   sbatch scripts/experiments/submit_phase3_reasoning.sh DeepSeek-R1-Distill-Qwen-7B
#   sbatch scripts/experiments/submit_phase3_reasoning.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase3_reasoning.sh Qwen3-4B-Instruct-2507

#SBATCH --job-name=p3-reason
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase3_reasoning_%j.out
#SBATCH --error=logs/phase3_reasoning_%j.err

MODEL="${1:-DeepSeek-R1-Distill-Qwen-7B}"

module load python/3.12.1
module load py-pyarrow/18.1.0_py312

if [ -f /home/groups/barbarae/molofsky/ml-env/bin/activate ]; then
    source /home/groups/barbarae/molofsky/ml-env/bin/activate
elif [ -f ~/sae-env/bin/activate ]; then
    source ~/sae-env/bin/activate
else
    echo "WARNING: Could not find virtualenv, using system python"
fi

export HF_HOME=$SCRATCH/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

echo "=========================================="
echo "Phase 3: Reasoning-Phase Degradation"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

srun python3 scripts/experiments/phase3_reasoning_degradation.py \
    --model "$MODEL" --device cuda --wandb-project patience-degradation

echo "Finished: $(date), exit code: $?"
