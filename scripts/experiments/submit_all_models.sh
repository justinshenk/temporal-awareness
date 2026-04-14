#!/usr/bin/bash
# ── Submit one experiment across all 5 models with correct GPU constraints ──
#
# Usage:
#   bash scripts/experiments/submit_all_models.sh refusal
#   bash scripts/experiments/submit_all_models.sh confound
#   bash scripts/experiments/submit_all_models.sh patching
#   bash scripts/experiments/submit_all_models.sh trajectory
#   bash scripts/experiments/submit_all_models.sh early_detection
#   bash scripts/experiments/submit_all_models.sh attention
#   bash scripts/experiments/submit_all_models.sh cross_model
#   bash scripts/experiments/submit_all_models.sh steering
#
# Submit ALL Wave 1 experiments (no dependencies):
#   for exp in refusal confound trajectory early_detection attention; do
#       bash scripts/experiments/submit_all_models.sh $exp
#   done
#
# Submit ALL Wave 2 experiments (after refusal finishes):
#   for exp in patching steering cross_model; do
#       bash scripts/experiments/submit_all_models.sh $exp
#   done

set -euo pipefail

EXPERIMENT="${1:?Usage: submit_all_models.sh <experiment_name>}"

# Map experiment name to SLURM script
declare -A SCRIPT_MAP=(
    ["refusal"]="submit_phase3_refusal.sh"
    ["confound"]="submit_phase3_confound.sh"
    ["patching"]="submit_phase3_patching.sh"
    ["trajectory"]="submit_phase3_trajectory.sh"
    ["early_detection"]="submit_phase3_early_detection.sh"
    ["attention"]="submit_phase3_attention.sh"
    ["cross_model"]="submit_phase3_cross_model.sh"
    ["steering"]="submit_phase4_steering.sh"
    ["causal_bridge"]="submit_phase4_causal_bridge.sh"
    ["prompt_dimensions"]="submit_phase3_prompt_dimensions.sh"
    ["safety"]="submit_phase5_safety.sh"
    ["benchmark"]="submit_benchmark_backends.sh"
    ["implicit"]="submit_phase3_implicit.sh"
)

SCRIPT="${SCRIPT_MAP[$EXPERIMENT]:-}"
if [ -z "$SCRIPT" ]; then
    echo "ERROR: Unknown experiment '$EXPERIMENT'"
    echo "Valid options: refusal, confound, patching, trajectory, early_detection, attention, cross_model, steering, causal_bridge, prompt_dimensions, safety, benchmark, implicit"
    exit 1
fi

SCRIPT_PATH="scripts/experiments/$SCRIPT"

# Models and their GPU/time requirements
# 8B models: 32GB GPU, standard time
# 30B MoE:   48GB GPU, 2x time
MODELS_8B=("Llama-3.1-8B-Instruct" "Qwen3-8B" "DeepSeek-R1-Distill-Qwen-7B" "Ouro-2.6B" "Llama-3.1-8B")
MODEL_30B="Qwen3-30B-A3B"

# Time overrides for generation-heavy experiments
declare -A TIME_8B=(
    ["refusal"]="04:00:00"
    ["confound"]="03:00:00"
    ["patching"]="04:00:00"
    ["trajectory"]="03:00:00"
    ["early_detection"]="03:00:00"
    ["attention"]="04:00:00"
    ["cross_model"]="03:00:00"
    ["steering"]="05:00:00"
    ["causal_bridge"]="06:00:00"
    ["prompt_dimensions"]="04:00:00"
    ["safety"]="08:00:00"
    ["benchmark"]="04:00:00"
    ["implicit"]="06:00:00"
)
declare -A TIME_30B=(
    ["refusal"]="08:00:00"
    ["confound"]="06:00:00"
    ["patching"]="08:00:00"
    ["trajectory"]="06:00:00"
    ["early_detection"]="06:00:00"
    ["attention"]="08:00:00"
    ["cross_model"]="06:00:00"
    ["steering"]="10:00:00"
    ["causal_bridge"]="12:00:00"
    ["prompt_dimensions"]="08:00:00"
    ["safety"]="14:00:00"
    ["benchmark"]="08:00:00"
    ["implicit"]="12:00:00"
)

echo "=========================================="
echo "Submitting: $EXPERIMENT (all 5 models)"
echo "=========================================="

# Submit 8B models on 32GB GPUs
for MODEL in "${MODELS_8B[@]}"; do
    TIME="${TIME_8B[$EXPERIMENT]}"
    JOB_ID=$(sbatch -C GPU_MEM:32GB --time=$TIME "$SCRIPT_PATH" "$MODEL" 2>&1 | grep -oP '\d+')
    echo "  $MODEL → job $JOB_ID (32GB, $TIME)"
done

# Submit 30B model on 80GB GPU (A100) — 30B MoE needs ~60GB in float16
TIME="${TIME_30B[$EXPERIMENT]}"
JOB_ID=$(sbatch -C GPU_MEM:80GB --time=$TIME "$SCRIPT_PATH" "$MODEL_30B" 2>&1 | grep -oP '\d+')
echo "  $MODEL_30B → job $JOB_ID (80GB/A100, $TIME)"

echo "=========================================="
echo "Done. Check status: squeue -u \$USER"
echo "=========================================="
