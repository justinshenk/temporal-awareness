#!/usr/bin/bash
# ── Submit ALL experiments across ALL models in one shot ──
#
# Wave 1: Independent experiments (no dependencies)
# Wave 2: Depends on refusal direction .npy files
#         Submitted with --dependency=afterok on the refusal jobs
# Wave 3: Safety evaluations (depends on refusal directions for alignment test)
#         Can run independently if refusal directions already exist
#
# Usage:
#   bash scripts/experiments/submit_all_waves.sh
#
# To submit only Wave 1:
#   bash scripts/experiments/submit_all_waves.sh wave1
#
# To submit only Wave 2 (if refusal already done):
#   bash scripts/experiments/submit_all_waves.sh wave2
#
# To submit only Wave 3 (safety evaluations):
#   bash scripts/experiments/submit_all_waves.sh wave3

set -euo pipefail

WAVE="${1:-all}"
SCRIPT_DIR="scripts/experiments"

MODELS_8B=("Llama-3.1-8B-Instruct" "Qwen3-8B" "DeepSeek-R1-Distill-Qwen-7B" "Llama-3.1-8B")
MODEL_30B="Qwen3-30B-A3B"

# Time configs: experiment -> 8B time, 30B time
declare -A TIME_8B=(
    ["refusal"]="04:00:00"    ["confound"]="03:00:00"
    ["patching"]="04:00:00"   ["trajectory"]="03:00:00"
    ["early_detection"]="03:00:00"  ["attention"]="04:00:00"
    ["cross_model"]="03:00:00"     ["steering"]="05:00:00"
    ["prompt_dimensions"]="04:00:00" ["causal_bridge"]="06:00:00"
    ["safety"]="08:00:00"
)
declare -A TIME_30B=(
    ["refusal"]="08:00:00"    ["confound"]="06:00:00"
    ["patching"]="08:00:00"   ["trajectory"]="06:00:00"
    ["early_detection"]="06:00:00"  ["attention"]="08:00:00"
    ["cross_model"]="06:00:00"     ["steering"]="10:00:00"
    ["prompt_dimensions"]="08:00:00" ["causal_bridge"]="12:00:00"
    ["safety"]="14:00:00"
)

# Script map
declare -A SCRIPT_MAP=(
    ["refusal"]="submit_phase3_refusal.sh"
    ["confound"]="submit_phase3_confound.sh"
    ["patching"]="submit_phase3_patching.sh"
    ["trajectory"]="submit_phase3_trajectory.sh"
    ["early_detection"]="submit_phase3_early_detection.sh"
    ["attention"]="submit_phase3_attention.sh"
    ["cross_model"]="submit_phase3_cross_model.sh"
    ["steering"]="submit_phase4_steering.sh"
    ["prompt_dimensions"]="submit_phase3_prompt_dimensions.sh"
    ["causal_bridge"]="submit_phase4_causal_bridge.sh"
    ["safety"]="submit_phase5_safety.sh"
)

WAVE1_EXPS=("refusal" "confound" "trajectory" "early_detection" "attention" "prompt_dimensions")
WAVE2_EXPS=("patching" "steering" "cross_model" "causal_bridge")
WAVE3_EXPS=("safety")

TOTAL=0
REFUSAL_JOBS=()

submit_experiment() {
    local EXP="$1"
    local DEP="${2:-}"  # optional dependency string

    local SCRIPT="${SCRIPT_MAP[$EXP]}"
    local SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT"

    echo ""
    echo "  ── $EXP ──"

    for MODEL in "${MODELS_8B[@]}"; do
        local TIME="${TIME_8B[$EXP]}"
        local DEP_FLAG=""
        [ -n "$DEP" ] && DEP_FLAG="--dependency=$DEP"
        local JOB_ID
        JOB_ID=$(sbatch -C GPU_MEM:32GB --time="$TIME" $DEP_FLAG "$SCRIPT_PATH" "$MODEL" 2>&1 | grep -oP '\d+')
        echo "    $MODEL → job $JOB_ID (32GB, $TIME)"

        # Track refusal jobs for Wave 2 dependency
        if [ "$EXP" = "refusal" ]; then
            REFUSAL_JOBS+=("$JOB_ID")
        fi
        TOTAL=$((TOTAL + 1))
    done

    local TIME="${TIME_30B[$EXP]}"
    local DEP_FLAG=""
    [ -n "$DEP" ] && DEP_FLAG="--dependency=$DEP"
    local JOB_ID
    JOB_ID=$(sbatch -C GPU_MEM:80GB --time="$TIME" $DEP_FLAG "$SCRIPT_PATH" "$MODEL_30B" 2>&1 | grep -oP '\d+')
    echo "    $MODEL_30B → job $JOB_ID (80GB/A100, $TIME)"

    if [ "$EXP" = "refusal" ]; then
        REFUSAL_JOBS+=("$JOB_ID")
    fi
    TOTAL=$((TOTAL + 1))
}

echo "=========================================="
echo "TEMPORAL AWARENESS — FULL EXPERIMENT SUBMISSION"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "=========================================="

# ── Wave 1 ──
if [ "$WAVE" = "all" ] || [ "$WAVE" = "wave1" ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  WAVE 1: Independent experiments     ║"
    echo "╚══════════════════════════════════════╝"

    for EXP in "${WAVE1_EXPS[@]}"; do
        submit_experiment "$EXP"
    done
fi

# ── Wave 2 ──
if [ "$WAVE" = "all" ] || [ "$WAVE" = "wave2" ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  WAVE 2: Depends on refusal results  ║"
    echo "╚══════════════════════════════════════╝"

    if [ "$WAVE" = "all" ] && [ ${#REFUSAL_JOBS[@]} -gt 0 ]; then
        # Build dependency string: afterok:job1:job2:job3...
        DEP_STR="afterok"
        for JID in "${REFUSAL_JOBS[@]}"; do
            DEP_STR="${DEP_STR}:${JID}"
        done
        echo ""
        echo "  Waiting for refusal jobs: ${REFUSAL_JOBS[*]}"
        echo "  Dependency: $DEP_STR"

        for EXP in "${WAVE2_EXPS[@]}"; do
            submit_experiment "$EXP" "$DEP_STR"
        done
    else
        # Wave 2 standalone — assume refusal already done
        echo ""
        echo "  (No dependency — assuming refusal results exist)"

        for EXP in "${WAVE2_EXPS[@]}"; do
            submit_experiment "$EXP"
        done
    fi
fi

# ── Wave 3 ──
if [ "$WAVE" = "all" ] || [ "$WAVE" = "wave3" ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  WAVE 3: Safety evaluations          ║"
    echo "╚══════════════════════════════════════╝"

    if [ "$WAVE" = "all" ] && [ ${#REFUSAL_JOBS[@]} -gt 0 ]; then
        # Safety evals benefit from refusal directions but can also extract them fresh
        DEP_STR="afterok"
        for JID in "${REFUSAL_JOBS[@]}"; do
            DEP_STR="${DEP_STR}:${JID}"
        done
        echo ""
        echo "  Waiting for refusal jobs: ${REFUSAL_JOBS[*]}"
        echo "  Dependency: $DEP_STR"

        for EXP in "${WAVE3_EXPS[@]}"; do
            submit_experiment "$EXP" "$DEP_STR"
        done
    else
        # Wave 3 standalone — assume refusal results exist or will extract fresh
        echo ""
        echo "  (No dependency — safety evals will extract directions if needed)"

        for EXP in "${WAVE3_EXPS[@]}"; do
            submit_experiment "$EXP"
        done
    fi
fi

echo ""
echo "=========================================="
echo "TOTAL JOBS SUBMITTED: $TOTAL"
echo "Check status: squeue -u \$USER"
echo "=========================================="
