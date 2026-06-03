#!/usr/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_model.sh — Run pipeline on one or more models sequentially.
#
# USAGE:
#   Run the full list defined below:
#     chmod +x run_model.sh
#     ./run_model.sh
#
#   Or override and run a single model ad-hoc:
#     MODEL="meta-llama/Llama-3.2-3B-Instruct" NAME="llama32_3b" ./run_model.sh
# ═══════════════════════════════════════════════════════════════

# ── Models to run (format: "hf/repo:short_name", comment out to skip) ─
MODELS=(
    "Qwen/Qwen3-4B-Instruct-2507:qwen3_4b_instruct"
    "Qwen/Qwen2.5-3B-Instruct:qwen25_3b_instruct"
    # "unsloth/Llama-3.2-3B-Instruct:llama32_3b_inst"
    # "unsloth/Llama-3.1-8B-Instruct:llama31_8b_inst"
    # "allenai/OLMo-20B-0824:olmo_20b"
)

source /data/datasets/.envs/safe_env/bin/activate

SCRIPT="knowing_saying_gap.py"
DATASET="/data/b22ai063/.mech_interp/temporal-awareness/data/raw/extended_contrastive_math_dataset.json"

# Sample sizes — tune down for larger models / less VRAM
N_PROBE=100         # Stage 1: traces for probe training (~90min for 3B)
N_INTERV_BATCH=75   # Stage 4: traces PER BATCH (2 batches = 150 total)
N_SENS=70           # Stage 5: traces for sensitivity ablation

run_pipeline() {
    local MODEL=$1
    local NAME=$2
    local OUTDIR="results_${NAME}"

    echo "════════════════════════════════════════════════════════════"
    echo "  Model:   $MODEL"
    echo "  Name:    $NAME"
    echo "  Output:  $OUTDIR"
    echo "════════════════════════════════════════════════════════════"

    # ── Stage 1: Collect activations ──────────────────────────
    echo ""
    echo "━━ Stage 1: Collect activations (n=$N_PROBE) ━━"
    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 1 --n_probe $N_PROBE \
        --output_dir $OUTDIR

    # ── Stage 2: Train probes ─────────────────────────────────
    echo ""
    echo "━━ Stage 2: Train probes ━━"
    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 2 \
        --output_dir $OUTDIR

    # ── Stage 3: Behavioral analysis ──────────────────────────
    echo ""
    echo "━━ Stage 3: Behavioral analysis ━━"
    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 3 \
        --output_dir $OUTDIR

    # ── Stage 4: Intervention (2 batches) ─────────────────────
    echo ""
    echo "━━ Stage 4: Intervention (2 batches of $N_INTERV_BATCH) ━━"

    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 4 --n_intervene $N_INTERV_BATCH \
        --seed 42 \
        --output_dir ${OUTDIR}/s4b1

    echo "  Batch 1 done. Cooling 10s..."
    sleep 10

    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 4 --n_intervene $N_INTERV_BATCH \
        --seed 142 \
        --output_dir ${OUTDIR}/s4b2

    echo "  Batch 2 done. Merging..."

    python -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob('${OUTDIR}/s4b*/intervention_records.csv'))]
if dfs:
    m = pd.concat(dfs).drop_duplicates(subset=['base_id','condition'], keep='first')
    m.to_csv('${OUTDIR}/intervention_records.csv', index=False)
    print(f'  Merged: {len(m)} records -> ${OUTDIR}/intervention_records.csv')
else:
    print('  WARNING: No intervention CSVs found')
"

    # ── Stage 6: Figures ──────────────────────────────────────
    echo ""
    echo "━━ Stage 6: Paper figures ━━"
    python $SCRIPT \
        --dataset $DATASET --model "$MODEL" \
        --stages 6 \
        --output_dir $OUTDIR

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  DONE: $MODEL"
    echo "  Results: $OUTDIR/"
    ls -lh $OUTDIR/*.csv $OUTDIR/*.json $OUTDIR/*.npz 2>/dev/null
    ls -lh $OUTDIR/figures/*.pdf 2>/dev/null
    echo "════════════════════════════════════════════════════════════"
}

# ── Dispatch: env-var override runs once; otherwise loop ─────────
if [[ -n "${MODEL:-}" && -n "${NAME:-}" ]]; then
    run_pipeline "$MODEL" "$NAME"
else
    for entry in "${MODELS[@]}"; do
        run_pipeline "${entry%%:*}" "${entry##*:}"
    done
fi
