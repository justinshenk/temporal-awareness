# LoRA vs ICL — Activation-Subspace Comparison (DDXPlus)

Tests whether **LoRA finetuning** and **in-context learning (ICL)** push Gemma-2-9B-IT's
residual stream along the **same direction / subspace** on the DDXPlus MCQ task.

Both effects are measured as an activation shift at the prediction site (final prompt token,
where the model emits its answer letter), relative to the shared `base + clean` baseline:

```
icl_shift  = resid(base, case WITH accumulated DDXPlus context) - resid(base, case clean)
lora_shift = resid(LoRA, case clean)                            - resid(base, case clean)
```

Primary metric: **per-layer cosine of the mean shifts + principal angles between their top-k
PCA subspaces** (`src/probes/lora_icl/subspace_metrics.py`). Controls: random null (±1/√d),
shuffled-label LoRA (task-specificity), and the per-layer profile.

Design brief: [`tasks/current_task.md`](../../tasks/current_task.md).
Config: [`configs/lora_icl/ddxplus_gemma_lora.yaml`](../../configs/lora_icl/ddxplus_gemma_lora.yaml).

## Run (needs an H200-class GPU + `HF_TOKEN` for gated Gemma)

```bash
export HF_TOKEN=...   # gated google/gemma-2-9b-it
CFG=configs/lora_icl/ddxplus_gemma_lora.yaml

# 1. Train the real adapter + the shuffled-label control adapter
uv run python -m scripts.lora_icl.train_ddxplus_lora --config $CFG
uv run python -m scripts.lora_icl.train_ddxplus_lora --config $CFG --shuffle-labels

# 2. Extract shifts (real run also writes the shared icl_shift; control reuses it)
uv run python -m scripts.lora_icl.extract_shifts --config $CFG \
    --adapter results/lora_icl/adapter --tag real
uv run python -m scripts.lora_icl.extract_shifts --config $CFG \
    --adapter results/lora_icl/adapter_shuffled --tag shuffled --skip-icl

# 3. Compare + write the report
uv run python -m scripts.lora_icl.run_subspace_comparison --config $CFG
```

Output: `results/lora_icl/2026-06-01-subspace-comparison.{md,json}`.

## Artifacts

| Path | Committed? | Notes |
|------|:----------:|-------|
| `adapter/`, `adapter_shuffled/` | no (gitignored) | LoRA weights — regenerate from step 1 |
| `shifts/*.npy`, `shifts/meta_*.json` | no (gitignored) | shift tensors — regenerate from step 2 |
| `2026-06-01-subspace-comparison.{md,json}` | yes | the result |

Adapters/tensors are deterministic (seed 42), so the committed report is reproducible from the
two committed inputs (config + code).

## Code

| Module | Purpose |
|--------|---------|
| `src/probes/lora_icl/subspace_metrics.py` | cosine, PCA, principal angles, `LayerSubspaceResult` |
| `src/probes/lora_icl/shift_extraction.py` | model-free shift-set assembly |
| `src/probes/lora_icl/ddxplus_cases.py` | shared, deterministic DDXPlus MCQ case builder |
| `scripts/lora_icl/train_ddxplus_lora.py` | PEFT LoRA trainer (+ `--shuffle-labels` control) |
| `scripts/lora_icl/extract_shifts.py` | ICL + LoRA shift extraction at the prediction site |
| `scripts/lora_icl/run_subspace_comparison.py` | metrics + markdown/JSON report |
| `tests/probes/lora_icl/` | unit + integration + offline plumbing tests (34) |
