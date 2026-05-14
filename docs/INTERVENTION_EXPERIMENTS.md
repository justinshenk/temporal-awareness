# Temporal Intervention Experiments

This runbook covers the post-probing work package: steering, activation
patching, attribution patching, and component ablations across the four-model
comparison set.

The implementation is additive and lives in:

```bash
scripts/experiments/multimodel_temporal_interventions.py
```

Outputs are written under:

```bash
results/temporal_interventions/
```

## Research Story

The probing results show that temporal preference is decodable. These
intervention experiments test whether the representation is causally relevant.

1. Probe-guided steering: does adding a temporal direction shift short/long
   answer logits?
2. Activation patching: do clean activations recover the clean answer on a
   corrupted prompt?
3. Attribution patching: do gradient-based patch estimates agree with
   activation patching?
4. Component ablation: do residual, attention, or MLP outputs matter for the
   temporal answer?

## Smoke Tests

Run one small GPT-2 check before launching larger sweeps:

```bash
.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment steering \
  --models gpt2 \
  --layers 0 \
  --components resid \
  --max-pairs 1 \
  --direction-max-pairs 1 \
  --strengths 0 \
  --local-files-only
```

Then test the other modes:

```bash
.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment activation_patching \
  --models gpt2 \
  --layers 0 \
  --components resid \
  --max-pairs 1 \
  --local-files-only

.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment attribution_patching \
  --models gpt2 \
  --layers 0 \
  --components resid \
  --max-pairs 1 \
  --local-files-only

.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment ablation \
  --models gpt2 \
  --layers 0 \
  --components resid \
  --max-pairs 1 \
  --local-files-only
```

## Four-Model Sweep

Start with the best LR-selected layer for each model:

```bash
.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment all \
  --models gpt2 qwen3-4b phi-3-mini-4k-instruct llama-3.2-3b \
  --layers best \
  --layer-source-method lr \
  --components resid attn mlp \
  --max-pairs 7 \
  --direction-max-pairs 7 \
  --strengths -3 -2 -1 0 1 2 3 \
  --local-files-only \
  --attn-implementation eager
```

Then expand to the top three DMM layers:

```bash
.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment all \
  --models gpt2 qwen3-4b phi-3-mini-4k-instruct llama-3.2-3b \
  --layers top-k \
  --top-k-layers 3 \
  --layer-source-method dmm \
  --components resid attn mlp \
  --max-pairs 7 \
  --direction-max-pairs 7 \
  --strengths -3 -2 -1 0 1 2 3 \
  --local-files-only \
  --attn-implementation eager
```

## Expected Presentation Claims

Use these only if the results support them:

- Steering: temporal directions causally shift short-vs-long logits.
- Activation patching: probe-selected layers recover clean behavior on
  corrupted prompts.
- Attribution patching: first-order estimates identify similar layers or
  components as activation patching.
- Ablation: temporal behavior depends more on residual/MLP/attention components
  in specific model families.

## Notes

- The default classification datasets are the existing aligned patching pairs
  for Qwen and Phi. GPT-2 and Llama use the Qwen-format pairs unless a dataset
  is passed explicitly with `--dataset`.
- `--layers best` and `--layers top-k` read from existing probe CSVs in
  `research/results/{lr,dmm,attn}`.
- The current ablation mode is component-level (`resid`, `attn`, `mlp`), which
  generalizes across Hugging Face decoder models. Per-head ablation is a good
  follow-up when using TransformerLens-supported models.
