# RQ47 Experiment Suite

Issue #47 asks whether temporal representations are useful as real-time
oversight signals. The work should be presented as a progression from static
decodability to causal and online evidence.

## Experiment Map

| Phase | Question | Status / Script | Main Output |
| --- | --- | --- | --- |
| 1 | Can probes detect temporal horizon? | Existing probe training/validation | Probe accuracy by layer/model/method |
| 2 | Does the signal generalize across models? | Existing multimodel probe notebooks | Cross-model comparison figures |
| 3 | Is the signal causal? | `scripts/experiments/multimodel_temporal_interventions.py` | Steering, patching, attribution, ablation CSV/JSON |
| 4 | Does the signal move during unfolding generation? | `scripts/experiments/temporal_probe_trajectory_monitor.py` | Token-level temporal probe trajectories |
| 5 | Does drift precede safety-relevant behavior? | Trajectory monitor now, classifier join later | Precursor timeline and lead-time metrics |

## New Code

```text
scripts/experiments/multimodel_temporal_interventions.py
scripts/experiments/temporal_probe_trajectory_monitor.py
data/raw/temporal_oversight_sequences.json
docs/INTERVENTION_EXPERIMENTS.md
```

Outputs go to:

```text
results/temporal_interventions/
results/temporal_oversight/
```

## Step 0: Sanity Checks

```bash
python -m py_compile \
  scripts/experiments/multimodel_temporal_interventions.py \
  scripts/experiments/temporal_probe_trajectory_monitor.py
```

Check CLI help:

```bash
python scripts/experiments/multimodel_temporal_interventions.py --help
python scripts/experiments/temporal_probe_trajectory_monitor.py --help
```

## Step 1: Causal Interventions, Tiny Smoke Test

Run all four causal modes on one GPT-2 prompt/layer:

```bash
.venv/bin/python scripts/experiments/multimodel_temporal_interventions.py \
  --experiment all \
  --models gpt2 \
  --layers 0 \
  --components resid \
  --max-pairs 1 \
  --direction-max-pairs 1 \
  --strengths 0 \
  --local-files-only
```

Expected output folders:

```text
results/temporal_interventions/steering/{lr,dmm,attn}/
results/temporal_interventions/activation_patching/{lr,dmm,attn}/
results/temporal_interventions/attribution_patching/{lr,dmm,attn}/
results/temporal_interventions/ablation/{lr,dmm,attn}/
```

Filenames and CSV/JSON metadata include the layer selector, for example:

```text
gpt2_steering_layers-best_from-lr_YYYYMMDDTHHMMSSZ.csv
```

## Step 2: Causal Interventions, Four-Model Main Run

Start with the best LR-selected layer. This is the fastest defensible main run:

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

Then run the top three DMM layers if compute allows:

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

Presentation claims to test:

- Steering: adding a temporal direction shifts short-vs-long logits.
- Activation patching: clean temporal activations recover clean behavior.
- Attribution patching: first-order estimates agree with activation patching.
- Ablation: removing selected components reduces temporal behavior.

## Step 3: Online Temporal Oversight, Tiny Smoke Test

```bash
.venv/bin/python scripts/experiments/temporal_probe_trajectory_monitor.py \
  --models gpt2 \
  --layers 0 \
  --max-prompts 1 \
  --max-new-tokens 2 \
  --local-files-only
```

Expected output:

```text
results/temporal_oversight/*_trajectory.csv
results/temporal_oversight/*_summary.csv
```

The trajectory file has one row per model/prompt/layer/token step. The summary
file has one row per model/prompt/layer with initial score, final score, max
score, event step, and pre-event drift.

## Step 4: Online Temporal Oversight, Main Run

Run the best LR layer across the four models:

```bash
.venv/bin/python scripts/experiments/temporal_probe_trajectory_monitor.py \
  --models gpt2 qwen3-4b phi-3-mini-4k-instruct llama-3.2-3b \
  --layers best \
  --layer-source-method lr \
  --max-new-tokens 48 \
  --temperature 0 \
  --max-prompts 6 \
  --local-files-only \
  --attn-implementation eager
```

If there is time, expand to the top three LR layers:

```bash
.venv/bin/python scripts/experiments/temporal_probe_trajectory_monitor.py \
  --models gpt2 qwen3-4b phi-3-mini-4k-instruct llama-3.2-3b \
  --layers top-k \
  --top-k-layers 3 \
  --layer-source-method lr \
  --max-new-tokens 48 \
  --temperature 0 \
  --max-prompts 6 \
  --local-files-only \
  --attn-implementation eager
```

## Step 5: Analysis To Present

Minimum tables/plots to produce from the CSVs:

1. Steering dose-response by model and layer.
2. Activation patching normalized recovery by model/component/layer.
3. Attribution-vs-activation patching agreement.
4. Ablation effect by model/component/layer.
5. Temporal probe trajectory over generated token index.
6. Pre-event drift table: `delta_before_event` and `first_event_step`.

## Framing For Slides

Use this story:

1. Static probes show temporal horizon is decodable.
2. Cross-model validation shows it is not only a GPT-2 artifact.
3. Steering and patching test whether the direction is causally active.
4. Trajectory monitoring tests whether the signal can act as online oversight.
5. The safety detector is currently keyword-based for offline smoke tests; the
   same output schema can be joined with LlamaGuard or another classifier for
   the final safety-precursor claim.

## Known Limitations

- The current safety event detector is keyword-based. This is acceptable for
  pipeline validation, but not for the final safety claim.
- Ablation is component-level across Hugging Face models. Per-head ablation is
  a follow-up for TransformerLens-supported models.
- The trajectory monitor recomputes full prefixes for clarity and portability;
  it is slower than a KV-cache implementation.
