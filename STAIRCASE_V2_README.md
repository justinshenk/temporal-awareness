# Staircase v2 — EMNLP push

Position-baseline staircase diagnostic, EMNLP version. Extends the ICML
mech-interp workshop submission with:

* Multi-domain coverage (code, rhyme, QA-suggestive, QA-neutral, trivia)
* 26 model variants from Maar et al. (ICLR 2026) for direct ground-truth
  comparison + 11 workshop models for code anchor
* Per-position probe curves (not just aggregated baselines)
* Causal ablation (zero + mean) at N+P positions
* MLP-probe robustness pass (Hewitt & Liang control)
* Pre-registration check built into every result

## The thesis (locked)

> **The staircase is a discriminator for genuine planning vs surface
> confounds, validated against Maar et al.'s causally-established
> ground truth.**

Pre-registered prediction matrix (encoded in `src/lookahead/domains/`):

| Domain          | Predicted gap        | Why                                                    |
|-----------------|----------------------|--------------------------------------------------------|
| code            | NEGATIVE             | Workshop result; signature carries return-type info    |
| trivia          | NEGATIVE             | Constructed negative control                           |
| qa\_suggestive  | NEAR\_ZERO           | Surface content partially carries the answer           |
| qa\_neutral     | STRONG POSITIVE      | Same prompt text for both pair members — planning only |
| rhyme           | STRONG POSITIVE      | Maar's primary causal-validation domain                |

## Architecture

```
src/lookahead/
  domains/__init__.py             ← Domain specs + position resolvers
  datasets/
    maar_data.py                  ← Loads Maar's rhyme + QA + neutral
    trivia.py                     ← 500-question negative control
    rhyme.py                      ← Workshop's curated rhyme (legacy)
    code_return.py                ← Workshop's 500-signature code dataset
  probing/
    hf_activation_extraction.py   ← HF-based (replaces TL for new models)
    activation_extraction.py      ← TransformerLens-based (legacy)
    commitment_probes.py          ← Per-position probe training
    comprehensive_baselines.py    ← BoW / PCA / shuffle / etc.
    np_ablation.py                ← Zero / mean ablation of N+P positions
    mlp_probe.py                  ← sklearn-compatible MLP probe
    staircase_headline.py         ← Headline + pre-registration check

scripts/lookahead/experiments/
  run_staircase_v2.py             ← Unified domain-agnostic runner (CLI)
  launch_partition.py             ← 4-GPU partition orchestrator
  bootstrap_vastai.sh             ← Vast.ai one-shot setup
  run_rq4_final.py                ← Workshop's runner (kept for reference)
```

## GPU partitions

121 jobs total across 4 GPUs:

| Partition | GPU                  | Models                                     | Jobs | Est. cost   |
|-----------|----------------------|--------------------------------------------|------|-------------|
| A         | A100 / H100 80GB     | 27B / 32B / Llama-3.3-70B (8 variants)     | 32   | $130–200    |
| B         | RTX 6000 Ada 48GB    | 8B / 12B / 14B (10 variants)               | 40   | $30–50      |
| C         | RTX 6000 Ada 48GB    | 1B / 4B (8) + workshop 11 (code)           | 43   | $30–50      |
| D         | A6000 48GB           | Pythia 410M / 1B / 2.8B (negative controls)| 6    | $15–35      |
| **Total** |                      |                                            | 121  | **~$235–365 + buffer** |

## Quickstart

### On each Vast.ai instance

```bash
# 1) Bootstrap (clones repo, installs deps, runs smoke test)
export HF_TOKEN=hf_...
curl -sSL $REPO/raw/branch/scripts/lookahead/experiments/bootstrap_vastai.sh \
    | bash -s -- A     # partition letter

# OR manually after cloning:
bash scripts/lookahead/experiments/bootstrap_vastai.sh A
```

### Manual single-job invocation

```bash
python scripts/lookahead/experiments/run_staircase_v2.py \
    --model google/gemma-2-9b-it \
    --domain rhyme \
    --output_dir results/v2 \
    --quantization bf16 \
    --layer_mode maar_range \
    --probe_types linear
```

### Re-run with MLP probe (rigor pass)

```bash
python scripts/lookahead/experiments/launch_partition.py \
    --partition B \
    --probe_types linear,mlp \
    --output_dir results/v2_mlp
```

## Per-model output

Each `(model, domain)` pair produces one JSON:
```
results/v2/{model_slug}__{domain}__staircase.json
```

Shape:
```python
{
  "meta": {model, domain, n_examples, layers, predicted_gap, ...},
  "baselines": {chance, bag_of_words_accuracy},
  "per_layer": {
    "linear__L0":   {per_position: {0: {cv_accuracy_mean,...}}, ...},
    "linear__L5":   {...},
    ...
  },
  "headlines": [
    {
      "layer": 5,
      "resolver": "last_word_before_newline",
      "target_accuracy": 0.85,
      "max_earlier_accuracy": 0.45,
      "headline_gap": 0.40,
      "pre_registration_check": {
        "observed_sign": "strong_positive",
        "predicted_sign": "strong_positive",
        "matches": true
      }
    },
    ...
  ]
}
```

## Maar et al. supplementary material

The rhyme + QA datasets come from the ICLR 2026 supplementary materials
of Maar, Paperno, McDougall, Nanda. Extract their ZIP to
`data/maar_supplementary_material/`. Loader expects:

```
data/maar_supplementary_material/
  train/
    rhyme_family_lines.json
    noun_qa.json
  test/
    rhyme_family_lines.json
    noun_qa.json
    noun_qa_neutral_filtered.json
```

Set `MAAR_DATA_ROOT` env var if extracting elsewhere.
