# Temporal Reasoning

Research on detecting and steering temporal scope representations in LLMs.

## Overview

This project investigates how LLMs encode temporal reasoning (immediate vs long-term thinking) and whether we can steer this behavior using Contrastive Activation Addition (CAA).

**Key findings:**
- GPT-2 encodes temporal scope with 99% linear separability (explicit markers)
- Probe accuracy drops to ~50% without explicit temporal keywords
- Steering vectors can shift model outputs toward immediate or long-term framing

## Setup

```bash
pip install -e .
cp .env.example .env  # Add API keys
```

## Structure

```
temporal-reasoning/
├── data/
│   ├── raw/                 # Original datasets
│   ├── validated/           # Human-validated
│   ├── processed/           # Train/val/test splits
│   └── adversarial/         # Edge cases
├── scripts/
│   ├── data/                # Dataset generation
│   ├── extraction/          # Activation extraction
│   ├── probes/              # Probe training
│   ├── circuits/            # Causal analysis
│   └── analysis/            # Figures, metrics
├── src/temporal_reasoning/  # Package code
├── configs/                 # Experiment configs
├── experiments/             # Tracked runs
├── results/                 # Tables, figures, checkpoints
├── notebooks/               # Analysis notebooks
├── docs/                    # Documentation
└── paper/                   # Manuscript
```

## Quick Start

```python
from temporal_reasoning import SteeringFramework, get_model_config
import json

# Load dataset
with open("data/raw/temporal_scope_caa.json") as f:
    data = json.load(f)

# Use latents library for extraction and steering
```

```bash
# Train probes
python scripts/probes/train_temporal_probes_caa.py

# Validate
python scripts/probes/validate_dataset_split.py
```

## Related Work

See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) for literature review:
- Time-R1: Temporal reasoning framework ([arXiv:2505.13508](https://arxiv.org/abs/2505.13508))
- CAA: Contrastive Activation Addition ([ACL 2024](https://arxiv.org/abs/2312.06681))
- Temporal Alignment via Activation Engineering ([arXiv:2505.14158](https://arxiv.org/abs/2505.14158))

## Public Datasets

| Dataset | Source | Link |
|---------|--------|------|
| Time-Bench | Time-R1 | [HuggingFace](https://huggingface.co/datasets/ulab-ai/Time-Bench) |
| Test of Time | Google | [HuggingFace](https://huggingface.co/datasets/baharef/ToT) |
| TIME | NeurIPS 2025 | [arXiv:2505.12891](https://arxiv.org/abs/2505.12891) |

## License

MIT
