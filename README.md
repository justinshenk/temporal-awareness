# Temporal Awareness

> "I want my time to grant me what time can't grant itself."
> — Al-Mutannabi

Research on detecting and steering temporal awareness in LLMs.

## Overview

This project investigates how LLMs encode temporal reasoning and whether we can:
1. **Detect** temporal preference from internal representations
2. **Steer** temporal orientation via activation engineering
3. **Measure** divergence between stated and internal time horizons

**Key findings:**
- GPT-2 encodes temporal scope with 92.5% linear separability
- Steering validation: r=0.935 correlation between steering and probe predictions
- Late layers (6-11) encode semantic temporal features robust to keyword removal

## Program

[https://github.com/justinshenk/temporal-awareness/blob/main/docs/RESEARCH_PROGRAM.md]

## Framework

We ground temporal awareness in **intertemporal preference**:

```
U(o_i; θ) = u(r_i) · D(t_i; θ)     # Value function
t_internal = inf{t : D(t) ≤ α}     # Internal horizon
```

**Key questions:**
- Does `t_internal ≈ t_h` (stated horizon)?
- Can we detect divergence between stated and internal preference?

See [docs/research_plan.md](docs/research_plan.md) for full framework.

## Setup

```bash
pip install -e .
cp .env.example .env  # Add API keys
```

## Structure

```
temporal-awareness/
├── data/
│   ├── raw/                 # Intertemporal preference datasets
│   ├── validated/           # Human-validated
│   └── processed/           # Train/val/test splits
├── scripts/
│   ├── probes/              # Probe training & validation
│   └── analysis/            # Figures, metrics
├── results/checkpoints/     # Trained probes & steering vectors
├── docs/
│   ├── research_plan.md     # Full framework & roadmap
│   └── RELATED_WORK.md      # Literature review
└── paper/                   # Manuscript
```

## Quick Start

```python
from latents import SteeringFramework
from latents.model_adapter import get_model_config

# Use latents library for extraction and steering
```

```bash
# Train probes
python scripts/probes/train_temporal_probes_caa.py
```

## Related Work

See [docs/RELATED_WORK.md](docs/RELATED_WORK.md):
- Zhu et al. 2025: Steering Risk Preferences via Behavioral-Neural Alignment
- Mazyaki et al. 2025: Temporal Preferences in LLMs for Long-Horizon Assistance
- Time-R1: Comprehensive temporal reasoning ([arXiv:2505.13508](https://arxiv.org/abs/2505.13508))

## Public Datasets

| Dataset | Source | Link |
|---------|--------|------|
| Time-Bench | Time-R1 | [HuggingFace](https://huggingface.co/datasets/ulab-ai/Time-Bench) |
| Test of Time | Google | [HuggingFace](https://huggingface.co/datasets/baharef/ToT) |

## License

MIT
