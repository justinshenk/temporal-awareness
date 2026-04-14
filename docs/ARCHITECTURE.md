# Architecture

## Repository Structure

```
temporal-awareness/
├── src/                          # Core library
│   ├── activation_api/           # Activation extraction engine
│   │   ├── extractor.py          # ActivationExtractor — dual backend (TL / HF)
│   │   ├── config.py             # ExtractionConfig dataclass
│   │   ├── hooks.py              # HookManager for forward-hook orchestration
│   │   └── result.py             # ActivationResult container
│   ├── inference/                # Model loading and generation
│   │   ├── model_runner.py       # ModelRunner — unified API across 5 backends
│   │   ├── backends/             # TransformerLens, NNsight, Pyvene, HF, MLX
│   │   ├── interventions.py      # Causal intervention definitions
│   │   └── generated_trajectory.py
│   ├── common/                   # Shared utilities
│   │   ├── device_utils.py       # GPU/MPS/CPU detection and memory tracking
│   │   ├── profiler.py           # Timing profiler
│   │   └── token_trajectory.py   # Base trajectory class
│   ├── experiments/              # Experiment orchestration
│   └── intertemporal/            # SAE pipeline (temporal preference analysis)
│       ├── sae/                  # SAE training, evaluation, clustering
│       ├── formatting/           # Prompt formatting configs
│       └── data/                 # Dataset configs and loaders
├── scripts/
│   ├── experiments/              # Phase 3-5 experiment scripts + SLURM jobs
│   ├── probes/                   # Probe training and validation
│   ├── extraction/               # Activation extraction pipelines
│   └── data/                     # Dataset preparation
├── configs/                      # YAML/JSON experiment and model configs
├── notebooks/                    # Analysis and visualization notebooks
├── tests/                        # Test suite (pytest)
├── results/                      # Experiment outputs (git-ignored)
└── docs/                         # Documentation
```

## Key Abstractions

### ActivationExtractor (`src/activation_api/extractor.py`)

The core activation extraction engine. Supports two backends:

1. **TransformerLens** — HookedTransformer with clean hook interface. Preferred for mechanistic interpretability (supports `resid_pre`, `resid_post`, `attn_out`, `mlp_out` natively).
2. **HuggingFace** — Raw `AutoModelForCausalLM` with PyTorch forward hooks. Works with any model, including those not supported by TransformerLens.

Both backends produce numerically equivalent results (divergence < 10^-5 validated by `benchmark_extraction_backends.py`).

### ModelRunner (`src/inference/model_runner.py`)

Unified inference API that abstracts over five backends: TransformerLens, NNsight, Pyvene, HuggingFace, and MLX. Provides:

- `generate()` — text generation with optional interventions
- `run_with_cache()` — forward pass + activation capture
- `run_with_intervention()` — causal interventions (activation patching/steering)
- `run_with_intervention_and_cache()` — combined intervention + capture
- `generate_trajectory()` — next-token probability trajectories

### Experiment Scripts (`scripts/experiments/`)

Each Phase 3-5 experiment follows a consistent pattern:

1. **MODEL_CONFIGS** dict — maps model short names to HF names, architecture info, and layer selections
2. **Prompt generation** — creates contrastive prompt pairs (e.g., rep=1 vs rep=N)
3. **Activation extraction** — uses `ActivationExtractor` to capture residual stream activations
4. **Analysis** — computes direction vectors, trains probes, runs PCA
5. **Logging** — results to W&B and local JSON

### SLURM Integration

Each experiment has a corresponding `submit_phase*.sh` script that handles:

- Virtual environment activation (separate envs for Ouro-2.6B due to `transformers` version constraint)
- GPU memory constraint selection (32GB for ≤8B models, 80GB for 30B MoE)
- W&B project configuration

The `submit_all_models.sh` meta-script submits all 7 models for any experiment.

## Data Flow

```
Prompts (contrastive pairs)
    → ActivationExtractor (forward hooks)
    → Residual stream activations [batch, layers, d_model]
    → Contrastive mean-difference direction (Arditi et al. 2024)
    → Linear probes (LogisticRegression)
    → PCA / cosine similarity analysis
    → W&B logging + JSON results
```

## Design Decisions

- **Dual backend validation**: PyTorch hooks and TransformerLens are benchmarked against each other to ensure activation extraction correctness.
- **Contrastive direction extraction**: Following Arditi et al. (2024), we use mean-difference directions rather than PCA to isolate specific behavioral dimensions.
- **Architecture coverage**: 7 models across 4 families (Llama, Qwen, DeepSeek, Ouro) covering dense, MoE, distilled-reasoning, and looped-recurrent architectures.
- **`use_cache=False`**: Required for Ouro-2.6B compatibility — the model's looped architecture conflicts with HF's KV cache assumptions.
