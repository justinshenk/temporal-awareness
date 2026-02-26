# Distribution Shift Datasets for Probe Reliability Testing

Datasets for systematically evaluating temporal probe degradation under
distribution shift. Used by `scripts/experiments/sae_feature_stability.py`.

## Shift Conditions

| Condition | Description | N pairs | Source |
|-----------|-------------|---------|--------|
| `domain_shift` | Financial, medical, environmental, personal contexts | 20 | Generated |
| `register_shift` | Formal, casual, technical, poetic, bureaucratic | 14 | Generated |
| `negation` | "not short-term" → long-term and vice versa | 16 | Generated |
| `paraphrase` | Creative/unusual temporal expressions | 16 | Generated |
| `ambiguous` | Genuinely ambiguous (for calibration analysis) | 10 | Generated |
| `implicit_only` | No explicit time words (from existing dataset) | 50 | `temporal_scope_implicit.json` |
| `cross_dataset_clean` | Full CAA format, pure temporal markers | 100 | `temporal_scope_clean.json` |
| `cross_dataset_caa` | Original CAA pairs | 50 | `temporal_scope_caa.json` |

## Schema

Each JSON file follows the same schema:

```json
{
  "metadata": {
    "name": "condition_name",
    "description": "...",
    "n_samples": N,
    "created": "2026-02-26",
    "purpose": "distribution_shift_evaluation"
  },
  "samples": [
    {"prompt": "text", "label": 0, "label_name": "immediate"},
    {"prompt": "text", "label": 1, "label_name": "long_term"}
  ]
}
```

## Usage

These datasets are loaded automatically by the SAE feature stability
experiment. They can also be used independently for other probe
reliability experiments.
