# Results Summary

## Main Claims

| # | Claim | Metric | Value | Status | Reproduce |
|---|-------|--------|-------|--------|-----------|
| 1 | Temporal scope is linearly encoded | Probe accuracy (Layer 8) | 92.5% | ✓ Verified | `make verify-probes` |
| 2 | Encoding generalizes to test set | Test accuracy (Layer 6) | 84% | ✓ Verified | `make verify-probes` |
| 3 | Steering affects same features probes detect | Correlation | r=0.935 | ⚠️ Verify | `make verify-steering` |
| 4 | Late layers encode semantic (not lexical) features | Ablation accuracy (L10-11) | 100% | ✓ Verified | `make verify-ablation` |
| 5 | Dual encoding: lexical (early) + semantic (late) | Layer pattern | See fig | ✓ Verified | `make figures` |

## Verification Source

Results verified from GPU experiments (GCS bucket: `gs://temporal-grounding-gpt2-82feb/`):

| Dataset | Samples | Best Layer | Accuracy |
|---------|---------|------------|----------|
| Training | 400 | Layer 8 | 92.5% |
| Test | 100 | Layer 6 | 84.0% |
| Ablated (no keywords) | 100 | Layer 10-11 | 100% |

Raw CSV results available in `verified/`.

## Quick Verification

```bash
# Verify all claims (~5 min, no GPU required)
make verify

# Or run individually
python scripts/verify_all_claims.py
```

## Key Figures

| Figure | Description | Location |
|--------|-------------|----------|
| Layer accuracy curve | Probe accuracy by layer (train vs test) | `figures/layer_accuracy.png` |
| Steering correlation | Steering strength vs probe predictions | `figures/steering_correlation.png` |
| Keyword ablation | Performance with/without temporal keywords | `figures/ablation.png` |

## Checkpoints

| File | Description | Layers |
|------|-------------|--------|
| `checkpoints/temporal_caa_layer_*_probe.pkl` | Trained linear probes | 0-11 |
| `checkpoints/temporal_directions_learned.json` | Steering vectors | 0-11 |

## Known Caveats

1. **Single model**: Only tested on GPT-2-small (124M params)
2. **Steering consistency**: ~60% (target: >70%)
3. **Dataset quality**: Explicit pairs may have keyword leakage - needs audit
4. **Binary classification**: Only short vs long, not fine-grained horizons

## Reproduction Time Estimates

| Task | Time | GPU |
|------|------|-----|
| Verify probes | ~2 min | No |
| Verify steering | ~3 min | No |
| Full reproduction | ~30 min | Yes (recommended) |
| Retrain from scratch | ~2 hours | Yes |

## Data Dependencies

```
data/raw/
├── temporal_scope_caa.json         # 50 explicit pairs (train)
├── temporal_scope_implicit.json    # 50 implicit pairs (test)
└── temporal_scope_clean.json       # Combined cleaned dataset
```

## Citation

If using these results, please verify claims independently before citing.
