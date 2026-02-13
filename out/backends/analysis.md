# Backend Performance Analysis

Benchmark: `uv run python3 scripts/intertemporal/run_full_experiment.py --test`
Model: Qwen/Qwen2.5-1.5B-Instruct on Apple Silicon (MPS)
Samples: 3

## Results Summary

| Backend        | Total Time | vs Fastest | Status |
|----------------|------------|------------|--------|
| PYVENE         | 14,958ms   | baseline   | OK     |
| HUGGINGFACE    | 16,078ms   | +7.5%      | OK     |
| TRANSFORMERLENS| 24,059ms   | +60.8%     | OK     |
| NNSIGHT        | -          | -          | FAILED |
| MLX            | -          | -          | NOT TESTED (requires mlx package) |

## Detailed Breakdown

### PYVENE (Fastest)
```
Total: 14957.5ms
  step_load_model: 2255.8ms (4x, avg 564.0ms)
  step_preference_data: 13374.1ms
    generate: 9397.8ms (3x, avg 3132.6ms)
    run_with_cache: 107.6ms (3x, avg 35.9ms)
    get_prob_trajectories_for_batch: 447.5ms (3x, avg 149.2ms)
```

### HUGGINGFACE
```
Total: 16078.3ms
  step_load_model: 3033.0ms (4x, avg 758.3ms)
  step_preference_data: 14504.3ms
    generate: 9695.6ms (3x, avg 3231.9ms)
    run_with_cache: 489.5ms (3x, avg 163.2ms)
    get_prob_trajectories_for_batch: 1221.4ms (3x, avg 407.1ms)
```

### TRANSFORMERLENS (Slowest working backend)
```
Total: 24058.5ms
  step_load_model: 3950.0ms (4x, avg 987.5ms)
  step_preference_data: 20851.5ms
    generate: 15215.4ms (3x, avg 5071.8ms)
    run_with_cache: 220.1ms (3x, avg 73.4ms)
    get_prob_trajectories_for_batch: 517.6ms (3x, avg 172.5ms)
```

### NNSIGHT (Failed)
Error: `RuntimeError: Inference tensors cannot be saved for backward`
Issue with autograd/inference_mode interaction in generate().

## Key Observations

1. **PYVENE is fastest overall** - 61% faster than TransformerLens
2. **HuggingFace is a good middle ground** - Simple API, decent speed (+7.5% slower than PyVene)
3. **TransformerLens is slowest** - The extra hook infrastructure adds overhead
4. **Generation is the bottleneck** - ~62-63% of total time across all backends

## Memory Usage (Peak)

| Backend        | Peak Memory |
|----------------|-------------|
| TRANSFORMERLENS| 8.07 GB     |
| HUGGINGFACE    | 6.73 GB     |
| PYVENE         | 6.84 GB     |

TransformerLens uses ~1.3GB more memory due to its additional hook infrastructure.

## Recommendations

- **For production/inference**: Use PYVENE or HUGGINGFACE
- **For interpretability research**: Use TRANSFORMERLENS (full hook support)
- **For Apple Silicon**: MLX should be tested when mlx package is installed
