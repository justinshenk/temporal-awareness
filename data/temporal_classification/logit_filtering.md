# Logit-Diff Filtering for Activation Patching

## Problem
The 200-pair dataset was filtered to 135 pairs by first-token correctness. However, many of these 135 pairs have weak logit differences — the model technically outputs the right first token but is barely confident. 23 pairs even have the wrong sign on the corrupted side (model's logit for "short" > logit for "long" despite outputting "long" as first token, what is an issue in different settings between classification validation and activation patching experiments). These weak pairs add noise to activation patching.

## Solution
Filter to pairs where the model is confidently correct on BOTH sides:
- Clean logit_diff (logit "short" − logit "long") > 1.0
- Corrupted logit_diff (logit "short" − logit "long") < −1.0

This yields 57 high-confidence pairs with balanced baselines:
- Clean mean logit_diff: +3.76
- Corrupted mean logit_diff: −2.73
- Ratio: 1.38 (acceptable; original 135-pair ratio was 2.59)

## Files
- `strong_pairs.json`: 57 pair IDs to use for primary activation patching analysis
- `failing_pairs.json`: 65 pair IDs removed by first-token check (from 200 → 135)
- `dataset_200.json`: full dataset (keep for secondary analyses)

## Usage
```python
import json

with open("dataset_200.json") as f:
    dataset = json.load(f)

with open("strong_pairs.json") as f:
    strong = json.load(f)

filtered = [dataset[i] for i in strong["strong_pair_ids"]]
```
