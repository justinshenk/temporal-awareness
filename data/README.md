# Data

## Structure

- `raw/` - LLM-generated and original prompt pairs
- `validated/` - Human-validated subset
- `processed/` - Train/val/test splits ready for training
- `adversarial/` - Paraphrased and edge case examples

## Schema

Each JSONL file contains pairs in CAA format:

```json
{
  "category": "planning",
  "question": "When deciding on a solution, focus on:",
  "immediate": " (A) What works right now",
  "long_term": " (B) What creates lasting value"
}
```

## Datasets

| File | Pairs | Description |
|------|-------|-------------|
| temporal_scope_caa.json | 50 | Explicit temporal markers (train) |
| temporal_scope_implicit.json | 50 | Semantic cues only (test) |
| temporal_scope_clean.json | 100 | Cleaned, decontaminated |
