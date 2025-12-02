# Contributing

## Setup

```bash
git clone https://github.com/justinshenk/temporal-reasoning
cd temporal-reasoning
pip install -e ".[dev]"
```

## Running Experiments

```bash
# Train probe
python scripts/probes/train_temporal_probes_caa.py

# Evaluate
python scripts/probes/validate_dataset_split.py
```

## Adding Data

1. Add pairs to `data/raw/`
2. Validate with `scripts/data/validate_batch.py`
3. Create splits with `scripts/data/create_splits.py`

## Code Style

```bash
black .
ruff check .
```
