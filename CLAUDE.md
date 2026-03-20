# Project Guidelines

## Key Entry Points

1. **Main experiment script**: `scripts/intertemporal/run_intertemporal_experiment.py`
   - Run experiments: `uv run python scripts/intertemporal/run_intertemporal_experiment.py`
   - Use cached data: `--cache` or `--cache experiment_name`
   - Regenerate viz: `--viz '{"regenerate_one": "experiment_name"}'`
   - Multilabel mode: `--multilabel`
   - Read this script to understand the experiment pipeline

## Code Style

1. **All imports always on top** - Never use inline imports or imports within functions unless absolutely necessary for circular dependency resolution.

2. **Use auto-export in ALL `__init__` files** - Every `__init__.py` should automatically export all public symbols from submodules.

3. **Code quality standards:**
   - **Clean code** - No dead code, no commented-out code, no debug prints
   - **Code re-use** - No duplicate code; extract common patterns into shared utilities
   - **No legacy/backwards compatibility** - Remove deprecated code, don't maintain backwards compatibility shims
   - **Maximum readability and modularity** - Break large files into smaller modules, use clear naming, keep functions focused

## Architecture Patterns

1. **Use BaseSchema for all dataclasses** - Inherit from `BaseSchema` (in `src/common/base_schema.py`) for automatic `.to_dict()`, `.from_dict()`, and serialization support. This applies to:
   - All analysis dataclasses
   - Any dataclass that needs serialization
