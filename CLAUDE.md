# Project Guidelines

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
