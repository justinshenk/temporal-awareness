# Project Guidelines

## Key Entry Points

1. **Main experiment script**: `scripts/intertemporal/run_intertemporal_experiment.py`
   - Run experiments: `uv run python scripts/intertemporal/run_intertemporal_experiment.py`
   - Use cached data: `--cache` or `--cache experiment_name`
   - Regenerate viz: `--viz '{"regenerate_one": "experiment_name"}'`
   - Multilabel mode: `--multilabel`
   - Read this script to understand the experiment pipeline

2. **SAE pipeline**: `scripts/intertemporal/run_sae_pipeline.py`
   - Show targets: `--show-targets`
   - Test iteration: `--test-iter`
   - Priority layers: `--priority high`

## Code Style

1. **All imports always on top** - Never use inline imports or imports within functions unless absolutely necessary for circular dependency resolution.

2. **Use auto-export in ALL `__init__` files** - Every `__init__.py` should automatically export all public symbols from submodules.

3. **Python file names must be multi-word** - Never use single-word .py file names (e.g., use `geo_viz_data.py` not `data.py`). This improves clarity and avoids import collisions.

4. **Code quality standards:**
   - **Clean code** - No dead code, no commented-out code, no debug prints
   - **Code re-use** - No duplicate code; extract common patterns into shared utilities
   - **No legacy/backwards compatibility** - Remove deprecated code, don't maintain backwards compatibility shims
   - **Maximum readability and modularity** - Break large files into smaller modules, use clear naming, keep functions focused

## Architecture Patterns

1. **Use BaseSchema for all dataclasses** - Inherit from `BaseSchema` (in `src/common/base_schema.py`) for automatic `.to_dict()`, `.from_dict()`, and serialization support. This applies to:
   - All analysis dataclasses
   - Any dataclass that needs serialization

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
