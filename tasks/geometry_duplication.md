# Geometry Scripts Duplication Analysis

Analysis of code duplication across:
- `scripts/intertemporal/compute_geometry_analysis.py`
- `scripts/intertemporal/compute_linear_probes.py`
- `scripts/intertemporal/visualize_geometry_analysis.py`
- `scripts/intertemporal/generate_geometry_samples.py`

---

## 1. Exact Duplicate Functions

### 1.1 `target_keys()` - EXACT DUPLICATE

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 27-29 |
| `compute_linear_probes.py` | 27-29 |

```python
def target_keys() -> list[str]:
    """Generate combined (aggregated) target keys only."""
    return [f"L{l}_{c}_{p}" for l in LAYERS for c in COMPONENTS for p in POSITIONS]
```

**Recommendation:** Extract to `src/intertemporal/geometry/target_keys.py` or a shared `geometry_utils.py` module.

---

### 1.2 `discover_datasets()` - EXACT DUPLICATE

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 675-683 |
| `compute_linear_probes.py` | 402-410 |
| `visualize_geometry_analysis.py` | 127-135 |

```python
def discover_datasets(base_dir: Path) -> list[Path]:
    """Discover all valid dataset directories under base_dir."""
    datasets = []
    if not base_dir.exists():
        return datasets
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and (subdir / "data" / "samples").exists():
            datasets.append(subdir)
    return datasets
```

**Recommendation:** Extract to `src/intertemporal/geometry/dataset_discovery.py` or include in the shared module.

---

### 1.3 `cache_position_mappings()` - EXACT DUPLICATE

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 159-172 |
| `compute_linear_probes.py` | 79-92 |

```python
def cache_position_mappings(data_dir: Path) -> tuple[list[Path], dict[int, dict]]:
    """Cache all position mappings once."""
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    mapping_cache = {}
    for i, d in enumerate(sample_dirs):
        mapping_file = d / "position_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapping_cache[i] = json.load(f)
    return sample_dirs, mapping_cache
```

**Recommendation:** Extract to shared module - this is critical for performance and should be unified.

---

### 1.4 `get_abs_pos()` - EXACT DUPLICATE

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 117-128 |
| `compute_linear_probes.py` | 50-57 |

```python
def get_abs_pos(mapping: dict, pos: str) -> int | list[int] | None:
    """Get absolute position(s) from mapping."""
    if "named_positions" not in mapping:
        raise KeyError("named_positions missing from mapping")
    abs_pos = mapping["named_positions"].get(pos)
    if abs_pos is None:
        return None
    return abs_pos
```

**Recommendation:** Extract to shared module.

---

### 1.5 `find_activation_file()` - EXACT DUPLICATE

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 131-156 |
| `compute_linear_probes.py` | 60-76 |

```python
def find_activation_file(sample_dir: Path, layer: int, comp: str, abs_pos: int | list[int]) -> Path | None:
    """Find activation file, supporting both old and new formats."""
    if isinstance(abs_pos, list):
        for p in abs_pos:
            f = sample_dir / f"L{layer}" / f"{comp}_{p}.npy"
            if f.exists():
                return f
            f = sample_dir / f"L{layer}_{comp}_{p}.npy"
            if f.exists():
                return f
        return None
    else:
        f = sample_dir / f"L{layer}" / f"{comp}_{abs_pos}.npy"
        if f.exists():
            return f
        f = sample_dir / f"L{layer}_{comp}_{abs_pos}.npy"
        return f if f.exists() else None
```

**Recommendation:** Extract to shared module - this handles file format compatibility.

---

## 2. Similar Functions That Could Be Merged

### 2.1 `load_target()` - SIMILAR with minor differences

| File | Lines | Differences |
|------|-------|-------------|
| `compute_geometry_analysis.py` | 175-235 | Handles `rel_pos` parameter for per-token keys |
| `compute_linear_probes.py` | 95-130 | Simpler version without `rel_pos` support |

**Recommendation:** The `compute_geometry_analysis.py` version is a superset. Use that version in shared module with `rel_pos` as optional parameter (defaults to None for backwards compatibility).

---

### 2.2 `parse_key()` - SIMILAR with different return types

| File | Lines | Returns |
|------|-------|---------|
| `compute_geometry_analysis.py` | 91-114 | `tuple[int, str, str, int \| None]` (layer, comp, pos, rel_pos) |
| `compute_linear_probes.py` | 36-47 | `tuple[int, str, str]` (layer, comp, pos) |

**Recommendation:** Use the more comprehensive version from `compute_geometry_analysis.py` that handles `rel_pos`. Callers that don't need `rel_pos` can ignore the 4th element.

---

### 2.3 `load_horizons()` - SIMILAR with minor error message differences

| File | Lines | Differences |
|------|-------|-------------|
| `compute_geometry_analysis.py` | 238-269 | More detailed error messages |
| `compute_linear_probes.py` | 133-154 | Simpler error messages |

```python
# compute_geometry_analysis.py version has better error messages:
raise FileNotFoundError(
    f"choice.json missing for sample {i}: {f}\n"
    "Re-run data extraction to regenerate choice.json files."
)
```

**Recommendation:** Use the `compute_geometry_analysis.py` version with better error messages.

---

### 2.4 `build_targets()` - SIMILAR implementations

| File | Lines | Differences |
|------|-------|-------------|
| `visualize_geometry_analysis.py` | 47-52 | Inline, uses hardcoded LAYERS |
| `generate_geometry_samples.py` | 95-106 | Takes layers/components/positions as parameters |

```python
# visualize_geometry_analysis.py:
def build_targets() -> list[TargetSpec]:
    return [
        TargetSpec(layer=l, component=c, position=p)
        for l in LAYERS for c in COMPONENTS for p in POSITIONS
    ]

# generate_geometry_samples.py (more flexible):
def build_targets(layers, components, positions) -> list[TargetSpec]:
    return [
        TargetSpec(layer=layer, component=component, position=position)
        for layer in layers
        for component in components
        for position in positions
    ]
```

**Recommendation:** Use the parameterized version from `generate_geometry_samples.py` in shared module.

---

### 2.5 `main()` and CLI Argument Parsing - SIMILAR structure

All four scripts follow the same pattern:
1. Parse arguments with `argparse`
2. Resolve dataset path(s)
3. Process dataset(s) in loop
4. Return exit code

**Recommendation:** Consider a shared CLI base class or helper function that handles:
- Common arguments (`--base-dir`, `--force`, dataset positional arg)
- Dataset discovery
- Main loop structure

---

## 3. Repeated Code Patterns

### 3.1 Boilerplate Header Pattern

All 4 files have identical boilerplate:

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
```

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 12-19 |
| `compute_linear_probes.py` | 13-20 |
| `visualize_geometry_analysis.py` | 25-40 |
| `generate_geometry_samples.py` | 47-65 |

**Recommendation:** These scripts are entry points, so some boilerplate is expected. However, consider using a standardized logging setup from `src/common/`.

---

### 3.2 Sample Directory Iteration Pattern

Repeated pattern for iterating over sample directories:

```python
samples_dir = data_dir / "data" / "samples"
sample_dirs = sorted(
    [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
    key=lambda x: int(x.name.split("_")[1])
)
```

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 39, 161-164, 245-248 |
| `compute_linear_probes.py` | 81-84, 136-138 |

**Recommendation:** Extract to `get_sample_dirs(data_dir: Path) -> list[Path]` helper.

---

### 3.3 Progress Logging Pattern

Similar progress logging pattern:

```python
log.info(f"  [{i}/{len(keys)}] {key} - LOADING...")
# ... do work ...
log.info(f"  [{i}/{len(keys)}] {key} - COMPLETE ({total_time:.1f}s total)")
```

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 300, 306, 309, 347 |
| `compute_linear_probes.py` | 187, 193, 196, 248 |

**Recommendation:** Consider a shared progress logger or context manager.

---

### 3.4 Caching/Skip Pattern

Both analysis scripts use the same pattern for checking cached results:

```python
if not force and cache_file.exists():
    with open(cache_file) as f:
        results[key] = json.load(f)
    skipped_count += 1
    if skipped_count % 100 == 0:
        log.info(f"  SKIP {skipped_count} cached targets...")
    continue
```

| File | Lines |
|------|-------|
| `compute_geometry_analysis.py` | 291-297 |
| `compute_linear_probes.py` | 178-184 |

**Recommendation:** Extract to a helper function or decorator pattern.

---

## 4. Magic Numbers That Should Be Constants

### 4.1 LAYERS Array - DUPLICATED

| File | Lines | Value |
|------|-------|-------|
| `compute_geometry_analysis.py` | 21 | `[0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]` |
| `compute_linear_probes.py` | 22 | `[0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]` |
| `visualize_geometry_analysis.py` | 42 | `[0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]` |
| `generate_geometry_samples.py` | 73-86 | `[0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]` (with comments) |
| `src/intertemporal/common/semantic_positions.py` | 8 | `DEFAULT_LAYERS = [8, 19, 21, 24, 28, 31, 34, 35]` (DIFFERENT!) |

**Critical Issue:** The LAYERS constant differs from DEFAULT_LAYERS in `semantic_positions.py`!

**Recommendation:** Define a single `GEOMETRY_LAYERS` constant in `semantic_positions.py` and import everywhere. The scripts add layers 0, 1, 3 that aren't in the default.

---

### 4.2 COMPONENTS Array - DUPLICATED

| File | Lines | Value |
|------|-------|-------|
| `compute_geometry_analysis.py` | 22 | `["resid_pre", "attn_out", "mlp_out", "resid_post"]` |
| `compute_linear_probes.py` | 23 | `["resid_pre", "attn_out", "mlp_out", "resid_post"]` |
| `visualize_geometry_analysis.py` | 43 | `["resid_pre", "attn_out", "mlp_out", "resid_post"]` |
| `generate_geometry_samples.py` | 89 | `["resid_pre", "attn_out", "mlp_out", "resid_post"]` |

**Recommendation:** Define `GEOMETRY_COMPONENTS` in `semantic_positions.py` and import everywhere.

---

### 4.3 POSITIONS Combination - DUPLICATED

| File | Lines | Value |
|------|-------|-------|
| `compute_geometry_analysis.py` | 23 | `PROMPT_POSITIONS + RESPONSE_POSITIONS` |
| `compute_linear_probes.py` | 24 | `PROMPT_POSITIONS + RESPONSE_POSITIONS` |
| `visualize_geometry_analysis.py` | 44 | `PROMPT_POSITIONS + RESPONSE_POSITIONS` |
| `generate_geometry_samples.py` | 92 | `PROMPT_POSITIONS + RESPONSE_POSITIONS` (as `ALL_POSITIONS`) |

**Note:** `semantic_positions.py` already defines `ALL_TRAJECTORY_POSITIONS` with same value!

**Recommendation:** Use `ALL_TRAJECTORY_POSITIONS` from `semantic_positions.py` instead of redefining.

---

### 4.4 Other Magic Numbers

| Constant | Value | Files | Recommendation |
|----------|-------|-------|----------------|
| `n_pca_components` | 10 | compute_geometry_analysis.py:318, generate_geometry_samples.py:115 | Define as `DEFAULT_PCA_COMPONENTS` |
| Minimum samples | 4 | compute_geometry_analysis.py:227, compute_linear_probes.py:123 | Define as `MIN_SAMPLES_FOR_ANALYSIS` |
| Minimum CV samples | 10 | compute_linear_probes.py:204 | Define as `MIN_SAMPLES_FOR_CV` |
| Max t-SNE samples | 2000 | compute_geometry_analysis.py:391 | Define as `MAX_TSNE_SAMPLES` |
| Default seed | 42 | All files | Define as `DEFAULT_SEED` |
| t-SNE max_iter | 300 | compute_geometry_analysis.py:401 | Define as `TSNE_MAX_ITER` |
| UMAP neighbors | 15 | compute_geometry_analysis.py:387 | Define as `UMAP_N_NEIGHBORS` |

---

## 5. Summary of Recommendations

### High Priority (Exact Duplicates)

1. **Create `src/intertemporal/geometry/geometry_io.py`** with:
   - `target_keys()`
   - `cache_position_mappings()`
   - `get_abs_pos()`
   - `find_activation_file()`
   - `load_target()`
   - `load_horizons()`
   - `get_sample_dirs()`

2. **Create `src/intertemporal/geometry/dataset_discovery.py`** with:
   - `discover_datasets()`

3. **Update `src/intertemporal/common/semantic_positions.py`** to include:
   - `GEOMETRY_LAYERS` (the expanded list with 0, 1, 3)
   - `GEOMETRY_COMPONENTS`
   - Already has `ALL_TRAJECTORY_POSITIONS` - just use it

### Medium Priority (Similar Functions)

4. **Consolidate `parse_key()` and `build_targets()`** into the shared module

5. **Create constants file for magic numbers** in `src/intertemporal/geometry/constants.py`:
   - `DEFAULT_PCA_COMPONENTS = 10`
   - `MIN_SAMPLES_FOR_ANALYSIS = 4`
   - `MIN_SAMPLES_FOR_CV = 10`
   - `MAX_TSNE_SAMPLES = 2000`
   - `DEFAULT_SEED = 42`

### Low Priority (Patterns)

6. **Consider shared CLI utilities** for common argument parsing and main loop structure

---

## 6. Estimated Savings

| Category | Lines Duplicated | Files Affected |
|----------|------------------|----------------|
| Exact duplicates | ~150 lines | 4 files |
| Similar functions | ~100 lines | 3 files |
| Magic numbers | ~20 definitions | 4 files |
| **Total** | **~270 lines** | **4 files** |

After deduplication, each script would shrink by approximately 50-100 lines and become more maintainable.
