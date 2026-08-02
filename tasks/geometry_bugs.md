# Geometry Scripts Bug Report

Analysis of potential bugs in the geometry pipeline scripts:
- `scripts/intertemporal/compute_geometry_analysis.py`
- `scripts/intertemporal/compute_linear_probes.py`
- `scripts/intertemporal/visualize_geometry_analysis.py`
- `scripts/intertemporal/generate_geometry_samples.py`

---

## Bug Summary

| ID | File | Line | Severity | Category |
|----|------|------|----------|----------|
| BUG-001 | compute_geometry_analysis.py | 311 | HIGH | Index Error / Data Mismatch |
| BUG-002 | compute_geometry_analysis.py | 327 | MEDIUM | Index Error |
| BUG-003 | compute_geometry_analysis.py | 45-46 | MEDIUM | Missing Error Handling |
| BUG-004 | compute_geometry_analysis.py | 170 | MEDIUM | Missing Error Handling |
| BUG-005 | compute_geometry_analysis.py | 360-361 | MEDIUM | Edge Case |
| BUG-006 | compute_linear_probes.py | 198-200 | HIGH | Index Error / Data Mismatch |
| BUG-007 | compute_linear_probes.py | 90 | MEDIUM | Missing Error Handling |
| BUG-008 | compute_linear_probes.py | 145 | MEDIUM | Missing Error Handling |
| BUG-009 | compute_linear_probes.py | 316 | LOW | Empty List |
| BUG-010 | visualize_geometry_analysis.py | 111-113 | MEDIUM | Race Condition / File Operation |
| BUG-011 | visualize_geometry_analysis.py | 194-196 | MEDIUM | Race Condition / File Operation |
| BUG-012 | visualize_geometry_analysis.py | 69 | LOW | Empty Array Handling |
| BUG-013 | visualize_geometry_analysis.py | 121-123 | LOW | Empty Directory |
| BUG-014 | generate_geometry_samples.py | 325-331 | LOW | Parsing Logic |
| BUG-015 | compute_geometry_analysis.py | 163-164 | MEDIUM | ValueError on Empty Directory |
| BUG-016 | compute_geometry_analysis.py | 86 | LOW | Import Location |
| BUG-017 | compute_geometry_analysis.py | 403-407 | MEDIUM | Subsampled t-SNE Output Misleading |
| BUG-018 | compute_linear_probes.py | 257-263 | MEDIUM | Data Race on summary.json |

---

## Detailed Bug Descriptions

### BUG-001: y_sub index mismatch with X in PCA analysis
**File:** `compute_geometry_analysis.py`
**Line:** 311
**Severity:** HIGH
**Category:** Index Error / Data Mismatch

**Description:**
```python
y_sub = y[:len(X)]
```
The code assumes `y` (horizons) has the same indexing as `X` (activations). However, `X` is loaded from `valid_files` which only contains samples that have the specific activation file. The `y` array is indexed 0 to N-1 for all samples. If sample indices 0, 2, 5 have valid activations, `X` will be shape (3, dim) but `y[:3]` will give horizons for samples 0, 1, 2 - not 0, 2, 5.

**Suggested Fix:**
Track which sample indices contributed to `X` during `load_target()` and use those indices to slice `y`. Return a tuple `(X, valid_sample_indices)` from `load_target()`.

---

### BUG-002: PCA valid_mask applied to wrong-sized Xp
**File:** `compute_geometry_analysis.py`
**Line:** 327
**Severity:** MEDIUM
**Category:** Index Error

**Description:**
```python
Xp_valid = Xp[valid_mask]
```
`Xp` has shape `(len(X), n)` and `valid_mask` is derived from `y_sub[:len(X)]`. This compounds the BUG-001 issue - if X doesn't align with y, the mask will be incorrect.

**Suggested Fix:**
Fix BUG-001 first, then this will be correct.

---

### BUG-003: Unclosed file handle in get_max_relpos_counts
**File:** `compute_geometry_analysis.py`
**Lines:** 45-46
**Severity:** MEDIUM
**Category:** Missing Error Handling

**Description:**
```python
with open(mapping_file) as f:
    mapping = json.load(f)
```
If `json.load()` raises a `JSONDecodeError` (malformed JSON), the exception propagates but the file handle is closed correctly due to `with`. However, there's no error handling - a single corrupt file will crash the entire analysis.

**Suggested Fix:**
Wrap in try/except and log warning for corrupt files, continue processing other files:
```python
try:
    with open(mapping_file) as f:
        mapping = json.load(f)
except json.JSONDecodeError as e:
    log.warning(f"Corrupt JSON in {mapping_file}: {e}")
    continue
```

---

### BUG-004: Unclosed file handle in cache_position_mappings
**File:** `compute_geometry_analysis.py`
**Line:** 170
**Severity:** MEDIUM
**Category:** Missing Error Handling

**Description:**
Same issue as BUG-003 - if JSON is malformed, no graceful handling.

**Suggested Fix:**
Add try/except around JSON loading.

---

### BUG-005: Empty pc_correlations crashes summary generation
**File:** `compute_geometry_analysis.py`
**Lines:** 360-361
**Severity:** MEDIUM
**Category:** Edge Case

**Description:**
```python
"pca": {k: {"top_pc": v["pc_correlations"][0][0], "top_corr": v["pc_correlations"][0][1]}
       for k, v in pca_all.items() if v.get("pc_correlations")}
```
If `pc_correlations` is an empty list `[]`, `v.get("pc_correlations")` returns truthy `[]`, but `v["pc_correlations"][0]` will raise `IndexError`.

**Suggested Fix:**
Change condition to:
```python
if v.get("pc_correlations") and len(v["pc_correlations"]) > 0
```

---

### BUG-006: y_sub index mismatch with X in linear probes
**File:** `compute_linear_probes.py`
**Lines:** 198-200
**Severity:** HIGH
**Category:** Index Error / Data Mismatch

**Description:**
```python
y_sub = y[:len(X)]
valid_mask = ~np.isnan(y_sub)
X_valid = X[valid_mask]
```
Same issue as BUG-001. The `load_target()` function in this file also doesn't return which sample indices were used, so `y[:len(X)]` doesn't correspond to the actual samples in X.

**Suggested Fix:**
Same as BUG-001 - track and return valid sample indices.

---

### BUG-007: Unclosed file handle in cache_position_mappings (linear probes)
**File:** `compute_linear_probes.py`
**Line:** 90
**Severity:** MEDIUM
**Category:** Missing Error Handling

**Description:**
Same issue as BUG-003/BUG-004.

**Suggested Fix:**
Add try/except around JSON loading.

---

### BUG-008: Unclosed file handle in load_horizons (linear probes)
**File:** `compute_linear_probes.py`
**Line:** 145
**Severity:** MEDIUM
**Category:** Missing Error Handling

**Description:**
```python
with open(f) as fp:
    choice_data = json.load(fp)
```
If JSON is malformed, crashes the entire run.

**Suggested Fix:**
Add try/except, raise with more context or skip sample.

---

### BUG-009: Empty pos_r2 list causes division issues
**File:** `compute_linear_probes.py`
**Line:** 316
**Severity:** LOW
**Category:** Empty List

**Description:**
```python
means = [np.mean(layer_r2[l]) if layer_r2[l] else 0 for l in layers_sorted]
```
`np.mean([])` raises a warning and returns `nan`, but the code handles this with `if layer_r2[l] else 0`. However, at line 316:
```python
heatmap[i, j] = np.mean(vals) if vals else 0
```
If `vals` is empty, this is fine. But the issue is `np.mean([])` produces a warning.

**Suggested Fix:**
The current code is correct but could suppress the warning or use explicit empty check.

---

### BUG-010: cleanup_empty_dirs may fail on concurrent access
**File:** `visualize_geometry_analysis.py`
**Lines:** 111-113
**Severity:** MEDIUM
**Category:** Race Condition / File Operation

**Description:**
```python
for d in viz_dir.iterdir():
    if d.is_dir() and not any(d.iterdir()):
        d.rmdir()
```
Between checking `not any(d.iterdir())` and calling `d.rmdir()`, another process could create a file in the directory, causing `rmdir()` to fail with `OSError: Directory not empty`.

**Suggested Fix:**
Wrap in try/except:
```python
try:
    d.rmdir()
except OSError:
    pass  # Directory no longer empty or already removed
```

---

### BUG-011: shutil.rmtree followed by rename may fail
**File:** `visualize_geometry_analysis.py`
**Lines:** 194-196
**Severity:** MEDIUM
**Category:** Race Condition / File Operation

**Description:**
```python
if viz_dir.exists():
    shutil.rmtree(viz_dir)
plots_dir.rename(viz_dir)
```
If another process creates `viz_dir` between `rmtree` and `rename`, the rename will fail. Also, if `plots_dir` doesn't exist, `rename()` will raise `FileNotFoundError`.

**Suggested Fix:**
Add existence check for `plots_dir` and wrap in try/except:
```python
if plots_dir.exists():
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
    plots_dir.rename(viz_dir)
```

---

### BUG-012: Empty array components in PCAResult
**File:** `visualize_geometry_analysis.py`
**Line:** 69
**Severity:** LOW
**Category:** Empty Array Handling

**Description:**
```python
components = np.array([[]])
if (d / "components.npy").exists():
    components = np.load(d / "components.npy")
```
Creating `np.array([[]])` creates a 2D array with shape (1, 0). This may cause shape mismatches downstream if components are expected to have specific dimensions.

**Suggested Fix:**
Use `np.array([])` or `np.empty((0, 0))` for empty fallback, or better yet, set to `None` and check for it downstream.

---

### BUG-013: count_files may fail on non-existent viz_dir
**File:** `visualize_geometry_analysis.py`
**Lines:** 121-123
**Severity:** LOW
**Category:** Empty Directory

**Description:**
```python
for d in sorted(viz_dir.iterdir()):
    if d.is_dir():
        counts[d.name] = sum(1 for _ in d.rglob("*") if _.is_file())
```
The function has no protection if `viz_dir` doesn't exist. The caller checks `if viz_dir.exists()` but a race condition could cause the directory to be removed.

**Suggested Fix:**
Add try/except or check existence at start of function.

---

### BUG-014: Target key parsing may fail for complex position names
**File:** `generate_geometry_samples.py`
**Lines:** 325-331
**Severity:** LOW
**Category:** Parsing Logic

**Description:**
```python
for comp in COMPONENTS:
    comp_parts = comp.split("_")
    comp_len = len(comp_parts)
    if "_".join(parts[1 : 1 + comp_len]) == comp:
        position = "_".join(parts[1 + comp_len :])
```
This parsing assumes the component name appears right after the layer. If a key has format `L0_resid_pre_time_horizon`, with `parts = ["L0", "resid", "pre", "time", "horizon"]`, checking `"_".join(parts[1:3])` gives `"resid_pre"` which matches. Position becomes `"time_horizon"`. This works.

However, if there's ever a position name that starts with a component name prefix (e.g., hypothetically `resid_pre_something`), it could match incorrectly. Low severity because current position names don't have this issue.

**Suggested Fix:**
Use the `parse_key()` function from `compute_geometry_analysis.py` for consistency.

---

### BUG-015: ValueError on empty samples directory
**File:** `compute_geometry_analysis.py`
**Lines:** 163-164
**Severity:** MEDIUM
**Category:** ValueError on Empty Directory

**Description:**
```python
sample_dirs = sorted(
    [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
    key=lambda x: int(x.name.split("_")[1])
)
```
If there are no sample directories, `sample_dirs` will be empty. This is handled correctly later. But if there's a directory with malformed name like `sample_abc`, `int("abc")` will raise `ValueError`.

**Suggested Fix:**
Add validation:
```python
def safe_sample_idx(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (ValueError, IndexError):
        return -1  # Sort to beginning, will be filtered

sample_dirs = sorted(
    [d for d in samples_dir.iterdir()
     if d.is_dir() and d.name.startswith("sample_") and d.name.split("_")[1].isdigit()],
    key=lambda x: int(x.name.split("_")[1])
)
```

---

### BUG-016: Import not at top of file
**File:** `compute_geometry_analysis.py`
**Line:** 86
**Severity:** LOW
**Category:** Import Location

**Description:**
```python
import re
_KEY_PATTERN = re.compile(r"L(\d+)_(.+)")
```
This import is after line 84 (end of `count_files` function), not at the top of the file with other imports. While technically functional, it violates the project guideline "All imports always on top".

**Suggested Fix:**
Move `import re` to line 11 (with other standard library imports).

---

### BUG-017: Subsampled t-SNE output is misleading
**File:** `compute_geometry_analysis.py`
**Lines:** 403-407
**Severity:** MEDIUM
**Category:** Subsampled t-SNE Output Misleading

**Description:**
```python
if idx is not None:
    e = np.zeros((n, 3), dtype=np.float32)
    e[idx] = e_sub
else:
    e = e_sub
```
When t-SNE subsamples (n > 2000), it computes embeddings for a subset but then pads with zeros for non-selected samples. This means:
1. Non-selected samples have embedding `[0, 0, 0]` which is likely in the middle of the point cloud
2. No indication is saved that these are placeholder values
3. Downstream visualization will show these zeros as real data points

**Suggested Fix:**
Use NaN instead of zeros to indicate missing values:
```python
e = np.full((n, 3), np.nan, dtype=np.float32)
e[idx] = e_sub
```
Or save a mask indicating which samples were computed.

---

### BUG-018: Data race on summary.json
**File:** `compute_linear_probes.py`
**Lines:** 257-263
**Severity:** MEDIUM
**Category:** Data Race on summary.json

**Description:**
```python
if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)
else:
    summary = {}
summary["linear_probe"] = lp_all
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
```
If `compute_geometry_analysis.py` and `compute_linear_probes.py` run concurrently (or nearly concurrently), they both read/write `summary.json`. This can cause:
1. Lost updates (one overwrites the other's changes)
2. Corrupted JSON if write happens during read

**Suggested Fix:**
Use file locking or atomic writes:
```python
import tempfile
import shutil

# Write to temp file first
with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=summary_file.parent) as tf:
    json.dump(summary, tf, indent=2)
    temp_path = tf.name

# Atomic rename
shutil.move(temp_path, summary_file)
```

---

## Memory Leak Analysis

### Potential Memory Issues

1. **compute_geometry_analysis.py lines 230-234**: Arrays are created in a loop but `valid_files` list grows unboundedly before allocation. For very large datasets, this could consume significant memory before the actual array is allocated.

2. **compute_geometry_analysis.py line 486**: In `run_embeddings`, `batch_data` holds all loaded arrays for a batch. With `batch_size=20` and high-dimensional data, this could be several GB. The memory is freed after `ProcessPoolExecutor` completes, but during execution both parent and child processes hold copies.

3. **compute_linear_probes.py line 261**: `summary["linear_probe"] = lp_all` assigns the entire dictionary to summary. If `lp_all` is large, this doubles memory usage momentarily.

### Mitigations Present

The code does call `clear_gpu_memory(aggressive=True)` periodically (lines 350-351 in compute_geometry_analysis.py, line 251-252 in compute_linear_probes.py), and uses `del` to remove large arrays. These are good practices.

---

## Summary by Severity

### HIGH (2 bugs)
- BUG-001: y/X index mismatch in PCA (compute_geometry_analysis.py)
- BUG-006: y/X index mismatch in linear probes (compute_linear_probes.py)

### MEDIUM (9 bugs)
- BUG-002: PCA valid_mask cascading issue
- BUG-003: Missing JSON error handling in get_max_relpos_counts
- BUG-004: Missing JSON error handling in cache_position_mappings
- BUG-005: Empty pc_correlations IndexError
- BUG-007: Missing JSON error handling (linear probes)
- BUG-008: Missing JSON error handling in load_horizons
- BUG-010: Race condition in cleanup_empty_dirs
- BUG-011: Race condition in plots_dir rename
- BUG-015: ValueError on malformed sample directory names
- BUG-017: Subsampled t-SNE zeros are misleading
- BUG-018: Data race on summary.json

### LOW (4 bugs)
- BUG-009: Empty list warning in visualization
- BUG-012: Empty array shape in PCAResult
- BUG-013: count_files race condition
- BUG-014: Target key parsing edge case
- BUG-016: Import not at top of file
