"""Shared utilities for geometry analysis.

Constants and helper functions used across geometry analysis scripts:
- compute_geometry_analysis.py
- compute_linear_probes.py

This module provides:
- Standard layer, component, and method constants
- Target key generation and parsing
- Position mapping utilities
- Activation file discovery
- Time horizon loading
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# NaN VALIDATION UTILITIES
# =============================================================================


class ActivationNaNError(Exception):
    """Raised when NaN values are detected in activation data."""

    pass


def _print_activation_nan_warning(context: str, details: str) -> None:
    """Print a loud warning about NaN values in activation data."""
    warning_msg = f"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! CRITICAL: NaN VALUES IN ACTIVATION DATA !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! Context: {context}
!!! {details}
!!!
!!! This indicates CORRUPT or INVALID activation files
!!! The data extraction pipeline may have failed
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    print(warning_msg, file=sys.stderr)
    logging.getLogger(__name__).critical(warning_msg)


from ..common.semantic_positions import (
    PROMPT_POSITIONS,
    RESPONSE_POSITIONS,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Standard layers for analysis (early, middle, late layers)
LAYERS: list[int] = [0, 1, 3, 12, 18, 19, 20, 21, 23, 24, 25, 28, 31, 34, 35]

# Activation components at each layer
COMPONENTS: list[str] = ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]

# Dimensionality reduction methods
# METHODS: list[str] = ["pca", "umap", "tsne"]
METHODS: list[str] = ["pca"]

# All semantic positions (prompt + response)
POSITIONS: list[str] = PROMPT_POSITIONS + RESPONSE_POSITIONS


# =============================================================================
# Target Key Generation
# =============================================================================


def target_keys() -> list[str]:
    """Generate combined (aggregated) target keys.

    Target key format: L{layer}_{component}_{position}

    Returns:
        List of target keys for all layer/component/position combinations.

    Example:
        >>> keys = target_keys()
        >>> keys[0]
        'L0_resid_pre_time_horizon'
    """
    return [
        f"L{layer}_{comp}_{pos}"
        for layer in LAYERS
        for comp in COMPONENTS
        for pos in POSITIONS
    ]


# =============================================================================
# Target Key Parsing
# =============================================================================

# Regex patterns for parsing target keys
_KEY_PATTERN = re.compile(r"L(\d+)_(.+)")
_RELPOS_PATTERN = re.compile(r"(.+)_r(\d+)$")


def parse_key(key: str) -> tuple[int, str, str, int | None] | None:
    """Parse target key into (layer, component, position, rel_pos).

    Supports both combined keys and per-token keys:
    - Combined: L{layer}_{component}_{position}
    - Per-token: L{layer}_{component}_{position}_r{rel_pos}

    Args:
        key: Target key string (e.g., "L0_resid_pre_time_horizon" or
             "L0_resid_pre_time_horizon_r1")

    Returns:
        Tuple of (layer, component, position, rel_pos) where rel_pos is
        None for combined keys, or the token index for per-token keys.
        Returns None if the key cannot be parsed.

    Example:
        >>> parse_key("L12_attn_out_response_choice")
        (12, 'attn_out', 'response_choice', None)
        >>> parse_key("L12_attn_out_time_horizon_r1")
        (12, 'attn_out', 'time_horizon', 1)
    """
    match = _KEY_PATTERN.match(key)
    if not match:
        return None

    layer = int(match.group(1))
    rest = match.group(2)

    # Try to match each known component
    for comp in COMPONENTS:
        if rest.startswith(comp + "_"):
            pos_part = rest[len(comp) + 1 :]

            # Check for _r{N} suffix (per-token key)
            rel_match = _RELPOS_PATTERN.match(pos_part)
            if rel_match:
                position = rel_match.group(1)
                rel_pos = int(rel_match.group(2))
                return (layer, comp, position, rel_pos)
            else:
                return (layer, comp, pos_part, None)

    return None


# =============================================================================
# Position Mapping Utilities
# =============================================================================


def get_abs_pos(mapping: dict, pos: str) -> int | list[int] | None:
    """Get absolute position(s) from position mapping.

    Args:
        mapping: Position mapping dict with 'named_positions' key.
        pos: Semantic position name (e.g., "time_horizon").

    Returns:
        Single int, list of ints, or None if position not in this sample.

    Raises:
        KeyError: If 'named_positions' key is missing from mapping.

    Example:
        >>> mapping = {"named_positions": {"time_horizon": [42, 43, 44]}}
        >>> get_abs_pos(mapping, "time_horizon")
        [42, 43, 44]
    """
    if "named_positions" not in mapping:
        raise KeyError("named_positions missing from mapping")

    abs_pos = mapping["named_positions"].get(pos)
    return abs_pos  # May be None if position not in this sample


def cache_position_mappings(data_dir: Path) -> tuple[list[Path], dict[int, dict]]:
    """Cache all position mappings from sample directories.

    Reads position_mapping.json from each sample directory once and caches
    the results to avoid repeated JSON parsing.

    Args:
        data_dir: Dataset directory containing data/samples/.

    Returns:
        Tuple of (sample_dirs, mapping_cache) where:
        - sample_dirs: Sorted list of sample directory paths
        - mapping_cache: Dict mapping sample index to position mapping dict

    Example:
        >>> sample_dirs, mappings = cache_position_mappings(Path("out/geo/investment"))
        >>> len(mappings)
        500
    """
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        [
            d
            for d in samples_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        ],
        key=lambda x: int(x.name.split("_")[1]),
    )

    mapping_cache: dict[int, dict] = {}
    for i, sample_dir in enumerate(sample_dirs):
        mapping_file = sample_dir / "position_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapping_cache[i] = json.load(f)

    logger.debug(f"Cached {len(mapping_cache)} position mappings from {data_dir}")
    return sample_dirs, mapping_cache


# =============================================================================
# Activation File Discovery
# =============================================================================


def find_activation_file(
    sample_dir: Path,
    layer: int,
    comp: str,
    abs_pos: int | list[int],
) -> Path | None:
    """Find activation file for a given position.

    Supports both old and new file formats:
    - New format (layer subfolders): sample_0/L35/resid_post_129.npy
    - Old format (flat): sample_0/L35_resid_post_129.npy

    If abs_pos is a list, tries each position in order until finding one
    that exists.

    Args:
        sample_dir: Path to sample directory (e.g., .../samples/sample_0).
        layer: Transformer layer number.
        comp: Activation component (resid_pre, attn_out, mlp_out, resid_post).
        abs_pos: Absolute token position(s) to look for.

    Returns:
        Path to the activation file, or None if not found.

    Example:
        >>> find_activation_file(Path("out/geo/investment/data/samples/sample_0"), 35, "resid_post", 129)
        PosixPath('out/geo/investment/data/samples/sample_0/L35/resid_post_129.npy')
    """
    if isinstance(abs_pos, list):
        # Try each position in order until we find one that exists
        for pos in abs_pos:
            # Try new format first (layer subfolders)
            new_path = sample_dir / f"L{layer}" / f"{comp}_{pos}.npy"
            if new_path.exists():
                return new_path
            # Fall back to old format
            old_path = sample_dir / f"L{layer}_{comp}_{pos}.npy"
            if old_path.exists():
                return old_path
        return None
    else:
        # Single position
        new_path = sample_dir / f"L{layer}" / f"{comp}_{abs_pos}.npy"
        if new_path.exists():
            return new_path
        old_path = sample_dir / f"L{layer}_{comp}_{abs_pos}.npy"
        return old_path if old_path.exists() else None


# =============================================================================
# Time Horizon Loading
# =============================================================================


def load_horizons(data_dir: Path) -> np.ndarray:
    """Load log time horizons from choice.json files.

    Reads the time_horizon_months field from each sample's choice.json
    and returns log10(months + 1) for each sample.

    Args:
        data_dir: Dataset directory containing data/samples/.

    Returns:
        Array of log10(time_horizon_months + 1) values.
        NaN is used for samples with null time_horizon_months (no-horizon condition).

    Raises:
        FileNotFoundError: If choice.json is missing for any sample.
        KeyError: If time_horizon_months field is missing from choice.json.

    Example:
        >>> horizons = load_horizons(Path("out/geo/investment"))
        >>> horizons.shape
        (500,)
        >>> np.isnan(horizons).sum()  # Count no-horizon samples
        50
    """
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        [
            d
            for d in samples_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        ],
        key=lambda x: int(x.name.split("_")[1]),
    )

    horizons = np.empty(len(sample_dirs), dtype=np.float32)

    for i, sample_dir in enumerate(sample_dirs):
        choice_file = sample_dir / "choice.json"
        if not choice_file.exists():
            raise FileNotFoundError(
                f"choice.json missing for sample {i}: {choice_file}\n"
                "Re-run data extraction to regenerate choice.json files."
            )

        with open(choice_file) as f:
            choice_data = json.load(f)

        if "time_horizon_months" not in choice_data:
            raise KeyError(
                f"time_horizon_months missing from choice.json in sample {i}: {choice_file}\n"
                "This field is REQUIRED. Re-run data extraction to regenerate choice.json files."
            )

        val = choice_data["time_horizon_months"]
        if val is None:
            horizons[i] = np.nan  # No-horizon samples get NaN
        else:
            horizons[i] = np.log10(
                val + 1
            )  # Consistent offset with geometry_analysis.py

    logger.debug(f"Loaded {len(horizons)} time horizons from {data_dir}")
    return horizons


# =============================================================================
# Activation Loading
# =============================================================================


def load_target(
    data_dir: Path,
    key: str,
    sample_dirs: list[Path] | None = None,
    mapping_cache: dict[int, dict] | None = None,
) -> tuple[np.ndarray, list[int]] | None:
    """Load activations for ONE target.

    For combined keys (no _r{N} suffix): loads first available token per sample.
    For per-rel_pos keys (_r{N} suffix): loads only the specific rel_pos token.

    Args:
        data_dir: Dataset directory
        key: Target key (e.g., "L0_resid_pre_time_horizon" or "L0_resid_pre_time_horizon_r1")
        sample_dirs: Pre-cached list of sample directories (optional, loaded if None)
        mapping_cache: Pre-cached position mappings {sample_idx: mapping_dict} (optional)

    Returns:
        Tuple of (activations array, valid sample indices) or None if insufficient data.
        The valid_indices list contains the original sample indices that have valid activations,
        which must be used to index into the horizons array (y[valid_indices]).
    """
    parsed = parse_key(key)
    if not parsed:
        return None
    layer, comp, pos, rel_pos = parsed

    # Use cached mappings if provided, otherwise load fresh
    if sample_dirs is None or mapping_cache is None:
        sample_dirs, mapping_cache = cache_position_mappings(data_dir)

    # Single pass: load all valid activation files
    valid_indices: list[int] = []
    valid_files: list[Path] = []
    dim = None

    for i, d in enumerate(sample_dirs):
        if i not in mapping_cache:
            continue
        abs_pos = get_abs_pos(mapping_cache[i], pos)
        if abs_pos is None:
            continue

        # For per-rel_pos: select specific token index
        if rel_pos is not None:
            if isinstance(abs_pos, list):
                if rel_pos >= len(abs_pos):
                    continue  # This sample doesn't have this rel_pos
                abs_pos = abs_pos[rel_pos]
            elif rel_pos > 0:
                continue  # Single token, but asking for rel_pos > 0

        act_file = find_activation_file(d, layer, comp, abs_pos)
        if act_file:
            if dim is None:
                dim = np.load(act_file, mmap_mode="r").shape[0]
            valid_indices.append(i)
            valid_files.append(act_file)

    if len(valid_files) < 4 or dim is None:
        return None

    # Load all valid activations into pre-allocated array
    X = np.empty((len(valid_files), dim), dtype=np.float32)
    for idx, act_file in enumerate(valid_files):
        activation = np.load(act_file)

        # =====================================================================
        # CRITICAL: Validate each activation file as it's loaded
        # =====================================================================
        if np.any(~np.isfinite(activation)):
            nan_count = np.sum(np.isnan(activation))
            inf_count = np.sum(np.isinf(activation))
            _print_activation_nan_warning(
                "LOADING ACTIVATION FILE",
                f"Target: {key} | File: {act_file} | NaN: {nan_count}, Inf: {inf_count}",
            )
            raise ActivationNaNError(
                f"Activation file {act_file} contains invalid values. "
                f"NaN: {nan_count}, Inf: {inf_count}. "
                f"Re-run data extraction to regenerate activation files."
            )

        X[idx] = activation

    # =========================================================================
    # FINAL VALIDATION: Check entire loaded matrix
    # =========================================================================
    if np.any(~np.isfinite(X)):
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        _print_activation_nan_warning(
            "LOADED ACTIVATION MATRIX",
            f"Target: {key} | Total NaN: {nan_count}, Inf: {inf_count} | Shape: {X.shape}",
        )
        raise ActivationNaNError(
            f"Loaded activation matrix for {key} contains invalid values. "
            f"NaN: {nan_count}, Inf: {inf_count}."
        )

    return X, valid_indices
