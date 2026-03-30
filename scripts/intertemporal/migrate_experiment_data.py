#!/usr/bin/env python3
"""Migrate experiment data to new folder structure.

This script:
1. Renames att/ -> att_patching/ in pair_* folders
2. Renames component_comparison/ -> sweep_component_comparison/
3. Merges coarse_agg/ into agg/ (moves coarse_agg/*.json -> agg/coarse/)
4. Backfills contrastive_preference.json for each pair (requires preference data)

Usage:
    python scripts/intertemporal/migrate_experiment_data.py [--dry-run] [experiments_dir]

If experiments_dir is not specified, uses the default experiments directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def get_experiments_dir() -> Path:
    """Get the default experiments directory."""
    from src.intertemporal.common import get_experiment_dir
    return get_experiment_dir()


def migrate_att_folder(pair_dir: Path, dry_run: bool) -> bool:
    """Rename att/ -> att_patching/ if needed, or merge contents."""
    old_path = pair_dir / "att"
    new_path = pair_dir / "att_patching"

    if not old_path.exists():
        return False

    if not new_path.exists():
        # Simple rename
        if dry_run:
            print(f"  Would rename: {old_path} -> {new_path}")
        else:
            old_path.rename(new_path)
            print(f"  Renamed: {old_path.name} -> {new_path.name}")
        return True

    # Both exist - merge contents from old to new
    changed = False
    for item in old_path.iterdir():
        dest = new_path / item.name
        if not dest.exists():
            if dry_run:
                print(f"  Would move: {item.name} from att/ to att_patching/")
            else:
                shutil.move(str(item), str(dest))
                print(f"  Moved: {item.name} to att_patching/")
            changed = True

    # Remove old att/ if empty
    if not dry_run:
        try:
            old_path.rmdir()
            print(f"  Removed empty: att/")
        except OSError:
            pass  # Not empty

    return changed


def migrate_component_comparison(pair_dir: Path, dry_run: bool) -> bool:
    """Rename component_comparison/ -> sweep_component_comparison/ if needed."""
    old_path = pair_dir / "component_comparison"
    new_path = pair_dir / "sweep_component_comparison"

    if old_path.exists() and not new_path.exists():
        if dry_run:
            print(f"  Would rename: {old_path} -> {new_path}")
        else:
            old_path.rename(new_path)
            print(f"  Renamed: {old_path.name} -> {new_path.name}")
        return True
    return False


def migrate_agg_component_comparison(exp_dir: Path, dry_run: bool) -> bool:
    """Rename agg/*/component_comparison/ -> agg/*/sweep_component_comparison/."""
    agg_dir = exp_dir / "agg"
    if not agg_dir.exists():
        return False

    changed = False
    for slice_dir in agg_dir.iterdir():
        if slice_dir.is_dir():
            old_path = slice_dir / "component_comparison"
            new_path = slice_dir / "sweep_component_comparison"
            if old_path.exists() and not new_path.exists():
                if dry_run:
                    print(f"  Would rename: {old_path} -> {new_path}")
                else:
                    old_path.rename(new_path)
                    print(f"  Renamed: {old_path.relative_to(exp_dir)}")
                changed = True
    return changed


def migrate_coarse_agg(exp_dir: Path, dry_run: bool) -> bool:
    """Move coarse_agg/*.json -> agg/coarse/*.json."""
    coarse_agg_dir = exp_dir / "coarse_agg"
    if not coarse_agg_dir.exists():
        return False

    # Target directory
    new_coarse_dir = exp_dir / "agg" / "coarse"

    if dry_run:
        print(f"  Would move: {coarse_agg_dir} -> {new_coarse_dir}")
        return True

    new_coarse_dir.mkdir(parents=True, exist_ok=True)

    # Move all files
    for json_file in coarse_agg_dir.glob("*.json"):
        dest = new_coarse_dir / json_file.name
        if not dest.exists():
            shutil.move(str(json_file), str(dest))
            print(f"  Moved: {json_file.name} -> agg/coarse/")

    # Remove old directory if empty
    try:
        coarse_agg_dir.rmdir()
        print(f"  Removed empty: coarse_agg/")
    except OSError:
        pass  # Not empty

    return True


def backfill_contrastive_prefs_and_horizon(exp_dir: Path, dry_run: bool) -> bool:
    """Backfill contrastive_preference.json and horizon analysis for existing pairs.

    This requires reloading preference data, which may not be available.
    """
    # Check if any pairs need backfilling
    pairs_needing_backfill = []
    for pair_dir in sorted(exp_dir.glob("pair_*")):
        pref_path = pair_dir / "contrastive_preference.json"
        if not pref_path.exists():
            pairs_needing_backfill.append(pair_dir)

    # Check if horizon analysis exists
    horizon_path = exp_dir / "horizon_analysis.json"
    needs_horizon_analysis = not horizon_path.exists()

    if not pairs_needing_backfill and not needs_horizon_analysis:
        return False

    if dry_run:
        if pairs_needing_backfill:
            print(f"  Would backfill contrastive_preference.json for {len(pairs_needing_backfill)} pairs")
        if needs_horizon_analysis:
            print(f"  Would create horizon analysis files")
        return True

    # Try to rebuild from preference data
    # This requires loading the experiment context
    try:
        from src.intertemporal.experiments.experiment_config import ExperimentConfig
        from src.intertemporal.experiments.experiment_context import ExperimentContext
        from src.intertemporal.experiments.horizon_analysis import build_horizon_analysis, save_horizon_analysis
        from src.intertemporal.preference import load_and_merge_preference_data
        from src.intertemporal.common import get_pref_dataset_dir

        # Try to infer config from directory name
        # Format: pref_exp__<dataset>_<model>_<n_pairs>
        exp_name = exp_dir.name
        if not exp_name.startswith("pref_exp__"):
            print(f"  Cannot parse experiment name: {exp_name}")
            return False

        # Load preference data
        prefix = "_".join(exp_name.split("_")[2:-2])  # Extract dataset prefix
        pref_data = load_and_merge_preference_data(prefix, get_pref_dataset_dir())
        if not pref_data:
            print(f"  Cannot load preference data for: {prefix}")
            return False

        # Build minimal config
        model_name = exp_name.split("_")[-2]
        n_pairs = int(exp_name.split("_")[-1])

        cfg = ExperimentConfig(
            model=model_name,
            n_pairs=n_pairs,
            dataset_config={"types": [prefix.split("_")[-1]] if "_" in prefix else ["horizon"]},
        )
        cfg.att_patch["enabled"] = False
        cfg.coarse_patch["enabled"] = False
        cfg.viz["enabled"] = False

        ctx = ExperimentContext(cfg, output_dir=exp_dir)
        ctx.pref_data = pref_data

        # Build pairs to populate pref_pairs
        _ = ctx.pairs

        # Save contrastive preferences
        for pair_dir in pairs_needing_backfill:
            pair_idx = int(pair_dir.name.split("_")[1])
            ctx.save_contrastive_pref(pair_idx)
            print(f"  Saved: {pair_dir.name}/contrastive_preference.json")

        # Build and save horizon analysis
        if needs_horizon_analysis:
            horizon_analysis = build_horizon_analysis(ctx.pref_pairs)
            save_horizon_analysis(horizon_analysis, exp_dir)
            print(f"  Created horizon analysis files")

        return True

    except Exception as e:
        print(f"  Error backfilling: {e}")
        return False


def migrate_experiment(exp_dir: Path, dry_run: bool) -> None:
    """Migrate a single experiment directory."""
    print(f"\nMigrating: {exp_dir.name}")

    changes = 0

    # 1. Migrate pair_* folders
    for pair_dir in sorted(exp_dir.glob("pair_*")):
        if migrate_att_folder(pair_dir, dry_run):
            changes += 1
        if migrate_component_comparison(pair_dir, dry_run):
            changes += 1

    # 2. Migrate agg/*/component_comparison
    if migrate_agg_component_comparison(exp_dir, dry_run):
        changes += 1

    # 3. Merge coarse_agg/ into agg/
    if migrate_coarse_agg(exp_dir, dry_run):
        changes += 1

    # 4. Backfill contrastive_preference.json and horizon analysis
    if backfill_contrastive_prefs_and_horizon(exp_dir, dry_run):
        changes += 1

    if changes == 0:
        print("  (no changes needed)")


def main():
    parser = argparse.ArgumentParser(description="Migrate experiment data to new folder structure")
    parser.add_argument("experiments_dir", nargs="?", help="Path to experiments directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir)
    else:
        exp_dir = get_experiments_dir()

    print(f"Experiments directory: {exp_dir}")
    print(f"Dry run: {args.dry_run}")

    if not exp_dir.exists():
        print(f"Directory does not exist: {exp_dir}")
        return

    # Check if this is a single experiment or a directory of experiments
    if (exp_dir / "pair_0").exists():
        # Single experiment
        migrate_experiment(exp_dir, args.dry_run)
    else:
        # Directory of experiments
        experiments = [d for d in exp_dir.iterdir() if d.is_dir() and (d / "pair_0").exists()]
        print(f"Found {len(experiments)} experiments")

        for experiment in sorted(experiments):
            migrate_experiment(experiment, args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
