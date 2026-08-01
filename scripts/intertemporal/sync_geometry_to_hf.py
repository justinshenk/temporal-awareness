#!/usr/bin/env python3
"""Incrementally sync a geometry run directory to a Hugging Face dataset repo.

The rented GPU boxes have no persistent volume, so the Hub is the only thing that
survives a machine dying. This script pushes whatever exists on disk right now,
records what it pushed, and can be re-run (or left watching) so that later passes
push only what appeared since.

It runs independently of the extraction process. Point it at any run directory,
from any machine, at any time.

The manifest (``<run-dir>/.hf_sync_manifest.jsonl``) is an append-only log, because
a run can hold millions of files and rewriting the whole record after every commit
would not scale::

    {"version": 1}
    {"target": "dataset:unrulyabstractions/temporal-awareness:geometry/llama31_8b_health"}
    ["config.json", 42, 1785564756021633924, 1785564780]

A file is re-uploaded whenever its size or mtime differs from the recorded entry,
so a file that was captured mid-write gets repaired on a later pass. Entries are
appended and fsynced only after the commit carrying them succeeds, so an
interrupted sync never claims an upload that did not happen.

Usage:
    # one pass, then exit
    uv run python scripts/intertemporal/sync_geometry_to_hf.py \
        --run-dir out/geo/llama31_8b_health --prefix geometry/llama31_8b_health --once

    # follow a running extraction until it completes
    uv run python scripts/intertemporal/sync_geometry_to_hf.py \
        --run-dir out/geo/llama31_8b_health --prefix geometry/llama31_8b_health \
        --watch --interval 120

    # show what would upload, upload nothing
    uv run python scripts/intertemporal/sync_geometry_to_hf.py \
        --run-dir out/geo/llama31_8b_health --dry-run

    # re-list the remote tree and compare sizes against disk
    uv run python scripts/intertemporal/sync_geometry_to_hf.py \
        --run-dir out/geo/llama31_8b_health --once --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_REPO = "unrulyabstractions/temporal-awareness"
REPO_TYPE = "dataset"

MANIFEST_NAME = ".hf_sync_manifest.jsonl"
MANIFEST_VERSION = 1

# Pushed ahead of everything else so a partially synced run is still interpretable.
PRIORITY_FILES = (
    "config.json",
    "summary.json",
    "data/metadata.json",
    "data/prompt_dataset.json",
)

# summary.json is written by generate_geometry_samples.py once extraction ends,
# so its presence plus an idle pass means the run is fully captured.
DEFAULT_DONE_FILE = "summary.json"

EXCLUDED_NAMES = frozenset({MANIFEST_NAME, ".DS_Store", ".gitattributes"})
EXCLUDED_DIRS = frozenset({".git", "__pycache__", ".ipynb_checkpoints"})
EXCLUDED_SUFFIXES = (".tmp", ".partial", ".lock", ".swp")

# A full geometry run holds millions of per-sample .npy files, so a dry run lists a
# sample of them instead of drowning the terminal, and the scan reports as it goes.
DRY_RUN_PREVIEW = 50
SCAN_REPORT_EVERY = 250_000

DEFAULT_BATCH_FILES = 1000
DEFAULT_BATCH_BYTES = 512 * 1024 * 1024
DEFAULT_INTERVAL = 120.0
DEFAULT_RETRIES = 5
RETRY_BASE_SECONDS = 5.0
RETRY_CAP_SECONDS = 300.0


@dataclass
class SyncTarget:
    """Where a run directory goes on the Hub."""

    repo_id: str
    prefix: str
    repo_type: str = REPO_TYPE

    @property
    def key(self) -> str:
        """Manifest key, so one run dir can be synced to several destinations."""
        return f"{self.repo_type}:{self.repo_id}:{self.prefix}"

    def path_in_repo(self, rel_path: str) -> str:
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path


@dataclass
class PassStats:
    """What one sync pass did."""

    files: int = 0
    n_bytes: int = 0
    commits: int = 0
    elapsed: float = 0.0
    pending: list[str] = field(default_factory=list)


# =============================================================================
# Formatting helpers
# =============================================================================


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


# =============================================================================
# Local scan and manifest
# =============================================================================


def is_excluded(name: str) -> bool:
    return name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES)


def scan_run_dir(run_dir: Path) -> dict[str, tuple[int, int]]:
    """Map every syncable file to (size, mtime_ns), keyed by posix relative path.

    A finished run holds millions of per-sample files, so this walks with scandir
    (one syscall per entry, no Path objects) and reports progress as it goes.
    """
    found: dict[str, tuple[int, int]] = {}
    root = str(run_dir)
    started = time.time()
    next_report = SCAN_REPORT_EVERY
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, PermissionError):
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in EXCLUDED_DIRS:
                    stack.append(entry.path)
                continue
            if is_excluded(entry.name):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                # The extractor replaced the file mid-scan; the next pass picks it up.
                continue
            found[entry.path[len(root) + 1 :]] = (stat.st_size, stat.st_mtime_ns)
        if len(found) >= next_report:
            logger.info("Scanning: %d files in %s", len(found), human_time(time.time() - started))
            next_report += SCAN_REPORT_EVERY
    return found


def load_manifest(manifest_path: Path) -> dict[str, dict[str, list]]:
    """Replay the append-only log into {target key: {path: [size, mtime_ns, uploaded_at]}}.

    A line is either a header object (``{"version": n}`` or ``{"target": key}``) or a
    file record ``[path, size, mtime_ns, uploaded_at]`` belonging to the last header.
    A truncated tail from a killed box is dropped rather than fatal.
    """
    targets: dict[str, dict[str, list]] = {}
    if not manifest_path.exists():
        return targets

    current: dict[str, list] | None = None
    dropped = 0
    with manifest_path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if isinstance(record, dict):
                if "target" in record:
                    current = targets.setdefault(record["target"], {})
                elif record.get("version") != MANIFEST_VERSION:
                    logger.warning("Manifest %s has unknown version %s; ignoring it", manifest_path, record)
                    return {}
            elif current is not None and len(record) == 4:
                current[record[0]] = record[1:]
            else:
                dropped += 1
    if dropped:
        logger.warning("Ignored %d unreadable manifest line(s) in %s", dropped, manifest_path)
    return targets


class ManifestLog:
    """Append-only record of what reached the Hub.

    A finished run holds millions of files, so entries are appended per commit.
    Rewriting the whole manifest would cost O(run size) on every commit.
    """

    def __init__(self, manifest_path: Path, target: SyncTarget):
        self.path = manifest_path
        self.target = target
        self._handle = None

    def append(self, entries: list[tuple[str, int, int, int]]) -> None:
        if self._handle is None:
            is_new = not self.path.exists()
            self._handle = self.path.open("a")
            if is_new:
                self._handle.write(json.dumps({"version": MANIFEST_VERSION}) + "\n")
            self._handle.write(json.dumps({"target": self.target.key}) + "\n")
        for entry in entries:
            self._handle.write(json.dumps(entry) + "\n")
        # fsync so a box dying right after a commit cannot lose the record of it.
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def pending_files(local: dict[str, tuple[int, int]], recorded: dict[str, list]) -> list[str]:
    """Paths needing upload, metadata first, then the rest in stable path order."""
    changed = set()
    for rel_path, (size, mtime_ns) in local.items():
        entry = recorded.get(rel_path)
        if entry is None or entry[0] != size or entry[1] != mtime_ns:
            changed.add(rel_path)
    priority = [p for p in PRIORITY_FILES if p in changed]
    rest = sorted(changed.difference(priority))
    return priority + rest


def batched(
    paths: list[str],
    local: dict[str, tuple[int, int]],
    max_files: int,
    max_bytes: int,
) -> list[list[str]]:
    """Split paths into commits bounded by both file count and total bytes."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for rel_path in paths:
        size = local[rel_path][0]
        if current and (len(current) >= max_files or current_bytes + size > max_bytes):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(rel_path)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


# =============================================================================
# Upload
# =============================================================================


def commit_batch(
    api: HfApi,
    target: SyncTarget,
    run_dir: Path,
    rel_paths: list[str],
    message: str,
    retries: int,
) -> None:
    """Commit one batch, retrying with exponential backoff.

    Operations are rebuilt on every attempt because a failed create_commit may
    have already consumed the file handles inside them.
    """
    for attempt in range(retries):
        operations = [
            CommitOperationAdd(
                path_in_repo=target.path_in_repo(rel_path),
                path_or_fileobj=str(run_dir / rel_path),
            )
            for rel_path in rel_paths
        ]
        try:
            api.create_commit(
                repo_id=target.repo_id,
                repo_type=target.repo_type,
                operations=operations,
                commit_message=message,
            )
            return
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            if attempt == retries - 1:
                raise
            delay = min(RETRY_BASE_SECONDS * (2**attempt), RETRY_CAP_SECONDS) + random.uniform(0, 1)
            logger.warning(
                "Commit failed (attempt %d/%d): %s -- retrying in %.1fs",
                attempt + 1,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)


def sync_once(
    api: HfApi,
    target: SyncTarget,
    run_dir: Path,
    manifest_path: Path,
    *,
    dry_run: bool,
    max_files: int,
    max_bytes: int,
    retries: int,
) -> PassStats:
    """Upload everything new or changed since the last recorded pass."""
    started = time.time()
    local = scan_run_dir(run_dir)
    recorded = load_manifest(manifest_path).get(target.key, {})
    pending = pending_files(local, recorded)

    total_bytes = sum(local[p][0] for p in pending)
    logger.info(
        "Scanned %s: %d files on disk, %d to upload (%s)",
        run_dir,
        len(local),
        len(pending),
        human_bytes(total_bytes),
    )

    if not pending:
        return PassStats(elapsed=time.time() - started)

    if dry_run:
        for rel_path in pending[:DRY_RUN_PREVIEW]:
            print(f"  WOULD UPLOAD  {human_bytes(local[rel_path][0]):>10}  {target.path_in_repo(rel_path)}")
        if len(pending) > DRY_RUN_PREVIEW:
            print(f"  ... and {len(pending) - DRY_RUN_PREVIEW} more files")
        logger.info(
            "[dry-run] %d files, %s would be uploaded to %s/%s -- nothing was sent",
            len(pending),
            human_bytes(total_bytes),
            target.repo_id,
            target.prefix,
        )
        return PassStats(elapsed=time.time() - started, pending=pending)

    batches = batched(pending, local, max_files, max_bytes)
    stats = PassStats()
    manifest_log = ManifestLog(manifest_path, target)
    try:
        for i, rel_paths in enumerate(batches, start=1):
            batch_bytes = sum(local[p][0] for p in rel_paths)
            batch_started = time.time()
            commit_batch(
                api,
                target,
                run_dir,
                rel_paths,
                message=f"sync {target.prefix or run_dir.name} [{i}/{len(batches)}] ({len(rel_paths)} files)",
                retries=retries,
            )
            uploaded_at = int(time.time())
            manifest_log.append([(p, local[p][0], local[p][1], uploaded_at) for p in rel_paths])

            stats.files += len(rel_paths)
            stats.n_bytes += batch_bytes
            stats.commits += 1
            batch_elapsed = time.time() - batch_started
            rate = batch_bytes / batch_elapsed if batch_elapsed > 0 else 0.0
            logger.info(
                "[commit %d/%d] %d files, %s in %s (%s/s) | pass total %d/%d files, %s, %s",
                i,
                len(batches),
                len(rel_paths),
                human_bytes(batch_bytes),
                human_time(batch_elapsed),
                human_bytes(int(rate)),
                stats.files,
                len(pending),
                human_bytes(stats.n_bytes),
                human_time(time.time() - started),
            )
    finally:
        manifest_log.close()

    stats.elapsed = time.time() - started
    return stats


# =============================================================================
# Verification
# =============================================================================


def verify_remote(api: HfApi, target: SyncTarget, run_dir: Path) -> bool:
    """Compare every local file against the sizes the Hub actually reports."""
    remote: dict[str, int] = {}
    try:
        for entry in api.list_repo_tree(
            target.repo_id,
            target.prefix or None,
            recursive=True,
            repo_type=target.repo_type,
        ):
            if isinstance(entry, RepoFile):
                rel_path = entry.path[len(target.prefix) + 1 :] if target.prefix else entry.path
                remote[rel_path] = entry.size
    except Exception as exc:  # noqa: BLE001 - a missing path means nothing landed
        logger.error("Could not list %s/%s: %s", target.repo_id, target.prefix, exc)
        return False

    local = scan_run_dir(run_dir)
    missing = [p for p in local if p not in remote]
    mismatched = [p for p in local if p in remote and remote[p] != local[p][0]]
    logger.info(
        "Verify %s/%s: %d local, %d remote, %d missing, %d size-mismatched",
        target.repo_id,
        target.prefix,
        len(local),
        len(remote),
        len(missing),
        len(mismatched),
    )
    for rel_path in missing[:20]:
        logger.error("MISSING on Hub: %s", rel_path)
    for rel_path in mismatched[:20]:
        logger.error("SIZE MISMATCH: %s local=%d remote=%d", rel_path, local[rel_path][0], remote[rel_path])
    return not missing and not mismatched


# =============================================================================
# Entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally sync a geometry run directory to the Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Geometry run directory, e.g. out/geo/<name>")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hub dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--prefix", default=None, help="Path inside the repo (default: geometry/<run-dir name>)")
    parser.add_argument("--watch", action="store_true", help="Keep syncing until the run completes")
    parser.add_argument("--once", action="store_true", help="Single pass, then exit (default)")
    parser.add_argument("--dry-run", action="store_true", help="List what would upload, upload nothing")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Seconds between watch passes")
    parser.add_argument(
        "--done-file",
        default=DEFAULT_DONE_FILE,
        help=f"Run-relative file whose presence marks completion (default: {DEFAULT_DONE_FILE})",
    )
    parser.add_argument("--batch-files", type=int, default=DEFAULT_BATCH_FILES, help="Max files per commit")
    parser.add_argument(
        "--batch-mb",
        type=float,
        default=DEFAULT_BATCH_BYTES / (1024 * 1024),
        help="Max megabytes per commit",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Attempts per commit before giving up")
    parser.add_argument("--create-repo", action="store_true", help="Create the repo if it does not exist")
    parser.add_argument("--verify", action="store_true", help="After syncing, compare Hub sizes against disk")
    args = parser.parse_args()

    if args.watch and args.once:
        parser.error("--watch and --once are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        logger.error("Run directory does not exist: %s", run_dir)
        return 1

    prefix = (args.prefix or f"geometry/{run_dir.name}").strip("/")
    target = SyncTarget(repo_id=args.repo, prefix=prefix)
    manifest_path = run_dir / MANIFEST_NAME

    api = HfApi()
    if not args.dry_run:
        if not api.repo_exists(target.repo_id, repo_type=target.repo_type):
            if not args.create_repo:
                logger.error(
                    "Repo %s (%s) does not exist. Re-run with --create-repo to create it.",
                    target.repo_id,
                    target.repo_type,
                )
                return 1
            api.create_repo(target.repo_id, repo_type=target.repo_type, exist_ok=True)
            logger.info("Created %s (%s)", target.repo_id, target.repo_type)

    logger.info("Syncing %s -> %s/%s (%s)", run_dir, target.repo_id, target.prefix, target.repo_type)

    max_bytes = int(args.batch_mb * 1024 * 1024)
    started = time.time()
    total_files = 0
    total_bytes = 0

    try:
        while True:
            stats = sync_once(
                api,
                target,
                run_dir,
                manifest_path,
                dry_run=args.dry_run,
                max_files=args.batch_files,
                max_bytes=max_bytes,
                retries=args.retries,
            )
            total_files += stats.files
            total_bytes += stats.n_bytes

            if not args.watch:
                break
            run_done = (run_dir / args.done_file).exists()
            if run_done and stats.files == 0 and not stats.pending:
                logger.info("%s is present and nothing is pending; the run is fully synced", args.done_file)
                break
            logger.info(
                "Waiting %.0fs for new files (done-file %s: %s)",
                args.interval,
                args.done_file,
                "present" if run_done else "absent",
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.warning("Interrupted; %d files (%s) are already on the Hub", total_files, human_bytes(total_bytes))
        return 130

    logger.info(
        "Done: %d files, %s in %s -> %s/%s",
        total_files,
        human_bytes(total_bytes),
        human_time(time.time() - started),
        target.repo_id,
        target.prefix,
    )

    if args.verify and not args.dry_run:
        return 0 if verify_remote(api, target, run_dir) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
