"""The HF sync must never re-upload what it already pushed, and never skip what changed."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "intertemporal" / "sync_geometry_to_hf.py"

# The sync tool is a script, not a package module, so load it by path. It has to be
# registered in sys.modules before exec so its @dataclass decorators can resolve.
_spec = importlib.util.spec_from_file_location("sync_geometry_to_hf", SCRIPT_PATH)
sync = importlib.util.module_from_spec(_spec)
sys.modules["sync_geometry_to_hf"] = sync
_spec.loader.exec_module(sync)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A miniature geometry run with metadata, a sample, and junk that must be skipped."""
    root = tmp_path / "run"
    (root / "data" / "samples" / "sample_0" / "L0").mkdir(parents=True)
    (root / "config.json").write_text('{"model": "x"}')
    (root / "summary.json").write_text('{"n_samples": 1}')
    (root / "data" / "metadata.json").write_text("{}")
    (root / "data" / "prompt_dataset.json").write_text("[]")
    (root / "data" / "samples" / "sample_0" / "choice.json").write_text('{"choice": "A"}')
    (root / "data" / "samples" / "sample_0" / "L0" / "resid_post_12.npy").write_bytes(b"\x00" * 64)
    (root / ".DS_Store").write_bytes(b"junk")
    (root / "data" / "samples" / "sample_0" / "L0" / "resid_post_13.npy.tmp").write_bytes(b"partial")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"junk")
    return root


def test_scan_skips_junk_and_finds_real_files(run_dir: Path):
    found = sync.scan_run_dir(run_dir)
    assert set(found) == {
        "config.json",
        "summary.json",
        "data/metadata.json",
        "data/prompt_dataset.json",
        "data/samples/sample_0/choice.json",
        "data/samples/sample_0/L0/resid_post_12.npy",
    }
    assert found["data/samples/sample_0/L0/resid_post_12.npy"][0] == 64


def test_metadata_is_uploaded_before_anything_else(run_dir: Path):
    pending = sync.pending_files(sync.scan_run_dir(run_dir), recorded={})
    assert pending[:4] == list(sync.PRIORITY_FILES)


def test_nothing_is_pending_once_recorded(run_dir: Path):
    local = sync.scan_run_dir(run_dir)
    recorded = {path: [size, mtime, 0] for path, (size, mtime) in local.items()}
    assert sync.pending_files(local, recorded) == []


def test_a_changed_file_is_re_uploaded(run_dir: Path):
    local = sync.scan_run_dir(run_dir)
    recorded = {path: [size, mtime, 0] for path, (size, mtime) in local.items()}

    target = run_dir / "data" / "samples" / "sample_0" / "L0" / "resid_post_12.npy"
    target.write_bytes(b"\x00" * 128)  # a mid-write file later completed by the extractor
    os.utime(target, (1_700_000_000, 1_700_000_000))

    assert sync.pending_files(sync.scan_run_dir(run_dir), recorded) == ["data/samples/sample_0/L0/resid_post_12.npy"]


def test_batches_respect_both_file_and_byte_limits():
    local = {f"f{i}.npy": (100, 0) for i in range(10)}
    paths = sorted(local)
    assert [len(b) for b in sync.batched(paths, local, max_files=4, max_bytes=10**9)] == [4, 4, 2]
    assert [len(b) for b in sync.batched(paths, local, max_files=1000, max_bytes=250)] == [2, 2, 2, 2, 2]


def test_batches_never_drop_a_file():
    local = {f"f{i}.npy": (i + 1, 0) for i in range(37)}
    paths = sorted(local)
    flat = [p for batch in sync.batched(paths, local, max_files=5, max_bytes=40) for p in batch]
    assert flat == paths


def test_one_oversized_file_still_gets_its_own_batch():
    local = {"big.npy": (10**9, 0), "small.npy": (10, 0)}
    batches = sync.batched(sorted(local), local, max_files=100, max_bytes=1024)
    assert batches == [["big.npy"], ["small.npy"]]


def test_manifest_round_trips(tmp_path: Path):
    path = tmp_path / sync.MANIFEST_NAME
    target = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x")
    assert sync.load_manifest(path) == {}

    log = sync.ManifestLog(path, target)
    log.append([("config.json", 12, 34, 56)])
    log.append([("summary.json", 78, 90, 12)])
    log.close()

    recorded = sync.load_manifest(path)[target.key]
    assert recorded == {"config.json": [12, 34, 56], "summary.json": [78, 90, 12]}


def test_a_truncated_manifest_tail_does_not_lose_earlier_entries(tmp_path: Path):
    """A box dying mid-append must not cost us the record of everything before it."""
    path = tmp_path / sync.MANIFEST_NAME
    target = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x")
    log = sync.ManifestLog(path, target)
    log.append([("config.json", 12, 34, 56)])
    log.close()
    with path.open("a") as handle:
        handle.write('["summary.json", 78, 9')

    assert sync.load_manifest(path)[target.key] == {"config.json": [12, 34, 56]}


def test_manifest_appends_rather_than_rewrites(tmp_path: Path):
    """The whole file must never be rewritten; a run holds millions of entries."""
    path = tmp_path / sync.MANIFEST_NAME
    log = sync.ManifestLog(path, sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x"))
    log.append([(f"f{i}.npy", i, i, 0) for i in range(100)])
    first_size = path.stat().st_size
    head = path.read_text()
    log.append([("later.npy", 1, 2, 3)])
    log.close()

    assert path.stat().st_size > first_size
    assert path.read_text().startswith(head)  # earlier bytes untouched


def test_two_prefixes_are_tracked_independently(tmp_path: Path):
    path = tmp_path / sync.MANIFEST_NAME
    a = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/a")
    b = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/b")
    log_a = sync.ManifestLog(path, a)
    log_a.append([("config.json", 1, 2, 3)])
    log_a.close()
    log_b = sync.ManifestLog(path, b)
    log_b.append([("summary.json", 4, 5, 6)])
    log_b.close()

    manifest = sync.load_manifest(path)
    assert manifest[a.key] == {"config.json": [1, 2, 3]}
    assert manifest[b.key] == {"summary.json": [4, 5, 6]}


def test_path_in_repo_joins_prefix():
    target = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x")
    assert target.path_in_repo("data/samples/sample_0/choice.json") == "geometry/x/data/samples/sample_0/choice.json"
    assert sync.SyncTarget(repo_id="ns/repo", prefix="").path_in_repo("config.json") == "config.json"


def test_dry_run_uploads_nothing_and_records_nothing(run_dir: Path, capsys):
    class ExplodingApi:
        def create_commit(self, *args, **kwargs):
            raise AssertionError("dry-run must not commit")

    manifest_path = run_dir / sync.MANIFEST_NAME
    stats = sync.sync_once(
        ExplodingApi(),
        sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x"),
        run_dir,
        manifest_path,
        dry_run=True,
        max_files=10,
        max_bytes=10**9,
        retries=1,
    )
    assert stats.files == 0
    assert len(stats.pending) == 6
    assert not manifest_path.exists()
    assert "WOULD UPLOAD" in capsys.readouterr().out


def test_dry_run_listing_is_capped(tmp_path: Path, capsys):
    """A real run holds millions of .npy files; the preview must not print them all."""
    root = tmp_path / "run"
    root.mkdir()
    for i in range(sync.DRY_RUN_PREVIEW + 20):
        (root / f"f{i:04d}.npy").write_bytes(b"x")

    sync.sync_once(
        object(),
        sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x"),
        root,
        root / sync.MANIFEST_NAME,
        dry_run=True,
        max_files=10,
        max_bytes=10**9,
        retries=1,
    )
    out = capsys.readouterr().out
    assert out.count("WOULD UPLOAD") == sync.DRY_RUN_PREVIEW
    assert "... and 20 more files" in out


def test_second_pass_uploads_nothing(run_dir: Path):
    """The whole point: re-running must not re-push the tree."""

    class RecordingApi:
        def __init__(self):
            self.committed: list[str] = []

        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            self.committed.extend(op.path_in_repo for op in operations)

    api = RecordingApi()
    target = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x")
    manifest_path = run_dir / sync.MANIFEST_NAME
    kwargs = dict(dry_run=False, max_files=2, max_bytes=10**9, retries=1)

    first = sync.sync_once(api, target, run_dir, manifest_path, **kwargs)
    assert first.files == 6
    assert first.commits == 3
    assert api.committed[0] == "geometry/x/config.json"

    second = sync.sync_once(api, target, run_dir, manifest_path, **kwargs)
    assert second.files == 0
    assert len(api.committed) == 6


def test_a_failed_commit_is_not_recorded(run_dir: Path):
    """A crash must never leave the manifest claiming an upload that did not land."""

    class FailingApi:
        def create_commit(self, **kwargs):
            raise RuntimeError("network down")

    manifest_path = run_dir / sync.MANIFEST_NAME
    with pytest.raises(RuntimeError):
        sync.sync_once(
            FailingApi(),
            sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x"),
            run_dir,
            manifest_path,
            dry_run=False,
            max_files=100,
            max_bytes=10**9,
            retries=1,
        )
    assert not manifest_path.exists()


def test_commit_retries_then_succeeds(run_dir: Path, monkeypatch):
    monkeypatch.setattr(sync.time, "sleep", lambda _: None)

    class FlakyApi:
        def __init__(self):
            self.attempts = 0

        def create_commit(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError("503")

    api = FlakyApi()
    sync.commit_batch(
        api,
        sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x"),
        run_dir,
        ["config.json"],
        message="test",
        retries=5,
    )
    assert api.attempts == 3


def test_manifest_partial_progress_is_kept_when_a_later_commit_fails(run_dir: Path):
    """Commit 1 lands, commit 2 dies: the first batch must stay recorded."""

    class HalfFailingApi:
        def __init__(self):
            self.calls = 0

        def create_commit(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("box died")

    manifest_path = run_dir / sync.MANIFEST_NAME
    target = sync.SyncTarget(repo_id="ns/repo", prefix="geometry/x")
    with pytest.raises(RuntimeError):
        sync.sync_once(
            HalfFailingApi(),
            target,
            run_dir,
            manifest_path,
            dry_run=False,
            max_files=2,
            max_bytes=10**9,
            retries=1,
        )
    assert set(sync.load_manifest(manifest_path)[target.key]) == {"config.json", "summary.json"}
