#!/usr/bin/env python
"""Verify experiment artifacts are complete on local disk.

This is the teardown gate. `cloud/reap.sh` will not destroy a box until this
exits 0, because a rented box's disk is the only copy of what it produced.

It opens artifacts and checks their content. It never treats "the path exists"
or "the job exited 0" as evidence.

    python scripts/verify_experiment_output.py --patching out/experiments/investment
    python scripts/verify_experiment_output.py --geometry out/geo/investment_geometry
    python scripts/verify_experiment_output.py --pulled          # every recorded pull
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PULL_MANIFEST = PROJECT_ROOT / "cloud" / ".pulled_runs"

VERIFIED, BROKEN, UNVERIFIED = "VERIFIED", "BROKEN", "UNVERIFIED"


@dataclass
class Check:
    name: str
    status: str
    evidence: str


@dataclass
class Report:
    target: str
    kind: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, evidence: str, soft: bool = False) -> None:
        status = VERIFIED if ok else (UNVERIFIED if soft else BROKEN)
        self.checks.append(Check(name, status, evidence))

    @property
    def ok(self) -> bool:
        return all(c.status != BROKEN for c in self.checks)

    def render(self) -> str:
        icon = {VERIFIED: "  ok  ", BROKEN: " BROKEN", UNVERIFIED: " UNVER"}
        head = f"[{self.kind}] {self.target}"
        body = "\n".join(
            f"  {icon[c.status]}  {c.name}: {c.evidence}" for c in self.checks
        )
        return f"{head}\n{body}"


def _load_json(path: Path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - reported as evidence
        return exc


def verify_patching(run_dir: Path) -> Report:
    """A coarse activation-patching run under out/experiments/<name>."""
    rep = Report(str(run_dir), "patching")

    if not run_dir.is_dir():
        rep.add("run directory", False, "does not exist")
        return rep

    cfg_path = run_dir / "working_config.json"
    cfg = _load_json(cfg_path) if cfg_path.exists() else None
    if isinstance(cfg, dict):
        rep.add("working_config.json", True, f"parsed, model={cfg.get('model')}")
    else:
        rep.add("working_config.json", False, f"missing or unparseable ({cfg})")

    pairs = sorted((run_dir / "pairs").glob("pair_*")) if (run_dir / "pairs").is_dir() else []
    rep.add("pairs/", bool(pairs), f"{len(pairs)} pair dir(s)")

    # Every pair must carry the contrastive record the pairing is rebuilt from.
    missing = [p.name for p in pairs if not (p / "contrastive_preference.json").exists()]
    if pairs:
        rep.add(
            "contrastive_preference.json per pair",
            not missing,
            "all present" if not missing else f"{len(missing)} missing e.g. {missing[:3]}",
        )

    agg = run_dir / "aggregated" / "coarse"
    comps = sorted(agg.glob("*.json")) if agg.is_dir() else []
    rep.add("aggregated/coarse/", bool(comps), f"{len(comps)} component file(s)")

    # The point of the sweep: which layers were actually patched.
    for comp in comps:
        data = _load_json(comp)
        if not isinstance(data, dict) or "by_sample" not in data:
            rep.add(f"{comp.name} layer coverage", False, "no by_sample block")
            continue
        by_sample = data["by_sample"]
        if not by_sample:
            rep.add(f"{comp.name} layer coverage", False, "by_sample is empty")
            continue
        first = next(iter(by_sample.values()))
        starts: set[int] = set()
        for step in (first.get("layer_results") or {}).values():
            starts.update(int(k) for k in (step.get("by_start") or {}))
        if starts:
            rep.add(
                f"{comp.name} layer coverage",
                True,
                f"{len(starts)} layers, {min(starts)}..{max(starts)} "
                f"over {len(by_sample)} sample(s)",
            )
        else:
            rep.add(f"{comp.name} layer coverage", False, "no swept layers recorded")

    return rep


def verify_geometry(run_dir: Path) -> Report:
    """A geometry extraction under out/geo/<name>."""
    rep = Report(str(run_dir), "geometry")

    if not run_dir.is_dir():
        rep.add("run directory", False, "does not exist")
        return rep

    summary = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else None
    if isinstance(summary, dict):
        n = summary.get("n_samples")
        rep.add(
            "summary.json",
            bool(n),
            f"n_samples={n}, layers={summary.get('layers')}, "
            f"components={summary.get('components')}",
        )
    else:
        rep.add("summary.json", False, f"missing or unparseable ({summary})")

    samples_dir = run_dir / "data" / "samples"
    samples = sorted(samples_dir.glob("sample_*")) if samples_dir.is_dir() else []

    declared = summary.get("n_samples") if isinstance(summary, dict) else None
    if declared and samples:
        # Analysis-only bundles legitimately ship zero .npy, so a sample-count
        # shortfall is the thing that matters, not raw-activation absence.
        rep.add(
            "sample count matches summary",
            len(samples) == declared,
            f"{len(samples)} on disk vs {declared} declared",
        )
    else:
        rep.add("data/samples/", bool(samples), f"{len(samples)} sample dir(s)")

    if samples:
        need = ["position_mapping.json", "prompt_sample.json", "choice.json"]
        bad = [
            s.name
            for s in samples[:200]
            if not all((s / f).exists() for f in need)
        ]
        rep.add(
            "per-sample JSON (first 200)",
            not bad,
            "complete" if not bad else f"{len(bad)} incomplete e.g. {bad[:3]}",
        )

    if samples and isinstance(summary, dict):
        _check_turn_positions(samples, summary, rep)

    analysis = run_dir / "analysis"
    pca = sorted((analysis / "pca").glob("*")) if (analysis / "pca").is_dir() else []
    # An extraction-only bundle carries raw activations and no analysis, which is
    # a legitimate shape: analysis runs later, off the box. Calling that BROKEN
    # would block teardown on every extraction run, so it is reported and not
    # failed. The mirror case, an analysis-only bundle with no .npy, is already
    # handled above.
    extraction_only = bool(samples) and any(
        next(s.rglob("*.npy"), None) is not None for s in samples[:5]
    )
    rep.add(
        "analysis/pca/",
        bool(pca),
        f"{len(pca)} target dir(s)"
        + (" (extraction-only bundle, analysis runs off the box)" if extraction_only else ""),
        soft=extraction_only,
    )

    return rep


# Tokens a chat template uses to end a turn and open the next one, per family.
# A turn window that decodes to none of these is not a turn window, whatever the
# mapping calls it.
TURN_TOKEN_MARKS = (
    "<|im_end|>", "<|im_start|>", "assistant",          # Qwen
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",  # Llama
    "<end_of_turn>", "<start_of_turn>", "model",        # Gemma
    "[/INST]",                                          # Mistral
)


def _check_turn_positions(samples: list[Path], summary: dict, rep: Report) -> None:
    """The turn window holds turn tokens, and every one of them has activations.

    A whole campaign of turn-transition figures was produced from activations
    that sat at response tokens while the mapping called them chat_suffix. The
    orderings had the same length, so every structural check passed. This reads
    the decoded token the mapping recorded at each turn position and refuses a
    run whose turn window contains no chat-control token at all.
    """
    turn_names = [
        p for p in (summary.get("positions") or [])
        if p.startswith("chat_suffix")
    ]
    if not turn_names:
        return

    checked = sorted(samples)[:50]
    windows: set[tuple[str, ...]] = set()
    missing_npy: list[str] = []
    layers = summary.get("layers") or []
    components = summary.get("components") or []

    for sample_dir in checked:
        mapping = _load_json(sample_dir / "position_mapping.json")
        if not isinstance(mapping, dict):
            rep.add("turn positions", False, f"{sample_dir.name}: unreadable mapping")
            return
        by_abs = {p["abs_pos"]: p for p in mapping.get("positions", [])}
        window: list[str] = []
        for name in turn_names:
            for abs_pos in mapping.get("named_positions", {}).get(name, []):
                info = by_abs.get(abs_pos)
                window.append(info["decoded_token"] if info else "<unmapped>")
                for layer in layers:
                    for comp in components:
                        npy = sample_dir / f"L{layer}" / f"{comp}_{abs_pos}.npy"
                        npz = npy.with_suffix(".npz")
                        if not npy.exists() and not npz.exists():
                            missing_npy.append(f"{sample_dir.name}/{npy.parent.name}/{npy.name}")
        windows.add(tuple(window))

    flat = {t for w in windows for t in w}
    marks = sorted(t for t in flat if t.strip() in TURN_TOKEN_MARKS)
    rep.add(
        "turn window holds turn tokens",
        bool(marks),
        f"{sorted(flat)} over {len(checked)} sample(s); recognised {marks}",
    )
    rep.add(
        "turn window identical across samples",
        len(windows) == 1,
        f"{len(windows)} distinct window(s)",
    )
    rep.add(
        "activations at every turn position",
        not missing_npy,
        "all present" if not missing_npy
        else f"{len(missing_npy)} missing e.g. {missing_npy[:3]}",
    )


def verify_pulled() -> list[Report]:
    """Verify every run recorded in cloud/.pulled_runs.

    An empty or absent manifest is NOT success: nothing has been proven pulled,
    so the caller must not be allowed to destroy anything.
    """
    if not PULL_MANIFEST.exists():
        rep = Report(str(PULL_MANIFEST), "pulled")
        rep.add("pull manifest", False, "does not exist; nothing proven captured")
        return [rep]

    entries = [
        line.strip().split("\t")
        for line in PULL_MANIFEST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not entries:
        rep = Report(str(PULL_MANIFEST), "pulled")
        rep.add("pull manifest", False, "empty; nothing proven captured")
        return [rep]

    reports = []
    for entry in entries:
        kind, path = (entry + ["", ""])[:2]
        target = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
        reports.append(
            verify_geometry(target) if kind == "geometry" else verify_patching(target)
        )
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patching", metavar="DIR", help="verify a patching run directory")
    ap.add_argument("--geometry", metavar="DIR", help="verify a geometry run directory")
    ap.add_argument("--pulled", action="store_true", help="verify every run in cloud/.pulled_runs")
    args = ap.parse_args()

    reports: list[Report] = []
    if args.patching:
        reports.append(verify_patching(Path(args.patching)))
    if args.geometry:
        reports.append(verify_geometry(Path(args.geometry)))
    if args.pulled:
        reports.extend(verify_pulled())

    if not reports:
        ap.error("nothing to verify: pass --patching, --geometry, or --pulled")

    for rep in reports:
        print(rep.render())
        print()

    broken = [r for r in reports if not r.ok]
    if broken:
        print(f"RESULT: BROKEN — {len(broken)} of {len(reports)} target(s) failed verification")
        return 1
    print(f"RESULT: VERIFIED — {len(reports)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
