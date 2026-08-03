#!/usr/bin/env python
"""Layer profile of a coarse activation-patching run.

One number per layer and component: the mean over contrastive pairs of
denoising recovery plus noising disruption. Patching a layer that carries the
choice both restores it from the corrupted run and destroys it in the clean
run, so the two effects add and the sum peaks where the decision lives.

It also reports the sanity block every pair must pass (full-patch recovery and
disruption of 1.0) and the clean-baseline logit difference behind each pair, so
a sweep can be gated before its numbers are read.

    python scripts/intertemporal/summarize_coarse_localization.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BAND_FRACTION = 0.5   # the band spans every layer reaching half the peak
EARLY_DEPTH = 0.2     # "early layers" are everything below this fractional depth


def layer_scores(agg_path: Path) -> dict[int, list[float]]:
    """Per-layer recovery+disruption, one entry per contrastive pair."""
    data = json.load(open(agg_path))
    scores: dict[int, list[float]] = {}
    for sample in data.get("by_sample", {}).values():
        for step in (sample.get("layer_results") or {}).values():
            for layer, cell in (step.get("by_start") or {}).items():
                recovery = (cell.get("denoising") or {}).get("recovery")
                disruption = (cell.get("noising") or {}).get("disruption")
                if recovery is None or disruption is None:
                    continue
                scores.setdefault(int(layer), []).append(recovery + disruption)
    return scores


def sanity_rows(run_dir: Path) -> list[dict]:
    """The per-pair sanity block and clean baseline logit difference."""
    rows = []
    for path in sorted((run_dir / "pairs").glob("pair_*/coarse/*/coarse_results.json")):
        result = json.load(open(path))
        sanity = result.get("sanity_result") or {}
        denoising = sanity.get("denoising") or {}
        noising = sanity.get("noising") or {}
        clean = denoising.get("baseline_clean_logprobs") or []
        rows.append(
            {
                "pair": path.parts[-4],
                "sweep": path.parts[-2],
                "recovery": denoising.get("recovery"),
                "disruption": noising.get("disruption"),
                "baseline_logit_diff": (clean[0] - clean[1]) if len(clean) == 2 else None,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.load(open(run_dir / "working_config.json"))
    agg_dir = run_dir / "aggregated" / "coarse"

    rows = sanity_rows(run_dir)
    clean_ok = [r for r in rows if r["recovery"] == 1.0 and r["disruption"] == 1.0]
    diffs = [r["baseline_logit_diff"] for r in rows if r["baseline_logit_diff"] is not None]
    print(f"model {config.get('model')}  dataset {config.get('dataset_config', {}).get('name')}")
    print(f"sanity: {len(clean_ok)}/{len(rows)} sweeps with recovery=1.0 and disruption=1.0")
    # Reported per sweep because patching every MLP is not a complete
    # intervention, so mlp_out legitimately misses the 1.0/1.0 block.
    by_sweep: dict[str, list[int]] = {}
    for row in rows:
        ok = row["recovery"] == 1.0 and row["disruption"] == 1.0
        counts = by_sweep.setdefault(row["sweep"], [0, 0])
        counts[0] += int(ok)
        counts[1] += 1
    for sweep, (ok, total) in sorted(by_sweep.items()):
        print(f"  {sweep}: {ok}/{total} clean")
    if diffs:
        print(
            f"clean baseline logit diff: min {min(diffs):.2f} max {max(diffs):.2f} "
            f"mean {sum(diffs) / len(diffs):.2f} over {len(diffs)} sweeps"
        )

    report = {
        "run_dir": str(run_dir),
        "model": config.get("model"),
        "n_sweeps": len(rows),
        "n_sanity_clean": len(clean_ok),
        "baseline_logit_diff": {
            "min": min(diffs) if diffs else None,
            "max": max(diffs) if diffs else None,
            "mean": sum(diffs) / len(diffs) if diffs else None,
        },
        "components": {},
    }

    for comp_path in sorted(agg_dir.glob("*.json")):
        comp = comp_path.stem
        scores = layer_scores(comp_path)
        if not scores:
            continue
        n_layers = max(scores) + 1
        means = {layer: sum(v) / len(v) for layer, v in sorted(scores.items())}
        peak_layer = max(means, key=lambda layer: means[layer])
        peak = means[peak_layer]
        threshold = peak * BAND_FRACTION
        band = sorted(layer for layer, value in means.items() if value >= threshold)
        early = [means[layer] for layer in means if layer / n_layers < EARLY_DEPTH]
        top5 = sorted(means, key=lambda layer: means[layer], reverse=True)[:5]

        report["components"][comp] = {
            "n_layers_swept": len(means),
            "n_pairs": len(next(iter(scores.values()))),
            "peak_layer": peak_layer,
            "peak_depth": peak_layer / n_layers,
            "peak_score": peak,
            "band_layers": [min(band), max(band)],
            "band_depth": [min(band) / n_layers, max(band) / n_layers],
            "early_mean": sum(early) / len(early) if early else None,
            "top5": [[layer, means[layer]] for layer in top5],
            "per_layer": {str(layer): value for layer, value in means.items()},
        }
        print(
            f"\n{comp}: {len(means)} layers, peak L{peak_layer} "
            f"({peak_layer / n_layers:.2f}) {peak:+.3f}, band L{min(band)}-L{max(band)} "
            f"({min(band) / n_layers:.2f}-{max(band) / n_layers:.2f}), "
            f"early<{EARLY_DEPTH} mean {report['components'][comp]['early_mean']:+.3f}"
        )
        print("  top5: " + ", ".join(f"L{layer} {means[layer]:+.3f}" for layer in top5))

    out_path = args.out or run_dir / "coarse_layer_profile.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
