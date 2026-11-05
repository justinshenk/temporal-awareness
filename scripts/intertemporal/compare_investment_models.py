#!/usr/bin/env python
"""Compare coarse-patching layer sweeps across all investment_* experiments.

Baseline: ``investment/`` (Qwen3-4B-Instruct-2507, the default model).
Compared variants: every sibling directory named ``investment_*``.

For each component (resid_post, attn_out, mlp_out), overlays per-layer mean
recovery and disruption across all models. Layer indices are plotted against
fractional depth (layer / n_layers) so different-sized models are comparable.

Outputs land in ``out/compare_intertemporal/``.

Usage:
    uv run python scripts/intertemporal/compare_investment_models.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.activation_patching.coarse.coarse_results import (
    CoarseActPatchAggregatedResults,
)


BASELINE_DIR_NAME = "investment"
COMPONENTS = ["resid_post", "attn_out", "mlp_out"]


@dataclass
class ModelRun:
    dirname: str
    label: str              # display name
    model: str              # HF model id from working_config.json
    n_layers: int           # number of layers (model depth)
    size_b: float           # parameter count in billions (parsed from model id)
    color: str
    is_baseline: bool = False


BASELINE_COLOR = "#FF1493"


def _size_b_from_model(model_id: str) -> float:
    """Extract parameter count (B) from a HF model id like 'Qwen/Qwen3-8B' → 8.0.

    Falls back to 0 when no size token is found.
    """
    import re
    # Search last size token (handles Qwen3.5-0.8B, Qwen3-14B, Qwen3-4B-Instruct, ...)
    m = re.findall(r"(\d+(?:\.\d+)?)[Bb](?![a-zA-Z0-9])", model_id)
    return float(m[-1]) if m else 0.0


def _short_label(dirname: str, model: str, size_b: float) -> str:
    """Human-readable label, e.g. 'Qwen3-4B-I (4.0B)'."""
    if dirname == BASELINE_DIR_NAME:
        return f"★ Qwen3-4B-Inst ({size_b:.1f}B)"
    stem = dirname.removeprefix("investment_").replace("_", "-")
    return f"{stem}  ({size_b:.1f}B)"


def _n_layers_from_dir(exp_dir: Path) -> int | None:
    """Best-effort layer count from aggregated coarse data (max seen layer + 1)."""
    for component in COMPONENTS:
        path = exp_dir / "aggregated" / "coarse" / f"{component}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        by_sample = data.get("by_sample", {})
        layers: set[int] = set()
        for sample in by_sample.values():
            for step_key, sweep in (sample.get("layer_results", {}) or {}).items():
                for layer_key in (sweep.get("by_start", {}) or {}).keys():
                    layers.add(int(layer_key))
        if layers:
            return max(layers) + 1
    return None


def _discover_runs(exp_root: Path) -> list[ModelRun]:
    """Find all investment / investment_* experiment folders and build ModelRun list."""
    candidates = sorted(
        d for d in exp_root.iterdir()
        if d.is_dir() and (d.name == BASELINE_DIR_NAME or d.name.startswith("investment_"))
    )
    runs: list[ModelRun] = []
    for d in candidates:
        if d.name.endswith("_backup") or "_backup_" in d.name:
            continue
        cfg_path = d / "working_config.json"
        if not cfg_path.exists():
            print(f"[skip] {d.name}: no working_config.json")
            continue
        model = json.loads(cfg_path.read_text()).get("model", "?")
        n_layers = _n_layers_from_dir(d)
        if n_layers is None:
            print(f"[skip] {d.name}: no aggregated coarse data yet")
            continue
        size_b = _size_b_from_model(model)
        runs.append(
            ModelRun(
                dirname=d.name,
                label=_short_label(d.name, model, size_b),
                model=model,
                n_layers=n_layers,
                size_b=size_b,
                color="",  # filled in below
                is_baseline=(d.name == BASELINE_DIR_NAME),
            )
        )

    # Sort strictly by parameter count (smallest → largest); this is also
    # the order used in every legend so the reader can scan bottom-up by size.
    runs.sort(key=lambda r: (r.size_b, r.dirname))

    # Colors: use a perceptually-uniform colormap (viridis) mapped to
    # log(size). Similar-size models get similar colors; very different
    # sizes get very different colors. Baseline keeps its fixed magenta
    # highlight on top of this.
    sizes = np.array([max(r.size_b, 0.1) for r in runs], dtype=float)
    log_sizes = np.log(sizes)
    lo, hi = log_sizes.min(), log_sizes.max()
    norm = (log_sizes - lo) / (hi - lo) if hi > lo else np.zeros_like(log_sizes)
    cmap = plt.get_cmap("viridis")
    for r, t in zip(runs, norm):
        r.color = BASELINE_COLOR if r.is_baseline else cmap(float(t))
    return runs


def _load_mean_curve(
    exp_dir: Path, component: str, metric: str
) -> tuple[list[int], list[float], int]:
    """Return (layers, values, n_pairs) for the chosen metric ('recovery' or 'disruption')."""
    path = exp_dir / "aggregated" / "coarse" / f"{component}.json"
    if not path.exists():
        return [], [], 0
    agg = CoarseActPatchAggregatedResults.from_json(path)
    n_pairs = agg.n_samples
    if n_pairs == 0:
        return [], [], 0
    sweep = agg.get_mean_layer_results()
    layers = sorted(sweep.keys())
    vals: list[float] = []
    for L in layers:
        tr = sweep.get(L)
        if tr is None:
            vals.append(0.0)
            continue
        v = tr.recovery if metric == "recovery" else tr.disruption
        vals.append(float(v) if v is not None else 0.0)
    return layers, vals, n_pairs


def _plot_panel(
    ax,
    runs: list[ModelRun],
    component: str,
    metric: str,
    fractional_depth: bool,
    exp_root: Path,
) -> None:
    for r in runs:
        layers, vals, n = _load_mean_curve(exp_root / r.dirname, component, metric)
        if not layers:
            continue
        if fractional_depth and r.n_layers > 1:
            xs = [L / (r.n_layers - 1) for L in layers]
        else:
            xs = layers
        lw = 3.0 if r.is_baseline else 1.6
        alpha = 1.0 if r.is_baseline else 0.75
        label = f"{r.label}  (n={n})"
        ax.plot(xs, vals, "-", color=r.color, linewidth=lw, alpha=alpha, label=label,
                marker="o" if r.is_baseline else None,
                markersize=4 if r.is_baseline else 0,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=10 if r.is_baseline else 3)

    ax.axhline(0, color="#AAAAAA", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlabel("Fractional layer depth" if fractional_depth else "Layer",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Mean {metric}", fontsize=11, fontweight="bold")
    ax.set_title(f"{component} — {metric}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")


def _plot_overview(runs: list[ModelRun], exp_root: Path, out_dir: Path) -> None:
    """Single figure: 3 components x 2 metrics = 6 panels."""
    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(14, 4.5 * len(COMPONENTS)))
    for i, component in enumerate(COMPONENTS):
        for j, metric in enumerate(["recovery", "disruption"]):
            _plot_panel(axes[i, j], runs, component, metric, fractional_depth=True,
                        exp_root=exp_root)
    # Single shared legend on the right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=False, title="Models")
    fig.suptitle(
        "Coarse Patching Across Investment Models\n"
        f"Baseline: {BASELINE_DIR_NAME} ({'★' if runs and runs[0].is_baseline else ''})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    path = out_dir / "overview_all_components.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _plot_per_component(
    runs: list[ModelRun], exp_root: Path, out_dir: Path
) -> None:
    """One figure per component, with recovery and disruption side-by-side."""
    for component in COMPONENTS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for j, metric in enumerate(["recovery", "disruption"]):
            _plot_panel(axes[j], runs, component, metric, fractional_depth=True,
                        exp_root=exp_root)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=9, frameon=False, title="Models")
        fig.suptitle(f"{component}: layer sweep across models",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.85, 0.94])
        path = out_dir / f"compare_{component}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def _plot_absolute_depth(
    runs: list[ModelRun], exp_root: Path, out_dir: Path
) -> None:
    """Overview using absolute layer index (not fractional) for comparison."""
    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(14, 4.5 * len(COMPONENTS)))
    for i, component in enumerate(COMPONENTS):
        for j, metric in enumerate(["recovery", "disruption"]):
            _plot_panel(axes[i, j], runs, component, metric, fractional_depth=False,
                        exp_root=exp_root)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=False, title="Models")
    fig.suptitle("Coarse Patching (absolute layer index)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    path = out_dir / "overview_absolute_depth.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main() -> int:
    exp_root = Path("out/experiments")
    if not exp_root.exists():
        print(f"Error: {exp_root} not found — run from the project root.")
        return 1

    runs = _discover_runs(exp_root)
    if not runs:
        print("No investment experiments found.")
        return 1
    print(f"Found {len(runs)} investment runs:")
    for r in runs:
        print(f"  {r.dirname:40s} model={r.model:40s} n_layers={r.n_layers}"
              f"{'  [baseline]' if r.is_baseline else ''}")

    out_dir = Path("out/compare_intertemporal")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    _plot_overview(runs, exp_root, out_dir)
    _plot_per_component(runs, exp_root, out_dir)
    _plot_absolute_depth(runs, exp_root, out_dir)

    # Emit a small summary of what each model recovered at its peak layer.
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Peak mean recovery per model, per component\n")
        f.write("=" * 70 + "\n")
        for component in COMPONENTS:
            f.write(f"\n-- {component} --\n")
            for r in runs:
                layers, vals, n = _load_mean_curve(exp_root / r.dirname, component, "recovery")
                if not layers:
                    continue
                peak_idx = int(np.argmax(vals))
                f.write(f"  {r.label:30s}  layer={layers[peak_idx]:>3d}  "
                        f"(depth={layers[peak_idx] / max(r.n_layers - 1, 1):.2f})  "
                        f"recovery={vals[peak_idx]:.3f}  n={n}\n")
    print(f"  saved {summary_path}")

    print(f"\nDone. Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
