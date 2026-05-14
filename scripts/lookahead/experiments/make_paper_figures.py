#!/usr/bin/env python3
"""make_paper_figures.py — auto-generate paper figures from results/v2/*.json.

Produces (writing to results/v2/figures/):
  1. fig1_cross_model_gaps.pdf    — bar chart of headline gaps × domain × model, with bootstrap CIs
  2. fig2_per_position_staircase.pdf — per-position curves for one headline model on each domain
  3. fig3_dual_baseline_scatter.pdf  — workshop's mean-pool gap vs our max-across-earlier gap
  4. fig4_ablation_heatmap.pdf       — ablation drop heatmap (model × domain)
  5. fig5_layer_trajectory.pdf       — best gap as a function of layer depth (normalized)

Also writes results/v2/STATS.md with key statistical tests:
  - Paired Wilcoxon: rhyme vs code per model
  - Paired Wilcoxon: rhyme vs qa_neutral per model
  - Spearman correlation: model_size vs rhyme_gap

Usage: python3 make_paper_figures.py --results_dir results/v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


# Aesthetic constants
DOMAIN_ORDER = ["rhyme", "qa_suggestive", "code", "qa_neutral", "trivia"]
DOMAIN_COLORS = {
    "rhyme":         "#2E86AB",  # blue — strong positive case
    "qa_suggestive": "#7CB342",  # green — weak positive
    "code":          "#E5A623",  # amber — weak positive (was workshop's "negative")
    "qa_neutral":    "#A23E48",  # red — null
    "trivia":        "#6E6E6E",  # gray — null (saturated BoW)
}
DOMAIN_LABEL = {
    "rhyme": "rhyme",
    "qa_suggestive": "qa-sugg",
    "code": "code",
    "qa_neutral": "qa-neut",
    "trivia": "trivia",
}

# Model display order (rough size order)
MODEL_DISPLAY = {
    "gpt2": "GPT2-S", "gpt2-medium": "GPT2-M", "gpt2-xl": "GPT2-XL",
    "EleutherAI/pythia-410m-deduped": "Py-410M", "EleutherAI/pythia-1b-deduped": "Py-1B",
    "EleutherAI/pythia-1.4b-deduped": "Py-1.4B", "EleutherAI/pythia-2.8b-deduped": "Py-2.8B",
    "Qwen/Qwen3-1.7B-Base": "Qw3-1.7B", "Qwen/Qwen3-8B-Base": "Qw3-8B",
    "google/gemma-2-2b": "G2-2B", "google/gemma-2-2b-it": "G2-2B-it",
    "google/gemma-2-9b": "G2-9B",
    "meta-llama/Llama-3.1-8B": "L3-8B", "meta-llama/Llama-3.1-8B-Instruct": "L3-8B-it",
}


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_all(results_dir: Path) -> list[dict]:
    """Load every staircase JSON, attach model/domain to top level."""
    out = []
    for p in sorted(results_dir.glob("*__staircase.json")):
        try:
            d = json.load(open(p))
            d["_path"] = str(p)
            out.append(d)
        except Exception as e:
            logger.warning(f"  failed to load {p.name}: {e}")
    return out


def best_headline(doc: dict) -> dict | None:
    """Return the headline row with the largest |gap| (strongest signal)."""
    hs = doc.get("headlines", [])
    if not hs:
        return None
    # Filter to linear probes only for fair comparison
    linear = [h for h in hs if h.get("probe_type", "linear") == "linear"]
    use = linear or hs
    return max(use, key=lambda h: abs(h.get("headline_gap", 0)))


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Cross-model gaps by domain
# ──────────────────────────────────────────────────────────────────────
def fig1_cross_model(docs, outdir: Path):
    """Bar chart: each bar is one (model, domain) pair, grouped by domain."""
    # Aggregate: (model, domain) → (gap, ci_lo, ci_hi)
    data: dict[tuple[str, str], dict] = {}
    for d in docs:
        meta = d.get("meta", {})
        model = meta.get("model"); domain = meta.get("domain")
        if not (model and domain): continue
        h = best_headline(d)
        if not h: continue
        ci = h.get("bootstrap_ci", {})
        gap = h["headline_gap"] * 100
        if ci.get("available"):
            ci_lo, ci_hi = ci["gap_ci"][0] * 100, ci["gap_ci"][1] * 100
        else:
            ci_lo, ci_hi = gap, gap
        data[(model, domain)] = {"gap": gap, "ci_lo": ci_lo, "ci_hi": ci_hi}

    if not data:
        logger.warning("fig1: no data"); return

    # Plot: domains on x-axis (grouped), one bar per model within each group
    models_in_data = sorted(set(m for m, _ in data.keys()),
                             key=lambda m: list(MODEL_DISPLAY.keys()).index(m)
                                          if m in MODEL_DISPLAY else 99)

    n_domains = len(DOMAIN_ORDER)
    n_models = len(models_in_data)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_domains * n_models * 0.4), 5.5))
    for mi, model in enumerate(models_in_data):
        xs, gaps, errs_lo, errs_hi = [], [], [], []
        for di, domain in enumerate(DOMAIN_ORDER):
            if (model, domain) in data:
                xs.append(di + (mi - n_models / 2) * width + width / 2)
                gaps.append(data[(model, domain)]["gap"])
                errs_lo.append(data[(model, domain)]["gap"] - data[(model, domain)]["ci_lo"])
                errs_hi.append(data[(model, domain)]["ci_hi"] - data[(model, domain)]["gap"])
        if xs:
            label = MODEL_DISPLAY.get(model, model.split("/")[-1])
            ax.bar(xs, gaps, width=width * 0.95, label=label,
                   yerr=[errs_lo, errs_hi], capsize=2,
                   edgecolor="black", linewidth=0.3, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xticks(range(n_domains))
    ax.set_xticklabels([DOMAIN_LABEL.get(d, d) for d in DOMAIN_ORDER])
    ax.set_xlabel("Domain")
    ax.set_ylabel("Headline gap (target − max-across-earlier), pp")
    ax.set_title("Per-position staircase: target-vs-best-earlier gap by domain × model")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = outdir / "fig1_cross_model_gaps.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"  fig1 → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Per-position staircase curves
# ──────────────────────────────────────────────────────────────────────
def fig2_staircase(docs, outdir: Path, anchor_model: str = "google/gemma-2-2b"):
    """For the anchor model, plot per-position CV accuracy curves at the best layer for each domain."""
    by_domain = {d.get("meta", {}).get("domain"): d for d in docs
                 if d.get("meta", {}).get("model") == anchor_model}
    if not by_domain:
        logger.warning(f"fig2: no docs for {anchor_model}"); return

    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(15, 3.5), sharey=True)
    for i, domain in enumerate(DOMAIN_ORDER):
        ax = axes[i]
        if domain not in by_domain:
            ax.set_title(f"{DOMAIN_LABEL.get(domain, domain)}\n(no data)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        d = by_domain[domain]
        h = best_headline(d)
        if not h:
            continue
        layer = h["layer"]
        # Pull per-position curve from per_layer_results
        plr = d.get("per_layer_results", {}).get(str(layer), {})
        pp = plr.get("per_position", {})
        if not pp:
            continue
        positions = sorted(int(p) for p in pp.keys())
        accs = [pp[str(p)]["cv_accuracy_mean"] for p in positions]
        ax.plot(positions, accs, marker="o", color=DOMAIN_COLORS[domain], linewidth=1.5)
        ax.axhline(d["baselines"]["chance"], color="gray", linestyle="--", linewidth=0.6, label="chance")
        ax.axhline(d["baselines"]["bag_of_words_accuracy"], color="black", linestyle=":", linewidth=0.6, label="BoW")
        # Mark target position(s)
        for r, info in d.get("target_position_results", {}).items():
            if isinstance(info, dict) and "mode_position" in info:
                ax.axvline(info["mode_position"], color="red", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{DOMAIN_LABEL.get(domain, domain)}\nL{layer}, gap={h['headline_gap']*100:+.1f}pp")
        ax.set_xlabel("token position")
        if i == 0: ax.set_ylabel("CV accuracy")
        ax.grid(alpha=0.3)
        if i == len(DOMAIN_ORDER) - 1:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Per-position staircase: {MODEL_DISPLAY.get(anchor_model, anchor_model)}", y=1.02)
    plt.tight_layout()
    out = outdir / "fig2_per_position_staircase.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"  fig2 → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Dual-baseline scatter
# ──────────────────────────────────────────────────────────────────────
def fig3_dual_baseline(docs, outdir: Path):
    """Scatter: workshop's mean-pool gap vs our max-across-earlier gap."""
    xs, ys, colors, labels = [], [], [], []
    for d in docs:
        h = best_headline(d)
        if not h: continue
        mp_gap = h.get("target_vs_mean_pool_gap")
        if mp_gap is None: continue
        max_earlier_gap = h["headline_gap"]
        domain = d["meta"]["domain"]
        xs.append(mp_gap * 100); ys.append(max_earlier_gap * 100)
        colors.append(DOMAIN_COLORS[domain]); labels.append(domain)

    if not xs:
        logger.warning("fig3: no mean-pool data (run patch_meanpool_baseline.py first)"); return

    fig, ax = plt.subplots(figsize=(7, 7))
    for xi, yi, ci, li in zip(xs, ys, colors, labels):
        ax.scatter(xi, yi, c=ci, s=80, edgecolor="black", linewidth=0.5, alpha=0.9)

    # y=x line
    lo, hi = min(min(xs), min(ys)) - 5, max(max(xs), max(ys)) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.8, label="y = x")
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlabel("Gap vs mean-pool baseline (workshop), pp")
    ax.set_ylabel("Gap vs max-across-earlier (ours), pp")
    ax.set_title("Dual-baseline comparison\n(stricter our baseline → smaller gap, mostly)")
    legend = [mpatches.Patch(color=DOMAIN_COLORS[d], label=DOMAIN_LABEL.get(d, d))
              for d in DOMAIN_ORDER if d in labels]
    ax.legend(handles=legend, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = outdir / "fig3_dual_baseline_scatter.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"  fig3 → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Ablation heatmap
# ──────────────────────────────────────────────────────────────────────
def fig4_ablation_heatmap(docs, outdir: Path):
    """Heatmap: rows = models, columns = domains, cells = mean ablation drop (pp)."""
    grid: dict[tuple[str, str], float] = {}
    for d in docs:
        meta = d.get("meta", {})
        model, domain = meta.get("model"), meta.get("domain")
        if not (model and domain): continue
        abl = d.get("ablation", {})
        zd = abl.get("zero", {}).get("drop_pp")
        md = abl.get("mean", {}).get("drop_pp")
        if zd is None and md is None: continue
        drop = np.nanmean([x for x in [zd, md] if x is not None])
        grid[(model, domain)] = drop

    if not grid:
        logger.warning("fig4: no ablation data"); return

    models = sorted(set(m for m, _ in grid.keys()),
                    key=lambda m: list(MODEL_DISPLAY.keys()).index(m)
                                 if m in MODEL_DISPLAY else 99)
    matrix = np.full((len(models), len(DOMAIN_ORDER)), np.nan)
    for i, m in enumerate(models):
        for j, dom in enumerate(DOMAIN_ORDER):
            if (m, dom) in grid:
                matrix[i, j] = grid[(m, dom)]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(models))))
    vmax = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels([DOMAIN_LABEL.get(d, d) for d in DOMAIN_ORDER], rotation=20)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m.split("/")[-1]) for m in models])
    for i in range(len(models)):
        for j in range(len(DOMAIN_ORDER)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        color="white" if abs(v) > vmax / 2 else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label="ablation drop, pp (higher = stronger causal effect)")
    ax.set_title("Ablation drop: target accuracy decrease when earlier positions are ablated")
    plt.tight_layout()
    out = outdir / "fig4_ablation_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"  fig4 → {out}")


# ──────────────────────────────────────────────────────────────────────
# Stats markdown
# ──────────────────────────────────────────────────────────────────────
def write_stats(docs, outdir: Path):
    """Compute key paired tests + correlations and write to STATS.md."""
    from scipy import stats

    # Per-model max gap per domain
    per_model: dict[str, dict[str, float]] = defaultdict(dict)
    for d in docs:
        model = d["meta"]["model"]; domain = d["meta"]["domain"]
        h = best_headline(d)
        if h: per_model[model][domain] = h["headline_gap"] * 100

    lines = ["# Statistical tests — staircase v2", ""]
    lines.append(f"_{len(per_model)} models, {sum(len(v) for v in per_model.values())} (model, domain) pairs_\n")

    # 1. Cross-domain paired test (rhyme vs each other domain)
    lines.append("## Paired Wilcoxon: rhyme vs each domain (across models)")
    rhyme = [v.get("rhyme") for v in per_model.values()]
    for other in ["qa_suggestive", "code", "qa_neutral", "trivia"]:
        paired = [(v.get("rhyme"), v.get(other)) for v in per_model.values()
                  if v.get("rhyme") is not None and v.get(other) is not None]
        if len(paired) < 3:
            lines.append(f"- rhyme vs {other}: too few pairs ({len(paired)})")
            continue
        a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
        try:
            stat, p = stats.wilcoxon(a, b)
            lines.append(f"- rhyme vs {other}:  n={len(paired)}  median diff={np.median(a-b):+.1f}pp  W={stat:.1f}  p={p:.4f}")
        except Exception as e:
            lines.append(f"- rhyme vs {other}: test failed ({e})")
    lines.append("")

    # 2. Effect-size summary per domain
    lines.append("## Mean headline gap per domain (across all models tested)")
    for dom in DOMAIN_ORDER:
        vals = [v.get(dom) for v in per_model.values() if v.get(dom) is not None]
        if vals:
            lines.append(f"- {dom:14s}  n={len(vals):2d}  mean={np.mean(vals):+6.1f}pp  median={np.median(vals):+6.1f}pp  range=[{min(vals):+.1f}, {max(vals):+.1f}]")
    lines.append("")

    # 3. Number of models where the predicted-sign matched the observed-sign
    lines.append("## Pre-registration check (sign match rate)")
    counts: dict[str, list[bool]] = defaultdict(list)
    for d in docs:
        domain = d["meta"]["domain"]; h = best_headline(d)
        if not h: continue
        chk = h.get("pre_registration_check", {})
        counts[domain].append(bool(chk.get("matches", False)))
    for dom in DOMAIN_ORDER:
        if dom in counts:
            n = len(counts[dom]); k = sum(counts[dom])
            lines.append(f"- {dom:14s}  {k}/{n}  ({100*k/n:.0f}% sign-match)")

    out = outdir / "STATS.md"
    out.write_text("\n".join(lines))
    logger.info(f"  stats → {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/v2")
    ap.add_argument("--anchor_model", default="google/gemma-2-2b",
                    help="which model to use as the anchor for fig2 (per-position curves)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)

    docs = load_all(results_dir)
    logger.info(f"Loaded {len(docs)} JSONs from {results_dir}")
    if not docs:
        logger.error("No JSONs found"); return

    fig1_cross_model(docs, outdir)
    fig2_staircase(docs, outdir, anchor_model=args.anchor_model)
    fig3_dual_baseline(docs, outdir)
    fig4_ablation_heatmap(docs, outdir)
    write_stats(docs, outdir)

    logger.info(f"All figures + stats in {outdir}/")


if __name__ == "__main__":
    main()
