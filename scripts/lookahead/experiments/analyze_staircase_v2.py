#!/usr/bin/env python3
"""analyze_staircase_v2.py — aggregate per-job JSONs into paper tables.

After `launch_partition.py` finishes writing one JSON per (model, domain),
this script collects them, picks the strongest headline per pair, and
produces:

  * results/v2/MASTER_TABLE.csv      — one row per (model, domain)
  * results/v2/DOMAIN_SUMMARY.csv    — mean gap and pre-registration
                                       match rate per domain
  * results/v2/SUPPLEMENTARY.csv     — full (model, domain, layer,
                                       resolver, probe_type) breakdown
  * results/v2/SUMMARY.md            — short markdown digest with the
                                       pre-reg matrix vs observed signs

The script is read-only on the JSON dir and idempotent — safe to re-run
after partial runs.

USAGE
-----
    python scripts/lookahead/experiments/analyze_staircase_v2.py \\
        --results_dir results/v2 \\
        --output_dir results/v2
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# JSON ingestion
# ──────────────────────────────────────────────────────────────────────

def collect_jsons(results_dir: Path) -> list[dict]:
    """Load every *__staircase.json under results_dir."""
    out = []
    for p in sorted(results_dir.glob("*__staircase.json")):
        try:
            with open(p) as f:
                doc = json.load(f)
            doc["_source_path"] = str(p)
            out.append(doc)
        except Exception as e:
            print(f"WARN: failed to load {p}: {e}")
    return out


def pick_strongest_headline(headlines: list[dict]) -> dict | None:
    """Return the headline row with the largest |gap| (most informative)."""
    if not headlines:
        return None
    return max(headlines, key=lambda r: abs(r.get("headline_gap", 0.0)))


# ──────────────────────────────────────────────────────────────────────
# Master table — one row per (model, domain)
# ──────────────────────────────────────────────────────────────────────

MASTER_COLS = [
    "model", "domain", "predicted_gap", "n_examples", "n_classes",
    "chance", "bag_of_words_acc",
    "best_layer", "best_resolver", "best_probe_type",
    "target_acc", "max_earlier_acc", "headline_gap_pp",
    "bootstrap_gap_pp", "bootstrap_ci_lo_pp", "bootstrap_ci_hi_pp",
    "p_gap_positive",
    "observed_sign", "pre_reg_matches",
    "ablation_zero_drop_pp", "ablation_mean_drop_pp",
    "elapsed_seconds",
]


def build_master_row(doc: dict) -> dict | None:
    meta = doc.get("meta", {})
    headlines = doc.get("headlines", [])
    h = pick_strongest_headline(headlines)
    if h is None:
        return None

    base = doc.get("baselines", {})
    ci = h.get("bootstrap_ci", {})
    has_ci = isinstance(ci, dict) and ci.get("available")
    prereg = h.get("pre_registration_check", {})
    abl = doc.get("ablation", {}) or {}

    def pp(v):
        return None if v is None else float(v) * 100.0

    return {
        "model": meta.get("model"),
        "domain": meta.get("domain"),
        "predicted_gap": meta.get("predicted_gap"),
        "n_examples": meta.get("n_examples"),
        "n_classes": meta.get("n_classes"),
        "chance": base.get("chance"),
        "bag_of_words_acc": base.get("bag_of_words_accuracy"),
        "best_layer": h.get("layer"),
        "best_resolver": h.get("resolver"),
        "best_probe_type": h.get("probe_type"),
        "target_acc": h.get("target_accuracy"),
        "max_earlier_acc": h.get("max_earlier_accuracy"),
        "headline_gap_pp": pp(h.get("headline_gap")),
        "bootstrap_gap_pp": pp(ci.get("gap_mean")) if has_ci else None,
        "bootstrap_ci_lo_pp": pp(ci.get("gap_ci", [None, None])[0]) if has_ci else None,
        "bootstrap_ci_hi_pp": pp(ci.get("gap_ci", [None, None])[1]) if has_ci else None,
        "p_gap_positive": ci.get("p_gap_positive") if has_ci else None,
        "observed_sign": prereg.get("observed_sign"),
        "pre_reg_matches": prereg.get("matches"),
        "ablation_zero_drop_pp": (abl.get("zero") or {}).get("drop_pp"),
        "ablation_mean_drop_pp": (abl.get("mean") or {}).get("drop_pp"),
        "elapsed_seconds": meta.get("total_seconds"),
    }


# ──────────────────────────────────────────────────────────────────────
# Domain summary
# ──────────────────────────────────────────────────────────────────────

def build_domain_summary(rows: list[dict]) -> list[dict]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("domain"):
            by_domain[r["domain"]].append(r)

    out = []
    for domain, items in sorted(by_domain.items()):
        valid_gaps = [it["headline_gap_pp"] for it in items
                      if isinstance(it.get("headline_gap_pp"), (int, float))]
        valid_match = [it["pre_reg_matches"] for it in items
                       if isinstance(it.get("pre_reg_matches"), bool)]
        out.append({
            "domain": domain,
            "predicted_gap": items[0].get("predicted_gap") if items else None,
            "n_models": len(items),
            "mean_gap_pp": (sum(valid_gaps) / len(valid_gaps)) if valid_gaps else None,
            "min_gap_pp": min(valid_gaps) if valid_gaps else None,
            "max_gap_pp": max(valid_gaps) if valid_gaps else None,
            "n_pre_reg_match": sum(valid_match),
            "n_pre_reg_total": len(valid_match),
            "match_rate": (sum(valid_match) / len(valid_match)) if valid_match else None,
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# Supplementary breakdown — every (layer, resolver, probe_type) row
# ──────────────────────────────────────────────────────────────────────

SUPP_COLS = [
    "model", "domain", "predicted_gap",
    "probe_type", "layer", "resolver",
    "target_acc", "max_earlier_acc", "headline_gap_pp",
    "observed_sign", "pre_reg_matches", "n_examples",
]


def build_supplementary_rows(doc: dict) -> list[dict]:
    meta = doc.get("meta", {})
    rows = []
    for h in doc.get("headlines", []):
        prereg = h.get("pre_registration_check", {})
        rows.append({
            "model": meta.get("model"),
            "domain": meta.get("domain"),
            "predicted_gap": meta.get("predicted_gap"),
            "probe_type": h.get("probe_type"),
            "layer": h.get("layer"),
            "resolver": h.get("resolver"),
            "target_acc": h.get("target_accuracy"),
            "max_earlier_acc": h.get("max_earlier_accuracy"),
            "headline_gap_pp": 100.0 * h.get("headline_gap", 0.0),
            "observed_sign": prereg.get("observed_sign"),
            "pre_reg_matches": prereg.get("matches"),
            "n_examples": h.get("n_examples"),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Markdown digest
# ──────────────────────────────────────────────────────────────────────

def build_markdown_summary(master_rows: list[dict], domain_rows: list[dict]) -> str:
    lines = []
    lines.append("# Staircase v2 — results summary\n")
    lines.append(f"_Aggregated from {len(master_rows)} (model, domain) pairs._\n")
    lines.append("## Pre-registration check by domain\n")
    lines.append("| Domain | Predicted | n_models | Mean gap (pp) | Min gap (pp) | Max gap (pp) | Pre-reg match rate |")
    lines.append("|--------|-----------|----------|---------------|--------------|--------------|---------------------|")
    for d in domain_rows:
        mean_str = f"{d['mean_gap_pp']:+.1f}" if d["mean_gap_pp"] is not None else "—"
        min_str = f"{d['min_gap_pp']:+.1f}" if d["min_gap_pp"] is not None else "—"
        max_str = f"{d['max_gap_pp']:+.1f}" if d["max_gap_pp"] is not None else "—"
        rate_str = (f"{d['n_pre_reg_match']}/{d['n_pre_reg_total']} "
                    f"({100*d['match_rate']:.0f}%)") if d["match_rate"] is not None else "—"
        lines.append(
            f"| {d['domain']:14s} | {d['predicted_gap']:18s} | "
            f"{d['n_models']:8d} | {mean_str:13s} | {min_str:12s} | "
            f"{max_str:12s} | {rate_str} |"
        )
    lines.append("\n## Best headline per (model, domain)\n")
    lines.append("| Model | Domain | Layer | Resolver | Target | Max-earlier | Gap (pp) | CI (pp) | Obs sign | ✓/✗ |")
    lines.append("|-------|--------|-------|----------|--------|-------------|----------|---------|----------|------|")
    for r in master_rows:
        gap = (f"{r['headline_gap_pp']:+.1f}"
               if isinstance(r.get('headline_gap_pp'), (int, float)) else "—")
        ci = ""
        if isinstance(r.get("bootstrap_ci_lo_pp"), (int, float)):
            ci = f"[{r['bootstrap_ci_lo_pp']:+.1f}, {r['bootstrap_ci_hi_pp']:+.1f}]"
        else:
            ci = "—"
        target = f"{r['target_acc']:.3f}" if isinstance(r.get('target_acc'), (int, float)) else "—"
        earlier = f"{r['max_earlier_acc']:.3f}" if isinstance(r.get('max_earlier_acc'), (int, float)) else "—"
        layer = str(r['best_layer']) if r.get('best_layer') is not None else "—"
        resolver = (r.get('best_resolver') or "—")[:28]
        match = "✓" if r.get("pre_reg_matches") else "✗"
        lines.append(
            f"| {r['model']} | {r['domain']:14s} | {layer:5s} | {resolver:28s} | "
            f"{target} | {earlier} | {gap} | {ci} | {r.get('observed_sign','—')} | {match} |"
        )
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path, cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Aggregate staircase-v2 JSONs")
    ap.add_argument("--results_dir", default="results/v2",
                    help="Directory containing *__staircase.json files")
    ap.add_argument("--output_dir", default="results/v2",
                    help="Where to write MASTER_TABLE.csv and friends")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    odir = Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    docs = collect_jsons(rdir)
    if not docs:
        print(f"No JSONs found in {rdir}")
        return 1
    print(f"Loaded {len(docs)} JSON result file(s)")

    # Master table: one row per (model, domain)
    master_rows = []
    for d in docs:
        r = build_master_row(d)
        if r is not None:
            master_rows.append(r)
    master_rows.sort(key=lambda r: (r["domain"], r["model"]))
    write_csv(master_rows, odir / "MASTER_TABLE.csv", MASTER_COLS)
    print(f"  → {odir / 'MASTER_TABLE.csv'}   ({len(master_rows)} rows)")

    # Domain summary
    domain_rows = build_domain_summary(master_rows)
    write_csv(
        domain_rows, odir / "DOMAIN_SUMMARY.csv",
        ["domain", "predicted_gap", "n_models",
         "mean_gap_pp", "min_gap_pp", "max_gap_pp",
         "n_pre_reg_match", "n_pre_reg_total", "match_rate"],
    )
    print(f"  → {odir / 'DOMAIN_SUMMARY.csv'}  ({len(domain_rows)} rows)")

    # Supplementary breakdown
    supp_rows: list[dict] = []
    for d in docs:
        supp_rows.extend(build_supplementary_rows(d))
    write_csv(supp_rows, odir / "SUPPLEMENTARY.csv", SUPP_COLS)
    print(f"  → {odir / 'SUPPLEMENTARY.csv'} ({len(supp_rows)} rows)")

    # Markdown
    md = build_markdown_summary(master_rows, domain_rows)
    (odir / "SUMMARY.md").write_text(md)
    print(f"  → {odir / 'SUMMARY.md'}")

    # Quick console digest
    print()
    print("=== Quick digest ===")
    for d in domain_rows:
        rate = f"{100*d['match_rate']:.0f}%" if d['match_rate'] is not None else "—"
        mg = f"{d['mean_gap_pp']:+.1f}pp" if d['mean_gap_pp'] is not None else "—"
        print(f"  {d['domain']:18s}  predicted={d['predicted_gap']:18s}  "
              f"n={d['n_models']:3d}  mean_gap={mg:>7s}  pre_reg_match={rate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
