"""
compare_models.py — Run after all models finish.
Reads results_*/  directories and produces:
  - cross_model_comparison.csv
  - figures/fig_cross_model.pdf (paper-ready comparison figure)

Usage:
    python compare_models.py
"""

import pandas as pd, numpy as np, json, glob, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "#fefdf9"
INK   = "#1a1a1a"
ASH   = "#8a8580"

COLORS = ["#c14a1d", "#3a5f8a", "#2a7a4a", "#6a4a7a", "#b8893a"]

def load_model_results(d):
    name = os.path.basename(d).replace("results_", "")
    row = {"model": name, "dir": d}

    # H3 comparison
    h3p = os.path.join(d, "h3_comparison.csv")
    if os.path.exists(h3p):
        h3 = pd.read_csv(h3p)
        p = h3[h3["signal"] == "probe_score"]
        if len(p):
            row["det_auroc"]  = p["auroc_detection"].values[0]
            row["fail_auroc"] = p["auroc_failure"].values[0]

    # Layer probe
    lp = os.path.join(d, "layer_probe_results.csv")
    if os.path.exists(lp):
        lpdf = pd.read_csv(lp)
        best = lpdf.loc[lpdf["auroc_mean"].idxmax()]
        row["best_layer"] = int(best["layer"])
        row["best_mode"]  = best["mode"]
        row["peak_auroc"] = best["auroc_mean"]
        # Store full profile for plotting
        row["_profile"] = lpdf

    # Intervention
    iv = os.path.join(d, "intervention_records.csv")
    if os.path.exists(iv):
        ivdf = pd.read_csv(iv)
        piv = ivdf.pivot_table(index="base_id", columns="condition",
                                values="final_correct", aggfunc="first").dropna()
        if "baseline" in piv.columns:
            bc = piv["baseline"].astype(bool)
            row["baseline_acc"] = bc.mean()
            row["n_interv"] = len(piv)
            if "branch_and_pick" in piv.columns:
                bp = piv["branch_and_pick"].astype(bool)
                row["bp_acc"]   = bp.mean()
                row["bp_delta"] = bp.mean() - bc.mean()
                row["rescued"]  = int((~bc & bp).sum())
                row["broken"]   = int((bc & ~bp).sum())

    # H4: verbalization
    s1 = os.path.join(d, "stage1_records.csv")
    if os.path.exists(s1):
        s1df = pd.read_csv(s1)
        verb = s1df[s1df["condition"] == "error_verbalized"]
        parsed = verb["verbalized_conf"].dropna()
        row["verb_compliance"] = len(parsed) / max(len(verb), 1)
        if len(parsed) > 0:
            v = parsed.values
            row["verb_binary_pct"] = float(((v <= 0.05) | (v >= 0.95)).mean())
            row["verb_unique_vals"] = len(set(round(x, 2) for x in v))
        row["hedge_rate"] = float(verb["hedged"].fillna(False).mean())

    # Persistence
    if os.path.exists(iv):
        ivdf = pd.read_csv(iv)
        base = ivdf[ivdf["condition"] == "baseline"]
        y = (~base["final_correct"].astype(bool)).astype(int)
        if "mean_probe_score" in base.columns and y.nunique() > 1:
            from sklearn.metrics import roc_auc_score
            try:
                row["persist_mean_auroc"] = roc_auc_score(y, base["mean_probe_score"])
                row["persist_max_auroc"]  = roc_auc_score(y, base["max_probe_score"])
            except:
                pass

    return row


def main():
    dirs = sorted(glob.glob("results_*"))
    dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith("figures")]

    if not dirs:
        print("No results_* directories found. Run models first.")
        return

    rows = [load_model_results(d) for d in dirs]
    df = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("  CROSS-MODEL COMPARISON")
    print("="*80)

    cols = ["model", "det_auroc", "fail_auroc", "best_layer", "peak_auroc",
            "baseline_acc", "bp_acc", "bp_delta", "rescued", "broken",
            "verb_binary_pct", "hedge_rate", "persist_mean_auroc"]

    print(f"\n  {'Model':<22} {'Det':>6} {'Fail':>6} {'Best':>5} {'Base':>6} "
          f"{'B+P':>6} {'Δ':>7} {'R':>3} {'B':>3} {'Bin%':>5} {'Hedge':>6} {'Pers':>5}")
    print("  " + "─"*78)

    for _, r in df.iterrows():
        def g(k, fmt=".3f"):
            v = r.get(k)
            if pd.isna(v) if isinstance(v, float) else v is None:
                return "  — "
            return f"{v:{fmt}}"

        print(f"  {r['model']:<22} {g('det_auroc'):>6} {g('fail_auroc'):>6} "
              f"{'L'+str(int(r.get('best_layer',0))) if pd.notna(r.get('best_layer')) else '  —':>5} "
              f"{g('baseline_acc','.1%'):>6} {g('bp_acc','.1%'):>6} "
              f"{g('bp_delta','+.1%'):>7} "
              f"{str(int(r.get('rescued',0))) if pd.notna(r.get('rescued')) else '—':>3} "
              f"{str(int(r.get('broken',0))) if pd.notna(r.get('broken')) else '—':>3} "
              f"{g('verb_binary_pct','.0%'):>5} "
              f"{g('hedge_rate','.0%'):>6} "
              f"{g('persist_mean_auroc','.3f'):>5}")

    # ── Key replication questions ─────────────────────────────────
    print(f"\n  REPLICATION QUESTIONS:")
    if "det_auroc" in df.columns:
        vals = df["det_auroc"].dropna()
        if len(vals) > 1:
            repl = (vals > 0.90).all()
            print(f"  Q1 Probe detection replicates?  {'YES' if repl else 'MIXED'} "
                  f"(range {vals.min():.3f}–{vals.max():.3f})")

    if "fail_auroc" in df.columns:
        vals = df["fail_auroc"].dropna()
        if len(vals) > 1:
            repl = (vals < 0.55).all()
            print(f"  Q2 Failure collapse replicates? {'YES' if repl else 'MIXED'} "
                  f"(range {vals.min():.3f}–{vals.max():.3f})")

    if "bp_delta" in df.columns:
        vals = df["bp_delta"].dropna()
        if len(vals) > 1:
            repl = (vals > 0).all()
            print(f"  Q3 Branch+pick helps?           {'YES' if repl else 'MIXED'} "
                  f"(range {vals.min():+.1%}–{vals.max():+.1%})")

    if "broken" in df.columns:
        vals = df["broken"].dropna()
        if len(vals) > 1:
            repl = (vals == 0).all()
            print(f"  Q4 Zero broken everywhere?      {'YES' if repl else 'NO'} "
                  f"(values: {vals.tolist()})")

    if "verb_binary_pct" in df.columns:
        vals = df["verb_binary_pct"].dropna()
        if len(vals) > 1:
            print(f"  Q5 Binary verbalization?        "
                  f"(values: {[f'{v:.0%}' for v in vals]})")

    # ── Save CSV ──────────────────────────────────────────────────
    out_cols = [c for c in cols if c in df.columns]
    df[out_cols].to_csv("cross_model_comparison.csv", index=False)
    print(f"\n  Saved: cross_model_comparison.csv")

    # ── Paper figure: layer profiles side by side ─────────────────
    profiles = [(r["model"], r["_profile"])
                for _, r in df.iterrows() if "_profile" in r and r["_profile"] is not None]

    if len(profiles) >= 2:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor(PAPER)
        ax.set_facecolor(PAPER)

        for i, (name, pdf) in enumerate(profiles):
            c = COLORS[i % len(COLORS)]
            # Use "last" mode if available, else whatever's there
            sub = pdf[pdf["mode"] == "last"] if "last" in pdf["mode"].values else pdf
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["auroc_mean"], color=c, lw=1.5,
                    label=f"{name}", marker=".", markersize=3)

        ax.axhline(0.5, color=ASH, ls=":", lw=0.6)
        ax.axhline(0.9, color="#2a7a4a", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Probe AUROC (5-fold CV)")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        ax.set_ylim(0.44, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/fig_cross_model_layers.pdf",
                    bbox_inches="tight", facecolor=PAPER)
        plt.close()
        print(f"  Saved: figures/fig_cross_model_layers.pdf")

    # ── Paper figure: intervention comparison ─────────────────────
    interv_models = df.dropna(subset=["bp_delta"])
    if len(interv_models) >= 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(PAPER)
        ax.set_facecolor(PAPER)

        x = np.arange(len(interv_models))
        w = 0.35
        ax.bar(x - w/2, interv_models["baseline_acc"] * 100,
               w, color=ASH, alpha=0.6, label="Baseline")
        ax.bar(x + w/2, interv_models["bp_acc"] * 100,
               w, color="#6a4a7a", alpha=0.8, label="Branch+pick")

        for i, (_, r) in enumerate(interv_models.iterrows()):
            delta = r["bp_delta"] * 100
            ax.text(i + w/2, r["bp_acc"]*100 + 1, f"+{delta:.1f}pp",
                    ha="center", fontsize=8, color="#6a4a7a", fontweight="bold")
            ax.text(i + w/2, r["bp_acc"]*100 - 4,
                    f"R={int(r['rescued'])} B={int(r['broken'])}",
                    ha="center", fontsize=7, color=INK)

        ax.set_xticks(x)
        ax.set_xticklabels(interv_models["model"], fontsize=8, rotation=15)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        fig.savefig("figures/fig_cross_model_intervention.pdf",
                    bbox_inches="tight", facecolor=PAPER)
        plt.close()
        print(f"  Saved: figures/fig_cross_model_intervention.pdf")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()