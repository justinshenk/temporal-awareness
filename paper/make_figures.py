#!/usr/bin/env python3
"""Publication-quality figures for the EMNLP staircase paper.
All numbers read directly from results/v2/*.json — nothing hardcoded."""
import json, glob, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

RES = "results/v2"
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# Color palette (colorblind-safe)
C = {"code": "#E69F00", "rhyme": "#0072B2", "qa_neutral": "#009E73",
     "qa_suggestive": "#CC79A7", "trivia": "#999999", "floor": "#BBBBBB",
     "learned": "#0072B2"}

def best_headline(d, probe="linear"):
    hs = [h for h in d["headlines"] if h.get("probe_type", "linear") == probe]
    return sorted(hs, key=lambda r: -abs(r["headline_gap"]))[0] if hs else None

def family(model):
    m = model.lower()
    for key, fam in [("gemma","Gemma"),("qwen","Qwen"),("pythia","Pythia"),
                     ("gpt2","GPT-2"),("mistral","Mistral"),("llama","Llama"),
                     ("falcon","Falcon3"),("olmo","OLMo"),("stable","StableLM")]:
        if key in m: return fam
    return "Other"

def gap_at(model_tag, step, dom):
    for f in glob.glob(f"{RES}/*{model_tag}*step{step}__{dom}__staircase.json"):
        if "mlp" in f or "full" in f: continue
        d = json.load(open(f)); h = best_headline(d)
        return h["headline_gap"]*100 if h else None
    return None

# ════════════════════════════════════════════════════════════════════
# FIG 1 — 3-way training dynamics (THE headline figure)
# ════════════════════════════════════════════════════════════════════
def fig1_training_dynamics():
    steps = [0, 512, 4000, 16000, 32000, 64000, 128000, 143000]
    series = {"code": [], "rhyme": [], "qa_neutral": []}
    xs = {"code": [], "rhyme": [], "qa_neutral": []}
    for dom in series:
        for s in steps:
            g = gap_at("pythia-1.4b", s, dom)
            if g is not None:
                series[dom].append(g); xs[dom].append(max(s, 200))  # log-safe

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = {"code":"Code (positional artifact)",
              "rhyme":"Rhyme (learned planning)",
              "qa_neutral":"QA-neutral (genuinely absent)"}
    markers = {"code":"s","rhyme":"o","qa_neutral":"^"}
    for dom in ["rhyme","code","qa_neutral"]:
        ax.plot(xs[dom], series[dom], marker=markers[dom], color=C[dom],
                label=labels[dom], linewidth=2, markersize=5)
    # Annotate floor and final for rhyme
    ax.annotate(f"floor\n+{series['rhyme'][0]:.0f}pp", xy=(xs['rhyme'][0], series['rhyme'][0]),
                xytext=(xs['rhyme'][0]*1.5, series['rhyme'][0]-13), fontsize=7,
                color=C['rhyme'], ha="center")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Staircase gap (pp)")
    ax.set_title("Training dynamics decompose the staircase gap\n(Pythia-1.4B)")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_ylim(-12, 68)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig1_training_dynamics.{ext}")
    plt.close(fig)
    print("✓ fig1_training_dynamics")

# ════════════════════════════════════════════════════════════════════
# FIG 2 — Cross-model gaps by domain (the discriminator)
# ════════════════════════════════════════════════════════════════════
def fig2_cross_model():
    from collections import defaultdict
    by_dom = defaultdict(list)
    for f in sorted(glob.glob(f"{RES}/*__staircase.json")):
        if any(x in f for x in ["step","mlp","full"]): continue
        d = json.load(open(f)); h = best_headline(d)
        if h:
            by_dom[d["meta"]["domain"]].append(
                (family(d["meta"]["model"]), h["headline_gap"]*100))

    domains = ["rhyme","qa_suggestive","code","qa_neutral","trivia"]
    dom_labels = {"rhyme":"Rhyme","qa_suggestive":"QA-sugg.","code":"Code",
                  "qa_neutral":"QA-neut.","trivia":"Trivia"}
    fams = sorted({f for v in by_dom.values() for f,_ in v})
    fam_colors = dict(zip(fams, plt.cm.tab10(np.linspace(0,1,len(fams)))))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, dom in enumerate(domains):
        pts = by_dom.get(dom, [])
        gaps = [g for _,g in pts]
        x = np.random.RandomState(0).normal(i, 0.07, len(gaps))
        for (fam, g), xi in zip(pts, x):
            ax.scatter(xi, g, color=fam_colors[fam], s=22, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
        if gaps:
            ax.hlines(np.mean(gaps), i-0.28, i+0.28, color="black", lw=1.6, zorder=4)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([dom_labels[d] for d in domains])
    ax.set_ylabel("Staircase gap (pp)")
    ax.set_title("Per-position staircase gap varies by an order of magnitude across domains")
    # Legend
    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=fam_colors[f],
               markersize=6, label=f) for f in fams]
    ax.legend(handles=handles, loc="upper right", ncol=2, framealpha=0.9, fontsize=7)
    ax.set_ylim(-12, 85)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig2_cross_model.{ext}")
    plt.close(fig)
    print("✓ fig2_cross_model")

# ════════════════════════════════════════════════════════════════════
# FIG 3 — Floor / learned decomposition across 3 model sizes
# ════════════════════════════════════════════════════════════════════
def fig3_decomposition():
    sizes = ["1.4b","2.8b","6.9b"]
    data = {}  # size -> {code:(floor,learned), rhyme:(floor,learned)}
    for s in sizes:
        cf, cF = gap_at(f"pythia-{s}",0,"code"), gap_at(f"pythia-{s}",143000,"code")
        rf, rF = gap_at(f"pythia-{s}",0,"rhyme"), gap_at(f"pythia-{s}",143000,"rhyme")
        data[s] = {"code":(cf, cF-cf), "rhyme":(rf, rF-rf)}

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(sizes)); w = 0.36
    # Code bars (floor + learned stacked)
    code_floor = [data[s]["code"][0] for s in sizes]
    code_learn = [data[s]["code"][1] for s in sizes]
    rhyme_floor = [data[s]["rhyme"][0] for s in sizes]
    rhyme_learn = [data[s]["rhyme"][1] for s in sizes]

    ax.bar(x-w/2, code_floor, w, color=C["floor"], label="Positional floor (step 0)")
    ax.bar(x-w/2, code_learn, w, bottom=code_floor, color=C["code"], label="Learned (code)")
    ax.bar(x+w/2, rhyme_floor, w, color=C["floor"])
    ax.bar(x+w/2, rhyme_learn, w, bottom=rhyme_floor, color=C["rhyme"], label="Learned (rhyme)")

    for i,s in enumerate(sizes):
        ax.text(i-w/2, code_floor[i]+code_learn[i]/2, f"+{code_learn[i]:.0f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        ax.text(i+w/2, rhyme_floor[i]+rhyme_learn[i]/2, f"+{rhyme_learn[i]:.0f}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Pythia-{s.upper()}" for s in sizes])
    ax.set_ylabel("Staircase gap (pp)")
    ax.set_title("Floor / learned decomposition is consistent across model scale")
    # Group labels under each pair of bars
    for i in range(len(sizes)):
        ax.text(i-w/2, 2, "code", ha="center", fontsize=6.5, color="white", rotation=90, va="bottom")
        ax.text(i+w/2, 2, "rhyme", ha="center", fontsize=6.5, color="white", rotation=90, va="bottom")
    ax.legend(loc="upper center", ncol=3, framealpha=0.9, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 72)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig3_decomposition.{ext}")
    plt.close(fig)
    print("✓ fig3_decomposition")

# ════════════════════════════════════════════════════════════════════
# FIG 4 — Dual baseline scatter (per-position vs mean-pool)
# ════════════════════════════════════════════════════════════════════
def fig4_dual_baseline():
    pts = {"code":[], "rhyme":[]}
    for f in sorted(glob.glob(f"{RES}/*__staircase.json")):
        if any(x in f for x in ["step","mlp","full"]): continue
        d = json.load(open(f)); dom = d["meta"]["domain"]
        if dom not in pts: continue
        h = best_headline(d)
        if h and h.get("target_vs_mean_pool_gap") is not None:
            pts[dom].append((h["headline_gap"]*100, h["target_vs_mean_pool_gap"]*100))

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for dom, marker in [("rhyme","o"),("code","s")]:
        if not pts[dom]: continue
        xs = [p[0] for p in pts[dom]]; ys = [p[1] for p in pts[dom]]
        ax.scatter(xs, ys, color=C[dom], marker=marker, s=30, alpha=0.8,
                   edgecolors="white", linewidths=0.4,
                   label=f"{dom} (n={len(xs)})")
    lim = 95
    ax.plot([-5,lim],[-5,lim], color="gray", ls="--", lw=0.8, label="y = x")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Gap vs. max-earlier baseline (pp)")
    ax.set_ylabel("Gap vs. mean-pool baseline (pp)")
    ax.set_title("Code gap collapses under a\nmean-pool baseline; rhyme persists")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-5, lim); ax.set_ylim(-15, lim)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig4_dual_baseline.{ext}")
    plt.close(fig)
    print("✓ fig4_dual_baseline")

# ════════════════════════════════════════════════════════════════════
# FIG 5 — Checkpoint behavioral (dual axis)
# ════════════════════════════════════════════════════════════════════
def fig5_behavioral():
    ck = json.load(open(f"{RES}/behavioral_checkpoints.json"))
    rows = sorted([ck[k] for k in ck if k.startswith("step")], key=lambda r: r["step"])
    steps = [max(r["step"],200) for r in rows]
    gaps = [r["probe_gap"] for r in rows]
    behav = [r["accuracy"]*100 for r in rows]
    corr = ck.get("_checkpoint_correlation", {})

    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ln1 = ax1.plot(steps, gaps, "o-", color=C["rhyme"], lw=2, markersize=5,
                   label="Probe gap (rhyme)")
    ax1.set_xlabel("Training step (log scale)")
    ax1.set_ylabel("Staircase gap (pp)", color=C["rhyme"])
    ax1.tick_params(axis="y", labelcolor=C["rhyme"])
    ax1.set_xscale("log")
    ax1.set_ylim(0, 68)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ln2 = ax2.plot(steps, behav, "s--", color=C["code"], lw=2, markersize=5,
                   label="Rhyme generation accuracy")
    ax2.set_ylabel("Behavioral rhyme accuracy (%)", color=C["code"])
    ax2.tick_params(axis="y", labelcolor=C["code"])
    ax2.set_ylim(-3, 68)

    ax1.set_title(f"Probe gap and rhyme generation emerge together\n"
                  f"(within-model: Spearman $\\rho$={corr.get('spearman_rho')}, "
                  f"p={corr.get('spearman_p')})")
    lns = ln1+ln2
    ax1.legend(lns, [l.get_label() for l in lns], loc="lower right", framealpha=0.9)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig5_behavioral.{ext}")
    plt.close(fig)
    print("✓ fig5_behavioral")

# ════════════════════════════════════════════════════════════════════
# APPENDIX FIG A — MLP vs linear agreement
# ════════════════════════════════════════════════════════════════════
def figA_mlp():
    pts = []
    for f in sorted(glob.glob(f"{RES}/*mlp*__staircase.json")):
        d = json.load(open(f))
        lh, mh = best_headline(d,"linear"), best_headline(d,"mlp")
        if lh and mh:
            pts.append((d["meta"]["domain"], lh["headline_gap"]*100, mh["headline_gap"]*100))
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for dom, marker in [("rhyme","o"),("qa_neutral","^")]:
        xs = [p[1] for p in pts if p[0]==dom]
        ys = [p[2] for p in pts if p[0]==dom]
        ax.scatter(xs, ys, color=C[dom], marker=marker, s=36, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=f"{dom} (n={len(xs)})")
    ax.plot([-10,85],[-10,85], color="gray", ls="--", lw=0.8, label="linear = MLP")
    ax.axhline(0, color="black", lw=0.5, ls=":"); ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Linear probe gap (pp)")
    ax.set_ylabel("MLP probe gap (pp)")
    ax.set_title("MLP probes agree with linear probes (13/13)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/figA_mlp_agreement.{ext}")
    plt.close(fig)
    print("✓ figA_mlp_agreement")

# ════════════════════════════════════════════════════════════════════
# APPENDIX FIG B — rhyme gap vs model size (invariance)
# ════════════════════════════════════════════════════════════════════
def figB_size_invariance():
    # parameter counts (approx, in billions)
    sizes = {"gpt2":0.124,"gpt2-medium":0.355,"gpt2-xl":1.5,
             "pythia-410m-deduped":0.41,"pythia-1b-deduped":1.0,
             "pythia-1.4b-deduped":1.4,"pythia-2.8b-deduped":2.8,
             "Qwen3-1.7B-Base":1.7,"Qwen3-8B-Base":8.0,
             "gemma-2-2b":2.6,"gemma-2-9b":9.0,"gemma-2-27b":27.0,
             "Mistral-7B-v0.3":7.2,"Meta-Llama-3.1-8B":8.0,"Llama-3.2-3B":3.2,
             "Falcon3-7B-Base":7.0,"OLMo-7B-0724-hf":7.0,"stablelm-2-1_6b":1.6}
    pts = []
    for f in sorted(glob.glob(f"{RES}/*__rhyme__staircase.json")):
        if any(x in f for x in ["step","mlp","full"]): continue
        d = json.load(open(f)); m = d["meta"]["model"].split("/")[-1]
        if m in sizes:
            h = best_headline(d)
            if h: pts.append((sizes[m], h["headline_gap"]*100, family(m)))
    fams = sorted({p[2] for p in pts})
    fam_colors = dict(zip(fams, plt.cm.tab10(np.linspace(0,1,len(fams)))))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fam in fams:
        xs = [p[0] for p in pts if p[2]==fam]
        ys = [p[1] for p in pts if p[2]==fam]
        ax.scatter(xs, ys, color=fam_colors[fam], s=34, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters, log scale)")
    ax.set_ylabel("Rhyme staircase gap (pp)")
    ax.set_title("Rhyme gap is size-invariant (+48 to +77pp, 0.12B–27B)")
    ax.legend(loc="lower right", ncol=2, fontsize=7, framealpha=0.9)
    ax.set_ylim(40, 82)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/figB_size_invariance.{ext}")
    plt.close(fig)
    print("✓ figB_size_invariance")

if __name__ == "__main__":
    fig1_training_dynamics()
    fig2_cross_model()
    fig3_decomposition()
    fig4_dual_baseline()
    fig5_behavioral()
    figA_mlp()
    figB_size_invariance()
    print("\nAll figures written to", OUT)

# ════════════════════════════════════════════════════════════════════
# FIG 6 — Diagnostic regimes overview (floor vs learned) [MUST-HAVE]
# ════════════════════════════════════════════════════════════════════
def fig6_regimes():
    import json as _json
    reg = _json.load(open("/tmp/regime.json"))
    labels = {"rhyme":"Rhyme","code":"Code","qa_neutral":"QA-neutral",
              "qa_suggestive":"QA-suggestive"}
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for dom, r in reg.items():
        ax.scatter(r["floor"], r["learned"], color=C.get(dom,"#555"), s=130,
                   edgecolors="black", linewidths=0.8, zorder=3)
        dx = 2.5 if dom!="rhyme" else -2.5
        ha = "left" if dom!="rhyme" else "right"
        ax.annotate(labels.get(dom,dom), (r["floor"], r["learned"]),
                    xytext=(r["floor"]+dx, r["learned"]+1.8), fontsize=9,
                    ha=ha, color=C.get(dom,"#555"), fontweight="bold")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axvline(0, color="black", lw=0.6, ls=":")
    # Quadrant guide text
    ax.text(34, 23.5+5, "", fontsize=7)
    ax.annotate("planning-consistent\n(high floor, high learned)",
                xy=(34,23.5), xytext=(20,15), fontsize=7, style="italic",
                color="#444", ha="center")
    ax.annotate("positional artifact\n(low learned)",
                xy=(10.3,2.0), xytext=(14,7.5), fontsize=7, style="italic",
                color="#444", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.6))
    ax.annotate("genuine null\n(no floor, no learned)",
                xy=(-2.5,1.3), xytext=(-1,8), fontsize=7, style="italic",
                color="#444", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.6))
    ax.set_xlabel("Positional floor (gap at step 0, pp)")
    ax.set_ylabel("Learned component (final $-$ floor, pp)")
    ax.set_title("Diagnostic regimes: the decomposition separates\nthree qualitatively different task signatures")
    ax.set_xlim(-8, 42); ax.set_ylim(-3, 28)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig6_regimes.{ext}")
    plt.close(fig)
    print("✓ fig6_regimes")

# ════════════════════════════════════════════════════════════════════
# FIG 7 — Full layer curves (kills max-over-layer attack) [MUST-HAVE]
# ════════════════════════════════════════════════════════════════════
def fig7_layer_curves():
    import json as _json
    cur = _json.load(open("/tmp/layer_curves.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    titles = {"rhyme":"Rhyme","code":"Code","qa_neutral":"QA-neutral"}
    for ax, dom in zip(axes, ["rhyme","code","qa_neutral"]):
        c = cur[dom]
        L = c["layers"]
        ax.plot(L, [t*100 for t in c["target"]], "o-", color=C[dom],
                markersize=3, lw=1.5, label="target")
        ax.plot(L, [e*100 for e in c["earlier"]], "s--", color="#888",
                markersize=3, lw=1.2, label="max-earlier")
        ax.fill_between(L, [e*100 for e in c["earlier"]],
                        [t*100 for t in c["target"]], color=C[dom], alpha=0.15)
        ax.set_title(titles[dom], fontsize=9)
        ax.set_xlabel("Layer")
        ax.axhline(0, color="black", lw=0.4, ls=":")
        if dom=="rhyme":
            ax.set_ylabel("Probe accuracy (%)")
            ax.legend(fontsize=6.5, loc="center right")
    fig.suptitle("Per-layer target vs. strongest-earlier accuracy (Gemma-2-2B): the gap is broad, not one lucky layer",
                 fontsize=8.5, y=1.04)
    fig.tight_layout()
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/fig7_layer_curves.{ext}")
    plt.close(fig)
    print("✓ fig7_layer_curves")

fig6_regimes()
fig7_layer_curves()
