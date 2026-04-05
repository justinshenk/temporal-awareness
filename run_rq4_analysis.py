#!/usr/bin/env python3
"""
RQ4 FINAL ANALYSIS: Save stats + Generate all figures
"""
import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/lookahead/final/figures", exist_ok=True)
os.makedirs("results/lookahead/final/stats", exist_ok=True)

# ================================================================
# ALL RESULTS
# ================================================================
results = {
    "GPT-2 Small": {
        "behavioral": 0.0, "params": "124M", "family": "GPT-2",
        "layers": {
            "L0":  {"probe": 0.698, "name": 0.637, "np": 0.809, "bow": 0.748, "p_name": 0.034, "p_np": 1.000},
            "L3":  {"probe": 0.732, "name": 0.617, "np": 0.827, "bow": 0.748, "p_name": 0.001, "p_np": 0.995},
            "L6":  {"probe": 0.769, "name": 0.617, "np": 0.846, "bow": 0.748, "p_name": 0.000, "p_np": 0.982},
            "L9":  {"probe": 0.803, "name": 0.607, "np": 0.842, "bow": 0.750, "p_name": 0.000, "p_np": 0.992},
            "L10": {"probe": 0.773, "name": 0.609, "np": 0.842, "bow": 0.750, "p_name": 0.000, "p_np": 0.995},
            "L11": {"probe": 0.791, "name": 0.617, "np": 0.832, "bow": 0.750, "p_name": 0.000, "p_np": 0.965},
        }
    },
    "GPT-2 Medium": {
        "behavioral": 0.0, "params": "345M", "family": "GPT-2",
        "layers": {
            "L0":  {"probe": 0.659, "name": 0.615, "np": 0.805, "bow": 0.748, "p_name": 0.176, "p_np": 1.000},
            "L6":  {"probe": 0.761, "name": 0.667, "np": 0.874, "bow": 0.746, "p_name": 0.007, "p_np": 0.997},
            "L12": {"probe": 0.775, "name": 0.633, "np": 0.854, "bow": 0.750, "p_name": 0.006, "p_np": 0.999},
            "L18": {"probe": 0.767, "name": 0.629, "np": 0.856, "bow": 0.748, "p_name": 0.004, "p_np": 0.995},
            "L22": {"probe": 0.773, "name": 0.651, "np": 0.876, "bow": 0.748, "p_name": 0.001, "p_np": 0.994},
            "L23": {"probe": 0.787, "name": 0.649, "np": 0.856, "bow": 0.748, "p_name": 0.000, "p_np": 0.990},
        }
    },
    "GPT-2 XL": {
        "behavioral": 0.0, "params": "1.5B", "family": "GPT-2",
        "layers": {
            "L0":  {"probe": 0.749, "name": 0.615, "np": 0.846, "bow": 0.748, "p_name": 0.000, "p_np": 0.998},
            "L12": {"probe": 0.771, "name": 0.635, "np": 0.864, "bow": 0.748, "p_name": 0.001, "p_np": 1.000},
            "L24": {"probe": 0.807, "name": 0.665, "np": 0.862, "bow": 0.748, "p_name": 0.001, "p_np": 0.993},
            "L36": {"probe": 0.789, "name": 0.657, "np": 0.874, "bow": 0.748, "p_name": 0.000, "p_np": 0.994},
            "L46": {"probe": 0.805, "name": 0.675, "np": 0.864, "bow": 0.748, "p_name": 0.004, "p_np": 0.997},
            "L47": {"probe": 0.789, "name": 0.681, "np": 0.858, "bow": 0.748, "p_name": 0.001, "p_np": 0.994},
        }
    },
    "Pythia-410M": {
        "behavioral": 0.267, "params": "410M", "family": "Pythia",
        "layers": {
            "L0":  {"probe": 0.639, "name": 0.607, "np": 0.830, "bow": 0.752, "p_name": 0.200, "p_np": 1.000},
            "L6":  {"probe": 0.771, "name": 0.631, "np": 0.892, "bow": 0.742, "p_name": 0.001, "p_np": 1.000},
            "L12": {"probe": 0.807, "name": 0.645, "np": 0.892, "bow": 0.752, "p_name": 0.001, "p_np": 0.999},
            "L18": {"probe": 0.793, "name": 0.667, "np": 0.876, "bow": 0.752, "p_name": 0.001, "p_np": 0.998},
            "L22": {"probe": 0.805, "name": 0.657, "np": 0.878, "bow": 0.752, "p_name": 0.000, "p_np": 0.998},
            "L23": {"probe": 0.809, "name": 0.643, "np": 0.864, "bow": 0.752, "p_name": 0.000, "p_np": 0.994},
        }
    },
    "Pythia-1B": {
        "behavioral": 0.320, "params": "1B", "family": "Pythia",
        "layers": {
            "L0":  {"probe": 0.704, "name": 0.645, "np": 0.842, "bow": 0.752, "p_name": 0.059, "p_np": 1.000},
            "L4":  {"probe": 0.763, "name": 0.641, "np": 0.901, "bow": 0.742, "p_name": 0.004, "p_np": 1.000},
            "L8":  {"probe": 0.823, "name": 0.653, "np": 0.901, "bow": 0.752, "p_name": 0.001, "p_np": 1.000},
            "L12": {"probe": 0.819, "name": 0.665, "np": 0.880, "bow": 0.752, "p_name": 0.000, "p_np": 0.990},
            "L14": {"probe": 0.821, "name": 0.659, "np": 0.870, "bow": 0.752, "p_name": 0.000, "p_np": 0.988},
            "L15": {"probe": 0.815, "name": 0.633, "np": 0.856, "bow": 0.752, "p_name": 0.000, "p_np": 0.984},
        }
    },
    "Pythia-1.4B": {
        "behavioral": 0.387, "params": "1.4B", "family": "Pythia",
        "layers": {
            "L0":  {"probe": 0.702, "name": 0.633, "np": 0.836, "bow": 0.752, "p_name": 0.130, "p_np": 1.000},
            "L6":  {"probe": 0.801, "name": 0.669, "np": 0.894, "bow": 0.752, "p_name": 0.001, "p_np": 0.999},
            "L12": {"probe": 0.832, "name": 0.690, "np": 0.900, "bow": 0.752, "p_name": 0.000, "p_np": 0.997},
            "L18": {"probe": 0.813, "name": 0.708, "np": 0.884, "bow": 0.752, "p_name": 0.001, "p_np": 0.998},
            "L22": {"probe": 0.801, "name": 0.696, "np": 0.874, "bow": 0.742, "p_name": 0.004, "p_np": 0.998},
            "L23": {"probe": 0.819, "name": 0.689, "np": 0.860, "bow": 0.742, "p_name": 0.001, "p_np": 0.994},
        }
    },
    "Pythia-2.8B": {
        "behavioral": 0.327, "params": "2.8B", "family": "Pythia",
        "layers": {
            "L0":  {"probe": 0.686, "name": 0.649, "np": 0.838, "bow": 0.752, "p_name": 0.121, "p_np": 1.000},
            "L8":  {"probe": 0.799, "name": 0.679, "np": 0.907, "bow": 0.665, "p_name": 0.001, "p_np": 1.000},
            "L16": {"probe": 0.826, "name": 0.677, "np": 0.911, "bow": 0.752, "p_name": 0.000, "p_np": 1.000},
            "L24": {"probe": 0.821, "name": 0.692, "np": 0.903, "bow": 0.665, "p_name": 0.000, "p_np": 1.000},
            "L30": {"probe": 0.832, "name": 0.663, "np": 0.905, "bow": 0.665, "p_name": 0.000, "p_np": 0.996},
            "L31": {"probe": 0.826, "name": 0.681, "np": 0.901, "bow": 0.742, "p_name": 0.000, "p_np": 0.999},
        }
    },
    "SantaCoder": {
        "behavioral": 0.413, "params": "1.1B", "family": "Code",
        "layers": {
            "L0":  {"probe": 0.680, "name": 0.680, "np": 0.870, "bow": 0.493, "p_name": 1.000, "p_np": 1.000},
            "L6":  {"probe": 0.858, "name": 0.667, "np": 0.919, "bow": 0.732, "p_name": 0.000, "p_np": 0.998},
            "L12": {"probe": 0.911, "name": 0.706, "np": 0.931, "bow": 0.732, "p_name": 0.000, "p_np": 0.949},
            "L18": {"probe": 0.874, "name": 0.708, "np": 0.927, "bow": 0.702, "p_name": 0.000, "p_np": 0.995},
            "L22": {"probe": 0.854, "name": 0.710, "np": 0.917, "bow": 0.663, "p_name": 0.000, "p_np": 0.994},
            "L23": {"probe": 0.856, "name": 0.704, "np": 0.909, "bow": 0.702, "p_name": 0.000, "p_np": 0.998},
        }
    },
    "CodeLlama-7B": {
        "behavioral": 0.427, "params": "7B", "family": "Code",
        "layers": {
            "L0":  {"probe": 0.677, "name": 0.629, "np": 0.848, "bow": 0.742, "p_name": 0.058, "p_np": 1.000},
            "L8":  {"probe": 0.860, "name": 0.700, "np": 0.935, "bow": 0.722, "p_name": 0.000, "p_np": 1.000},
            "L16": {"probe": 0.886, "name": 0.651, "np": 0.945, "bow": 0.637, "p_name": 0.000, "p_np": 1.000},
            "L24": {"probe": 0.870, "name": 0.685, "np": 0.919, "bow": 0.722, "p_name": 0.000, "p_np": 0.998},
            "L30": {"probe": 0.844, "name": 0.680, "np": 0.901, "bow": 0.722, "p_name": 0.000, "p_np": 1.000},
            "L31": {"probe": 0.824, "name": 0.675, "np": 0.901, "bow": 0.722, "p_name": 0.000, "p_np": 1.000},
        }
    },
    "Llama-3.2-1B": {
        "behavioral": 0.340, "params": "1.2B", "family": "Llama",
        "layers": {
            "L0":  {"probe": 0.718, "name": 0.718, "np": 0.852, "bow": 0.556, "p_name": 1.000, "p_np": 1.000},
            "L4":  {"probe": 0.840, "name": 0.718, "np": 0.931, "bow": 0.706, "p_name": 0.003, "p_np": 1.000},
            "L8":  {"probe": 0.866, "name": 0.748, "np": 0.937, "bow": 0.706, "p_name": 0.001, "p_np": 1.000},
            "L12": {"probe": 0.820, "name": 0.722, "np": 0.921, "bow": 0.769, "p_name": 0.022, "p_np": 1.000},
            "L14": {"probe": 0.822, "name": 0.714, "np": 0.901, "bow": 0.769, "p_name": 0.024, "p_np": 1.000},
            "L15": {"probe": 0.811, "name": 0.730, "np": 0.903, "bow": 0.769, "p_name": 0.021, "p_np": 1.000},
        }
    },
    "Llama-3.2-1B-Inst": {
        "behavioral": 0.340, "params": "1.2B", "family": "Llama",
        "layers": {
            "L0":  {"probe": 0.730, "name": 0.730, "np": 0.876, "bow": 0.556, "p_name": 1.000, "p_np": 1.000},
            "L4":  {"probe": 0.813, "name": 0.736, "np": 0.927, "bow": 0.706, "p_name": 0.009, "p_np": 1.000},
            "L8":  {"probe": 0.836, "name": 0.736, "np": 0.943, "bow": 0.769, "p_name": 0.021, "p_np": 1.000},
            "L12": {"probe": 0.811, "name": 0.773, "np": 0.927, "bow": 0.706, "p_name": 0.163, "p_np": 1.000},
            "L14": {"probe": 0.807, "name": 0.749, "np": 0.913, "bow": 0.769, "p_name": 0.099, "p_np": 1.000},
            "L15": {"probe": 0.813, "name": 0.742, "np": 0.909, "bow": 0.706, "p_name": 0.041, "p_np": 1.000},
        }
    },
}

# Generation-time decay data
gentime_data = {
    "GPT-2 XL": {"steps": list(range(20)),
        "L24": [0.872,0.870,0.880,0.880,0.878,0.878,0.878,0.886,0.888,0.878,0.880,0.872,0.884,0.872,0.884,0.880,0.866,0.850,0.870,0.854],
        "np": 0.876},
    "SantaCoder": {"steps": list(range(20)),
        "L12": [0.935,0.901,0.874,0.858,0.854,0.840,0.840,0.828,0.820,0.832,0.791,0.720,0.754,0.744,0.753,0.714,0.700,0.722,0.692,0.651],
        "np": 0.939},
    "CodeLlama-7B": {"steps": list(range(20)),
        "L31": [0.953,0.943,0.933,0.817,0.714,0.724,0.631,0.675,0.663,0.651,0.710,0.665,0.647,0.698,0.698,0.641,0.663,0.627,0.627,0.629],
        "np": 0.917},
}

# Misleading names data
misleading_data = {
    "SantaCoder": {"L12": {"acc": 0.220, "follows_name": 11, "follows_params": 28, "follows_neither": 11}},
    "Pythia-2.8B": {"L16": {"acc": 0.320, "follows_name": 16, "follows_params": 28, "follows_neither": 6}},
    "GPT-2 XL": {"L24": {"acc": 0.340, "follows_name": 17, "follows_params": 24, "follows_neither": 9}},
}

# Acrostic data
acrostic_data = {
    "GPT-2 XL": 0.12, "Pythia-2.8B": 0.08, "SantaCoder": 0.12,
    "CodeLlama-7B": 0.20, "Llama-3.2-1B": 0.08, "Llama-3.2-1B-Inst": 0.04,
}

# ================================================================
# COMPUTE STATS
# ================================================================
model_summaries = []
all_p_np = []
all_p_name = []
for model_name, data in results.items():
    best_layer = max(data["layers"].items(), key=lambda x: x[1]["probe"])
    best_probe = best_layer[1]["probe"]
    best_np = max(ld["np"] for ld in data["layers"].values())
    gap = best_layer[1]["probe"] - best_layer[1]["np"]
    min_p_np = min(ld["p_np"] for ld in data["layers"].values())
    
    model_summaries.append({
        "model": model_name, "behavioral": data["behavioral"],
        "params": data["params"], "family": data["family"],
        "best_probe": best_probe, "best_np": best_np,
        "gap_np": gap, "min_p_np": min_p_np,
        "best_layer": best_layer[0],
    })
    for ld in data["layers"].values():
        all_p_np.append(ld["p_np"])
        all_p_name.append(ld["p_name"])

behavs = np.array([s["behavioral"] for s in model_summaries])
probes = np.array([s["best_probe"] for s in model_summaries])
gaps = np.array([s["gap_np"] for s in model_summaries])

rho_beh_probe, p_beh_probe = stats.spearmanr(behavs, probes)
rho_beh_gap, p_beh_gap = stats.spearmanr(behavs, gaps)

# FDR
all_p_np = np.array(all_p_np)
all_p_name = np.array(all_p_name)
n_tests = len(all_p_np)
sorted_pnp = np.sort(all_p_np)
bh_thresh = np.array([(i+1)/n_tests * 0.05 for i in range(n_tests)])
n_sig_np = np.sum(sorted_pnp <= bh_thresh)
sorted_pname = np.sort(all_p_name)
n_sig_name = np.sum(sorted_pname <= bh_thresh)

# Save stats JSON
stats_json = {
    "fdr_correction": {
        "n_tests": int(n_tests),
        "probe_vs_name_params": {"n_significant": int(n_sig_np), "min_p": float(all_p_np.min())},
        "probe_vs_name_only": {"n_significant": int(n_sig_name), "min_p": float(all_p_name.min())},
    },
    "spearman_correlations": {
        "behavioral_vs_probe": {"rho": float(rho_beh_probe), "p": float(p_beh_probe)},
        "behavioral_vs_gap": {"rho": float(rho_beh_gap), "p": float(p_beh_gap)},
    },
    "model_summaries": model_summaries,
}
with open("results/lookahead/final/stats/statistical_analysis.json", "w") as f:
    json.dump(stats_json, f, indent=2)
print("Saved stats/statistical_analysis.json")

# ================================================================
# FIGURE 1: BASELINE STAIRCASE (the money figure)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 7))
models = [s["model"] for s in model_summaries]
x = np.arange(len(models))
width = 0.18

# Get best-layer values for each model
chance = [0.20] * len(models)
bows = []
names = []
nps = []
probes_fig = []
for s in model_summaries:
    data = results[s["model"]]
    best_l = s["best_layer"]
    ld = data["layers"][best_l]
    bows.append(ld["bow"])
    names.append(ld["name"])
    nps.append(ld["np"])
    probes_fig.append(ld["probe"])

bars1 = ax.bar(x - 1.5*width, chance, width, label='Chance (20%)', color='#d3d3d3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x - 0.5*width, names, width, label='Name-Only', color='#87CEEB', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + 0.5*width, probes_fig, width, label='Probe (best layer)', color='#FFA500', edgecolor='black', linewidth=0.5)
bars4 = ax.bar(x + 1.5*width, nps, width, label='Name+Params', color='#FF4444', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Baseline Staircase: Name+Params Explains All Probing Signal', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
ax.grid(axis='y', alpha=0.3)

# Add behavioral accuracy as text on top
for i, s in enumerate(model_summaries):
    ax.text(i, 0.02, f"Beh:{s['behavioral']:.0%}", ha='center', fontsize=7, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig1_baseline_staircase.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig1_baseline_staircase.pdf", bbox_inches='tight')
plt.close()
print("Saved fig1_baseline_staircase")

# ================================================================
# FIGURE 2: GENERATION-TIME DECAY
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, (model_name, gdata) in enumerate(gentime_data.items()):
    ax = axes[idx]
    layer_key = list(gdata.keys())[0] if list(gdata.keys())[0] != "steps" and list(gdata.keys())[0] != "np" else list(gdata.keys())[1]
    for k, v in gdata.items():
        if k not in ("steps", "np"):
            layer_key = k
            break
    
    steps = gdata["steps"]
    probe_accs = gdata[layer_key]
    np_val = gdata["np"]
    
    ax.plot(steps, probe_accs, 'o-', color='#FFA500', linewidth=2, markersize=4, label=f'Probe ({layer_key})')
    ax.axhline(y=np_val, color='#FF4444', linestyle='--', linewidth=2, label=f'Name+Params ({np_val:.1%})')
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.fill_between(steps, probe_accs, np_val, alpha=0.15, color='red')
    
    ax.set_xlabel('Generation Step', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Probe Accuracy', fontsize=12)
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.15, 1.0)
    ax.grid(alpha=0.3)

plt.suptitle('Generation-Time Commitment: Probe Accuracy Decays During Code Generation', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig2_gentime_decay.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig2_gentime_decay.pdf", bbox_inches='tight')
plt.close()
print("Saved fig2_gentime_decay")

# ================================================================
# FIGURE 3: MISLEADING NAMES — WHAT DOES THE MODEL FOLLOW?
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (model_name, mdata) in enumerate(misleading_data.items()):
    ax = axes[idx]
    layer_key = list(mdata.keys())[0]
    d = mdata[layer_key]
    
    categories = ['Follows\nName', 'Follows\nParams', 'Follows\nNeither']
    values = [d["follows_name"], d["follows_params"], d["follows_neither"]]
    colors = ['#87CEEB', '#FF4444', '#d3d3d3']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Count (out of 50)' if idx == 0 else '', fontsize=12)
    ax.set_title(f'{model_name} ({layer_key})\nAcc: {d["acc"]:.0%}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Misleading Names: Models Follow Parameter Names Over Function Semantics',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig3_misleading_names.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig3_misleading_names.pdf", bbox_inches='tight')
plt.close()
print("Saved fig3_misleading_names")

# ================================================================
# FIGURE 4: SPEARMAN — BEHAVIORAL vs PROBE vs GAP
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

family_colors = {"GPT-2": "#1f77b4", "Pythia": "#ff7f0e", "Code": "#2ca02c", "Llama": "#d62728"}
family_markers = {"GPT-2": "o", "Pythia": "s", "Code": "D", "Llama": "^"}

# Left: Behavioral vs Best Probe
for s in model_summaries:
    ax1.scatter(s["behavioral"], s["best_probe"], 
                c=family_colors[s["family"]], marker=family_markers[s["family"]],
                s=100, zorder=5, edgecolors='black', linewidth=0.5)
    ax1.annotate(s["model"], (s["behavioral"], s["best_probe"]),
                 fontsize=7, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')

ax1.set_xlabel('Behavioral Accuracy', fontsize=12)
ax1.set_ylabel('Best Probe Accuracy', fontsize=12)
ax1.set_title(f'Behavioral vs Probe\nρ={rho_beh_probe:.3f}, p={p_beh_probe:.4f}', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Right: Behavioral vs Gap (probe - N+P)
for s in model_summaries:
    ax2.scatter(s["behavioral"], s["gap_np"],
                c=family_colors[s["family"]], marker=family_markers[s["family"]],
                s=100, zorder=5, edgecolors='black', linewidth=0.5)
    ax2.annotate(s["model"], (s["behavioral"], s["gap_np"]),
                 fontsize=7, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')

ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_xlabel('Behavioral Accuracy', fontsize=12)
ax2.set_ylabel('Gap: Probe − Name+Params', fontsize=12)
ax2.set_title(f'Behavioral vs Planning Gap\nρ={rho_beh_gap:.3f}, p={p_beh_gap:.4f}', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# Legend
handles = [plt.scatter([], [], c=c, marker=m, s=80, edgecolors='black', linewidth=0.5, label=f)
           for f, (c, m) in zip(family_colors.keys(), 
                                 zip(family_colors.values(), family_markers.values()))]
ax2.legend(handles=handles, labels=list(family_colors.keys()), fontsize=10, loc='lower left')

plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig4_spearman.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig4_spearman.pdf", bbox_inches='tight')
plt.close()
print("Saved fig4_spearman")

# ================================================================
# FIGURE 5: ACROSTIC BEHAVIORAL
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
acr_models = list(acrostic_data.keys())
acr_vals = [acrostic_data[m] for m in acr_models]

bars = ax.bar(acr_models, acr_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#2ca02c', '#d62728', '#d62728'],
              edgecolor='black', linewidth=0.5)
ax.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, label='Chance (1/5 words)')
ax.set_ylabel('Acrostic Accuracy', fontsize=12)
ax.set_title('Acrostic Generation: All Models Near or Below Chance', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, acr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig5_acrostic.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig5_acrostic.pdf", bbox_inches='tight')
plt.close()
print("Saved fig5_acrostic")

# ================================================================
# FIGURE 6: SUMMARY TABLE (as figure for paper)
# ================================================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

columns = ['Model', 'Family', 'Params', 'Behav.', 'Probe', 'Name', 'N+P', 'Gap', 'p(FDR)']
rows = []
for s in model_summaries:
    data = results[s["model"]]
    best_l = s["best_layer"]
    ld = data["layers"][best_l]
    rows.append([
        s["model"], s["family"], s["params"],
        f'{s["behavioral"]:.0%}', f'{s["best_probe"]:.1%}',
        f'{ld["name"]:.1%}', f'{ld["np"]:.1%}',
        f'{s["gap_np"]:+.1%}', 'n.s.'
    ])

table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)

# Color header
for j in range(len(columns)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color gap column red (all negative)
for i in range(1, len(rows) + 1):
    table[i, 7].set_text_props(color='red', fontweight='bold')

ax.set_title('Summary: Name+Params Baseline Explains All Probing Signal (0/66 significant after FDR)',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("results/lookahead/final/figures/fig6_summary_table.png", dpi=300, bbox_inches='tight')
plt.savefig("results/lookahead/final/figures/fig6_summary_table.pdf", bbox_inches='tight')
plt.close()
print("Saved fig6_summary_table")

print("\n" + "=" * 50)
print("ALL FIGURES AND STATS SAVED")
print("=" * 50)
print(f"Stats: results/lookahead/final/stats/")
print(f"Figures: results/lookahead/final/figures/")
print(f"Total figures: 6 (PNG + PDF)")
