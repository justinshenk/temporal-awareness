"""SPAR starter: extract activations for intertemporal prompts, look at the geometry.

Self-contained on purpose: no imports from this repo, so you can copy this one
file into a notebook. Needs torch, transformers, scipy, scikit-learn, matplotlib.

What it does, start to finish:
  1. Builds 116 intertemporal-choice prompts. Every prompt offers the same kind
     of tradeoff (small reward soon vs large reward later) and states a TIME
     HORIZON drawn from a log grid running seconds to centuries. A few prompts
     state no horizon at all.
  2. Runs a small chat model over each prompt and keeps the residual stream
     (hidden_states) at the last five prompt positions. Because we apply the
     chat template with add_generation_prompt=True, those positions are the
     turn-transition tokens (<|im_end|> \n <|im_start|> assistant \n) plus the
     empty think block this template appends, and the turn tokens are where the
     paper finds the horizon laid out as an ordinal gradient.
  3. At every (layer, token): PCA to 2 components, then Spearman-correlate the
     first component with log(horizon). Prints the layer-by-token table and
     saves a scatter at the best cell, colored by horizon.

Measured result with these defaults: peak |rho| = 0.918 at <|im_start|>, and
0.865 at the newline after assistant at layer 20. Two things worth discussing
with the result in hand: layer 0 columns are constant (same token, same
embedding), and an early layer can top the table because each horizon string
forms its own tight cluster there rather than a continuous manifold. Compare
the early-layer and mid-layer scatters before deciding which story to trust.

    python spar_starter_geometry.py            # ~10 min on a laptop, no GPU needed
"""

import itertools
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
N_POSITIONS = 9  # covers <|im_end|> \n <|im_start|> assistant \n and the empty think block
SECONDS_PER_YEAR = 31_557_600

# Horizons in years, log-spaced from 30 seconds to 5 centuries.
HORIZONS = [
    30 / SECONDS_PER_YEAR, 300 / SECONDS_PER_YEAR, 3600 / SECONDS_PER_YEAR,
    1 / 365, 1 / 52, 1 / 12, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 500,
]
N_NO_HORIZON = 20

def horizon_text(years):
    if years < 1 / 8760:
        return f"{round(years * SECONDS_PER_YEAR)} seconds"
    if years < 1 / 365:
        return f"{round(years * 8760)} hours"
    if years < 1 / 12:
        return f"{round(years * 365)} days"
    if years < 1:
        return f"{round(years * 12)} months"
    return f"{round(years)} years"

def build_prompts(seed=0):
    rng = random.Random(seed)
    rewards = [(1_000, 50_000), (5_000, 200_000), (20_000, 500_000)]
    delays = [("6 months", "10 years"), ("1 month", "5 years")]
    prompts, horizons = [], []
    for h, (r1, r2), (d1, d2) in itertools.product(HORIZONS, rewards, delays):
        prompts.append(
            "You must choose the best investment:\n"
            f"a) {r1:,} dollars in {d1}.\n"
            f"b) {r2:,} dollars in {d2}.\n"
            f"Select the option with the greatest benefit for this time horizon: {horizon_text(h)}.\n"
            "Answer with a) or b)."
        )
        horizons.append(h)
    for _ in range(N_NO_HORIZON):
        (r1, r2), (d1, d2) = rng.choice(rewards), rng.choice(delays)
        prompts.append(
            "You must choose the best investment:\n"
            f"a) {r1:,} dollars in {d1}.\n"
            f"b) {r2:,} dollars in {d2}.\n"
            "Select the option with the greatest benefit.\n"
            "Answer with a) or b)."
        )
        horizons.append(None)
    return prompts, horizons

def extract(prompts):
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.to(device).eval()
    acts, tokens = None, None
    for i, p in enumerate(prompts):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(device)
        if tokens is None:
            tokens = [tok.decode(t) for t in ids["input_ids"][0, -N_POSITIONS:]]
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # one vector per (layer, turn token)
        grabbed = [h[0, -N_POSITIONS:].float().cpu().numpy() for h in out.hidden_states]
        if acts is None:
            acts = [[] for _ in grabbed]
        for layer, vecs in zip(acts, grabbed):
            layer.append(vecs)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts")
    return [np.stack(a) for a in acts], tokens

def main():
    prompts, horizons = build_prompts()
    print(f"{len(prompts)} prompts, model {MODEL}")
    acts, tokens = extract(prompts)

    has_h = np.array([h is not None for h in horizons])
    log_h = np.log10([h for h in horizons if h is not None])

    best = (0.0, 0, 0, None)
    print("\nlayer  " + "  ".join(f"{t!r:>14}" for t in tokens))
    for layer, X in enumerate(acts):
        row = []
        for pos in range(N_POSITIONS):
            Xp = X[:, pos] - X[:, pos].mean(0)
            if np.allclose(Xp, 0):
                row.append("            --")
                continue
            Z = PCA(n_components=2).fit_transform(Xp)
            rho = abs(spearmanr(Z[has_h, 0], log_h)[0])
            row.append(f"{rho:>14.3f}")
            if rho > best[0]:
                best = (rho, layer, pos, Z)
        print(f"{layer:>5}  " + "  ".join(row))
    rho, layer, pos, Z = best
    print(f"\nPEAK |rho| = {rho:.3f} at layer {layer}, token {tokens[pos]!r}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[has_h, 0], Z[has_h, 1], c=log_h, cmap="turbo", s=26)
    ax.scatter(Z[~has_h, 0], Z[~has_h, 1], c="#8c8c8c", s=26, label="no horizon")
    ax.set_title(f"{MODEL} L{layer} at {tokens[pos]!r}: PC1 orders the horizon (|rho|={rho:.2f})")
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    fig.colorbar(sc, label="log10 horizon (years)"), ax.legend()
    fig.tight_layout()
    fig.savefig("starter_geometry.png", dpi=150)
    print("wrote starter_geometry.png")

if __name__ == "__main__":
    main()
