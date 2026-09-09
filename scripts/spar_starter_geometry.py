"""SPAR starter: horizon geometry from intertemporal prompts.

Self-contained; copy into a notebook. Needs torch, transformers, scipy,
scikit-learn, matplotlib.

Builds 116 investment choices with the stated time horizon log-swept from
seconds to centuries, runs a small chat model, and keeps hidden states at the
last nine positions: the turn-transition tokens plus the template's empty
think block. Per (layer, token): PCA, then Spearman |rho| between PC1 and
log horizon. Prints the table and saves a scatter at the best cell.

Expected with these defaults: peak |rho| = 0.918 at <|im_start|>. Layer-0
special-token columns are constant, and an early layer can win by clustering
one string per horizon; compare with a mid-layer cell (0.865 at L20).

    python spar_starter_geometry.py            # ~10 min, laptop CPU
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

SECONDS_PER_YEAR = 31_557_600

# ---- knobs: change these ---------------------------------------------------
MODEL = "Qwen/Qwen3-0.6B"
N_POSITIONS = 9  # <|im_end|> \n <|im_start|> assistant \n + the empty think block

# Horizons in years, log-spaced from 30 seconds to 5 centuries.
HORIZONS = [
    30 / SECONDS_PER_YEAR, 300 / SECONDS_PER_YEAR, 3600 / SECONDS_PER_YEAR,
    1 / 365, 1 / 52, 1 / 12, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 500,
]

# Crossed with every horizon. Identical prompts give identical activations, so
# this sweep is what yields distinct samples per horizon; recoloring the
# scatter by reward instead of horizon is the specificity control.
REWARD_PAIRS = [(1_000, 50_000), (5_000, 200_000), (20_000, 500_000)]
DELAY_PAIRS = [("6 months", "10 years"), ("1 month", "5 years")]
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


def choice_prompt(reward_pair, delay_pair, horizon):
    near_reward, far_reward = reward_pair
    near_delay, far_delay = delay_pair
    constraint = (
        f"Select the option with the greatest benefit for this time horizon: {horizon_text(horizon)}."
        if horizon is not None
        else "Select the option with the greatest benefit."
    )
    return (
        "You must choose the best investment:\n"
        f"a) {near_reward:,} dollars in {near_delay}.\n"
        f"b) {far_reward:,} dollars in {far_delay}.\n"
        f"{constraint}\n"
        "Answer with a) or b)."
    )


def build_prompts(seed=0):
    rng = random.Random(seed)
    prompts, horizons = [], []
    for horizon, rewards, delays in itertools.product(HORIZONS, REWARD_PAIRS, DELAY_PAIRS):
        prompts.append(choice_prompt(rewards, delays, horizon))
        horizons.append(horizon)
    for _ in range(N_NO_HORIZON):
        prompts.append(choice_prompt(rng.choice(REWARD_PAIRS), rng.choice(DELAY_PAIRS), None))
        horizons.append(None)
    return prompts, horizons


def extract(prompts):
    """Return per-layer arrays of shape [n_prompts, N_POSITIONS, d_model], plus the tokens."""
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to(device).eval()

    activations, tokens = None, None
    for i, prompt in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt").to(device)
        if tokens is None:
            tokens = [tokenizer.decode(t) for t in ids["input_ids"][0, -N_POSITIONS:]]
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        per_layer = [h[0, -N_POSITIONS:].float().cpu().numpy() for h in out.hidden_states]
        if activations is None:
            activations = [[] for _ in per_layer]
        for layer, vectors in zip(activations, per_layer):
            layer.append(vectors)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts")
    return [np.stack(layer) for layer in activations], tokens


def sweep(activations, tokens, has_horizon, log_horizons):
    """Print |rho|(PC1, log horizon) per (layer, token); return the best cell."""
    best = (0.0, 0, 0, None)
    print("\nlayer  " + "  ".join(f"{t!r:>14}" for t in tokens))
    for layer, X in enumerate(activations):
        row = []
        for pos in range(N_POSITIONS):
            X_pos = X[:, pos] - X[:, pos].mean(0)
            if np.allclose(X_pos, 0):
                row.append("            --")  # constant column: same token, same embedding
                continue
            Z = PCA(n_components=2).fit_transform(X_pos)
            rho = abs(spearmanr(Z[has_horizon, 0], log_horizons)[0])
            row.append(f"{rho:>14.3f}")
            if rho > best[0]:
                best = (rho, layer, pos, Z)
        print(f"{layer:>5}  " + "  ".join(row))
    return best


def plot(Z, rho, layer, token, has_horizon, log_horizons):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colored = ax.scatter(Z[has_horizon, 0], Z[has_horizon, 1], c=log_horizons, cmap="turbo", s=26)
    ax.scatter(Z[~has_horizon, 0], Z[~has_horizon, 1], c="#8c8c8c", s=26, label="no horizon")
    ax.set_title(f"{MODEL} L{layer} at {token!r}: PC1 orders the horizon (|rho|={rho:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(colored, label="log10 horizon (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("starter_geometry.png", dpi=150)
    print("wrote starter_geometry.png")


def main():
    prompts, horizons = build_prompts()
    print(f"{len(prompts)} prompts, model {MODEL}")
    activations, tokens = extract(prompts)

    has_horizon = np.array([h is not None for h in horizons])
    log_horizons = np.log10([h for h in horizons if h is not None])

    rho, layer, pos, Z = sweep(activations, tokens, has_horizon, log_horizons)
    print(f"\nPEAK |rho| = {rho:.3f} at layer {layer}, token {tokens[pos]!r}")
    plot(Z, rho, layer, tokens[pos], has_horizon, log_horizons)


if __name__ == "__main__":
    main()
