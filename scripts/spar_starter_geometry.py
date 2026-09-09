"""SPAR starter: horizon geometry from intertemporal prompts.

Self-contained; copy into a notebook. Needs torch, transformers, scipy,
scikit-learn, matplotlib.

Builds 116 investment choices with the stated time horizon log-swept from
seconds to centuries. For each prompt the model greedily generates its answer,
then one forward pass keeps hidden states at every turn-transition token (the
template suffix, including its empty think block) and at the generated
response tokens through the answer. Per (layer, token): PCA, then Spearman
|rho| between PC1 and log horizon. Prints the table and saves one scatter per
token at the best display layer, colored by horizon.

Expected with these defaults: peak |rho| = 0.955 at a turn newline (L12), a
display layer of 10, and the ordering visible in every panel, including the
answer token and the model's own closing <|im_end|>. Layer-0 special-token
columns are constant, and an early layer can win a single cell by clustering
one string per horizon; the display layer maximizes the MEAN |rho| across
tokens instead. If the answer token varies with the choice (it is starred),
structure there is the choice, not the horizon.

After the geometry, the same bank measures behavior: temporal reasoning
(picking the only option that can deliver by the deadline), order stability,
and label stability, each printed as a share. Measured with these defaults:
reasoning 100%, order stability 0%, label stability 10%. Read those three
together: the model answers a) every time, and the deliverable option sits in
slot a) in the base order, so the perfect reasoning score is position bias.
The paper's instrument counterbalances order for exactly this reason; average
the base and swapped runs and the reasoning share lands at chance.

    python spar_starter_geometry.py            # ~25 min, laptop CPU
"""

import itertools
import random
from collections import Counter

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
MODEL = "Qwen/Qwen3.5-0.8B"
N_POSITIONS = 9   # turn suffix: <|im_end|> \n <|im_start|> assistant \n + empty think block
N_RESPONSE = 6    # generated tokens to keep: covers "I choose: a)"

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
DELAY_YEARS = {"6 months": 0.5, "10 years": 10.0, "1 month": 1 / 12, "5 years": 5.0}
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


def choice_prompt(reward_pair, delay_pair, horizon, labels=("a", "b"), swap=False):
    """The base prompt; swap=True exchanges option order, labels relabels them."""
    near = f"{reward_pair[0]:,} dollars in {delay_pair[0]}."
    far = f"{reward_pair[1]:,} dollars in {delay_pair[1]}."
    first, second = (far, near) if swap else (near, far)
    constraint = (
        f"Select the option with the greatest benefit for this time horizon: {horizon_text(horizon)}."
        if horizon is not None
        else "Select the option with the greatest benefit."
    )
    return (
        "You must choose the best investment:\n"
        f"{labels[0]}) {first}\n"
        f"{labels[1]}) {second}\n"
        f"{constraint}\n"
        f"Answer with {labels[0]}) or {labels[1]})."
    )


def build_prompts(seed=0):
    rng = random.Random(seed)
    records = []
    for horizon, rewards, delays in itertools.product(HORIZONS, REWARD_PAIRS, DELAY_PAIRS):
        records.append({"prompt": choice_prompt(rewards, delays, horizon),
                        "rewards": rewards, "delays": delays, "horizon": horizon})
    for _ in range(N_NO_HORIZON):
        rewards, delays = rng.choice(REWARD_PAIRS), rng.choice(DELAY_PAIRS)
        records.append({"prompt": choice_prompt(rewards, delays, None),
                        "rewards": rewards, "delays": delays, "horizon": None})
    return records


def load_model():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to(device).eval()
    return tokenizer, model, device


def generate_answer(tokenizer, model, device, prompt):
    """Greedy answer text for one prompt (no hidden states)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        full = model.generate(
            **ids, max_new_tokens=N_RESPONSE, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(full[0, ids["input_ids"].shape[1]:])


def parse_choice(answer_text, labels=("a", "b")):
    """0 for the first label, 1 for the second, None if neither appears first."""
    hits = {i: answer_text.find(f"{lab})") for i, lab in enumerate(labels)}
    hits = {i: pos for i, pos in hits.items() if pos != -1}
    return min(hits, key=hits.get) if hits else None


def extract(tokenizer, model, device, prompts):
    """Generate each answer, then keep hidden states at the turn tokens and the
    response tokens. Returns per-layer arrays [n_prompts, n_positions, d_model],
    one token label per position (majority label; '*' marks positions whose
    token varies across prompts, such as the answer token), and each prompt's
    generated answer text."""
    n_positions = N_POSITIONS + N_RESPONSE
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    activations, label_counts = None, [Counter() for _ in range(n_positions)]
    turn_len = N_RESPONSE  # shortest generated turn (through its closing token)
    answers = []
    for i, prompt in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = ids["input_ids"].shape[1]
        with torch.no_grad():
            full = model.generate(
                **ids, max_new_tokens=N_RESPONSE, min_new_tokens=N_RESPONSE,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
            out = model(input_ids=full, output_hidden_states=True)
        response = full[0, prompt_len:].tolist()
        answers.append(tokenizer.decode(full[0, prompt_len:]))
        if end_id in response:
            turn_len = min(turn_len, response.index(end_id) + 1)
        keep = slice(prompt_len - N_POSITIONS, prompt_len + N_RESPONSE)
        for pos, tok_id in enumerate(full[0, keep]):
            label_counts[pos][tokenizer.decode(tok_id)] += 1
        per_layer = [h[0, keep].float().cpu().numpy() for h in out.hidden_states]
        if activations is None:
            activations = [[] for _ in per_layer]
        for layer, vectors in zip(activations, per_layer):
            layer.append(vectors)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts")

    # Drop positions past the model's own turn close: they belong to a next
    # turn the model hallucinates, not to this answer.
    n_keep = N_POSITIONS + turn_len
    print(f"answers: {dict(label_counts[N_POSITIONS])}  (turn closes after {turn_len} tokens)")
    tokens = [c.most_common(1)[0][0] + ("*" if len(c) > 1 else "")
              for c in label_counts[:n_keep]]
    return [np.stack(layer)[:, :n_keep] for layer in activations], tokens, answers


def sweep(activations, tokens, has_horizon, log_horizons):
    """Print |rho|(PC1, log horizon) per (layer, token). Returns rho[layer][pos]
    (nan where the column is constant) and the embeddings Z[layer][pos]."""
    n_positions = len(tokens)
    rho = np.full((len(activations), n_positions), np.nan)
    Z_all = [[None] * n_positions for _ in activations]
    print("\nlayer  " + "  ".join(f"{t!r:>10}" for t in tokens))
    for layer, X in enumerate(activations):
        row = []
        for pos in range(n_positions):
            X_pos = X[:, pos] - X[:, pos].mean(0)
            if np.allclose(X_pos, 0):
                row.append("        --")  # constant column: same token, same embedding
                continue
            Z = PCA(n_components=2).fit_transform(X_pos)
            rho[layer, pos] = abs(spearmanr(Z[has_horizon, 0], log_horizons)[0])
            Z_all[layer][pos] = Z
            row.append(f"{rho[layer, pos]:>10.3f}")
        print(f"{layer:>5}  " + "  ".join(row))
    return rho, Z_all


def plot(display_layer, rho, Z_all, tokens, has_horizon, log_horizons):
    """One scatter per token at the display layer, colored by horizon."""
    n_positions = len(tokens)
    cols = 3
    rows = -(-n_positions // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows))
    for pos in range(rows * cols):
        ax = axes.flat[pos]
        if pos >= n_positions or Z_all[display_layer][pos] is None:
            ax.axis("off")
            continue
        Z = Z_all[display_layer][pos]
        ax.scatter(Z[has_horizon, 0], Z[has_horizon, 1], c=log_horizons, cmap="turbo", s=14)
        ax.scatter(Z[~has_horizon, 0], Z[~has_horizon, 1], c="#8c8c8c", s=14)
        region = "turn" if pos < N_POSITIONS else "response"
        if pos == N_POSITIONS:
            region = "response, answer"
        ax.set_title(f"{tokens[pos]!r} ({region})  |rho|={rho[display_layer, pos]:.2f}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{MODEL}, layer {display_layer}: PC1 vs horizon at every position "
                 "(gray = no horizon; * = token varies, e.g. the answer)", fontsize=11)
    fig.tight_layout()
    fig.savefig("starter_geometry.png", dpi=150)
    print("wrote starter_geometry.png")


def behavior(tokenizer, model, device, records, base_answers):
    """The paper's three behavioral measures, on this prompt bank.

    Temporal reasoning: among prompts whose horizon lets only the near option
    deliver in time, the share choosing the near option. Order stability: the
    share whose chosen CONTENT survives swapping the option order. Label
    stability: the share whose choice survives relabeling a/b as 1/2."""
    reasoning_hits, reasoning_n = 0, 0
    order_same, order_n = 0, 0
    label_same, label_n = 0, 0
    for i, rec in enumerate(records):
        base = parse_choice(base_answers[i])
        if base is None:
            continue
        near_delay = DELAY_YEARS[rec["delays"][0]]
        far_delay = DELAY_YEARS[rec["delays"][1]]
        h = rec["horizon"]
        if h is not None and near_delay <= h < far_delay:
            reasoning_n += 1
            reasoning_hits += base == 0  # only the near option delivers in time
        swapped = parse_choice(generate_answer(
            tokenizer, model, device,
            choice_prompt(rec["rewards"], rec["delays"], h, swap=True)))
        if swapped is not None:
            order_n += 1
            order_same += (1 - swapped) == base  # same content, opposite slot
        relabeled = parse_choice(generate_answer(
            tokenizer, model, device,
            choice_prompt(rec["rewards"], rec["delays"], h, labels=("1", "2"))),
            labels=("1", "2"))
        if relabeled is not None:
            label_n += 1
            label_same += relabeled == base
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)} behavior prompts")
    print("\nbehavior:")
    print(f"  temporal reasoning  {reasoning_hits}/{reasoning_n}"
          f"  ({reasoning_hits / max(reasoning_n, 1):.0%} pick the only option that delivers in time)")
    print(f"  order stability     {order_same}/{order_n}"
          f"  ({order_same / max(order_n, 1):.0%} keep their choice when the options swap places)")
    print(f"  label stability     {label_same}/{label_n}"
          f"  ({label_same / max(label_n, 1):.0%} keep their choice when a/b becomes 1/2)")


def main():
    records = build_prompts()
    print(f"{len(records)} prompts, model {MODEL}")
    tokenizer, model, device = load_model()
    prompts = [r["prompt"] for r in records]
    horizons = [r["horizon"] for r in records]
    activations, tokens, answers = extract(tokenizer, model, device, prompts)

    has_horizon = np.array([h is not None for h in horizons])
    log_horizons = np.log10([h for h in horizons if h is not None])

    rho, Z_all = sweep(activations, tokens, has_horizon, log_horizons)
    peak = np.unravel_index(np.nanargmax(rho), rho.shape)
    print(f"\nPEAK |rho| = {rho[peak]:.3f} at layer {peak[0]}, token {tokens[peak[1]]!r}")
    with np.errstate(invalid="ignore"):
        mean_rho = np.array([np.nanmean(r) if not np.all(np.isnan(r)) else -np.inf
                             for r in rho])
    display_layer = int(np.argmax(mean_rho))
    print(f"display layer (best mean |rho| across tokens): {display_layer}")
    plot(display_layer, rho, Z_all, tokens, has_horizon, log_horizons)
    behavior(tokenizer, model, device, records, answers)


if __name__ == "__main__":
    main()
