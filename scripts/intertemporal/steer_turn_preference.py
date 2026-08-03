#!/usr/bin/env python
"""CAA steering sweep for turn preference, with a random-vector control.

Method (paper Appendix on contrastive steering, with improvements):
- Build a CAA vector per sweep layer as the mean difference of resid_post
  activations at the last token of long-term-choice vs immediate-choice
  completions over the 300-pair implicit set. Unit-normalize per layer.
- Sweep fractional depths x alpha values, injecting the vector at all
  positions of the target layer (mode="add").
- Metric: forced-choice S = mean[logp(long) - logp(immediate)] on held-out
  explicit prompts, label-order counterbalanced (both orders reported).
- Control: identical sweep with a random unit vector of matched norm.
  Steering must beat the control to count.

Usage:
    uv run python scripts/intertemporal/steer_turn_preference.py \
        --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.base_schema import BaseSchema
from src.common.device_utils import clear_gpu_memory
from src.inference import ModelBackend, ModelRunner
from src.inference.interventions import Intervention, InterventionTarget

LAYER_FRACS = (0.50, 0.55, 0.58, 0.61, 0.65)
ALPHAS = (10.0, 20.0, 35.0, 50.0)

CAA_DATA = Path("data/raw/temporal_scope_AB_randomized/temporal_scope_implicit_expanded_300.json")
CAA_DATA_FALLBACK = Path(
    "data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json"
)
EVAL_DATA = Path("data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json")

LABEL_RE = re.compile(r"^\s*\(([AB])\)\s*")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAA steering sweep with random-vector control")
    parser.add_argument("--model", type=str, required=True, help="HF model name")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/steering/extreme_sweep"),
        help="Output directory (a per-model subdirectory is created inside)",
    )
    parser.add_argument("--n-eval", type=int, default=20, help="Held-out explicit prompts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the control vector")
    return parser.parse_args()


def parse_option(option: str) -> tuple[str, str]:
    """Split ' (A) some text' into ('A', 'some text')."""
    match = LABEL_RE.match(option)
    if match is None:
        raise ValueError(f"Option has no (A)/(B) label: {option!r}")
    return match.group(1), option[match.end():].strip()


def build_prompt(question: str, text_a: str, text_b: str) -> str:
    return f"{question}\n(A) {text_a}\n(B) {text_b}\n\nAnswer:"


@dataclass
class ScoredSequence:
    """A tokenized prompt+completion sequence with the completion start index."""

    token_ids: list[int]
    completion_start: int


@dataclass
class EvalItem:
    """One forced-choice eval prompt in one label order."""

    pair_id: int
    order: str  # "long_A" or "long_B"
    long_seq: ScoredSequence
    imm_seq: ScoredSequence


@dataclass
class SweepRow(BaseSchema):
    """One sweep configuration result."""

    layer_frac: float
    layer: int
    alpha: float
    S: float
    S_long_A: float
    S_long_B: float
    S_ctrl: float
    S_ctrl_long_A: float
    S_ctrl_long_B: float
    lift: float
    lift_ctrl: float
    beats_ctrl: bool


@dataclass
class OrderScores:
    """Forced-choice score, overall and per label order.

    per_prompt holds the raw per-prompt logp differences behind each mean,
    keyed by label order, in eval-item order. Means are computed from these
    same lists, so downstream bootstrap CIs use exactly the scored values.
    """

    S: float
    long_A: float
    long_B: float
    per_prompt: dict[str, list[float]] | None = None


def load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["pairs"]


def tokenize_scored(runner: ModelRunner, prompt: str, completion: str) -> ScoredSequence:
    """Tokenize chat-formatted prompt + completion, tracking the completion start."""
    formatted = runner.apply_chat_template(prompt) + runner.skip_thinking_prefix
    prompt_ids = runner.encode_ids(formatted)
    full_ids = runner.encode_ids(formatted + completion)
    # Guard against tokenizer merges at the boundary
    start = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        start += 1
    if start < len(prompt_ids) - 2:
        raise ValueError(f"Tokenizer merged prompt/completion boundary at {start}")
    return ScoredSequence(token_ids=full_ids, completion_start=start)


def build_caa_sequences(
    runner: ModelRunner, pairs: list[dict]
) -> tuple[list[ScoredSequence], list[ScoredSequence]]:
    """Chat-formatted prompt + chosen-option completion for both choices of each pair."""
    long_seqs: list[ScoredSequence] = []
    imm_seqs: list[ScoredSequence] = []
    for pair in pairs:
        imm_label, imm_text = parse_option(pair["immediate"])
        lt_label, lt_text = parse_option(pair["long_term"])
        if imm_label == lt_label:
            raise ValueError(f"Pair {pair['id']} has duplicate labels")
        text_a = imm_text if imm_label == "A" else lt_text
        text_b = imm_text if imm_label == "B" else lt_text
        prompt = build_prompt(pair["question"], text_a, text_b)
        long_seqs.append(tokenize_scored(runner, prompt, f" ({lt_label}) {lt_text}"))
        imm_seqs.append(tokenize_scored(runner, prompt, f" ({imm_label}) {imm_text}"))
    return long_seqs, imm_seqs


def extract_caa_vectors(
    runner: ModelRunner,
    long_seqs: list[ScoredSequence],
    imm_seqs: list[ScoredSequence],
    layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Unit-normalized mean-difference vectors (long - immediate) per layer."""
    hook_names = {layer: f"blocks.{layer}.hook_resid_post" for layer in layers}
    wanted = set(hook_names.values())

    def names_filter(name: str) -> bool:
        return name in wanted

    def mean_last_token(seqs: list[ScoredSequence]) -> dict[int, np.ndarray]:
        sums = {layer: np.zeros(runner.d_model, dtype=np.float64) for layer in layers}
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            trajs = runner.compute_trajectories_batch_with_intervention_and_cache(
                [s.token_ids for s in batch], intervention=None, names_filter=names_filter
            )
            for traj in trajs:
                for layer in layers:
                    act = traj.internals[hook_names[layer]][0, -1]
                    sums[layer] += act.float().cpu().numpy()
                traj.pop_heavy()
            clear_gpu_memory()
        return {layer: sums[layer] / len(seqs) for layer in layers}

    mean_long = mean_last_token(long_seqs)
    mean_imm = mean_last_token(imm_seqs)
    vectors: dict[int, np.ndarray] = {}
    for layer in layers:
        diff = (mean_long[layer] - mean_imm[layer]).astype(np.float32)
        norm = float(np.linalg.norm(diff))
        if norm == 0.0:
            raise ValueError(f"Zero CAA vector at layer {layer}")
        vectors[layer] = diff / norm
    return vectors


def build_eval_items(runner: ModelRunner, pairs: list[dict]) -> list[EvalItem]:
    """Both label orders for each held-out explicit pair."""
    items: list[EvalItem] = []
    for pair in pairs:
        _, imm_text = parse_option(pair["immediate"])
        _, lt_text = parse_option(pair["long_term"])
        for order in ("long_A", "long_B"):
            if order == "long_A":
                text_a, text_b = lt_text, imm_text
                long_label, imm_label = "A", "B"
            else:
                text_a, text_b = imm_text, lt_text
                long_label, imm_label = "B", "A"
            prompt = build_prompt(pair["question"], text_a, text_b)
            items.append(
                EvalItem(
                    pair_id=pair["id"],
                    order=order,
                    long_seq=tokenize_scored(runner, prompt, f" ({long_label}) {lt_text}"),
                    imm_seq=tokenize_scored(runner, prompt, f" ({imm_label}) {imm_text}"),
                )
            )
    return items


def score_items(
    runner: ModelRunner,
    items: list[EvalItem],
    intervention,
    batch_size: int,
) -> OrderScores:
    """Forced-choice S = mean[logp(long) - logp(immediate)], per label order."""
    seqs: list[ScoredSequence] = []
    for item in items:
        seqs.extend([item.long_seq, item.imm_seq])

    logps: list[float] = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        trajs = runner.compute_trajectories_batch_with_intervention(
            [s.token_ids for s in batch], intervention=intervention
        )
        for seq, traj in zip(batch, trajs):
            logps.append(float(sum(traj.logprobs[seq.completion_start :])))
            traj.pop_heavy()
    clear_gpu_memory()

    diffs_by_order: dict[str, list[float]] = {"long_A": [], "long_B": []}
    for idx, item in enumerate(items):
        diff = logps[2 * idx] - logps[2 * idx + 1]
        diffs_by_order[item.order].append(diff)
    long_a = float(np.mean(diffs_by_order["long_A"]))
    long_b = float(np.mean(diffs_by_order["long_B"]))
    return OrderScores(
        S=(long_a + long_b) / 2.0,
        long_A=long_a,
        long_B=long_b,
        per_prompt=diffs_by_order,
    )


def plot_heatmap(
    rows: list[SweepRow],
    layer_fracs: list[float],
    alphas: list[float],
    baseline: OrderScores,
    model_name: str,
    out_path: Path,
) -> None:
    """One diverging heatmap: S per (alpha, layer), control S annotated per cell."""
    grid = np.zeros((len(alphas), len(layer_fracs)))
    ctrl = np.zeros_like(grid)
    layer_of: dict[float, int] = {}
    for row in rows:
        i = alphas.index(row.alpha)
        j = layer_fracs.index(row.layer_frac)
        grid[i, j] = row.S
        ctrl[i, j] = row.S_ctrl
        layer_of[row.layer_frac] = row.layer

    vmax = max(abs(grid).max(), abs(baseline.S), 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(layer_fracs)))
    ax.set_xticklabels([f"{f:.2f}\n(L{layer_of[f]})" for f in layer_fracs])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel("Fractional depth (layer)")
    ax.set_ylabel("Steering alpha")
    ax.set_title(
        f"{model_name}: forced-choice S under CAA steering\n"
        f"baseline S = {baseline.S:.3f} (cells: S, control in parentheses)",
        fontsize=10,
    )
    for i in range(len(alphas)):
        for j in range(len(layer_fracs)):
            rel = abs(grid[i, j]) / vmax
            color = "white" if rel > 0.6 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}\n({ctrl[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("S = mean[logp(long) - logp(immediate)]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = get_args()
    model_short = args.model.split("/")[-1]
    out_dir = args.out_dir / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    caa_path = CAA_DATA if CAA_DATA.exists() else CAA_DATA_FALLBACK
    if not caa_path.exists():
        print(f"Error: no CAA dataset found at {CAA_DATA} or {CAA_DATA_FALLBACK}")
        return 1
    caa_pairs = load_pairs(caa_path)
    eval_pairs = sorted(load_pairs(EVAL_DATA), key=lambda p: p["id"])[: args.n_eval]

    # process_weights=False: TransformerLens advises no_processing at reduced
    # precision, and the fp32 processing pass peaks at 4-5x model size in host
    # RAM. CAA mean differences are invariant to the skipped reparametrization.
    runner = ModelRunner(
        args.model, backend=ModelBackend.TRANSFORMERLENS, process_weights=False
    )
    n_layers = runner.n_layers
    layer_fracs = list(LAYER_FRACS)
    frac_to_layer = {
        frac: min(n_layers - 1, max(0, int(round(frac * n_layers)))) for frac in layer_fracs
    }
    layers = sorted(set(frac_to_layer.values()))
    print(f"Sweep layers for {args.model} (n_layers={n_layers}): {frac_to_layer}")

    print(f"Building CAA sequences from {caa_path} ({len(caa_pairs)} pairs)...")
    long_seqs, imm_seqs = build_caa_sequences(runner, caa_pairs)
    print("Extracting CAA vectors...")
    vectors = extract_caa_vectors(runner, long_seqs, imm_seqs, layers, args.batch_size)
    np.savez(
        out_dir / "caa_vectors.npz",
        **{f"layer_{layer}": vec for layer, vec in vectors.items()},
    )

    rng = np.random.default_rng(args.seed)
    controls = {}
    for layer in layers:
        g = rng.standard_normal(runner.d_model).astype(np.float32)
        controls[layer] = g / np.linalg.norm(g)
    np.savez(
        out_dir / "control_vectors.npz",
        **{f"layer_{layer}": vec for layer, vec in controls.items()},
    )

    print(f"Building {len(eval_pairs)} held-out eval prompts (both label orders)...")
    items = build_eval_items(runner, eval_pairs)

    print("Scoring baseline (alpha=0)...")
    baseline = score_items(runner, items, intervention=None, batch_size=args.batch_size)
    print(
        f"  baseline S={baseline.S:.4f} "
        f"(long_A={baseline.long_A:.4f}, long_B={baseline.long_B:.4f})"
    )

    rows: list[SweepRow] = []
    for frac in layer_fracs:
        layer = frac_to_layer[frac]
        for alpha in ALPHAS:
            scores = {}
            for name, vec in (("steer", vectors[layer]), ("ctrl", controls[layer])):
                intervention = Intervention(
                    layer=layer,
                    mode="add",
                    values=(alpha * vec).astype(np.float32),
                    target=InterventionTarget.all(),
                    component="resid_post",
                )
                scores[name] = score_items(runner, items, intervention, args.batch_size)
            steer, ctrl = scores["steer"], scores["ctrl"]
            row = SweepRow(
                layer_frac=frac,
                layer=layer,
                alpha=alpha,
                S=steer.S,
                S_long_A=steer.long_A,
                S_long_B=steer.long_B,
                S_ctrl=ctrl.S,
                S_ctrl_long_A=ctrl.long_A,
                S_ctrl_long_B=ctrl.long_B,
                lift=steer.S - baseline.S,
                lift_ctrl=ctrl.S - baseline.S,
                beats_ctrl=steer.S > ctrl.S,
            )
            rows.append(row)
            print(
                f"  frac={frac:.2f} L{layer} alpha={alpha:g}: "
                f"S={row.S:.4f} S_ctrl={row.S_ctrl:.4f} lift={row.lift:+.4f} "
                f"beats_ctrl={row.beats_ctrl}"
            )

    csv_path = out_dir / "steering_sweep.csv"
    fieldnames = list(SweepRow.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())

    best = max(rows, key=lambda r: r.S)
    summary = {
        "model": args.model,
        "backend": "transformerlens",
        "process_weights": False,
        "dtype": str(runner.dtype),
        "n_layers": n_layers,
        "d_model": runner.d_model,
        "caa_dataset": str(caa_path),
        "n_caa_pairs": len(caa_pairs),
        "eval_dataset": str(EVAL_DATA),
        "n_eval_pairs": len(eval_pairs),
        "eval_pair_ids": [p["id"] for p in eval_pairs],
        "layer_fracs": layer_fracs,
        "frac_to_layer": {str(k): v for k, v in frac_to_layer.items()},
        "alphas": list(ALPHAS),
        "control_seed": args.seed,
        "baseline": {"S": baseline.S, "long_A": baseline.long_A, "long_B": baseline.long_B},
        "best": {"layer_frac": best.layer_frac, "layer": best.layer, "alpha": best.alpha,
                 "S": best.S, "S_ctrl": best.S_ctrl, "lift": best.lift},
        "rows": [row.to_dict() for row in rows],
    }
    with open(out_dir / "steering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig_path = out_dir / "steering_heatmap.png"
    plot_heatmap(rows, layer_fracs, list(ALPHAS), baseline, model_short, fig_path)

    print(f"\nWrote {csv_path}, {out_dir / 'steering_summary.json'}, {fig_path}")
    print(
        f"Best: frac={best.layer_frac} L{best.layer} alpha={best.alpha:g} "
        f"S={best.S:.4f} (ctrl {best.S_ctrl:.4f}, baseline {baseline.S:.4f})"
    )

    runner.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
