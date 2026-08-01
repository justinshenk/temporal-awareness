#!/usr/bin/env python3
"""Per-layer logistic probes for immediate vs long-term turn preference (App. G/U).

For every pair in the implicit A/B dataset, two prompts are built with the same
"question + chosen answer" template the steering pipeline uses (the answer
strings already carry their randomized " (A)/(B)" labels). One forward pass per
prompt caches resid_post at every layer; the probe reads the last prompt token.
Per layer, a StandardScaler + LogisticRegression(C=0.1, max_iter=2000) probe is
trained on an 80/20 pair-aware split (both members of a pair land on the same
side). Controls: 10x shuffled-label probes per layer (chance ~50%), and
zero-shot transfer of every layer's probe to the explicit 500 set.

Usage:
    uv run python scripts/intertemporal/probe_turn_preference.py
    uv run python scripts/intertemporal/probe_turn_preference.py \
        --models google/gemma-2-9b-it
    uv run python scripts/intertemporal/probe_turn_preference.py --plot-only
    uv run python scripts/intertemporal/probe_turn_preference.py --plot-only --upload
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformer_lens.loading_from_pretrained import get_official_model_name

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import ModelRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
SHORT_NAMES = {
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b_instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b_instruct",
    "google/gemma-2-9b-it": "gemma2_9b_it",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b_instruct_v03",
}
TRAIN_DATA = REPO_ROOT / "data/raw/temporal_scope_AB_randomized/temporal_scope_implicit_expanded_300.json"
TRAIN_DATA_FALLBACK = REPO_ROOT / "data/raw/refined_datasets/temporal_scope_implicit_expanded_500_debiased.json"
TRANSFER_DATA = REPO_ROOT / "data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json"

TEST_FRAC = 0.2
PROBE_C = 0.1
PROBE_MAX_ITER = 2000

# First four categorical slots of the validated reference palette (all-pairs safe).
MODEL_COLORS = {
    "qwen3_4b_instruct": "#2a78d6",
    "llama31_8b_instruct": "#eb6834",
    "gemma2_9b_it": "#1baf7a",
    "mistral_7b_instruct_v03": "#eda100",
}


# -------------------- Data --------------------


def load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    pairs = data["pairs"] if isinstance(data, dict) else data
    for pair in pairs:
        for key in ("question", "immediate", "long_term"):
            if key not in pair:
                raise ValueError(f"{path}: pair missing key {key!r}: {pair}")
    return pairs


def build_samples(pairs: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Two prompts per pair: question + chosen answer. Label 0=immediate, 1=long_term."""
    texts, labels, pair_ids = [], [], []
    for i, pair in enumerate(pairs):
        for label, key in ((0, "immediate"), (1, "long_term")):
            texts.append(pair["question"] + pair[key])
            labels.append(label)
            pair_ids.append(i)
    return texts, np.array(labels), np.array(pair_ids)


# -------------------- Extraction --------------------


def select_backend(model_name: str) -> ModelBackend:
    """TransformerLens where the architecture is supported, HuggingFace hooks otherwise.

    Both backends expose identical blocks.{L}.hook_resid_post cache keys through
    ModelRunner.run_with_cache.
    """
    if model_name == "Qwen/Qwen3-4B-Instruct-2507":
        return ModelBackend.TRANSFORMERLENS  # mapped to the Qwen3-4B config inside ModelRunner
    try:
        get_official_model_name(model_name)
        return ModelBackend.TRANSFORMERLENS
    except ValueError:
        return ModelBackend.HUGGINGFACE


def extract_last_token_resid(runner: ModelRunner, texts: list[str], tag: str) -> np.ndarray:
    """resid_post at the last prompt token, every layer, one forward pass per prompt.

    Returns float32 array [n_prompts, n_layers, d_model].
    """
    n_layers = runner.n_layers
    wanted = {f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)}
    names_filter = wanted.__contains__

    rows = []
    t0 = time.time()
    for i, text in enumerate(texts):
        _, cache = runner.run_with_cache(text, names_filter=names_filter)
        row = torch.stack(
            [cache[f"blocks.{layer}.hook_resid_post"][0, -1] for layer in range(n_layers)]
        )
        rows.append(row.float().cpu().numpy())
        del cache
        if (i + 1) % 100 == 0 or i + 1 == len(texts):
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{tag}] {i + 1}/{len(texts)} prompts ({rate:.1f}/s)", flush=True)
    return np.stack(rows)


# -------------------- Probing --------------------


def pair_aware_split(pair_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """80/20 split over pairs; both prompts of a pair land on the same side."""
    rng = np.random.default_rng(seed)
    unique_pairs = np.unique(pair_ids)
    perm = rng.permutation(unique_pairs)
    n_test = max(1, round(TEST_FRAC * len(unique_pairs)))
    test_pairs = set(perm[:n_test].tolist())
    is_test = np.array([pid in test_pairs for pid in pair_ids])
    return np.where(~is_test)[0], np.where(is_test)[0]


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=seed),
    )
    probe.fit(X_train, y_train)
    return probe


def probe_all_layers(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
    X_transfer: np.ndarray,
    y_transfer: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> list[dict]:
    """Per layer: test accuracy, mean shuffled-label accuracy, zero-shot transfer accuracy."""
    train_idx, test_idx = pair_aware_split(pair_ids, seed)
    n_layers = X.shape[1]
    rng = np.random.default_rng(seed + 1)
    results = []
    for layer in range(n_layers):
        Xl = X[:, layer, :]
        probe = fit_probe(Xl[train_idx], y[train_idx], seed)
        acc = probe.score(Xl[test_idx], y[test_idx])
        transfer_acc = probe.score(X_transfer[:, layer, :], y_transfer)

        shuffled_accs = []
        for _ in range(n_shuffles):
            y_shuffled = rng.permutation(y)
            shuffled_probe = fit_probe(Xl[train_idx], y_shuffled[train_idx], seed)
            shuffled_accs.append(shuffled_probe.score(Xl[test_idx], y_shuffled[test_idx]))
        acc_shuffled = float(np.mean(shuffled_accs))

        results.append(
            dict(layer=layer, acc=float(acc), acc_shuffled=acc_shuffled, transfer_acc=float(transfer_acc))
        )
        print(
            f"  layer {layer:2d}: acc={acc:.4f} shuffled={acc_shuffled:.4f} transfer={transfer_acc:.4f}",
            flush=True,
        )
    return results


# -------------------- Per-model driver --------------------


def run_model(
    model_name: str,
    train_pairs: list[dict],
    transfer_pairs: list[dict],
    out_dir: Path,
    n_shuffles: int,
    seed: int,
    device: str | None,
) -> None:
    short = SHORT_NAMES.get(model_name, model_name.split("/")[-1].lower().replace("-", "_").replace(".", ""))
    csv_path = out_dir / f"{short}.csv"
    t0 = time.time()

    backend = select_backend(model_name)
    print(f"\n=== {model_name} (backend={backend}) ===", flush=True)
    runner = ModelRunner(model_name, device=device, backend=backend)

    texts, y, pair_ids = build_samples(train_pairs)
    texts_tr, y_tr, _ = build_samples(transfer_pairs)
    X = extract_last_token_resid(runner, texts, f"{short}/train")
    X_transfer = extract_last_token_resid(runner, texts_tr, f"{short}/transfer")
    n_layers, d_model = runner.n_layers, runner.d_model

    runner.unload()
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = probe_all_layers(X, y, pair_ids, X_transfer, y_tr, n_shuffles, seed)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "acc", "acc_shuffled", "transfer_acc"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                dict(
                    layer=row["layer"],
                    acc=f"{row['acc']:.4f}",
                    acc_shuffled=f"{row['acc_shuffled']:.4f}",
                    transfer_acc=f"{row['transfer_acc']:.4f}",
                )
            )

    best = max(results, key=lambda r: r["acc"])
    meta = dict(
        model=model_name,
        short_name=short,
        backend=str(backend),
        n_layers=n_layers,
        d_model=d_model,
        n_train_pairs=len(train_pairs),
        n_transfer_pairs=len(transfer_pairs),
        probe=dict(C=PROBE_C, max_iter=PROBE_MAX_ITER, test_frac=TEST_FRAC, n_shuffles=n_shuffles, seed=seed),
        best_layer=best["layer"],
        best_acc=best["acc"],
        best_layer_acc_shuffled=best["acc_shuffled"],
        best_layer_transfer_acc=best["transfer_acc"],
        elapsed_sec=time.time() - t0,
    )
    with open(out_dir / f"{short}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{short}] best layer {best['layer']} acc={best['acc']:.4f} "
        f"shuffled={best['acc_shuffled']:.4f} transfer={best['transfer_acc']:.4f} "
        f"({meta['elapsed_sec']:.0f}s) -> {csv_path}",
        flush=True,
    )


# -------------------- Figure --------------------


def plot_overlay(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_offsets = [(0, 9), (0, -16), (18, 9), (-18, 9)]
    for series_idx, (short, color) in enumerate(MODEL_COLORS.items()):
        csv_path = out_dir / f"{short}.csv"
        if not csv_path.exists():
            print(f"[plot] missing {csv_path}, skipping", flush=True)
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        layers = np.array([int(r["layer"]) for r in rows])
        accs = np.array([float(r["acc"]) for r in rows])
        n_layers = len(layers)
        depth = (layers + 1) / n_layers
        ax.plot(depth, accs, color=color, linewidth=2, label=short)
        peak = int(np.argmax(accs))
        ax.plot(depth[peak], accs[peak], "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(
            f"L{layers[peak]} ({accs[peak]:.2f})",
            (depth[peak], accs[peak]),
            textcoords="offset points",
            xytext=annotation_offsets[series_idx % len(annotation_offsets)],
            ha="center",
            fontsize=8,
            color="#52514e",
        )
    ax.axhline(0.5, color="#b3b2ac", linewidth=1, linestyle="--", zorder=0)
    ax.text(0.005, 0.503, "chance", fontsize=8, color="#52514e")
    ax.set_xlabel("Fractional depth (layer / n_layers)")
    ax.set_ylabel("Probe test accuracy")
    ax.set_title("Turn preference probes: accuracy vs depth")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.grid(color="#ecebe6", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig_path = out_dir / "turn_preference_probe_accuracy.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {fig_path}", flush=True)
    return fig_path


# -------------------- Upload --------------------


def upload_results(out_dir: Path, repo_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    files = sorted(
        p for p in out_dir.iterdir() if p.suffix in (".csv", ".json", ".png") and p.is_file()
    )
    if not files:
        raise RuntimeError(f"nothing to upload in {out_dir}")
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=prefix,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.csv", "*.json", "*.png"],
        commit_message=f"Turn preference probes: {', '.join(p.name for p in files)}",
    )
    remote_paths = [f"{prefix}/{p.name}" for p in files]
    infos = api.get_paths_info(repo_id, remote_paths, repo_type="dataset")
    found = {info.path: info.size for info in infos}
    for p in files:
        rp = f"{prefix}/{p.name}"
        local_size = p.stat().st_size
        if rp not in found:
            raise RuntimeError(f"upload verification FAILED: {rp} missing on hub")
        if found[rp] != local_size:
            raise RuntimeError(
                f"upload verification FAILED: {rp} size {found[rp]} != local {local_size}"
            )
        print(f"[upload] verified {rp} ({found[rp]} bytes)", flush=True)


# -------------------- Main --------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out/probing/turn_preference")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--transfer-data", type=Path, default=TRANSFER_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shuffles", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if the model CSV exists")
    parser.add_argument("--plot-only", action="store_true", help="Only rebuild the overlay figure")
    parser.add_argument("--upload", action="store_true", help="Upload CSVs, meta and figure to the Hub")
    parser.add_argument("--hf-repo", default="unrulyabstractions/temporal-awareness")
    parser.add_argument("--hf-prefix", default="probing/turn_preference")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        train_path = args.train_data if args.train_data.exists() else TRAIN_DATA_FALLBACK
        if train_path != args.train_data:
            print(f"[data] {args.train_data} missing, falling back to {train_path}", flush=True)
        train_pairs = load_pairs(train_path)
        transfer_pairs = load_pairs(args.transfer_data)
        print(
            f"[data] train={train_path} ({len(train_pairs)} pairs) "
            f"transfer={args.transfer_data} ({len(transfer_pairs)} pairs)",
            flush=True,
        )
        for model_name in args.models:
            short = SHORT_NAMES.get(model_name, model_name.split("/")[-1].lower().replace("-", "_").replace(".", ""))
            if (args.out_dir / f"{short}.csv").exists() and not args.force:
                print(f"[skip] {args.out_dir / f'{short}.csv'} exists (use --force)", flush=True)
                continue
            run_model(
                model_name,
                train_pairs,
                transfer_pairs,
                args.out_dir,
                args.n_shuffles,
                args.seed,
                args.device,
            )

    plot_overlay(args.out_dir)

    if args.upload:
        upload_results(args.out_dir, args.hf_repo, args.hf_prefix)


if __name__ == "__main__":
    main()
