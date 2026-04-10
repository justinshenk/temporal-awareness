#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "output_root_relative": "results/qwen3_32b/question_only_probe_variations",
    "use_chat_template": True,
    "disable_thinking_trace": True,
    "selected_layers": [24, 28, 32, 36, 40, 44, 48],
    "split_random_state": 42,
    "explicit_train_fraction": 0.8,
    "implicit_train_fraction": 0.7,
    "strip_option_letters_for_probe_training": True,
    "normalize_probe_vectors": True,
    "probe_batch_size": 1,
    "whiten_reg": 1e-2,
    "require_cuda": True,
    "quick_mode": False,
    "explicit_path": None,
    "implicit_path": None,
}

FEATURE_SPECS = [
    {"name": "mean_answer_tokens", "pooling": "mean"},
    {"name": "last_answer_token", "pooling": "last"},
]

PROBE_FAMILIES = ["lr", "wlr", "mm", "wmm"]

STYLE_LAST_ONLY = {
    "lr": {"label": "LR", "color": "C0", "marker": "o"},
    "wlr": {"label": "WLR", "color": "C1", "marker": "o"},
    "mm": {"label": "MM", "color": "C2", "marker": "o"},
    "wmm": {"label": "WMM", "color": "C3", "marker": "o"},
}
STYLE_MEAN_ONLY = STYLE_LAST_ONLY
STYLE_COMPARISON = {
    "lr": {
        "last": {"color": "C0", "marker": "o", "linestyle": "-", "label": "LR, last completion token"},
        "mean": {"color": "C0", "marker": "o", "linestyle": "--", "label": "LR, mean completion tokens"},
    },
    "wlr": {
        "last": {"color": "C1", "marker": "o", "linestyle": "-", "label": "WLR, last completion token"},
        "mean": {"color": "C1", "marker": "o", "linestyle": "--", "label": "WLR, mean completion tokens"},
    },
    "mm": {
        "last": {"color": "C2", "marker": "o", "linestyle": "-", "label": "MM, last completion token"},
        "mean": {"color": "C2", "marker": "o", "linestyle": "--", "label": "MM, mean completion tokens"},
    },
    "wmm": {
        "last": {"color": "C3", "marker": "o", "linestyle": "-", "label": "WMM, last completion token"},
        "mean": {"color": "C3", "marker": "o", "linestyle": "--", "label": "WMM, mean completion tokens"},
    },
}


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / "pyproject.toml").exists() and (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root from current working directory.")


def pick_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + str([str(path) for path in paths]))


def load_pairs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "pairs" in data:
        return data.get("metadata", {}), data["pairs"]
    return {}, data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_cuda_runtime(*, require_cuda: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this run, but torch.cuda.is_available() is False. "
            "Refusing to fall back to CPU/MPS."
        )


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_option_letter(option_text):
    match = re.search(r"\(([ABab])\)", option_text or "")
    return match.group(1).upper() if match else None


def strip_option_label(option_text):
    return re.sub(r"^\s*\([ABab]\)\s*", "", option_text or "").strip()


def build_question_only_probe_prompt(question_text):
    question_text = (question_text or "").strip()
    if not question_text:
        raise ValueError("Question text is empty; cannot build question-only probe prompt.")
    return question_text


def get_pair_option_payload(pair):
    immediate_letter = extract_option_letter(pair["immediate"])
    long_term_letter = extract_option_letter(pair["long_term"])
    if not immediate_letter or not long_term_letter or immediate_letter == long_term_letter:
        raise ValueError(f"Could not resolve A/B option ordering for pair: {pair!r}")

    return {
        "candidate_immediate_text": strip_option_label(pair["immediate"]),
        "candidate_long_term_text": strip_option_label(pair["long_term"]),
    }


def build_teacher_forced_examples_from_pairs(pairs, strip_option_letters=True):
    examples = []
    labels = []
    for question_idx, pair in enumerate(pairs):
        option_payload = get_pair_option_payload(pair)
        prompt = build_question_only_probe_prompt(pair["question"])
        immediate_continuation = option_payload["candidate_immediate_text"]
        long_term_continuation = option_payload["candidate_long_term_text"]
        if not strip_option_letters:
            immediate_continuation = pair["immediate"]
            long_term_continuation = pair["long_term"]

        examples.append({
            "prompt": prompt,
            "continuation": immediate_continuation,
            "label": 0,
            "question_idx": int(question_idx),
        })
        labels.append(0)
        examples.append({
            "prompt": prompt,
            "continuation": long_term_continuation,
            "label": 1,
            "question_idx": int(question_idx),
        })
        labels.append(1)

    return examples, np.array(labels, dtype=np.int64)


def format_prompt_for_model(tokenizer, user_prompt, *, use_chat_template=True, disable_thinking_trace=True):
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not expose apply_chat_template, but use_chat_template=True was requested.")
        messages = [{"role": "user", "content": user_prompt}]
        if disable_thinking_trace:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return templated + "<think>\n</think>\n\n"
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def get_model_device(model):
    return next(model.parameters()).device


def assert_model_on_cuda(model) -> torch.device:
    model_device = get_model_device(model)
    if model_device.type != "cuda":
        raise RuntimeError(
            f"Model is not on CUDA. Expected CUDA execution, but model device is {model_device}."
        )
    return model_device


def move_batch_to_model_device(model, batch):
    model_device = assert_model_on_cuda(model)
    return {key: value.to(model_device) for key, value in batch.items()}


def extract_answer_token_activations_qwen(
    *,
    model,
    tokenizer,
    selected_layers,
    feature_specs,
    examples,
    use_chat_template=True,
    disable_thinking_trace=True,
    batch_size=1,
    dataset_name="dataset",
):
    activations = {
        feature_spec["name"]: {layer: [] for layer in selected_layers}
        for feature_spec in feature_specs
    }
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model_device = assert_model_on_cuda(model)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if model_device.type == "cuda"
        else nullcontext()
    )

    progress = tqdm(
        range(0, len(examples), batch_size),
        total=(len(examples) + batch_size - 1) // batch_size,
        desc=f"Extracting {dataset_name}",
        unit="batch",
    )

    for start in progress:
        batch_examples = examples[start:start + batch_size]
        prompt_ids_batch = []
        full_ids_batch = []
        seq_lengths = []
        answer_spans = []

        for example in batch_examples:
            model_prompt = format_prompt_for_model(
                tokenizer,
                example["prompt"],
                use_chat_template=use_chat_template,
                disable_thinking_trace=disable_thinking_trace,
            )
            prompt_ids = tokenizer(
                model_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            full_ids = tokenizer(
                model_prompt + example["continuation"],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            continuation_token_count = int(full_ids.shape[0] - prompt_ids.shape[0])
            if continuation_token_count <= 0:
                raise ValueError(f"Empty continuation for training example: {example!r}")
            prompt_ids_batch.append(prompt_ids)
            full_ids_batch.append(full_ids)
            seq_lengths.append(int(full_ids.shape[0]))

        max_seq_len = max(seq_lengths)
        input_ids = torch.full((len(batch_examples), max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_examples), max_seq_len), dtype=torch.long)

        for row_idx, (prompt_ids, seq) in enumerate(zip(prompt_ids_batch, full_ids_batch)):
            seq_len = int(seq.shape[0])
            answer_start = int(prompt_ids.shape[0])
            answer_end = seq_len
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1
            answer_spans.append((answer_start, answer_end))

        batch = move_batch_to_model_device(model, {"input_ids": input_ids, "attention_mask": attention_mask})
        batch_devices = {value.device.type for value in batch.values()}
        if batch_devices != {"cuda"}:
            raise RuntimeError(f"Tokenized batch is not fully on CUDA: {batch_devices}")

        with torch.inference_mode():
            with autocast_ctx:
                outputs = model(**batch, output_hidden_states=True, use_cache=False)

        for layer in selected_layers:
            hidden = outputs.hidden_states[layer + 1]
            pooled_mean_rows = []
            pooled_last_rows = []
            for row_idx, (answer_start, answer_end) in enumerate(answer_spans):
                answer_hidden = hidden[row_idx, answer_start:answer_end, :]
                pooled_mean_rows.append(answer_hidden.mean(dim=0).detach().float().cpu().numpy())
                pooled_last_rows.append(answer_hidden[-1, :].detach().float().cpu().numpy())
            activations["mean_answer_tokens"][layer].append(np.stack(pooled_mean_rows, axis=0))
            activations["last_answer_token"][layer].append(np.stack(pooled_last_rows, axis=0))

        del outputs
        clear_gpu_cache()

    for feature_name in activations:
        for layer in selected_layers:
            activations[feature_name][layer] = np.concatenate(
                activations[feature_name][layer],
                axis=0,
            ).astype(np.float32)

    return activations


def normalize_direction(direction):
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return direction.astype(np.float32), norm
    return (direction / norm).astype(np.float32), norm


def train_mm_probe(X_train, y_train):
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)
    return (mu1 - mu0).astype(np.float32)


def mm_predict(X, direction):
    scores = X @ direction
    return (scores > 0).astype(np.int64), scores


def fit_whitener(X_train, reg=1e-2):
    X_train = X_train.astype(np.float64, copy=False)
    mean_train = X_train.mean(axis=0)
    Xc = X_train - mean_train

    cov = np.cov(Xc, rowvar=False, bias=False)
    avg_var = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] > 0 else 1.0
    cov_reg = cov + (reg * avg_var) * np.eye(cov.shape[0], dtype=cov.dtype)

    precision = np.linalg.pinv(cov_reg)
    eigvals, eigvecs = np.linalg.eigh(cov_reg)
    eigvals = np.clip(eigvals, 1e-12, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    try:
        cond = float(np.linalg.cond(cov_reg))
    except Exception:
        cond = float("nan")

    return {
        "mean_train": mean_train.astype(np.float32),
        "precision": precision.astype(np.float32),
        "inv_sqrt": inv_sqrt.astype(np.float32),
        "reg": float(reg),
        "cov_reg_condition_number": cond,
    }


def apply_whitener(X, whitener):
    Xc = X.astype(np.float32, copy=False) - whitener["mean_train"]
    return Xc @ whitener["inv_sqrt"]


def train_whitened_mm_probe(X_train, y_train, reg=1e-2):
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)
    mm_direction = (mu1 - mu0).astype(np.float32)
    whitener = fit_whitener(X_train, reg=reg)
    effective_direction = (whitener["precision"] @ mm_direction).astype(np.float32)
    return {
        "mean_train": whitener["mean_train"],
        "effective_direction": effective_direction,
        "reg": whitener["reg"],
        "cov_reg_condition_number": whitener["cov_reg_condition_number"],
    }


def whitened_mm_predict(X, model):
    Xc = X - model["mean_train"]
    scores = Xc @ model["effective_direction"]
    return (scores > 0).astype(np.int64), scores


def split_question_indices(n_questions: int, train_fraction: float, random_state: int):
    question_indices = np.arange(n_questions, dtype=np.int64)
    train_idx, test_idx = train_test_split(
        question_indices,
        train_size=train_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return np.sort(train_idx.astype(np.int64)), np.sort(test_idx.astype(np.int64))


def question_indices_to_example_indices(question_idx):
    return np.sort(np.concatenate([
        2 * question_idx,
        2 * question_idx + 1,
    ]).astype(np.int64))


def add_line_only_legend(ax, *, loc="lower left", fontsize=8, ncol=1):
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([0], [0], color=handle.get_color(), linestyle=handle.get_linestyle(), linewidth=handle.get_linewidth())
        for handle in handles
    ]
    ax.legend(line_only_handles, labels, loc=loc, fontsize=fontsize, ncol=ncol)


def draw_pooling_only_figure(df, title_prefix, style_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "in_domain_holdout_acc", "In-domain held-out accuracy by layer"),
        (axes[1], "cross_domain_holdout_acc", "Cross-domain held-out accuracy by layer"),
    ]:
        for family in PROBE_FAMILIES:
            style = style_map[family]
            ax.plot(
                df["layer"],
                df[f"{family}_{metric}"],
                marker=style["marker"],
                linewidth=2,
                color=style["color"],
                label=style["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
        add_line_only_legend(ax, loc="lower left", fontsize=8)
    fig.suptitle(title_prefix, y=1.03)
    return fig


def draw_comparison_figure(last_df, mean_df, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "in_domain_holdout_acc", "In-domain held-out accuracy by layer"),
        (axes[1], "cross_domain_holdout_acc", "Cross-domain held-out accuracy by layer"),
    ]:
        for family in PROBE_FAMILIES:
            last_style = STYLE_COMPARISON[family]["last"]
            mean_style = STYLE_COMPARISON[family]["mean"]
            ax.plot(
                last_df["layer"],
                last_df[f"{family}_{metric}"],
                linewidth=2,
                marker=last_style["marker"],
                linestyle=last_style["linestyle"],
                color=last_style["color"],
                label=last_style["label"],
            )
            ax.plot(
                mean_df["layer"],
                mean_df[f"{family}_{metric}"],
                linewidth=2,
                marker=mean_style["marker"],
                linestyle=mean_style["linestyle"],
                color=mean_style["color"],
                label=mean_style["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
        add_line_only_legend(ax, loc="lower left", fontsize=8, ncol=2)
    fig.suptitle(title_prefix, y=1.03)
    return fig


def collect_family_metric_dict(*, lr_probe, wlr_effective_coef, wlr_effective_intercept, mm_direction, wmm_model, X, y):
    metrics = {}

    metrics["lr"] = float((lr_probe.predict(X) == y).mean())

    wlr_pred = ((X @ wlr_effective_coef) + wlr_effective_intercept > 0).astype(np.int64)
    metrics["wlr"] = float((wlr_pred == y).mean())

    mm_pred, _ = mm_predict(X, mm_direction)
    metrics["mm"] = float((mm_pred == y).mean())

    wmm_pred, _ = whitened_mm_predict(X, wmm_model)
    metrics["wmm"] = float((wmm_pred == y).mean())
    return metrics


def fit_probe_regime(
    *,
    selected_layers,
    normalize_probe_vectors,
    split_random_state,
    whiten_reg,
    feature_acts_by_domain,
    labels_by_domain,
    split_payloads,
    train_domain,
    train_dataset_name,
    train_dataset_label,
    progress_desc,
):
    rows = []
    artifact_rows = []
    other_domain = "implicit" if train_domain == "explicit" else "explicit"

    train_idx = split_payloads[train_domain]["train_example_idx"]
    layer_progress = tqdm(
        selected_layers,
        desc=progress_desc,
        unit="layer",
    )
    for layer in layer_progress:
        X_train = feature_acts_by_domain[train_domain][layer][train_idx]
        y_train = labels_by_domain[train_domain][train_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_probe = LogisticRegression(max_iter=1000, random_state=split_random_state)
            lr_probe.fit(X_train, y_train)

        whitener = fit_whitener(X_train, reg=whiten_reg)
        X_train_w = apply_whitener(X_train, whitener)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wlr_probe = LogisticRegression(max_iter=1000, random_state=split_random_state)
            wlr_probe.fit(X_train_w, y_train)

        mm_direction = train_mm_probe(X_train, y_train)
        if normalize_probe_vectors:
            mm_vector, mm_raw_norm = normalize_direction(mm_direction)
        else:
            mm_vector = mm_direction.astype(np.float32)
            mm_raw_norm = float(np.linalg.norm(mm_direction))

        wmm_model = train_whitened_mm_probe(X_train, y_train, reg=whiten_reg)
        if normalize_probe_vectors:
            wmm_vector, wmm_raw_norm = normalize_direction(wmm_model["effective_direction"])
        else:
            wmm_vector = wmm_model["effective_direction"].astype(np.float32)
            wmm_raw_norm = float(np.linalg.norm(wmm_model["effective_direction"]))

        lr_coef = lr_probe.coef_[0].astype(np.float32)
        lr_intercept = float(lr_probe.intercept_[0])

        wlr_whitened_coef = wlr_probe.coef_[0].astype(np.float32)
        wlr_effective_coef = (whitener["inv_sqrt"] @ wlr_whitened_coef).astype(np.float32)
        wlr_effective_intercept = float(
            wlr_probe.intercept_[0] - np.dot(whitener["mean_train"], wlr_effective_coef)
        )

        domain_metric_cache = {}
        for domain_name in ["explicit", "implicit"]:
            X_full = feature_acts_by_domain[domain_name][layer]
            y_full = labels_by_domain[domain_name]
            X_holdout = feature_acts_by_domain[domain_name][layer][split_payloads[domain_name]["test_example_idx"]]
            y_holdout = labels_by_domain[domain_name][split_payloads[domain_name]["test_example_idx"]]

            domain_metric_cache[(domain_name, "full")] = collect_family_metric_dict(
                lr_probe=lr_probe,
                wlr_effective_coef=wlr_effective_coef,
                wlr_effective_intercept=wlr_effective_intercept,
                mm_direction=mm_direction,
                wmm_model=wmm_model,
                X=X_full,
                y=y_full,
            )
            domain_metric_cache[(domain_name, "holdout")] = collect_family_metric_dict(
                lr_probe=lr_probe,
                wlr_effective_coef=wlr_effective_coef,
                wlr_effective_intercept=wlr_effective_intercept,
                mm_direction=mm_direction,
                wmm_model=wmm_model,
                X=X_holdout,
                y=y_holdout,
            )

        train_metrics = collect_family_metric_dict(
            lr_probe=lr_probe,
            wlr_effective_coef=wlr_effective_coef,
            wlr_effective_intercept=wlr_effective_intercept,
            mm_direction=mm_direction,
            wmm_model=wmm_model,
            X=X_train,
            y=y_train,
        )

        row = {
            "train_dataset": train_dataset_name,
            "train_dataset_label": train_dataset_label,
            "train_domain": train_domain,
            "cross_domain": other_domain,
            "layer": int(layer),
            "train_size": int(len(train_idx)),
            "explicit_train_question_count": int(len(split_payloads["explicit"]["train_question_idx"])),
            "explicit_test_question_count": int(len(split_payloads["explicit"]["test_question_idx"])),
            "implicit_train_question_count": int(len(split_payloads["implicit"]["train_question_idx"])),
            "implicit_test_question_count": int(len(split_payloads["implicit"]["test_question_idx"])),
            "whitener_cov_reg_condition_number": float(whitener["cov_reg_condition_number"]),
            "wmm_reg": float(wmm_model["reg"]),
            "mm_raw_norm": float(mm_raw_norm),
            "mm_vector_norm": float(np.linalg.norm(mm_vector)),
            "wmm_raw_norm": float(wmm_raw_norm),
            "wmm_vector_norm": float(np.linalg.norm(wmm_vector)),
        }

        for family in PROBE_FAMILIES:
            row[f"{family}_train_acc"] = float(train_metrics[family])
            row[f"{family}_explicit_holdout_acc"] = float(domain_metric_cache[("explicit", "holdout")][family])
            row[f"{family}_implicit_holdout_acc"] = float(domain_metric_cache[("implicit", "holdout")][family])
            row[f"{family}_explicit_full_acc"] = float(domain_metric_cache[("explicit", "full")][family])
            row[f"{family}_implicit_full_acc"] = float(domain_metric_cache[("implicit", "full")][family])
            row[f"{family}_in_domain_holdout_acc"] = float(domain_metric_cache[(train_domain, "holdout")][family])
            row[f"{family}_cross_domain_holdout_acc"] = float(domain_metric_cache[(other_domain, "holdout")][family])
            row[f"{family}_in_domain_full_acc"] = float(domain_metric_cache[(train_domain, "full")][family])
            row[f"{family}_cross_domain_full_acc"] = float(domain_metric_cache[(other_domain, "full")][family])

        rows.append(row)
        artifact_rows.append({
            "layer": int(layer),
            "lr_coef": lr_coef,
            "lr_intercept": np.float32(lr_intercept),
            "wlr_effective_coef": wlr_effective_coef,
            "wlr_effective_intercept": np.float32(wlr_effective_intercept),
            "mm_raw_direction": mm_direction.astype(np.float32),
            "mm_probe_vector": mm_vector.astype(np.float32),
            "wmm_effective_direction": wmm_model["effective_direction"].astype(np.float32),
            "wmm_probe_vector": wmm_vector.astype(np.float32),
            "wmm_mean_train": wmm_model["mean_train"].astype(np.float32),
        })

        layer_progress.set_postfix(
            layer=int(layer),
            train_acc=f"{row['mm_train_acc']:.3f}",
            holdout=f"{row['mm_in_domain_holdout_acc']:.3f}",
            cross=f"{row['mm_cross_domain_holdout_acc']:.3f}",
        )
        clear_gpu_cache()

    rows_df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    artifact_rows = sorted(artifact_rows, key=lambda row: row["layer"])
    return rows_df, artifact_rows


def summarize_probe_metrics(metrics_df):
    summary_rows = []
    for train_dataset, dataset_df in metrics_df.groupby("train_dataset"):
        for feature_name, feature_df in dataset_df.groupby("feature_name"):
            summary_row = {
                "train_dataset": train_dataset,
                "train_dataset_label": str(feature_df["train_dataset_label"].iloc[0]),
                "train_domain": str(feature_df["train_domain"].iloc[0]),
                "cross_domain": str(feature_df["cross_domain"].iloc[0]),
                "feature_name": feature_name,
            }
            for family in PROBE_FAMILIES:
                for metric_name in [
                    "in_domain_holdout_acc",
                    "cross_domain_holdout_acc",
                    "explicit_full_acc",
                    "implicit_full_acc",
                ]:
                    best_idx = feature_df[f"{family}_{metric_name}"].idxmax()
                    summary_row[f"best_{family}_{metric_name}_layer"] = int(feature_df.loc[best_idx, "layer"])
                    summary_row[f"best_{family}_{metric_name}"] = float(feature_df.loc[best_idx, f"{family}_{metric_name}"])
            summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)


def materialize_artifact_arrays(*, selected_layers, hidden_size, regime_names, regime_labels, feature_names, artifact_store):
    artifact_shape = (len(regime_names), len(feature_names), len(selected_layers), hidden_size)
    scalar_shape = (len(regime_names), len(feature_names), len(selected_layers))

    lr_coef = np.zeros(artifact_shape, dtype=np.float32)
    lr_intercept = np.zeros(scalar_shape, dtype=np.float32)
    wlr_effective_coef = np.zeros(artifact_shape, dtype=np.float32)
    wlr_effective_intercept = np.zeros(scalar_shape, dtype=np.float32)
    mm_raw_directions = np.zeros(artifact_shape, dtype=np.float32)
    mm_probe_vectors = np.zeros(artifact_shape, dtype=np.float32)
    wmm_effective_directions = np.zeros(artifact_shape, dtype=np.float32)
    wmm_probe_vectors = np.zeros(artifact_shape, dtype=np.float32)
    wmm_mean_train = np.zeros(artifact_shape, dtype=np.float32)

    for regime_idx, regime_name in enumerate(regime_names):
        for feature_idx, feature_name in enumerate(feature_names):
            artifact_rows = artifact_store[regime_name][feature_name]
            for layer_idx, layer in enumerate(selected_layers):
                row = artifact_rows[layer_idx]
                if row["layer"] != layer:
                    raise ValueError(
                        f"Artifact row order mismatch for {regime_name} / {feature_name}: "
                        f"expected layer {layer}, got {row['layer']}"
                    )
                lr_coef[regime_idx, feature_idx, layer_idx, :] = row["lr_coef"]
                lr_intercept[regime_idx, feature_idx, layer_idx] = row["lr_intercept"]
                wlr_effective_coef[regime_idx, feature_idx, layer_idx, :] = row["wlr_effective_coef"]
                wlr_effective_intercept[regime_idx, feature_idx, layer_idx] = row["wlr_effective_intercept"]
                mm_raw_directions[regime_idx, feature_idx, layer_idx, :] = row["mm_raw_direction"]
                mm_probe_vectors[regime_idx, feature_idx, layer_idx, :] = row["mm_probe_vector"]
                wmm_effective_directions[regime_idx, feature_idx, layer_idx, :] = row["wmm_effective_direction"]
                wmm_probe_vectors[regime_idx, feature_idx, layer_idx, :] = row["wmm_probe_vector"]
                wmm_mean_train[regime_idx, feature_idx, layer_idx, :] = row["wmm_mean_train"]

    return {
        "lr_coef": lr_coef,
        "lr_intercept": lr_intercept,
        "wlr_effective_coef": wlr_effective_coef,
        "wlr_effective_intercept": wlr_effective_intercept,
        "mm_raw_directions": mm_raw_directions,
        "mm_probe_vectors": mm_probe_vectors,
        "wmm_effective_directions": wmm_effective_directions,
        "wmm_probe_vectors": wmm_probe_vectors,
        "wmm_mean_train": wmm_mean_train,
        "layers": np.asarray(selected_layers, dtype=np.int64),
        "train_regimes": np.asarray(regime_names),
        "train_regime_labels": np.asarray(regime_labels),
        "feature_names": np.asarray(feature_names),
    }


def save_stage_outputs(
    *,
    save_dir: Path,
    run_id: str,
    stage_suffix: str,
    stage_label: str,
    model_name: str,
    selected_layers,
    feature_names,
    all_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    artifact_store,
    regime_results,
    regime_names,
    regime_labels,
    hidden_size: int,
    split_payloads,
    strip_option_letters_for_probe_training: bool,
    use_chat_template: bool,
    disable_thinking_trace: bool,
    normalize_probe_vectors: bool,
    whiten_reg: float,
    probe_batch_size: int,
    explicit_path: Path,
    implicit_path: Path,
    explicit_hash: str,
    implicit_hash: str,
):
    suffix = f"_{stage_suffix}" if stage_suffix else ""

    metrics_path = save_dir / f"qwen3_32b_question_only_probe_metrics_{run_id}{suffix}.csv"
    summary_path = save_dir / f"qwen3_32b_question_only_probe_summary_{run_id}{suffix}.csv"
    artifact_path = save_dir / f"qwen3_32b_question_only_probe_artifacts_{run_id}{suffix}.npz"
    metadata_path = save_dir / f"qwen3_32b_question_only_probe_metadata_{run_id}{suffix}.json"
    figure_index_path = save_dir / f"qwen3_32b_question_only_probe_figures_{run_id}{suffix}.csv"

    all_metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    arrays = materialize_artifact_arrays(
        selected_layers=selected_layers,
        hidden_size=hidden_size,
        regime_names=regime_names,
        regime_labels=regime_labels,
        feature_names=feature_names,
        artifact_store=artifact_store,
    )
    np.savez_compressed(
        artifact_path,
        train_regimes=arrays["train_regimes"],
        train_regime_labels=arrays["train_regime_labels"],
        feature_names=arrays["feature_names"],
        layers=arrays["layers"],
        explicit_train_question_indices=np.asarray(split_payloads["explicit"]["train_question_idx"], dtype=np.int64),
        explicit_test_question_indices=np.asarray(split_payloads["explicit"]["test_question_idx"], dtype=np.int64),
        explicit_train_example_indices=np.asarray(split_payloads["explicit"]["train_example_idx"], dtype=np.int64),
        explicit_test_example_indices=np.asarray(split_payloads["explicit"]["test_example_idx"], dtype=np.int64),
        implicit_train_question_indices=np.asarray(split_payloads["implicit"]["train_question_idx"], dtype=np.int64),
        implicit_test_question_indices=np.asarray(split_payloads["implicit"]["test_question_idx"], dtype=np.int64),
        implicit_train_example_indices=np.asarray(split_payloads["implicit"]["train_example_idx"], dtype=np.int64),
        implicit_test_example_indices=np.asarray(split_payloads["implicit"]["test_example_idx"], dtype=np.int64),
        lr_coef=arrays["lr_coef"],
        lr_intercept=arrays["lr_intercept"],
        wlr_effective_coef=arrays["wlr_effective_coef"],
        wlr_effective_intercept=arrays["wlr_effective_intercept"],
        mm_raw_directions=arrays["mm_raw_directions"],
        mm_probe_vectors=arrays["mm_probe_vectors"],
        wmm_effective_directions=arrays["wmm_effective_directions"],
        wmm_probe_vectors=arrays["wmm_probe_vectors"],
        wmm_mean_train=arrays["wmm_mean_train"],
    )

    metadata_payload = {
        "run_id": run_id,
        "stage_label": stage_label,
        "stage_suffix": stage_suffix,
        "model_name": model_name,
        "selected_layers": list(selected_layers),
        "feature_names": list(feature_names),
        "train_regimes": list(regime_names),
        "train_regime_labels": list(regime_labels),
        "explicit_split_granularity": "question",
        "explicit_split_strategy": "question_level_80_20",
        "implicit_split_granularity": "question",
        "implicit_split_strategy": "question_level_70_30",
        "explicit_train_question_count": int(len(split_payloads["explicit"]["train_question_idx"])),
        "explicit_test_question_count": int(len(split_payloads["explicit"]["test_question_idx"])),
        "implicit_train_question_count": int(len(split_payloads["implicit"]["train_question_idx"])),
        "implicit_test_question_count": int(len(split_payloads["implicit"]["test_question_idx"])),
        "strip_option_letters_for_probe_training": bool(strip_option_letters_for_probe_training),
        "use_chat_template": bool(use_chat_template),
        "probe_prompt_use_chat_template": bool(use_chat_template),
        "disable_thinking_trace": bool(disable_thinking_trace),
        "probe_prompt_disable_thinking_trace": bool(disable_thinking_trace),
        "normalize_probe_vectors": bool(normalize_probe_vectors),
        "whiten_reg": float(whiten_reg),
        "probe_batch_size": int(probe_batch_size),
        "prompt_family": "question_only_teacher_forced_answers",
        "prompt_format_description": "question-only prompt; no options shown to model; immediate/long-term continuations teacher-forced with A/B stripped",
        "explicit_expanded_path": str(explicit_path),
        "implicit_expanded_path": str(implicit_path),
        "explicit_expanded_sha256": explicit_hash,
        "implicit_expanded_sha256": implicit_hash,
        "artifact_format_version": 1,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")

    figure_rows = []
    for regime_name in regime_names:
        safe_regime_name = regime_name.replace("/", "_")
        last_df = regime_results[regime_name]["last_answer_token"]
        mean_df = regime_results[regime_name]["mean_answer_tokens"]

        last_fig = draw_pooling_only_figure(
            last_df,
            f"{model_name}: {regime_name} | last completion-token probes | {stage_label}",
            STYLE_LAST_ONLY,
        )
        mean_fig = draw_pooling_only_figure(
            mean_df,
            f"{model_name}: {regime_name} | mean completion-token probes | {stage_label}",
            STYLE_MEAN_ONLY,
        )
        comparison_fig = draw_comparison_figure(
            last_df,
            mean_df,
            f"{model_name}: {regime_name} | pooling-method comparison | {stage_label}",
        )

        last_fig_path = save_dir / f"{safe_regime_name}_last_answer_token_{run_id}{suffix}.png"
        mean_fig_path = save_dir / f"{safe_regime_name}_mean_answer_tokens_{run_id}{suffix}.png"
        comparison_fig_path = save_dir / f"{safe_regime_name}_comparison_{run_id}{suffix}.png"

        last_fig.savefig(last_fig_path, dpi=200, bbox_inches="tight")
        mean_fig.savefig(mean_fig_path, dpi=200, bbox_inches="tight")
        comparison_fig.savefig(comparison_fig_path, dpi=200, bbox_inches="tight")
        plt.close(last_fig)
        plt.close(mean_fig)
        plt.close(comparison_fig)

        figure_rows.extend([
            {"train_dataset": regime_name, "figure_type": "last_answer_token", "path": str(last_fig_path)},
            {"train_dataset": regime_name, "figure_type": "mean_answer_tokens", "path": str(mean_fig_path)},
            {"train_dataset": regime_name, "figure_type": "comparison", "path": str(comparison_fig_path)},
        ])

    figure_df = pd.DataFrame(figure_rows)
    figure_df.to_csv(figure_index_path, index=False)

    print(f"[{stage_label}] Saved metrics  :", metrics_path)
    print(f"[{stage_label}] Saved summary  :", summary_path)
    print(f"[{stage_label}] Saved artifacts:", artifact_path)
    print(f"[{stage_label}] Saved metadata :", metadata_path)
    print(f"[{stage_label}] Saved figures  :", figure_index_path)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "figure_index_path": figure_index_path,
    }


def run_experiment(config_overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    root = find_repo_root(Path.cwd())
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    require_cuda_runtime(require_cuda=bool(cfg["require_cuda"]))

    output_root = root / cfg["output_root_relative"]
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    save_dir = output_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    explicit_path = (
        Path(cfg["explicit_path"]).expanduser().resolve()
        if cfg.get("explicit_path")
        else pick_first_existing([
            root / "data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json",
            root / "data/raw/temporal_scope/temporal_scope_explicit_expanded_500.json",
            root / "data/raw/temporal_scope_explicit_expanded_500.json",
        ])
    )
    implicit_path = (
        Path(cfg["implicit_path"]).expanduser().resolve()
        if cfg.get("implicit_path")
        else pick_first_existing([
            root / "data/raw/temporal_scope_AB_randomized/temporal_scope_implicit_expanded_300.json",
            root / "data/raw/temporal_scope/temporal_scope_implicit_expanded_300.json",
            root / "data/raw/temporal_scope_implicit_expanded_300.json",
        ])
    )

    exp_meta, explicit_pairs = load_pairs(explicit_path)
    imp_meta, implicit_pairs = load_pairs(implicit_path)
    explicit_hash = sha256(explicit_path)
    implicit_hash = sha256(implicit_path)

    print("Repo root:", root)
    print("Save dir:", save_dir)
    print("Expanded explicit dataset:", explicit_path)
    print("Expanded implicit dataset:", implicit_path)
    print("Expanded explicit metadata:", exp_meta)
    print("Expanded implicit metadata:", imp_meta)
    print("Expanded explicit questions:", len(explicit_pairs))
    print("Expanded implicit questions:", len(implicit_pairs))

    strip_option_letters = bool(cfg["strip_option_letters_for_probe_training"])
    explicit_examples, y_explicit = build_teacher_forced_examples_from_pairs(
        explicit_pairs,
        strip_option_letters=strip_option_letters,
    )
    implicit_examples, y_implicit = build_teacher_forced_examples_from_pairs(
        implicit_pairs,
        strip_option_letters=strip_option_letters,
    )

    print("Expanded explicit samples:", len(y_explicit), "| class balance:", np.bincount(y_explicit))
    print("Expanded implicit samples:", len(y_implicit), "| class balance:", np.bincount(y_implicit))
    print("Prompt example:", repr(explicit_examples[0]["prompt"]))
    print("Continuation example:", repr(explicit_examples[0]["continuation"]))

    selected_layers = [int(layer) for layer in cfg["selected_layers"]]
    if bool(cfg.get("quick_mode", False)):
        warnings.warn("quick_mode=True: reducing selected layers for faster debugging.")
        selected_layers = selected_layers[:2]

    model_name = str(cfg["model_name"])
    split_random_state = int(cfg["split_random_state"])
    explicit_train_fraction = float(cfg["explicit_train_fraction"])
    implicit_train_fraction = float(cfg["implicit_train_fraction"])
    use_chat_template = bool(cfg["use_chat_template"])
    disable_thinking_trace = bool(cfg["disable_thinking_trace"])
    normalize_probe_vectors = bool(cfg["normalize_probe_vectors"])
    probe_batch_size = int(cfg["probe_batch_size"])
    whiten_reg = float(cfg["whiten_reg"])

    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = model.to("cuda")
    model.eval()
    model_device = assert_model_on_cuda(model)

    n_layers = len(model.model.layers)
    hidden_size = int(model.config.hidden_size)
    invalid_layers = [layer for layer in selected_layers if layer < 0 or layer >= n_layers]
    if invalid_layers:
        raise ValueError(f"selected_layers contains invalid layers: {invalid_layers}; n_layers={n_layers}")

    print(
        "[cuda] confirmed GPU execution:",
        f"device={model_device}",
        f"| name={torch.cuda.get_device_name(model_device)}",
        f"| total_memory_gb={torch.cuda.get_device_properties(model_device).total_memory / (1024 ** 3):.1f}",
    )
    print("Loaded", model_name, "| n_layers =", n_layers, "| hidden_size =", hidden_size)
    print("Selected layers:", selected_layers)

    explicit_train_question_idx, explicit_test_question_idx = split_question_indices(
        len(explicit_pairs),
        explicit_train_fraction,
        split_random_state,
    )
    implicit_train_question_idx, implicit_test_question_idx = split_question_indices(
        len(implicit_pairs),
        implicit_train_fraction,
        split_random_state,
    )

    split_payloads = {
        "explicit": {
            "train_question_idx": explicit_train_question_idx,
            "test_question_idx": explicit_test_question_idx,
            "train_example_idx": question_indices_to_example_indices(explicit_train_question_idx),
            "test_example_idx": question_indices_to_example_indices(explicit_test_question_idx),
        },
        "implicit": {
            "train_question_idx": implicit_train_question_idx,
            "test_question_idx": implicit_test_question_idx,
            "train_example_idx": question_indices_to_example_indices(implicit_train_question_idx),
            "test_example_idx": question_indices_to_example_indices(implicit_test_question_idx),
        },
    }

    for domain_name, payload in split_payloads.items():
        if np.intersect1d(payload["train_example_idx"], payload["test_example_idx"]).size != 0:
            raise ValueError(f"{domain_name} train/test example indices overlap after question-level split.")

    print(
        "Explicit question split:",
        f"train = {len(explicit_train_question_idx)}",
        f"| test = {len(explicit_test_question_idx)}",
    )
    print(
        "Implicit question split:",
        f"train = {len(implicit_train_question_idx)}",
        f"| test = {len(implicit_test_question_idx)}",
    )

    print("Extracting explicit Qwen activations...")
    explicit_feature_acts = extract_answer_token_activations_qwen(
        model=model,
        tokenizer=tokenizer,
        selected_layers=selected_layers,
        feature_specs=FEATURE_SPECS,
        examples=explicit_examples,
        use_chat_template=use_chat_template,
        disable_thinking_trace=disable_thinking_trace,
        batch_size=probe_batch_size,
        dataset_name="explicit_expanded",
    )
    print("Extracting implicit Qwen activations...")
    implicit_feature_acts = extract_answer_token_activations_qwen(
        model=model,
        tokenizer=tokenizer,
        selected_layers=selected_layers,
        feature_specs=FEATURE_SPECS,
        examples=implicit_examples,
        use_chat_template=use_chat_template,
        disable_thinking_trace=disable_thinking_trace,
        batch_size=probe_batch_size,
        dataset_name="implicit_expanded",
    )

    labels_by_domain = {
        "explicit": y_explicit,
        "implicit": y_implicit,
    }

    regime_specs = [
        {
            "name": "explicit_train_only",
            "label": "explicit train only",
            "train_domain": "explicit",
            "stage_suffix": "explicit_only_checkpoint",
            "stage_label": "explicit_only_checkpoint",
        },
        {
            "name": "implicit_train_only",
            "label": "implicit train only",
            "train_domain": "implicit",
            "stage_suffix": "implicit_only_checkpoint",
            "stage_label": "implicit_only_checkpoint",
        },
    ]

    regime_results = {}
    artifact_store = {}
    all_metric_parts = []
    stage_outputs = {}

    for regime in regime_specs:
        regime_name = regime["name"]
        regime_results[regime_name] = {}
        artifact_store[regime_name] = {}
        stage_metric_parts = []
        for feature_spec in FEATURE_SPECS:
            feature_name = feature_spec["name"]
            print(f"Training {feature_name} probes for regime={regime_name}")
            metrics_df, artifact_rows = fit_probe_regime(
                selected_layers=selected_layers,
                normalize_probe_vectors=normalize_probe_vectors,
                split_random_state=split_random_state,
                whiten_reg=whiten_reg,
                feature_acts_by_domain={
                    "explicit": explicit_feature_acts[feature_name],
                    "implicit": implicit_feature_acts[feature_name],
                },
                labels_by_domain=labels_by_domain,
                split_payloads=split_payloads,
                train_domain=regime["train_domain"],
                train_dataset_name=regime_name,
                train_dataset_label=regime["label"],
                progress_desc=f"Training {regime_name} | {feature_name}",
            )
            metrics_df.insert(2, "feature_name", feature_name)
            regime_results[regime_name][feature_name] = metrics_df.copy()
            artifact_store[regime_name][feature_name] = artifact_rows
            stage_metric_parts.append(metrics_df)
            all_metric_parts.append(metrics_df)

        stage_metrics_df = pd.concat(stage_metric_parts, ignore_index=True).sort_values(
            ["train_dataset", "feature_name", "layer"]
        ).reset_index(drop=True)
        stage_summary_df = summarize_probe_metrics(stage_metrics_df)
        stage_outputs[regime_name] = save_stage_outputs(
            save_dir=save_dir,
            run_id=run_id,
            stage_suffix=regime["stage_suffix"],
            stage_label=regime["stage_label"],
            model_name=model_name,
            selected_layers=selected_layers,
            feature_names=[spec["name"] for spec in FEATURE_SPECS],
            all_metrics_df=stage_metrics_df,
            summary_df=stage_summary_df,
            artifact_store={regime_name: artifact_store[regime_name]},
            regime_results={regime_name: regime_results[regime_name]},
            regime_names=[regime_name],
            regime_labels=[regime["label"]],
            hidden_size=hidden_size,
            split_payloads=split_payloads,
            strip_option_letters_for_probe_training=strip_option_letters,
            use_chat_template=use_chat_template,
            disable_thinking_trace=disable_thinking_trace,
            normalize_probe_vectors=normalize_probe_vectors,
            whiten_reg=whiten_reg,
            probe_batch_size=probe_batch_size,
            explicit_path=explicit_path,
            implicit_path=implicit_path,
            explicit_hash=explicit_hash,
            implicit_hash=implicit_hash,
        )

    all_metrics_df = pd.concat(all_metric_parts, ignore_index=True).sort_values(
        ["train_dataset", "feature_name", "layer"]
    ).reset_index(drop=True)
    summary_df = summarize_probe_metrics(all_metrics_df)
    final_outputs = save_stage_outputs(
        save_dir=save_dir,
        run_id=run_id,
        stage_suffix="",
        stage_label="final_all_regimes",
        model_name=model_name,
        selected_layers=selected_layers,
        feature_names=[spec["name"] for spec in FEATURE_SPECS],
        all_metrics_df=all_metrics_df,
        summary_df=summary_df,
        artifact_store=artifact_store,
        regime_results=regime_results,
        regime_names=[regime["name"] for regime in regime_specs],
        regime_labels=[regime["label"] for regime in regime_specs],
        hidden_size=hidden_size,
        split_payloads=split_payloads,
        strip_option_letters_for_probe_training=strip_option_letters,
        use_chat_template=use_chat_template,
        disable_thinking_trace=disable_thinking_trace,
        normalize_probe_vectors=normalize_probe_vectors,
        whiten_reg=whiten_reg,
        probe_batch_size=probe_batch_size,
        explicit_path=explicit_path,
        implicit_path=implicit_path,
        explicit_hash=explicit_hash,
        implicit_hash=implicit_hash,
    )

    run_config_path = save_dir / f"qwen3_32b_question_only_probe_run_config_{run_id}.json"
    run_config_payload = {
        "run_id": run_id,
        "save_dir": str(save_dir),
        "config": cfg,
        "explicit_dataset_path": str(explicit_path),
        "implicit_dataset_path": str(implicit_path),
        "explicit_dataset_sha256": explicit_hash,
        "implicit_dataset_sha256": implicit_hash,
        "explicit_stage_artifact": str(stage_outputs["explicit_train_only"]["artifact_path"]),
        "implicit_stage_artifact": str(stage_outputs["implicit_train_only"]["artifact_path"]),
        "final_artifact": str(final_outputs["artifact_path"]),
    }
    run_config_path.write_text(json.dumps(run_config_payload, indent=2) + "\n", encoding="utf-8")
    print("Saved run config:", run_config_path)

    return {
        "run_id": run_id,
        "save_dir": save_dir,
        "final_metrics_path": final_outputs["metrics_path"],
        "final_summary_path": final_outputs["summary_path"],
        "final_artifact_path": final_outputs["artifact_path"],
        "run_config_path": run_config_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train question-only LR/MM/WLR/WMM probe variants for Qwen3-32B on explicit and implicit temporal datasets."
    )
    parser.add_argument("--explicit-path", type=str, default=None, help="Optional explicit dataset JSON path.")
    parser.add_argument("--implicit-path", type=str, default=None, help="Optional implicit dataset JSON path.")
    parser.add_argument("--probe-batch-size", type=int, default=None, help="Optional extraction batch size override.")
    parser.add_argument("--quick-mode", action="store_true", help="Run only the first two selected layers for faster debugging.")
    parser.add_argument("--output-root-relative", type=str, default=None, help="Optional output dir relative to repo root.")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.explicit_path is not None:
        overrides["explicit_path"] = args.explicit_path
    if args.implicit_path is not None:
        overrides["implicit_path"] = args.implicit_path
    if args.probe_batch_size is not None:
        overrides["probe_batch_size"] = int(args.probe_batch_size)
    if args.output_root_relative is not None:
        overrides["output_root_relative"] = args.output_root_relative
    if args.quick_mode:
        overrides["quick_mode"] = True

    result = run_experiment(overrides)
    print("Run complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
