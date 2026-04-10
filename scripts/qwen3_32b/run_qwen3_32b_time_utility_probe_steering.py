#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION = 1


DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "use_chat_template": True,
    "disable_thinking_trace": True,
    "x_now": 100,
    "y_values": [100, 101, 110, 150, 200, 500, 1000, 10000],
    "t_values": ["1 hour", "8 hours", "tomorrow", "week", "month", "year", "2 years", "3 years", "5 years", "10 years"],
    "n_repeats": 1,
    "max_new_tokens": 128,
    "require_cuda": True,
    "artifact_search_roots": None,
    "artifact_path": None,
    "metadata_path": None,
    "train_regime": "explicit_train_only",
    "feature_name": "mean_answer_tokens",
    "vector_key": "mm_probe_vectors",
    "steering_layer": 40,
    "steering_conditions": [
        {"condition": "steer_long_term_plus8", "signed_strength": 8.0},
        {"condition": "steer_immediate_minus8", "signed_strength": -8.0},
        {"condition": "steer_long_term_plus16", "signed_strength": 16.0},
        {"condition": "steer_immediate_minus16", "signed_strength": -16.0},
        {"condition": "steer_long_term_plus32", "signed_strength": 32.0},
        {"condition": "steer_immediate_minus32", "signed_strength": -32.0},
        {"condition": "steer_long_term_plus64", "signed_strength": 64.0},
        {"condition": "steer_immediate_minus64", "signed_strength": -64.0},
        {"condition": "steer_long_term_plus128", "signed_strength": 128.0},
        {"condition": "steer_immediate_minus128", "signed_strength": -128.0},
        {"condition": "steer_long_term_plus256", "signed_strength": 256.0},
        {"condition": "steer_immediate_minus256", "signed_strength": -256.0},
        {"condition": "steer_long_term_plus512", "signed_strength": 512.0},
        {"condition": "steer_immediate_minus512", "signed_strength": -512.0},
    ],
    "include_baseline_condition": True,
    "patch_prompt_last_only": True,
    "patch_generation_tokens": True,
    "output_root_relative": "results/qwen3_32b/time_utility_experiment_probe_steered",
    "run_id": None,
}

THINKING_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from current working directory.")


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def require_cuda_runtime(*, require_cuda: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this run, but torch.cuda.is_available() is False. "
            "Refusing to fall back to CPU/MPS."
        )


def make_model_slug(model_name: str) -> str:
    model_leaf = (model_name or "").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "_", model_leaf.lower()).strip("_")


def pick_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + str([str(path) for path in paths]))


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def decode_str_array(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def resolve_metadata_path_for_artifact(artifact_path: Path, metadata_candidates):
    candidate = artifact_path.with_name(
        artifact_path.name.replace("_probe_artifacts_", "_probe_metadata_").replace(".npz", ".json")
    )
    if candidate.exists():
        return candidate
    for metadata_path in metadata_candidates:
        if metadata_path.stem.replace("_probe_metadata_", "_probe_artifacts_") == artifact_path.stem:
            return metadata_path
    return None


def load_probe_metadata(metadata_path):
    if metadata_path is None or not Path(metadata_path).exists():
        return None
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def metadata_is_compatible(metadata):
    if not metadata:
        return False
    if int(metadata.get("artifact_format_version", 0)) < REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION:
        return False
    if metadata.get("prompt_family") != "question_only_teacher_forced_answers":
        return False
    if metadata.get("explicit_split_granularity") != "question":
        return False
    if metadata.get("implicit_split_granularity") != "question":
        return False
    if not bool(metadata.get("probe_prompt_use_chat_template", metadata.get("use_chat_template", False))):
        return False
    if not bool(metadata.get("probe_prompt_disable_thinking_trace", metadata.get("disable_thinking_trace", False))):
        return False
    return True


def artifact_contains_train_regime(artifact_path: Path, required_train_regime: str | None) -> bool:
    if required_train_regime is None:
        return True
    try:
        with np.load(artifact_path) as bundle:
            train_regimes = decode_str_array(bundle["train_regimes"])
    except Exception:
        return False
    return required_train_regime in train_regimes


def locate_latest_probe_artifacts(
    search_roots,
    artifact_path_override: str | None = None,
    metadata_path_override: str | None = None,
    required_train_regime: str | None = None,
):
    if artifact_path_override is not None:
        artifact_path = Path(artifact_path_override).expanduser().resolve()
        if metadata_path_override is not None:
            metadata_path = Path(metadata_path_override).expanduser().resolve()
        else:
            metadata_path = resolve_metadata_path_for_artifact(artifact_path, [])
        if not artifact_contains_train_regime(artifact_path, required_train_regime):
            raise ValueError(
                f"Artifact {artifact_path} does not contain requested train_regime={required_train_regime!r}."
            )
        return artifact_path, metadata_path

    artifact_candidates = []
    metadata_candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        artifact_candidates.extend(sorted(root.rglob("qwen3_32b_question_only_probe_artifacts_*.npz")))
        metadata_candidates.extend(sorted(root.rglob("qwen3_32b_question_only_probe_metadata_*.json")))

    artifact_candidates = unique_paths(artifact_candidates)
    metadata_candidates = unique_paths(metadata_candidates)

    if not artifact_candidates:
        roots_text = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            "Could not find any Qwen3-32B probe-artifact bundles under: "
            f"{roots_text}. Set artifact_path explicitly if needed."
        )

    compatible_candidates = []
    for artifact_path in reversed(artifact_candidates):
        metadata_path = resolve_metadata_path_for_artifact(artifact_path, metadata_candidates)
        metadata = load_probe_metadata(metadata_path)
        if metadata_is_compatible(metadata) and artifact_contains_train_regime(artifact_path, required_train_regime):
            compatible_candidates.append((artifact_path, metadata_path))
    if not compatible_candidates:
        roots_text = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            "Found Qwen3-32B probe artifacts, but none were compatible with the current question-only + question-split + chat-template format "
            f"(required artifact_format_version >= {REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION}). Search roots: {roots_text}"
        )
    return compatible_candidates[0]


def load_probe_vector(*, artifact_path: Path, metadata_path: Path | None, train_regime: str, feature_name: str, vector_key: str, layer: int) -> dict[str, object]:
    bundle = np.load(artifact_path)
    try:
        metadata = load_probe_metadata(metadata_path) or {}
        if not metadata_is_compatible(metadata):
            raise ValueError(
                f"Artifact metadata at {metadata_path} is not compatible with the current question-only + question-split + chat-template format."
            )

        train_regimes = decode_str_array(bundle["train_regimes"])
        feature_names = decode_str_array(bundle["feature_names"])
        available_layers = bundle["layers"].astype(int)

        regime_matches = [idx for idx, name in enumerate(train_regimes) if name == train_regime]
        feature_matches = [idx for idx, name in enumerate(feature_names) if name == feature_name]
        layer_matches = np.where(available_layers == int(layer))[0]

        if len(regime_matches) != 1:
            raise ValueError(f"Train regime {train_regime!r} not found exactly once. Available: {train_regimes}")
        if len(feature_matches) != 1:
            raise ValueError(f"Feature name {feature_name!r} not found exactly once. Available: {feature_names}")
        if layer_matches.size != 1:
            raise ValueError(f"Layer {layer} not found exactly once in artifact. Available layers: {available_layers.tolist()}")
        if vector_key not in bundle.files:
            raise KeyError(f"{vector_key} not found in {artifact_path}. Available keys: {bundle.files}")

        regime_idx = int(regime_matches[0])
        feature_idx = int(feature_matches[0])
        layer_idx = int(layer_matches[0])

        raw_key = None
        if vector_key == "mm_probe_vectors" and "mm_raw_directions" in bundle.files:
            raw_key = "mm_raw_directions"
        elif vector_key == "wmm_probe_vectors" and "wmm_effective_directions" in bundle.files:
            raw_key = "wmm_effective_directions"

        steering_vector = bundle[vector_key][regime_idx, feature_idx, layer_idx, :].astype(np.float32)
        raw_vector = bundle[raw_key][regime_idx, feature_idx, layer_idx, :].astype(np.float32) if raw_key is not None else steering_vector
        if not np.isfinite(steering_vector).all():
            raise ValueError("Steering vector contains non-finite values.")
        steering_vector_norm = float(np.linalg.norm(steering_vector))
        raw_vector_norm = float(np.linalg.norm(raw_vector))
        if steering_vector_norm == 0.0:
            raise ValueError("Steering vector has zero norm.")

        return {
            "metadata": metadata,
            "steering_vector": steering_vector,
            "raw_vector": raw_vector,
            "steering_vector_norm": steering_vector_norm,
            "raw_vector_norm": raw_vector_norm,
            "available_train_regimes": train_regimes,
            "available_feature_names": feature_names,
            "available_layers": available_layers.tolist(),
        }
    finally:
        bundle.close()


def build_prompt(x: int, y: int, t: str) -> str:
    if (t or "").strip().lower() == "tomorrow":
        return f"What would you prefer {x}$ now or {y}$ tomorrow? Answer in just a few words."
    return f"What would you prefer {x}$ now or {y}$ in {t}? Answer in just a few words."


def format_for_model(tokenizer, user_prompt: str, *, use_chat_template: bool, disable_thinking_trace: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
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


def get_model_device(model) -> torch.device:
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


def strip_thinking_trace_for_parse(response: str, *, disable_thinking_trace: bool) -> str:
    text = (response or "").strip()
    if not text or disable_thinking_trace:
        return text

    cleaned = THINKING_BLOCK_RE.sub(" ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned:
        return cleaned

    close_match = re.search(r"</think\s*>", text, flags=re.IGNORECASE)
    if close_match is not None:
        trailing = text[close_match.end():].strip()
        if trailing:
            return trailing

    return text


def prepare_response_for_parse(response: str, *, disable_thinking_trace: bool) -> str:
    return strip_thinking_trace_for_parse(response, disable_thinking_trace=disable_thinking_trace).strip()


def normalize_parse_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_it_depends_response(normalized_text: str) -> bool:
    ambiguity_patterns = [
        r"\bit depends\b",
        r"\bdepends on\b",
        r"\bdepending on\b",
        r"\bdepends\b",
        r"\bhard to say\b",
        r"\b(can(?:not|'t)|unable to)\s+say\b",
        r"\bneed more (?:info|information|context)\b",
        r"\bmore (?:info|information|context) (?:is )?needed\b",
        r"\b(no|without) (?:clear )?(?:preference|way to choose)\b",
        r"\beither (?:one|option)\b",
        r"\bcase by case\b",
        r"\bcontext[- ]dependent\b",
        r"\bsituation[- ]dependent\b",
    ]
    return any(re.search(pattern, normalized_text) for pattern in ambiguity_patterns)


def parse_choice(response: str, x: int, y: int, t: str, *, disable_thinking_trace: bool) -> str:
    text = prepare_response_for_parse(response, disable_thinking_trace=disable_thinking_trace)
    if not text:
        return "unparsed"

    lower = normalize_parse_text(text)
    t_lower = normalize_parse_text(t)
    now_score = 0
    later_score = 0

    x_number_pattern = rf"(?<!\d)(?:\$\s*)?{x}(?!\d)"
    y_number_pattern = rf"(?<!\d)(?:\$\s*)?{y}(?!\d)"

    if y != x:
        if re.search(x_number_pattern, lower):
            now_score += 1
        if re.search(y_number_pattern, lower):
            later_score += 1

    if re.search(r"\b(now|today|immediately|right now|upfront|up front)\b", lower):
        now_score += 1
    if re.search(r"\b(later|wait|waiting|future|after|eventually|defer|delayed)\b", lower):
        later_score += 1
    if t_lower and t_lower in lower:
        later_score += 1

    if re.search(
        rf"\b(prefer|choose|take|pick|go with|would take|i(?:'d| would) take|i(?:'d| would) choose)\b[^.!?\n]{{0,60}}\b({x}|now|today|immediately)\b",
        lower,
    ):
        now_score += 2
    if re.search(
        rf"\b(prefer|choose|take|pick|go with|would take|i(?:'d| would) take|i(?:'d| would) choose)\b[^.!?\n]{{0,80}}\b({y}|later|wait|{re.escape(t_lower)})\b",
        lower,
    ):
        later_score += 2

    if re.search(r"\b(worth|better)\s+to\s+wait\b", lower):
        later_score += 1
    if re.search(r"\b(can(?:not|'t)|do\s+not|don't|won't|wouldn't)\s+wait\b", lower):
        now_score += 1

    if now_score > later_score:
        return "x_now"
    if later_score > now_score:
        return "y_later"

    now_markers = [
        lower.find(" now"),
        lower.find("today"),
        lower.find("immediately"),
        lower.find("right now"),
    ]
    later_markers = [
        lower.find("later"),
        lower.find("wait"),
        lower.find("future"),
        lower.find(t_lower) if t_lower else -1,
    ]

    now_pos = min([p for p in now_markers if p >= 0], default=-1)
    later_pos = min([p for p in later_markers if p >= 0], default=-1)

    if now_pos >= 0 and (later_pos < 0 or now_pos < later_pos):
        return "x_now"
    if later_pos >= 0 and (now_pos < 0 or later_pos < now_pos):
        return "y_later"

    if is_it_depends_response(lower):
        return "it depends"

    return "unparsed"


def summarize_time_utility_results(results_df: pd.DataFrame, *, y_values: list[int], t_values: list[str]) -> pd.DataFrame:
    summary_df = (
        results_df
        .groupby(["condition", "y_later", "t_delay"], as_index=False)
        .agg(
            n_total=("choice", "size"),
            n_choose_x=("choice", lambda values: int((values == "x_now").sum())),
            n_choose_y=("choice", lambda values: int((values == "y_later").sum())),
            n_it_depends=("choice", lambda values: int((values == "it depends").sum())),
            n_unparsed=("choice", lambda values: int((values == "unparsed").sum())),
        )
    )
    summary_df["n_parsed"] = summary_df["n_choose_x"] + summary_df["n_choose_y"] + summary_df["n_it_depends"]
    summary_df["prop_choose_x_parsed"] = np.where(
        summary_df["n_parsed"] > 0,
        (summary_df["n_choose_x"] + 0.5 * summary_df["n_it_depends"]) / summary_df["n_parsed"],
        np.nan,
    )
    summary_df["prop_choose_y_parsed"] = np.where(
        summary_df["n_parsed"] > 0,
        (summary_df["n_choose_y"] + 0.5 * summary_df["n_it_depends"]) / summary_df["n_parsed"],
        np.nan,
    )
    summary_df["prop_it_depends_parsed"] = np.where(
        summary_df["n_parsed"] > 0,
        summary_df["n_it_depends"] / summary_df["n_parsed"],
        np.nan,
    )
    summary_df["unparsed_rate"] = summary_df["n_unparsed"] / summary_df["n_total"]
    summary_df = summary_df.sort_values(["condition", "y_later", "t_delay"]).reset_index(drop=True)

    for condition_name, condition_df in summary_df.groupby("condition"):
        pivot = condition_df.pivot(index="y_later", columns="t_delay", values="n_total")
        pivot = pivot.reindex(index=y_values, columns=t_values)
        if pivot.isna().any().any():
            missing = pivot[pivot.isna()]
            raise ValueError(f"Summary is missing grid entries for condition={condition_name}: {missing.stack().index.tolist()}")

    return summary_df


def to_matrix(df: pd.DataFrame, value_col: str, *, y_values: list[int], t_values: list[str]) -> np.ndarray:
    pivot = df.pivot(index="y_later", columns="t_delay", values=value_col)
    pivot = pivot.reindex(index=y_values, columns=t_values)
    return pivot.to_numpy(dtype=float)


def draw_heatmap(ax, matrix, title: str, *, y_values: list[int], t_values: list[str], vmin=0.0, vmax=1.0, cmap="viridis", fmt="{:.2f}"):
    im = ax.imshow(matrix, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Delay t")
    ax.set_ylabel("Delayed amount y ($)")
    ax.set_xticks(np.arange(len(t_values)))
    ax.set_xticklabels(t_values, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels(y_values)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "nan" if np.isnan(value) else fmt.format(value)
            color = "white" if (not np.isnan(value) and value > 0.55) else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    return im


def maybe_register_time_utility_steering_hook(*, model, layer: int | None, direction: np.ndarray | None, strength: float, prompt_len: int, patch_prompt_last_only: bool, patch_decode_tokens: bool):
    if layer is None or direction is None or abs(float(strength)) == 0.0:
        return None

    target_layer = model.model.layers[int(layer)]
    layer_device = next(target_layer.parameters()).device
    vector = torch.tensor(direction, device=layer_device, dtype=torch.float32)

    def steering_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_mod = hidden.clone()
        delta = (float(strength) * vector).to(hidden_mod.dtype)

        if hidden_mod.shape[1] >= prompt_len:
            if patch_prompt_last_only:
                hidden_mod[:, prompt_len - 1, :] = hidden_mod[:, prompt_len - 1, :] + delta
            else:
                hidden_mod[:, :prompt_len, :] = hidden_mod[:, :prompt_len, :] + delta
        elif patch_decode_tokens:
            if patch_prompt_last_only:
                hidden_mod[:, -1, :] = hidden_mod[:, -1, :] + delta
            else:
                hidden_mod = hidden_mod + delta

        if isinstance(output, tuple):
            return (hidden_mod,) + output[1:]
        return hidden_mod

    return target_layer.register_forward_hook(steering_hook)


def generate_response_with_steering(
    *,
    model,
    tokenizer,
    prompt: str,
    generation_kwargs: dict[str, object],
    use_chat_template: bool,
    disable_thinking_trace: bool,
    layer: int | None,
    direction: np.ndarray | None,
    strength: float,
    patch_prompt_last_only: bool,
    patch_generation_tokens: bool,
) -> str:
    assert_model_on_cuda(model)
    model_input_text = format_for_model(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        disable_thinking_trace=disable_thinking_trace,
    )
    enc = tokenizer(model_input_text, return_tensors="pt")
    enc = move_batch_to_model_device(model, enc)
    batch_devices = {value.device.type for value in enc.values()}
    if batch_devices != {"cuda"}:
        raise RuntimeError(f"Tokenized batch is not fully on CUDA: {batch_devices}")
    prompt_len = int(enc["input_ids"].shape[1])

    hook_handle = maybe_register_time_utility_steering_hook(
        model=model,
        layer=layer,
        direction=direction,
        strength=strength,
        prompt_len=prompt_len,
        patch_prompt_last_only=patch_prompt_last_only,
        patch_decode_tokens=patch_generation_tokens,
    )
    try:
        with torch.inference_mode():
            out = model.generate(**enc, **generation_kwargs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    new_ids = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_output_paths(output_dir: Path, run_id: str) -> dict[str, Path]:
    return {
        "raw_path": output_dir / f"mmraz_time_utility_qwen3_32b_probe_steered_raw_{run_id}.csv",
        "summary_path": output_dir / f"mmraz_time_utility_qwen3_32b_probe_steered_summary_{run_id}.csv",
        "config_path": output_dir / f"mmraz_time_utility_qwen3_32b_probe_steered_config_{run_id}.json",
        "plot_index_path": output_dir / f"mmraz_time_utility_qwen3_32b_probe_steered_plots_{run_id}.csv",
        "artifact_index_path": output_dir / f"mmraz_time_utility_qwen3_32b_probe_steered_artifacts_{run_id}.csv",
        "partial_dir": output_dir / "partial",
    }


def build_run_condition_specs(cfg: dict[str, object]) -> list[dict[str, object]]:
    condition_specs: list[dict[str, object]] = []
    if bool(cfg.get("include_baseline_condition", True)):
        condition_specs.append(
            {
                "condition": "baseline",
                "signed_strength": 0.0,
                "is_baseline": True,
            }
        )

    for raw_spec in list(cfg["steering_conditions"]):
        spec = dict(raw_spec)
        spec["condition"] = str(spec["condition"])
        spec["signed_strength"] = float(spec["signed_strength"])
        spec["is_baseline"] = bool(spec.get("is_baseline", False))
        condition_specs.append(spec)

    deduped_specs: list[dict[str, object]] = []
    seen_conditions: set[str] = set()
    for spec in condition_specs:
        condition_name = str(spec["condition"])
        if condition_name in seen_conditions:
            continue
        deduped_specs.append(spec)
        seen_conditions.add(condition_name)
    return deduped_specs


def build_expected_prompt_grid(*, x_now: int, y_values: list[int], t_values: list[str], n_repeats: int) -> set[tuple[int, str, int, str]]:
    expected = set()
    for y in y_values:
        for t in t_values:
            prompt = build_prompt(int(x_now), int(y), str(t))
            for repeat_idx in range(1, int(n_repeats) + 1):
                expected.add((int(y), str(t), int(repeat_idx), prompt))
    return expected


def validate_cached_condition_df(
    condition_df: pd.DataFrame,
    *,
    condition_name: str,
    x_now: int,
    y_values: list[int],
    t_values: list[str],
    n_repeats: int,
    steering_layer: int,
    probe_train_regime: str,
    probe_feature_name: str,
    vector_key: str,
) -> tuple[bool, str]:
    required_columns = {
        "condition",
        "x_now",
        "y_later",
        "t_delay",
        "repeat_idx",
        "prompt",
        "response",
        "choice",
    }
    missing_columns = required_columns.difference(condition_df.columns)
    if missing_columns:
        return False, f"missing columns: {sorted(missing_columns)}"

    if condition_df.empty:
        return False, "cached dataframe is empty"

    observed_conditions = {str(value) for value in condition_df["condition"].astype(str).unique().tolist()}
    if observed_conditions != {str(condition_name)}:
        return False, f"unexpected condition labels: {sorted(observed_conditions)}"

    observed_x_values = {int(value) for value in condition_df["x_now"].astype(int).unique().tolist()}
    if observed_x_values != {int(x_now)}:
        return False, f"unexpected x_now values: {sorted(observed_x_values)}"

    if "steering_layer" in condition_df.columns:
        observed_layers = {int(value) for value in condition_df["steering_layer"].dropna().astype(int).unique().tolist()}
        if observed_layers != {int(steering_layer)}:
            return False, f"unexpected steering layers: {sorted(observed_layers)}"

    if "probe_train_regime" in condition_df.columns:
        observed_regimes = {str(value) for value in condition_df["probe_train_regime"].dropna().astype(str).unique().tolist()}
        if observed_regimes != {str(probe_train_regime)}:
            return False, f"unexpected probe train regimes: {sorted(observed_regimes)}"

    if "probe_feature_name" in condition_df.columns:
        observed_features = {str(value) for value in condition_df["probe_feature_name"].dropna().astype(str).unique().tolist()}
        if observed_features != {str(probe_feature_name)}:
            return False, f"unexpected probe feature names: {sorted(observed_features)}"

    if "vector_key" in condition_df.columns:
        observed_vector_keys = {str(value) for value in condition_df["vector_key"].dropna().astype(str).unique().tolist()}
        if observed_vector_keys != {str(vector_key)}:
            return False, f"unexpected vector keys: {sorted(observed_vector_keys)}"

    expected_prompt_grid = build_expected_prompt_grid(
        x_now=int(x_now),
        y_values=list(y_values),
        t_values=list(t_values),
        n_repeats=int(n_repeats),
    )
    observed_prompt_grid = {
        (int(row["y_later"]), str(row["t_delay"]), int(row["repeat_idx"]), str(row["prompt"]))
        for _, row in condition_df.iterrows()
    }

    if observed_prompt_grid != expected_prompt_grid:
        missing_rows = sorted(expected_prompt_grid.difference(observed_prompt_grid))
        extra_rows = sorted(observed_prompt_grid.difference(expected_prompt_grid))
        reason_bits = []
        if missing_rows:
            reason_bits.append(f"missing {len(missing_rows)} expected grid rows")
        if extra_rows:
            reason_bits.append(f"found {len(extra_rows)} unexpected grid rows")
        return False, ", ".join(reason_bits) or "prompt grid mismatch"

    return True, "ok"


def try_load_cached_condition_records(
    *,
    condition_name: str,
    paths: dict[str, Path],
    x_now: int,
    y_values: list[int],
    t_values: list[str],
    n_repeats: int,
    steering_layer: int,
    probe_train_regime: str,
    probe_feature_name: str,
    vector_key: str,
) -> tuple[pd.DataFrame | None, str | None]:
    partial_raw_path = paths["partial_dir"] / f"{condition_name}_raw.csv"
    full_raw_path = paths["raw_path"]

    candidate_sources = []
    if partial_raw_path.exists():
        candidate_sources.append(("partial_raw", partial_raw_path, None))
    if full_raw_path.exists():
        candidate_sources.append(("combined_raw", full_raw_path, condition_name))

    for source_name, source_path, required_condition in candidate_sources:
        try:
            cached_df = pd.read_csv(source_path)
        except Exception as exc:
            print(f"[{condition_name}] failed to read cached {source_name} at {source_path}: {exc}")
            continue

        if required_condition is not None:
            if "condition" not in cached_df.columns:
                print(f"[{condition_name}] cached {source_name} is missing 'condition'; will recompute.")
                continue
            cached_df = cached_df.loc[cached_df["condition"].astype(str) == str(required_condition)].copy()

        is_valid, reason = validate_cached_condition_df(
            cached_df,
            condition_name=str(condition_name),
            x_now=int(x_now),
            y_values=list(y_values),
            t_values=list(t_values),
            n_repeats=int(n_repeats),
            steering_layer=int(steering_layer),
            probe_train_regime=str(probe_train_regime),
            probe_feature_name=str(probe_feature_name),
            vector_key=str(vector_key),
        )
        if is_valid:
            cached_df = cached_df.sort_values(["condition", "y_later", "t_delay", "repeat_idx"]).reset_index(drop=True)
            return cached_df, source_name

        print(f"[{condition_name}] cached {source_name} at {source_path} is not reusable: {reason}")

    return None, None


def run_experiment(config_overrides: dict | None = None) -> dict[str, object]:
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    root = find_repo_root(Path.cwd())

    artifact_search_roots = cfg["artifact_search_roots"] or [
        root / "results" / "qwen3_32b" / "question_only_probe_variations",
        Path("/workspace/results/qwen3_32b/question_only_probe_variations"),
    ]
    artifact_path, metadata_path = locate_latest_probe_artifacts(
        artifact_search_roots,
        artifact_path_override=cfg.get("artifact_path"),
        metadata_path_override=cfg.get("metadata_path"),
        required_train_regime=str(cfg["train_regime"]),
    )
    probe_payload = load_probe_vector(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        train_regime=str(cfg["train_regime"]),
        feature_name=str(cfg["feature_name"]),
        vector_key=str(cfg["vector_key"]),
        layer=int(cfg["steering_layer"]),
    )

    output_root = root / str(cfg["output_root_relative"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.get("run_id") or time.strftime("%Y%m%d-%H%M%S"))
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(output_dir, run_id)
    paths["partial_dir"].mkdir(parents=True, exist_ok=True)
    existing_config_payload = {}
    if paths["config_path"].exists():
        try:
            existing_config_payload = json.loads(paths["config_path"].read_text(encoding="utf-8"))
        except Exception:
            existing_config_payload = {}

    condition_specs = build_run_condition_specs(cfg)
    y_values = list(cfg["y_values"])
    t_values = list(cfg["t_values"])
    n_repeats = int(cfg["n_repeats"])
    x_now = int(cfg["x_now"])

    all_records: list[dict[str, object]] = []
    plot_rows: list[dict[str, str]] = []
    reused_conditions: list[str] = []
    generated_conditions: list[str] = []
    pending_condition_specs: list[dict[str, object]] = []

    for condition_spec in condition_specs:
        condition_name = str(condition_spec["condition"])
        cached_df, cache_source = try_load_cached_condition_records(
            condition_name=condition_name,
            paths=paths,
            x_now=x_now,
            y_values=y_values,
            t_values=t_values,
            n_repeats=n_repeats,
            steering_layer=int(cfg["steering_layer"]),
            probe_train_regime=str(cfg["train_regime"]),
            probe_feature_name=str(cfg["feature_name"]),
            vector_key=str(cfg["vector_key"]),
        )
        if cached_df is not None:
            steering_enabled = not bool(condition_spec.get("is_baseline", False)) and abs(float(condition_spec["signed_strength"])) > 0.0
            cached_df = cached_df.copy()
            if "signed_strength" not in cached_df.columns:
                cached_df["signed_strength"] = float(condition_spec["signed_strength"])
            else:
                cached_df["signed_strength"] = cached_df["signed_strength"].fillna(float(condition_spec["signed_strength"]))
            if "steering_applied" not in cached_df.columns:
                cached_df["steering_applied"] = bool(steering_enabled)
            else:
                cached_df["steering_applied"] = cached_df["steering_applied"].fillna(bool(steering_enabled))
            if "steering_layer" not in cached_df.columns:
                cached_df["steering_layer"] = int(cfg["steering_layer"])
            if "probe_train_regime" not in cached_df.columns:
                cached_df["probe_train_regime"] = str(cfg["train_regime"])
            if "probe_feature_name" not in cached_df.columns:
                cached_df["probe_feature_name"] = str(cfg["feature_name"])
            if "vector_key" not in cached_df.columns:
                cached_df["vector_key"] = str(cfg["vector_key"])
            if "probe_artifact_path" not in cached_df.columns:
                cached_df["probe_artifact_path"] = str(artifact_path)
            all_records.extend(cached_df.to_dict(orient="records"))
            reused_conditions.append(condition_name)
            print(f"[{condition_name}] reusing cached results from {cache_source}.")
        else:
            pending_condition_specs.append(condition_spec)

    tokenizer = None
    model = None
    primary_model_device = str(existing_config_payload.get("device", "cached_only"))

    if pending_condition_specs:
        require_cuda_runtime(require_cuda=bool(cfg["require_cuda"]))
        tokenizer = AutoTokenizer.from_pretrained(str(cfg["model_name"]), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["model_name"]),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model = model.to("cuda")
        model.eval()
        primary_model_device = str(get_model_device(model))
        assert_model_on_cuda(model)
        cuda_device = torch.cuda.current_device()
        cuda_props = torch.cuda.get_device_properties(cuda_device)
        print(
            "[cuda] confirmed GPU execution:",
            f"device=cuda:{cuda_device}",
            f"| name={cuda_props.name}",
            f"| total_memory_gb={cuda_props.total_memory / (1024 ** 3):.1f}",
            f"| device_count={torch.cuda.device_count()}",
        )
    else:
        print("[resume] all requested conditions are already cached; skipping CUDA check, model load, and generation.")

    generation_kwargs = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id if tokenizer is not None else existing_config_payload.get("generation_kwargs", {}).get("pad_token_id"),
        "eos_token_id": tokenizer.eos_token_id if tokenizer is not None else existing_config_payload.get("generation_kwargs", {}).get("eos_token_id"),
    }

    print("Model:", cfg["model_name"])
    print("Probe artifact:", artifact_path)
    print("Probe metadata:", metadata_path)
    print("Train regime:", cfg["train_regime"])
    print("Feature name:", cfg["feature_name"])
    print("Vector key:", cfg["vector_key"])
    print("Steering layer:", cfg["steering_layer"])
    print("Vector norm:", probe_payload["steering_vector_norm"])
    print("Output dir:", output_dir)

    total_prompts = len(pending_condition_specs) * len(y_values) * len(t_values) * n_repeats
    progress_bar = tqdm(total=total_prompts, desc="Time utility steering", unit="prompt")
    try:
        for condition_spec in pending_condition_specs:
            condition_name = str(condition_spec["condition"])
            signed_strength = float(condition_spec["signed_strength"])
            condition_records: list[dict[str, object]] = []
            steering_enabled = not bool(condition_spec.get("is_baseline", False)) and abs(signed_strength) > 0.0

            for y in y_values:
                for t in t_values:
                    for repeat_idx in range(1, n_repeats + 1):
                        prompt = build_prompt(x_now, int(y), str(t))
                        response = generate_response_with_steering(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            generation_kwargs=generation_kwargs,
                            use_chat_template=bool(cfg["use_chat_template"]),
                            disable_thinking_trace=bool(cfg["disable_thinking_trace"]),
                            layer=int(cfg["steering_layer"]) if steering_enabled else None,
                            direction=probe_payload["steering_vector"] if steering_enabled else None,
                            strength=signed_strength if steering_enabled else 0.0,
                            patch_prompt_last_only=bool(cfg["patch_prompt_last_only"]),
                            patch_generation_tokens=bool(cfg["patch_generation_tokens"]),
                        )
                        response_for_parse = prepare_response_for_parse(
                            response,
                            disable_thinking_trace=bool(cfg["disable_thinking_trace"]),
                        )
                        parsed_choice = parse_choice(
                            response,
                            x_now,
                            int(y),
                            str(t),
                            disable_thinking_trace=bool(cfg["disable_thinking_trace"]),
                        )
                        row = {
                            "condition": condition_name,
                            "x_now": x_now,
                            "y_later": int(y),
                            "t_delay": str(t),
                            "repeat_idx": int(repeat_idx),
                            "prompt": prompt,
                            "response": response,
                            "response_for_parse": response_for_parse,
                            "thinking_trace_removed_for_parse": False,
                            "choice": parsed_choice,
                            "steering_layer": int(cfg["steering_layer"]),
                            "signed_strength": signed_strength,
                            "steering_applied": bool(steering_enabled),
                            "probe_train_regime": str(cfg["train_regime"]),
                            "probe_feature_name": str(cfg["feature_name"]),
                            "vector_key": str(cfg["vector_key"]),
                            "probe_artifact_path": str(artifact_path),
                        }
                        condition_records.append(row)
                        all_records.append(row)
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            condition=condition_name,
                            y_later=int(y),
                            t_delay=str(t),
                            parsed=parsed_choice,
                        )

            condition_df = pd.DataFrame(condition_records)
            condition_raw_path = paths["partial_dir"] / f"{condition_name}_raw.csv"
            condition_summary_path = paths["partial_dir"] / f"{condition_name}_summary.csv"
            condition_df.to_csv(condition_raw_path, index=False)
            condition_summary_df = summarize_time_utility_results(
                condition_df,
                y_values=y_values,
                t_values=t_values,
            )
            condition_summary_df.to_csv(condition_summary_path, index=False)
            generated_conditions.append(condition_name)
            print(f"[{condition_name}] saved partial raw:", condition_raw_path)
            print(f"[{condition_name}] saved partial summary:", condition_summary_path)
    finally:
        progress_bar.close()

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    clear_gpu_cache()

    results_df = pd.DataFrame(all_records)
    results_df = results_df.sort_values(["condition", "y_later", "t_delay", "repeat_idx"]).reset_index(drop=True)
    results_df.to_csv(paths["raw_path"], index=False)

    summary_df = summarize_time_utility_results(
        results_df,
        y_values=y_values,
        t_values=t_values,
    )
    summary_df.to_csv(paths["summary_path"], index=False)

    strength_lookup = {str(item["condition"]): float(item["signed_strength"]) for item in condition_specs}
    for condition_name in summary_df["condition"].drop_duplicates().tolist():
        condition_summary_df = summary_df.loc[summary_df["condition"] == condition_name].copy()
        prop_x_matrix = to_matrix(
            condition_summary_df,
            "prop_choose_x_parsed",
            y_values=y_values,
            t_values=t_values,
        )
        unparsed_matrix = to_matrix(
            condition_summary_df,
            "unparsed_rate",
            y_values=y_values,
            t_values=t_values,
        )

        if condition_name == "baseline":
            title_suffix = "baseline | no steering"
        else:
            title_suffix = (
                f"{condition_name} | layer {cfg['steering_layer']} | {cfg['train_regime']} | "
                f"{cfg['feature_name']} | signed strength {strength_lookup[condition_name]:g}"
            )

        fig_x, ax_x = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        im_x = draw_heatmap(
            ax_x,
            prop_x_matrix,
            "Choose x = 100 now (parsed only; 'it depends' = 0.5)",
            y_values=y_values,
            t_values=t_values,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )
        fig_x.suptitle(title_suffix, y=1.02)
        fig_x.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.04)

        fig_unparsed, ax_unparsed = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        im_unparsed = draw_heatmap(
            ax_unparsed,
            unparsed_matrix,
            "Unparsed rate",
            y_values=y_values,
            t_values=t_values,
            vmin=0.0,
            vmax=1.0,
            cmap="Greys",
        )
        fig_unparsed.suptitle(title_suffix, y=1.02)
        fig_unparsed.colorbar(im_unparsed, ax=ax_unparsed, fraction=0.046, pad=0.04)

        strength_tag = str(abs(int(strength_lookup[condition_name])))
        choose_x_plot_path = output_dir / f"qwen3_32b_{condition_name}_layer{cfg['steering_layer']}_strength{strength_tag}_choose_x_now.png"
        unparsed_plot_path = output_dir / f"qwen3_32b_{condition_name}_layer{cfg['steering_layer']}_strength{strength_tag}_unparsed_rate.png"
        fig_x.savefig(choose_x_plot_path, dpi=200, bbox_inches="tight")
        fig_unparsed.savefig(unparsed_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_unparsed)

        plot_rows.extend(
            [
                {"condition": condition_name, "plot_type": "choose_x_now", "plot_path": str(choose_x_plot_path)},
                {"condition": condition_name, "plot_type": "unparsed_rate", "plot_path": str(unparsed_plot_path)},
            ]
        )

    plot_index_df = pd.DataFrame(plot_rows)
    plot_index_df.to_csv(paths["plot_index_path"], index=False)

    config_payload = {
        "run_id": run_id,
        "model_name": str(cfg["model_name"]),
        "model_slug": make_model_slug(str(cfg["model_name"])),
        "device": primary_model_device,
        "use_chat_template": bool(cfg["use_chat_template"]),
        "disable_thinking_trace": bool(cfg["disable_thinking_trace"]),
        "prompt_format": "direct_time_utility_question",
        "prompt_template_default": "What would you prefer {x}$ now or {y}$ in {t}? Answer in just a few words.",
        "prompt_template_tomorrow": "What would you prefer {x}$ now or {y}$ tomorrow? Answer in just a few words.",
        "special_case_without_in": "tomorrow",
        "x_now": int(cfg["x_now"]),
        "y_values": list(cfg["y_values"]),
        "t_values": list(cfg["t_values"]),
        "n_repeats": int(cfg["n_repeats"]),
        "generation_kwargs": generation_kwargs,
        "parser_variant": "comma_normalized_direct_choice_with_it_depends_v2",
        "it_depends_choice_value": "it depends",
        "it_depends_heatmap_value": 0.5,
        "probe_artifact_path": str(artifact_path),
        "probe_metadata_path": str(metadata_path) if metadata_path is not None else None,
        "probe_artifact_format_version": int(probe_payload["metadata"].get("artifact_format_version", 0)),
        "probe_prompt_family": probe_payload["metadata"].get("prompt_family"),
        "probe_train_regime": str(cfg["train_regime"]),
        "probe_feature_name": str(cfg["feature_name"]),
        "probe_vector_key": str(cfg["vector_key"]),
        "steering_layer": int(cfg["steering_layer"]),
        "steering_vector_norm": float(probe_payload["steering_vector_norm"]),
        "raw_vector_norm": float(probe_payload["raw_vector_norm"]),
        "artifact_vectors_already_normalized": bool(probe_payload["metadata"].get("normalize_probe_vectors", False)),
        "patch_prompt_last_only": bool(cfg["patch_prompt_last_only"]),
        "patch_generation_tokens": bool(cfg["patch_generation_tokens"]),
        "include_baseline_condition": bool(cfg.get("include_baseline_condition", True)),
        "conditions": condition_specs,
        "steering_conditions": list(cfg["steering_conditions"]),
        "available_train_regimes": probe_payload["available_train_regimes"],
        "available_feature_names": probe_payload["available_feature_names"],
        "available_layers": probe_payload["available_layers"],
        "raw_path": str(paths["raw_path"]),
        "summary_path": str(paths["summary_path"]),
        "plot_index_path": str(paths["plot_index_path"]),
        "partial_dir": str(paths["partial_dir"]),
        "reused_conditions": reused_conditions,
        "generated_conditions": generated_conditions,
    }
    paths["config_path"].write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    artifact_rows = [
        {"artifact": "raw_csv", "path": str(paths["raw_path"])},
        {"artifact": "summary_csv", "path": str(paths["summary_path"])},
        {"artifact": "config_json", "path": str(paths["config_path"])},
        {"artifact": "plot_index_csv", "path": str(paths["plot_index_path"])},
        {"artifact": "partial_dir", "path": str(paths["partial_dir"])},
    ]
    artifact_rows.extend({"artifact": f"{row['condition']}_{row['plot_type']}", "path": row["plot_path"]} for row in plot_rows)
    artifact_index_df = pd.DataFrame(artifact_rows)
    artifact_index_df.to_csv(paths["artifact_index_path"], index=False)

    print("Saved raw outputs   :", paths["raw_path"])
    print("Saved summary table :", paths["summary_path"])
    print("Saved config        :", paths["config_path"])
    print("Saved plot index    :", paths["plot_index_path"])
    print("Saved artifact list :", paths["artifact_index_path"])

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "raw_path": str(paths["raw_path"]),
        "summary_path": str(paths["summary_path"]),
        "config_path": str(paths["config_path"]),
        "plot_index_path": str(paths["plot_index_path"]),
        "artifact_index_path": str(paths["artifact_index_path"]),
        "n_rows": int(len(results_df)),
        "reused_conditions": reused_conditions,
        "generated_conditions": generated_conditions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen3-32B direct-question time-utility steering experiment using stored question-only probe artifacts."
        )
    )
    parser.add_argument("--artifact-path", type=str, default=None, help="Optional explicit probe artifact .npz path.")
    parser.add_argument("--metadata-path", type=str, default=None, help="Optional explicit probe metadata .json path.")
    parser.add_argument("--train-regime", type=str, default=None, help="Optional probe train regime override.")
    parser.add_argument("--feature-name", type=str, default=None, help="Optional probe feature name override.")
    parser.add_argument("--vector-key", type=str, default=None, help="Optional probe vector key override.")
    parser.add_argument("--steering-layer", type=int, default=None, help="Optional steering layer override.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id for resuming into the same output directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {}
    if args.artifact_path is not None:
        overrides["artifact_path"] = args.artifact_path
    if args.metadata_path is not None:
        overrides["metadata_path"] = args.metadata_path
    if args.train_regime is not None:
        overrides["train_regime"] = args.train_regime
    if args.feature_name is not None:
        overrides["feature_name"] = args.feature_name
    if args.vector_key is not None:
        overrides["vector_key"] = args.vector_key
    if args.steering_layer is not None:
        overrides["steering_layer"] = int(args.steering_layer)
    if args.run_id is not None:
        overrides["run_id"] = args.run_id

    summary = run_experiment(overrides or None)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
