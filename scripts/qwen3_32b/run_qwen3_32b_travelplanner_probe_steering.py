#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_variants import default_probe_artifact_search_roots


REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION = 1
EXPERIMENT_NAME = "qwen3_32b_travelplanner_probe_steering"
RUN_FINGERPRINT_VERSION = 1
TRAVELPLANNER_PROMPT_FORMAT_VERSION = "reference_json_prompt_v1"
DEFAULT_TRAVELPLANNER_REPO = "out/travelplanner/TravelPlanner"
DEFAULT_HF_DATASET_DIR = "out/travelplanner/hf_dataset"
DEFAULT_OUTPUT_ROOT = "results/qwen3_32b/travelplanner_probe_steered"
DEFAULT_STRENGTHS = [-128.0, 128.0, -91.0, 91.0, -64.0, 64.0, -32.0, 32.0, -16.0, 16.0, -8.0, 8.0]

COMMONSENSE_KEYS = [
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
]

HARD_KEYS = [
    "valid_cost",
    "valid_room_rule",
    "valid_cuisine",
    "valid_room_type",
    "valid_transportation",
]

REQUIRED_PLAN_FIELDS = [
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
]

FIELD_ALIASES = {
    "day": "day",
    "days": "day",
    "current city": "current_city",
    "current_city": "current_city",
    "transportation": "transportation",
    "breakfast": "breakfast",
    "attraction": "attraction",
    "attractions": "attraction",
    "lunch": "lunch",
    "dinner": "dinner",
    "accommodation": "accommodation",
    "accommodations": "accommodation",
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from current working directory.")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def decode_str_array(values: Any) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def resolve_metadata_path_for_artifact(artifact_path: Path, metadata_candidates: list[Path]) -> Path | None:
    candidate = artifact_path.with_name(
        artifact_path.name.replace("_probe_artifacts_", "_probe_metadata_").replace(".npz", ".json")
    )
    if candidate.exists():
        return candidate
    for metadata_path in metadata_candidates:
        if metadata_path.stem.replace("_probe_metadata_", "_probe_artifacts_") == artifact_path.stem:
            return metadata_path
    return None


def load_probe_metadata(metadata_path: Path | None) -> dict[str, Any] | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_is_compatible(metadata: dict[str, Any] | None) -> bool:
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
    search_roots: list[Path],
    *,
    artifact_path_override: str | None,
    metadata_path_override: str | None,
    required_train_regime: str | None,
    artifact_file_prefix: str,
) -> tuple[Path, Path | None]:
    if artifact_path_override is not None:
        artifact_path = Path(artifact_path_override).expanduser().resolve()
        metadata_path = (
            Path(metadata_path_override).expanduser().resolve()
            if metadata_path_override is not None
            else resolve_metadata_path_for_artifact(artifact_path, [])
        )
        if not artifact_contains_train_regime(artifact_path, required_train_regime):
            raise ValueError(f"Artifact {artifact_path} does not contain train_regime={required_train_regime!r}.")
        return artifact_path, metadata_path

    artifact_candidates: list[Path] = []
    metadata_candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        artifact_candidates.extend(sorted(root.rglob(f"{artifact_file_prefix}_artifacts_*.npz")))
        metadata_candidates.extend(sorted(root.rglob(f"{artifact_file_prefix}_metadata_*.json")))

    artifact_candidates = unique_paths(artifact_candidates)
    metadata_candidates = unique_paths(metadata_candidates)
    if not artifact_candidates:
        roots_text = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"Could not find any {artifact_file_prefix} probe-artifact bundles under: {roots_text}."
        )

    compatible_candidates = []
    for artifact_path in reversed(artifact_candidates):
        metadata_path = resolve_metadata_path_for_artifact(artifact_path, metadata_candidates)
        metadata = load_probe_metadata(metadata_path)
        if metadata_is_compatible(metadata) and artifact_contains_train_regime(artifact_path, required_train_regime):
            compatible_candidates.append((artifact_path, metadata_path))

    if not compatible_candidates:
        roots_text = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"Found {artifact_file_prefix} artifacts, but none matched the required Qwen3 question-only "
            f"probe format. Search roots: {roots_text}"
        )
    return compatible_candidates[0]


def load_probe_vector(
    *,
    artifact_path: Path,
    metadata_path: Path | None,
    train_regime: str,
    feature_name: str,
    vector_key: str,
    layer: int,
) -> dict[str, Any]:
    with np.load(artifact_path) as bundle:
        metadata = load_probe_metadata(metadata_path) or {}
        if not metadata_is_compatible(metadata):
            raise ValueError(f"Probe metadata at {metadata_path} is not compatible with this runner.")

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
            raise ValueError(f"Layer {layer} not found exactly once. Available: {available_layers.tolist()}")
        if vector_key not in bundle.files:
            raise KeyError(f"{vector_key} not found in {artifact_path}. Available keys: {bundle.files}")

        regime_idx = int(regime_matches[0])
        feature_idx = int(feature_matches[0])
        layer_idx = int(layer_matches[0])
        vector = bundle[vector_key][regime_idx, feature_idx, layer_idx, :].astype(np.float32)

        if not np.isfinite(vector).all():
            raise ValueError("Steering vector contains non-finite values.")
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0:
            raise ValueError("Steering vector has zero norm.")

        return {
            "metadata": metadata,
            "steering_vector": vector,
            "steering_vector_norm": vector_norm,
            "available_train_regimes": train_regimes,
            "available_feature_names": feature_names,
            "available_layers": available_layers.tolist(),
        }


def parse_csv_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{('\"":
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def normalize_query_record(row: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    record = dict(row)
    record["dataset_index"] = int(dataset_index)
    record["benchmark_idx"] = int(dataset_index + 1)
    for key in ["date", "local_constraint", "reference_information", "annotated_plan"]:
        if key in record:
            record[key] = parse_csv_literal(record[key])
    if "local_constraint" not in record or not isinstance(record.get("local_constraint"), dict):
        record["local_constraint"] = None
    return record


def load_travelplanner_split(dataset_dir: Path, set_type: str) -> list[dict[str, Any]]:
    csv_path = dataset_dir / f"{set_type}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing TravelPlanner split CSV: {csv_path}. Expected files downloaded from "
            "https://huggingface.co/datasets/osunlp/TravelPlanner."
        )
    df = pd.read_csv(csv_path)
    return [normalize_query_record(row, idx) for idx, row in enumerate(df.to_dict(orient="records"))]


def parse_index_spec(index_spec: str, total: int) -> list[int]:
    indices: list[int] = []
    seen = set()
    for part in index_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range in --query-indices: {part!r}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for one_based in values:
            if one_based < 1 or one_based > total:
                raise ValueError(f"Query index {one_based} is outside valid range 1..{total}.")
            zero_based = one_based - 1
            if zero_based not in seen:
                indices.append(zero_based)
                seen.add(zero_based)
    return indices


def select_queries(
    records: list[dict[str, Any]],
    *,
    query_indices: str | None,
    query_start: int,
    max_queries: int | None,
) -> list[dict[str, Any]]:
    if query_indices:
        selected_indices = parse_index_spec(query_indices, len(records))
    else:
        if query_start < 1:
            raise ValueError("--query-start is 1-based and must be >= 1.")
        selected_indices = list(range(query_start - 1, len(records)))
        if max_queries is not None:
            selected_indices = selected_indices[: int(max_queries)]
    return [records[idx] for idx in selected_indices]


def format_reference_information(reference_information: Any) -> str:
    if isinstance(reference_information, list):
        sections = []
        for item in reference_information:
            if isinstance(item, dict) and "Description" in item and "Content" in item:
                sections.append(f"### {item['Description']}\n{item['Content']}")
            else:
                sections.append(json.dumps(item, ensure_ascii=False, indent=2))
        return "\n\n".join(sections)
    if isinstance(reference_information, dict):
        return json.dumps(reference_information, ensure_ascii=False, indent=2)
    return str(reference_information)


def build_travelplanner_prompt(query_record: dict[str, Any], reference_text: str) -> str:
    days = int(query_record.get("days", 0) or 0)
    return f"""You are a proficient travel planner. Create a complete, feasible itinerary for the user query using only the provided reference information.

All named flights, restaurants, attractions, accommodations, cities, prices, dates, and transport options in the plan must come from the reference information. Keep the trip within the user's requested budget and constraints. Use "-" only when that item is genuinely unnecessary, such as meals before departure or accommodation after returning on the final day.

Return ONLY valid JSON. Do not use markdown. The JSON must be a list with exactly {days} objects. Each object must use these keys:
"day", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation".

Formatting rules:
- "day" is the 1-based day number.
- "current_city" should be a city name, or "from ORIGIN to DESTINATION" when traveling between cities that day.
- "transportation" should include the flight number and route, or a self-driving/taxi route, or "-".
- Restaurant, attraction, and accommodation values should include the name and city when possible.
- Multiple attractions should be separated with semicolons and should end with a semicolon.

User query:
{query_record.get("query", "")}

Reference information:
{reference_text}

JSON plan:"""


def format_for_model(tokenizer: Any, user_prompt: str, *, use_chat_template: bool, disable_thinking_trace: bool) -> str:
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


def get_input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_decoder_layer(model: Any, layer: int) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[int(layer)]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[int(layer)]
    raise AttributeError("Could not locate decoder layers on model; expected model.model.layers or model.transformer.h.")


def maybe_register_steering_hook(
    *,
    model: Any,
    layer: int | None,
    direction: np.ndarray | None,
    strength: float,
    prompt_len: int,
    patch_prompt_last_only: bool,
    patch_decode_tokens: bool,
) -> Any | None:
    if layer is None or direction is None or abs(float(strength)) == 0.0:
        return None

    target_layer = get_decoder_layer(model, int(layer))
    layer_device = next(target_layer.parameters()).device
    vector = torch.tensor(direction, device=layer_device, dtype=torch.float32)

    def steering_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
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


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().strip()
    if normalized == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown --torch-dtype={dtype_name!r}.")
    return mapping[normalized]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_torch_dtype(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if not args.device_map or args.device_map.lower() == "none":
        if torch.cuda.is_available():
            model = model.to("cuda")
        elif torch.backends.mps.is_available() and not args.require_cuda:
            model = model.to("mps")
        else:
            model = model.to("cpu")
    model.eval()
    device_summary = str(get_input_device(model))
    if hasattr(model, "hf_device_map"):
        device_summary = f"{device_summary}; hf_device_map={getattr(model, 'hf_device_map')}"
    return model, tokenizer, device_summary


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    layer: int | None,
    direction: np.ndarray | None,
    strength: float,
) -> tuple[str, dict[str, Any]]:
    model_input_text = format_for_model(
        tokenizer,
        prompt,
        use_chat_template=bool(args.use_chat_template),
        disable_thinking_trace=bool(args.disable_thinking_trace),
    )
    tokenization_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if args.max_input_tokens is not None:
        tokenization_kwargs.update({"truncation": True, "max_length": int(args.max_input_tokens)})
    enc = tokenizer(model_input_text, **tokenization_kwargs)
    input_device = get_input_device(model)
    enc = move_batch_to_device(enc, input_device)
    prompt_len = int(enc["input_ids"].shape[1])

    hook_handle = maybe_register_steering_hook(
        model=model,
        layer=layer,
        direction=direction,
        strength=strength,
        prompt_len=prompt_len,
        patch_prompt_last_only=bool(args.patch_prompt_last_only),
        patch_decode_tokens=bool(args.patch_generation_tokens),
    )
    generation_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = float(args.temperature)
        generation_kwargs["top_p"] = float(args.top_p)

    start_time = time.time()
    try:
        with torch.inference_mode():
            output = model.generate(**enc, **generation_kwargs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    generation_seconds = time.time() - start_time

    new_ids = output[0, prompt_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    metadata = {
        "input_tokens": int(prompt_len),
        "output_tokens": int(new_ids.shape[0]),
        "generation_seconds": float(generation_seconds),
        "generation_kwargs": generation_kwargs,
    }
    return response, metadata


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def find_balanced_json_span(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote_char:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def loads_jsonish(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def extract_json_object(response: str) -> Any:
    stripped = strip_code_fences(response)
    for candidate in [stripped, find_balanced_json_span(stripped, "[", "]"), find_balanced_json_span(stripped, "{", "}")]:
        if not candidate:
            continue
        try:
            return loads_jsonish(candidate)
        except Exception:
            continue
    raise ValueError("Could not parse a JSON object or JSON list from the model response.")


def canonical_key(raw_key: Any) -> str | None:
    key = re.sub(r"\s+", " ", str(raw_key).strip().lower().replace("_", " "))
    return FIELD_ALIASES.get(key)


def stringify_plan_value(value: Any, *, field: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "-"
    if isinstance(value, list):
        parts = [stringify_plan_value(item, field=field) for item in value]
        parts = [part for part in parts if part and part != "-"]
        if not parts:
            return "-"
        separator = ";" if field == "attraction" else "; "
        text = separator.join(part.rstrip(";") for part in parts)
        if field == "attraction" and not text.endswith(";"):
            text += ";"
        return text
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text if text else "-"


def normalize_plan_item(item: Any, day_number: int) -> dict[str, Any]:
    normalized = {"day": int(day_number)}
    if isinstance(item, dict):
        for raw_key, value in item.items():
            key = canonical_key(raw_key)
            if key is None:
                continue
            if key == "day":
                try:
                    normalized["day"] = int(value)
                except Exception:
                    normalized["day"] = int(day_number)
            else:
                normalized[key] = stringify_plan_value(value, field=key)
    for field in REQUIRED_PLAN_FIELDS:
        normalized[field] = stringify_plan_value(normalized.get(field, "-"), field=field)
    return normalized


def unwrap_plan_object(parsed: Any) -> Any:
    if isinstance(parsed, dict):
        for key in ["plan", "travel_plan", "itinerary", "days"]:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return parsed


def parse_text_day_blocks(response: str) -> list[dict[str, Any]]:
    text = response.strip()
    chunks = re.split(r"(?im)^\s*Day\s*(\d+)\s*:?\s*", text)
    if len(chunks) < 3:
        return []
    items: list[dict[str, Any]] = []
    label_pattern = r"Current City|Transportation|Breakfast|Attraction|Attractions|Lunch|Dinner|Accommodation"
    for idx in range(1, len(chunks), 2):
        day_text = chunks[idx]
        body = chunks[idx + 1]
        try:
            day_number = int(day_text)
        except Exception:
            day_number = len(items) + 1
        item: dict[str, Any] = {"day": day_number}
        for label in re.findall(rf"(?im)^\s*({label_pattern})\s*:", body):
            key = canonical_key(label)
            if key is None:
                continue
            match = re.search(
                rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*(?:{label_pattern})\s*:|\Z)",
                body,
            )
            if match:
                item[key] = match.group(1).strip()
        items.append(item)
    return items


def parse_plan(response: str, expected_days: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parse_info: dict[str, Any] = {"parse_method": None, "parse_success": False, "parse_error": None}
    try:
        parsed = unwrap_plan_object(extract_json_object(response))
        if not isinstance(parsed, list):
            raise ValueError(f"Parsed JSON is {type(parsed).__name__}, expected a list.")
        plan = [normalize_plan_item(item, idx + 1) for idx, item in enumerate(parsed)]
        parse_info["parse_method"] = "json"
    except Exception as exc:
        text_plan = parse_text_day_blocks(response)
        if text_plan:
            plan = [normalize_plan_item(item, idx + 1) for idx, item in enumerate(text_plan)]
            parse_info["parse_method"] = "day_block_text"
            parse_info["parse_error"] = f"json_parse_failed: {type(exc).__name__}: {exc}"
        else:
            parse_info["parse_error"] = f"{type(exc).__name__}: {exc}"
            return [], parse_info

    if expected_days > 0:
        plan = plan[:expected_days]
        while len(plan) < expected_days:
            plan.append(normalize_plan_item({}, len(plan) + 1))
    parse_info["parse_success"] = True
    parse_info["n_plan_days"] = len(plan)
    return plan, parse_info


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_slug(value: Any, *, max_len: int = 80) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = "none"
    return text[:max_len].strip("_") or "none"


def query_selection_slug(selected_queries: list[dict[str, Any]]) -> str:
    indices = [int(row["dataset_index"]) for row in selected_queries]
    if not indices:
        return "qnone"
    one_based = [idx + 1 for idx in indices]
    contiguous = one_based == list(range(one_based[0], one_based[0] + len(one_based)))
    if contiguous:
        return f"q{one_based[0]}-{one_based[-1]}_n{len(one_based)}"
    digest = sha256_text(",".join(str(idx) for idx in one_based))[:10]
    return f"qset_{digest}_n{len(one_based)}"


def run_fingerprint_payload(
    *,
    args: argparse.Namespace,
    selected_queries: list[dict[str, Any]],
    artifact_path: Path,
    metadata_path: Path | None,
    artifact_sha256: str,
    metadata_sha256: str | None,
    dataset_split_sha256: str,
) -> dict[str, Any]:
    selected_indices = [int(row["dataset_index"]) for row in selected_queries]
    return {
        "experiment_name": EXPERIMENT_NAME,
        "run_fingerprint_version": RUN_FINGERPRINT_VERSION,
        "prompt_format_version": TRAVELPLANNER_PROMPT_FORMAT_VERSION,
        "model_name": args.model_name,
        "set_type": args.set_type,
        "dataset_split_sha256": dataset_split_sha256,
        "query_start": int(args.query_start),
        "max_queries": args.max_queries,
        "query_indices": args.query_indices,
        "selected_dataset_indices": selected_indices,
        "n_queries_selected": len(selected_indices),
        "artifact_filename": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "metadata_filename": metadata_path.name if metadata_path is not None else None,
        "metadata_sha256": metadata_sha256,
        "train_regime": args.train_regime,
        "feature_name": args.feature_name,
        "vector_key": args.vector_key,
        "steering_layer": int(args.steering_layer),
        "patch_prompt_last_only": bool(args.patch_prompt_last_only),
        "patch_generation_tokens": bool(args.patch_generation_tokens),
        "use_chat_template": bool(args.use_chat_template),
        "disable_thinking_trace": bool(args.disable_thinking_trace),
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if bool(args.do_sample) else None,
        "top_p": float(args.top_p) if bool(args.do_sample) else None,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
    }


def run_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_text(stable_json_dumps(payload))


def default_run_id_from_fingerprint(
    *,
    args: argparse.Namespace,
    selected_queries: list[dict[str, Any]],
    artifact_sha256: str,
    fingerprint: str,
) -> str:
    model_slug = make_slug(args.model_name.replace("/", "_"), max_len=32)
    query_slug = query_selection_slug(selected_queries)
    regime_slug = make_slug(args.train_regime, max_len=32)
    feature_slug = make_slug(args.feature_name, max_len=32)
    vector_slug = make_slug(args.vector_key, max_len=32)
    generation_slug = "sample" if bool(args.do_sample) else "greedy"
    thinking_slug = "nothink" if bool(args.disable_thinking_trace) else "think"
    chat_slug = "chat" if bool(args.use_chat_template) else "plain"
    return (
        f"{model_slug}_{args.set_type}_{query_slug}_layer{int(args.steering_layer)}_"
        f"{regime_slug}_{feature_slug}_{vector_slug}_art{artifact_sha256[:10]}_"
        f"{chat_slug}_{thinking_slug}_{generation_slug}_tok{int(args.max_new_tokens)}_{fingerprint[:10]}"
    )


def legacy_run_config_compatible(existing_config: dict[str, Any], current_payload: dict[str, Any]) -> bool:
    comparable_keys = [
        "model_name",
        "set_type",
        "query_start",
        "max_queries",
        "query_indices",
        "train_regime",
        "feature_name",
        "vector_key",
        "steering_layer",
        "patch_prompt_last_only",
        "patch_generation_tokens",
        "use_chat_template",
        "disable_thinking_trace",
        "max_input_tokens",
        "max_new_tokens",
        "do_sample",
        "torch_dtype",
        "device_map",
        "attn_implementation",
    ]
    for key in comparable_keys:
        if key in existing_config and existing_config.get(key) != current_payload.get(key):
            return False

    if "n_queries_selected" in existing_config and int(existing_config["n_queries_selected"]) != int(
        current_payload["n_queries_selected"]
    ):
        return False

    existing_artifact = existing_config.get("artifact_path")
    if existing_artifact and Path(str(existing_artifact)).name != str(current_payload.get("artifact_filename")):
        return False

    return True


def existing_run_config_compatible(existing_config: dict[str, Any], fingerprint: str, payload: dict[str, Any]) -> bool:
    existing_fingerprint = existing_config.get("run_fingerprint")
    if existing_fingerprint:
        return str(existing_fingerprint) == fingerprint
    return legacy_run_config_compatible(existing_config, payload)


def assert_output_dir_compatible(
    *,
    output_dir: Path,
    fingerprint: str,
    payload: dict[str, Any],
    allow_mismatch: bool,
) -> None:
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return
    try:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if allow_mismatch:
            print(f"Warning: could not read existing run config at {config_path}: {exc}")
            return
        raise RuntimeError(f"Existing run config is not readable: {config_path}") from exc
    if existing_run_config_compatible(existing_config, fingerprint, payload):
        return
    if allow_mismatch:
        print(f"Warning: existing run config does not match current fingerprint; continuing because override is set: {config_path}")
        return
    raise RuntimeError(
        "Refusing to write into an incompatible TravelPlanner run directory. "
        f"Directory: {output_dir}. Use a different --run-id or --allow-run-config-mismatch if this is intentional."
    )


def find_latest_compatible_run(
    *,
    output_root: Path,
    fingerprint: str,
    payload: dict[str, Any],
    preferred_run_id: str,
) -> Path | None:
    if not output_root.exists():
        return None
    candidates: list[tuple[int, float, Path]] = []
    for config_path in output_root.glob("*/run_config.json"):
        run_dir = config_path.parent
        if run_dir.name == preferred_run_id:
            continue
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not existing_run_config_compatible(existing_config, fingerprint, payload):
            continue
        try:
            mtime = max(config_path.stat().st_mtime, run_dir.stat().st_mtime)
        except OSError:
            mtime = 0.0
        record_count = len(list(run_dir.glob("conditions/*/records/query_*.json")))
        candidates.append((record_count, mtime, run_dir))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0][2]


def build_condition_specs(strengths: list[float], *, include_baseline: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if include_baseline:
        specs.append(
            {
                "condition": "baseline",
                "signed_strength": 0.0,
                "steering_applied": False,
            }
        )
    for strength in strengths:
        if abs(float(strength)) == 0.0:
            continue
        label = (
            f"steer_long_term_plus{abs(int(strength))}"
            if float(strength) > 0
            else f"steer_immediate_minus{abs(int(strength))}"
        )
        specs.append(
            {
                "condition": label,
                "signed_strength": float(strength),
                "steering_applied": True,
            }
        )
    return specs


def assert_baseline_runs_first(condition_specs: list[dict[str, Any]]) -> None:
    baseline_positions = [
        idx for idx, spec in enumerate(condition_specs)
        if str(spec.get("condition")) == "baseline"
    ]
    if baseline_positions and baseline_positions[0] != 0:
        raise RuntimeError(
            "Internal condition ordering error: baseline must run before steered conditions."
        )


def install_travelplanner_import_stubs() -> None:
    if "gradio" not in sys.modules:
        gradio_stub = types.ModuleType("gradio")

        class GradioError(Exception):
            pass

        gradio_stub.Error = GradioError
        sys.modules["gradio"] = gradio_stub

    try:
        importlib.import_module("requests")
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")
        exceptions_stub = types.ModuleType("requests.exceptions")

        class SSLError(Exception):
            pass

        def offline_request(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                "The TravelPlanner offline evaluator imported a requests stub. "
                "Network Google Distance Matrix calls are unavailable in this environment."
            )

        exceptions_stub.SSLError = SSLError
        requests_stub.exceptions = exceptions_stub
        requests_stub.get = offline_request
        sys.modules["requests"] = requests_stub
        sys.modules["requests.exceptions"] = exceptions_stub


def load_travelplanner_evaluators(travelplanner_repo: Path) -> tuple[Any, Any]:
    old_cwd = Path.cwd()
    evaluation_dir = travelplanner_repo / "evaluation"
    sys.path.insert(0, str(travelplanner_repo))
    sys.path.insert(0, str(evaluation_dir))
    install_travelplanner_import_stubs()
    try:
        os.chdir(evaluation_dir)
        commonsense_module = importlib.import_module("commonsense_constraint")
        hard_module = importlib.import_module("hard_constraint")
    finally:
        os.chdir(old_cwd)
    return commonsense_module.evaluation, hard_module.evaluation


def first_bool(value: Any) -> bool | None:
    if isinstance(value, (list, tuple)) and value:
        raw = value[0]
    else:
        raw = value
    if raw is None:
        return None
    return bool(raw)


def macro_pass(info: dict[str, Any] | None) -> bool:
    if info is None:
        return False
    for value in info.values():
        flag = first_bool(value)
        if flag is False:
            return False
    return True


def count_true_constraints(info: dict[str, Any] | None, keys: list[str]) -> int:
    if info is None:
        return 0
    count = 0
    for key in keys:
        flag = first_bool(info.get(key))
        if flag is True:
            count += 1
    return count


def hard_total_for_query(query_record: dict[str, Any]) -> int:
    if "budget" not in query_record or pd.isna(query_record.get("budget")):
        return 0
    total = 1
    local_constraint = query_record.get("local_constraint")
    if isinstance(local_constraint, dict):
        for key in ["house rule", "cuisine", "room type", "transportation"]:
            if local_constraint.get(key) is not None:
                total += 1
    return total


def make_eval_query(query_record: dict[str, Any]) -> dict[str, Any]:
    query = dict(query_record)
    query.pop("reference_information", None)
    query.pop("annotated_plan", None)
    query["days"] = int(query["days"])
    if "visiting_city_number" in query and not pd.isna(query["visiting_city_number"]):
        query["visiting_city_number"] = int(query["visiting_city_number"])
    if "people_number" in query and not pd.isna(query["people_number"]):
        query["people_number"] = int(query["people_number"])
    if "budget" in query and not pd.isna(query["budget"]):
        query["budget"] = int(query["budget"])
    return query


def evaluate_condition(
    *,
    condition_name: str,
    records: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
    travelplanner_repo: Path,
    output_dir: Path,
    set_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    if set_type not in {"train", "validation"}:
        scores = {
            "condition": condition_name,
            "set_type": set_type,
            "n_queries": len(records),
            "evaluation_skipped": True,
            "skip_reason": "TravelPlanner only releases local labels/evaluation inputs for train and validation.",
        }
        return scores, [], pd.DataFrame()

    commonsense_eval, hard_eval = load_travelplanner_evaluators(travelplanner_repo)
    query_by_index = {int(row["dataset_index"]): row for row in query_records}
    details: list[dict[str, Any]] = []

    for record in tqdm(records, desc=f"Evaluate {condition_name}", unit="plan"):
        query_record = query_by_index[int(record["dataset_index"])]
        query = make_eval_query(query_record)
        plan = record.get("plan") or []
        commonsense_info = None
        hard_info = None
        eval_error = None
        if plan:
            try:
                commonsense_info = commonsense_eval(query, plan)
                if (
                    commonsense_info
                    and first_bool(commonsense_info.get("is_not_absent")) is True
                    and first_bool(commonsense_info.get("is_valid_information_in_sandbox")) is True
                ):
                    hard_info = hard_eval(query, plan)
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"

        commonsense_pass = macro_pass(commonsense_info)
        hard_pass = macro_pass(hard_info)
        detail = {
            "condition": condition_name,
            "dataset_index": int(record["dataset_index"]),
            "benchmark_idx": int(record["benchmark_idx"]),
            "level": query_record.get("level"),
            "days": int(query_record.get("days")),
            "delivered_plan": bool(plan),
            "parse_success": bool(record.get("parse_success")),
            "commonsense_pass": bool(commonsense_pass),
            "hard_pass": bool(hard_pass),
            "final_pass": bool(commonsense_pass and hard_pass),
            "commonsense_true_count": count_true_constraints(commonsense_info, COMMONSENSE_KEYS),
            "commonsense_total_count": len(COMMONSENSE_KEYS),
            "hard_true_count": count_true_constraints(hard_info, HARD_KEYS),
            "hard_total_count": hard_total_for_query(query_record),
            "evaluation_error": eval_error,
            "commonsense_constraint": commonsense_info,
            "hard_constraint": hard_info,
        }
        details.append(json_safe(detail))

    n_queries = len(details)
    delivery_count = sum(1 for row in details if row["delivered_plan"])
    commonsense_true = sum(int(row["commonsense_true_count"]) for row in details)
    commonsense_total = sum(int(row["commonsense_total_count"]) for row in details)
    hard_true = sum(int(row["hard_true_count"]) for row in details)
    hard_total = sum(int(row["hard_total_count"]) for row in details)
    commonsense_macro_count = sum(1 for row in details if row["commonsense_pass"])
    hard_macro_count = sum(1 for row in details if row["hard_pass"])
    final_count = sum(1 for row in details if row["final_pass"])

    scores = {
        "condition": condition_name,
        "set_type": set_type,
        "n_queries": n_queries,
        "Delivery Rate": delivery_count / n_queries if n_queries else np.nan,
        "Commonsense Constraint Micro Pass Rate": commonsense_true / commonsense_total if commonsense_total else np.nan,
        "Commonsense Constraint Macro Pass Rate": commonsense_macro_count / n_queries if n_queries else np.nan,
        "Hard Constraint Micro Pass Rate": hard_true / hard_total if hard_total else np.nan,
        "Hard Constraint Macro Pass Rate": hard_macro_count / n_queries if n_queries else np.nan,
        "Final Pass Rate": final_count / n_queries if n_queries else np.nan,
        "delivery_count": delivery_count,
        "commonsense_true_count": commonsense_true,
        "commonsense_total_count": commonsense_total,
        "hard_true_count": hard_true,
        "hard_total_count": hard_total,
        "final_pass_count": final_count,
        "evaluation_skipped": False,
    }

    details_df = pd.DataFrame(details)
    group_rows = []
    if not details_df.empty:
        for (level, days), group_df in details_df.groupby(["level", "days"], dropna=False):
            group_rows.append(
                {
                    "condition": condition_name,
                    "level": level,
                    "days": int(days),
                    "n_queries": int(len(group_df)),
                    "Delivery Rate": float(group_df["delivered_plan"].mean()),
                    "Commonsense Constraint Micro Pass Rate": (
                        float(group_df["commonsense_true_count"].sum() / group_df["commonsense_total_count"].sum())
                        if int(group_df["commonsense_total_count"].sum()) > 0
                        else np.nan
                    ),
                    "Commonsense Constraint Macro Pass Rate": float(group_df["commonsense_pass"].mean()),
                    "Hard Constraint Micro Pass Rate": (
                        float(group_df["hard_true_count"].sum() / group_df["hard_total_count"].sum())
                        if int(group_df["hard_total_count"].sum()) > 0
                        else np.nan
                    ),
                    "Hard Constraint Macro Pass Rate": float(group_df["hard_pass"].mean()),
                    "Final Pass Rate": float(group_df["final_pass"].mean()),
                }
            )
    by_group_df = pd.DataFrame(group_rows)

    metrics_dir = output_dir / "metrics"
    write_json(metrics_dir / f"{condition_name}_scores.json", scores)
    write_jsonl(metrics_dir / f"{condition_name}_evaluation_details.jsonl", details)
    if not by_group_df.empty:
        by_group_df.to_csv(metrics_dir / f"{condition_name}_scores_by_level_day.csv", index=False)
    return scores, details, by_group_df


def condition_dir(output_dir: Path, condition_name: str) -> Path:
    return output_dir / "conditions" / condition_name


def record_path(output_dir: Path, condition_name: str, benchmark_idx: int) -> Path:
    return condition_dir(output_dir, condition_name) / "records" / f"query_{benchmark_idx:04d}.json"


def collect_condition_records(output_dir: Path, condition_name: str, selected_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for query_record in selected_queries:
        path = record_path(output_dir, condition_name, int(query_record["benchmark_idx"]))
        if not path.exists():
            raise FileNotFoundError(f"Missing generated record for condition={condition_name}: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def condition_records_complete(
    *,
    output_dir: Path,
    condition_spec: dict[str, Any],
    selected_queries: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    steering_layer: int,
) -> bool:
    condition_name = str(condition_spec["condition"])
    prompt_by_idx = {int(row["benchmark_idx"]): row for row in prompt_rows}
    for query_record in selected_queries:
        benchmark_idx = int(query_record["benchmark_idx"])
        path = record_path(output_dir, condition_name, benchmark_idx)
        if not path.exists():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if str(record.get("condition")) != condition_name:
            return False
        if int(record.get("benchmark_idx", -1)) != benchmark_idx:
            return False
        if int(record.get("dataset_index", -1)) != int(query_record["dataset_index"]):
            return False
        if bool(record.get("steering_applied")) != bool(condition_spec["steering_applied"]):
            return False
        if not np.isclose(float(record.get("signed_strength", np.nan)), float(condition_spec["signed_strength"])):
            return False
        if int(record.get("steering_layer", steering_layer)) != int(steering_layer):
            return False
        expected_prompt = prompt_by_idx.get(benchmark_idx, {})
        if expected_prompt and record.get("prompt_sha256") != expected_prompt.get("prompt_sha256"):
            return False
    return True


def load_existing_condition_evaluation(output_dir: Path, condition_name: str) -> tuple[dict[str, Any], pd.DataFrame] | None:
    metrics_dir = output_dir / "metrics"
    scores_path = metrics_dir / f"{condition_name}_scores.json"
    if not scores_path.exists():
        return None
    try:
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    by_group_path = metrics_dir / f"{condition_name}_scores_by_level_day.csv"
    if by_group_path.exists():
        try:
            by_group_df = pd.read_csv(by_group_path)
        except Exception:
            by_group_df = pd.DataFrame()
    else:
        by_group_df = pd.DataFrame()
    scores["evaluation_result_source"] = "reused_existing"
    return scores, by_group_df


def compact_record_for_progress(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "condition",
        "signed_strength",
        "steering_applied",
        "steering_layer",
        "set_type",
        "dataset_index",
        "benchmark_idx",
        "level",
        "days",
        "org",
        "dest",
        "visiting_city_number",
        "people_number",
        "budget",
        "parse_success",
        "parse_method",
        "parse_error",
        "input_tokens",
        "output_tokens",
        "generation_seconds",
        "prompt_path",
        "response_path",
        "plan_path",
    ]
    return {key: record.get(key) for key in keys if key in record}


def write_condition_progress(
    *,
    output_dir: Path,
    condition_spec: dict[str, Any],
    records: list[dict[str, Any]],
    total_queries: int,
    status: str,
) -> dict[str, str]:
    condition_name = str(condition_spec["condition"])
    cdir = condition_dir(output_dir, condition_name)
    progress_path = cdir / "progress.json"
    records_jsonl_path = cdir / "records_so_far.jsonl"
    summary_csv_path = cdir / "generation_summary_so_far.csv"
    submission_path = cdir / "evaluation_input_so_far.jsonl"

    ordered_records = sorted(records, key=lambda row: int(row.get("benchmark_idx", 0)))
    write_jsonl(records_jsonl_path, ordered_records)
    pd.DataFrame([compact_record_for_progress(row) for row in ordered_records]).to_csv(summary_csv_path, index=False)
    write_jsonl(
        submission_path,
        [
            {
                "idx": int(record["benchmark_idx"]),
                "query": record["query"],
                "plan": record.get("plan") or [],
            }
            for record in ordered_records
        ],
    )
    progress = {
        "condition": condition_name,
        "signed_strength": float(condition_spec["signed_strength"]),
        "steering_applied": bool(condition_spec["steering_applied"]),
        "status": status,
        "completed_queries": len(ordered_records),
        "total_queries": int(total_queries),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records_jsonl_path": str(records_jsonl_path),
        "summary_csv_path": str(summary_csv_path),
        "evaluation_input_so_far_path": str(submission_path),
    }
    write_json(progress_path, progress)
    return {
        "progress_path": str(progress_path),
        "records_jsonl_path": str(records_jsonl_path),
        "summary_csv_path": str(summary_csv_path),
        "evaluation_input_so_far_path": str(submission_path),
    }


def write_condition_submission(
    *,
    output_dir: Path,
    condition_name: str,
    records: list[dict[str, Any]],
) -> Path:
    rows = [
        {
            "idx": int(record["benchmark_idx"]),
            "query": record["query"],
            "plan": record.get("plan") or [],
        }
        for record in records
    ]
    submission_path = output_dir / "submissions" / f"{condition_name}_evaluation_input.jsonl"
    write_jsonl(submission_path, rows)
    return submission_path


def make_scores_dataframe(score_rows: list[dict[str, Any]]) -> pd.DataFrame:
    scores_df = pd.DataFrame(score_rows)
    if "Final Pass Rate" in scores_df.columns and (scores_df["condition"] == "baseline").any():
        baseline_final = float(scores_df.loc[scores_df["condition"] == "baseline", "Final Pass Rate"].iloc[0])
        scores_df["Final Pass Rate Delta vs Baseline"] = scores_df["Final Pass Rate"] - baseline_final
    return scores_df


def generate_condition_records(
    *,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    output_dir: Path,
    condition_spec: dict[str, Any],
    selected_queries: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    steering_vector: np.ndarray,
) -> list[dict[str, Any]]:
    condition_name = str(condition_spec["condition"])
    signed_strength = float(condition_spec["signed_strength"])
    steering_applied = bool(condition_spec["steering_applied"])
    records: list[dict[str, Any]] = []

    cdir = condition_dir(output_dir, condition_name)
    (cdir / "responses").mkdir(parents=True, exist_ok=True)
    (cdir / "plans").mkdir(parents=True, exist_ok=True)
    (cdir / "records").mkdir(parents=True, exist_ok=True)
    write_condition_progress(
        output_dir=output_dir,
        condition_spec=condition_spec,
        records=records,
        total_queries=len(selected_queries),
        status="running",
    )

    for query_record, prompt_row in tqdm(
        list(zip(selected_queries, prompt_rows)),
        desc=f"Generate {condition_name}",
        unit="query",
    ):
        benchmark_idx = int(query_record["benchmark_idx"])
        rpath = record_path(output_dir, condition_name, benchmark_idx)
        if args.resume_existing and rpath.exists():
            records.append(json.loads(rpath.read_text(encoding="utf-8")))
            write_condition_progress(
                output_dir=output_dir,
                condition_spec=condition_spec,
                records=records,
                total_queries=len(selected_queries),
                status="running",
            )
            continue

        prompt = Path(prompt_row["prompt_path"]).read_text(encoding="utf-8")
        if args.dry_run:
            response = ""
            generation_metadata = {
                "input_tokens": None,
                "output_tokens": 0,
                "generation_seconds": 0.0,
                "generation_kwargs": {},
                "dry_run": True,
            }
        else:
            response, generation_metadata = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                args=args,
                layer=int(args.steering_layer) if steering_applied else None,
                direction=steering_vector if steering_applied else None,
                strength=signed_strength if steering_applied else 0.0,
            )

        plan, parse_info = parse_plan(response, expected_days=int(query_record.get("days", 0) or 0))
        response_path = cdir / "responses" / f"response_{benchmark_idx:04d}.txt"
        plan_path = cdir / "plans" / f"plan_{benchmark_idx:04d}.json"
        response_path.write_text(response + "\n", encoding="utf-8")
        write_json(plan_path, plan)

        record = {
            "condition": condition_name,
            "signed_strength": signed_strength,
            "steering_applied": steering_applied,
            "steering_layer": int(args.steering_layer),
            "dataset_index": int(query_record["dataset_index"]),
            "benchmark_idx": benchmark_idx,
            "set_type": args.set_type,
            "level": query_record.get("level"),
            "days": int(query_record.get("days", 0) or 0),
            "org": query_record.get("org"),
            "dest": query_record.get("dest"),
            "visiting_city_number": query_record.get("visiting_city_number"),
            "people_number": query_record.get("people_number"),
            "budget": query_record.get("budget"),
            "local_constraint": query_record.get("local_constraint"),
            "query": query_record.get("query"),
            "prompt_path": prompt_row["prompt_path"],
            "prompt_sha256": prompt_row["prompt_sha256"],
            "response_path": str(response_path),
            "response_sha256": sha256_text(response),
            "plan_path": str(plan_path),
            "plan": plan,
            **parse_info,
            **generation_metadata,
        }
        write_json(rpath, record)
        records.append(record)
        write_condition_progress(
            output_dir=output_dir,
            condition_spec=condition_spec,
            records=records,
            total_queries=len(selected_queries),
            status="running",
        )
    write_condition_progress(
        output_dir=output_dir,
        condition_spec=condition_spec,
        records=records,
        total_queries=len(selected_queries),
        status="complete",
    )
    return records


def prepare_prompts(output_dir: Path, selected_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for query_record in selected_queries:
        benchmark_idx = int(query_record["benchmark_idx"])
        reference_text = format_reference_information(query_record.get("reference_information"))
        prompt = build_travelplanner_prompt(query_record, reference_text)
        prompt_path = prompt_dir / f"query_{benchmark_idx:04d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        rows.append(
            {
                "dataset_index": int(query_record["dataset_index"]),
                "benchmark_idx": benchmark_idx,
                "prompt_path": str(prompt_path),
                "prompt_sha256": sha256_text(prompt),
                "reference_information_sha256": sha256_text(reference_text),
                "prompt_chars": len(prompt),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "prompt_index.csv", index=False)
    return rows


def parse_strengths(raw: str | None) -> list[float]:
    if raw is None:
        return list(DEFAULT_STRENGTHS)
    strengths = []
    for part in raw.split(","):
        text = part.strip()
        if text:
            strengths.append(float(text))
    return strengths


def parse_artifact_search_roots(raw: str | None, root: Path, result_root_name: str) -> list[Path]:
    if raw is None:
        return default_probe_artifact_search_roots(root, result_root_name)
    return [Path(part).expanduser().resolve() for part in raw.split(",") if part.strip()]


def add_bool_argument(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str | None = None) -> None:
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def dataset_manifest(dataset_dir: Path, travelplanner_repo: Path) -> dict[str, Any]:
    files = {}
    for path in [
        dataset_dir / "train.csv",
        dataset_dir / "validation.csv",
        dataset_dir / "test.csv",
        travelplanner_repo / "database" / "train_ref_info.jsonl",
        travelplanner_repo / "database" / "validation_ref_info.jsonl",
        travelplanner_repo / "database" / "test_ref_info.jsonl",
    ]:
        if path.exists():
            files[str(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    db_files = [
        "database/accommodations/clean_accommodations_2022.csv",
        "database/attractions/attractions.csv",
        "database/background/citySet.txt",
        "database/background/citySet_with_states.txt",
        "database/flights/clean_Flights_2022.csv",
        "database/googleDistanceMatrix/distance.csv",
        "database/restaurants/clean_restaurant_2022.csv",
    ]
    for relative in db_files:
        path = travelplanner_repo / relative
        if path.exists():
            files[str(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    git_commit = None
    git_dir = travelplanner_repo / ".git"
    if git_dir.exists():
        try:
            import subprocess

            git_commit = (
                subprocess.check_output(
                    ["git", "-C", str(travelplanner_repo), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
            )
        except Exception:
            git_commit = None
    return {
        "travelplanner_repo": str(travelplanner_repo),
        "travelplanner_git_commit": git_commit,
        "dataset_dir": str(dataset_dir),
        "files": files,
        "sources": {
            "github": "https://github.com/OSU-NLP-Group/TravelPlanner",
            "huggingface_dataset": "https://huggingface.co/datasets/osunlp/TravelPlanner",
            "huggingface_leaderboard_database_mirror": "https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = find_repo_root(Path.cwd())
    travelplanner_repo = (root / args.travelplanner_repo).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    if not travelplanner_repo.exists():
        raise FileNotFoundError(f"TravelPlanner repository not found: {travelplanner_repo}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"TravelPlanner Hugging Face CSV directory not found: {dataset_dir}")

    records_all = load_travelplanner_split(dataset_dir, args.set_type)
    selected_queries = select_queries(
        records_all,
        query_indices=args.query_indices,
        query_start=int(args.query_start),
        max_queries=args.max_queries,
    )
    if not selected_queries:
        raise ValueError("No TravelPlanner queries selected.")

    artifact_search_roots = parse_artifact_search_roots(args.artifact_search_roots, root, args.result_root_name)
    artifact_path, metadata_path = locate_latest_probe_artifacts(
        artifact_search_roots,
        artifact_path_override=args.artifact_path,
        metadata_path_override=args.metadata_path,
        required_train_regime=args.train_regime,
        artifact_file_prefix=args.artifact_file_prefix,
    )
    probe_payload = load_probe_vector(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        train_regime=args.train_regime,
        feature_name=args.feature_name,
        vector_key=args.vector_key,
        layer=int(args.steering_layer),
    )

    strengths = parse_strengths(args.strengths)
    condition_specs = build_condition_specs(strengths, include_baseline=not args.no_baseline)
    if args.conditions:
        requested = {name.strip() for name in args.conditions.split(",") if name.strip()}
        condition_specs = [spec for spec in condition_specs if spec["condition"] in requested]
        missing = requested - {spec["condition"] for spec in condition_specs}
        if missing:
            raise ValueError(f"Requested condition(s) not available after strength expansion: {sorted(missing)}")
    if not condition_specs:
        raise ValueError("No conditions selected.")
    assert_baseline_runs_first(condition_specs)

    artifact_sha256 = sha256_file(artifact_path)
    metadata_sha256 = sha256_file(metadata_path) if metadata_path is not None and metadata_path.exists() else None
    dataset_split_sha256 = sha256_file(dataset_dir / f"{args.set_type}.csv")
    fingerprint_payload = run_fingerprint_payload(
        args=args,
        selected_queries=selected_queries,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        artifact_sha256=artifact_sha256,
        metadata_sha256=metadata_sha256,
        dataset_split_sha256=dataset_split_sha256,
    )
    fingerprint = run_fingerprint(fingerprint_payload)
    default_run_id = default_run_id_from_fingerprint(
        args=args,
        selected_queries=selected_queries,
        artifact_sha256=artifact_sha256,
        fingerprint=fingerprint,
    )
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id_strategy = "explicit"
    if args.run_id:
        run_id = args.run_id
        output_dir = output_root / run_id
    else:
        run_id_strategy = "config_fingerprint"
        output_dir = output_root / default_run_id
        if bool(args.auto_resume_compatible_run) and bool(args.resume_existing) and not output_dir.exists():
            compatible_run_dir = find_latest_compatible_run(
                output_root=output_root,
                fingerprint=fingerprint,
                payload=fingerprint_payload,
                preferred_run_id=default_run_id,
            )
            if compatible_run_dir is not None:
                output_dir = compatible_run_dir
                run_id_strategy = "auto_resumed_latest_compatible"
    run_id = output_dir.name
    assert_output_dir_compatible(
        output_dir=output_dir,
        fingerprint=fingerprint,
        payload=fingerprint_payload,
        allow_mismatch=bool(args.allow_run_config_mismatch),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = prepare_prompts(output_dir, selected_queries)
    selected_query_rows = [
        {
            "dataset_index": int(row["dataset_index"]),
            "benchmark_idx": int(row["benchmark_idx"]),
            "set_type": args.set_type,
            "level": row.get("level"),
            "days": row.get("days"),
            "org": row.get("org"),
            "dest": row.get("dest"),
            "visiting_city_number": row.get("visiting_city_number"),
            "people_number": row.get("people_number"),
            "budget": row.get("budget"),
            "local_constraint": row.get("local_constraint"),
            "query": row.get("query"),
        }
        for row in selected_queries
    ]
    write_jsonl(output_dir / "selected_queries.jsonl", selected_query_rows)

    complete_generation_conditions = set()
    missing_generation_conditions: list[str] = []
    if not args.skip_generation:
        for condition_spec in condition_specs:
            condition_name = str(condition_spec["condition"])
            if bool(args.resume_existing) and condition_records_complete(
                output_dir=output_dir,
                condition_spec=condition_spec,
                selected_queries=selected_queries,
                prompt_rows=prompt_rows,
                steering_layer=int(args.steering_layer),
            ):
                complete_generation_conditions.add(condition_name)
            else:
                missing_generation_conditions.append(condition_name)

    model = None
    tokenizer = None
    device_summary = "not_loaded"
    needs_model = (not args.dry_run) and (not args.skip_generation) and bool(missing_generation_conditions)
    if needs_model:
        model, tokenizer, device_summary = load_model_and_tokenizer(args)
        print("Model:", args.model_name)
        print("Device:", device_summary)
    else:
        if args.dry_run:
            print("[dry-run] Skipping model load.")
        elif args.skip_generation:
            print("[skip-generation] Skipping model load; collecting existing records only.")
        else:
            print("[reuse] All selected condition records already exist; skipping model load.")
    if complete_generation_conditions:
        print("[reuse] Complete generated conditions:", ", ".join(sorted(complete_generation_conditions)))
    if missing_generation_conditions:
        print("[generate] Missing/incomplete conditions:", ", ".join(missing_generation_conditions))

    run_config = {
        "run_id": run_id,
        "run_id_strategy": run_id_strategy,
        "default_run_id": default_run_id,
        "run_fingerprint": fingerprint,
        "run_fingerprint_payload": fingerprint_payload,
        "run_fingerprint_note": (
            "Strengths and selected condition subsets are intentionally not part of the run fingerprint; "
            "condition directories are additive within a compatible run."
        ),
        "model_name": args.model_name,
        "set_type": args.set_type,
        "n_queries_selected": len(selected_queries),
        "query_start": args.query_start,
        "max_queries": args.max_queries,
        "query_indices": args.query_indices,
        "conditions": condition_specs,
        "strengths": strengths,
        "resume_existing": bool(args.resume_existing),
        "auto_resume_compatible_run": bool(args.auto_resume_compatible_run),
        "reuse_existing_evaluation": bool(args.reuse_existing_evaluation),
        "complete_generation_conditions_at_start": sorted(complete_generation_conditions),
        "missing_generation_conditions_at_start": missing_generation_conditions,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "metadata_sha256": metadata_sha256,
        "dataset_split_sha256": dataset_split_sha256,
        "train_regime": args.train_regime,
        "feature_name": args.feature_name,
        "vector_key": args.vector_key,
        "steering_layer": int(args.steering_layer),
        "steering_vector_norm": float(probe_payload["steering_vector_norm"]),
        "available_train_regimes": probe_payload["available_train_regimes"],
        "available_feature_names": probe_payload["available_feature_names"],
        "available_layers": probe_payload["available_layers"],
        "patch_prompt_last_only": bool(args.patch_prompt_last_only),
        "patch_generation_tokens": bool(args.patch_generation_tokens),
        "use_chat_template": bool(args.use_chat_template),
        "disable_thinking_trace": bool(args.disable_thinking_trace),
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "device_summary": device_summary,
        "travelplanner_repo": str(travelplanner_repo),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "probe_metadata.json", probe_payload["metadata"])
    write_json(output_dir / "dataset_manifest.json", dataset_manifest(dataset_dir, travelplanner_repo))

    score_rows: list[dict[str, Any]] = []
    all_group_dfs: list[pd.DataFrame] = []
    artifact_rows: list[dict[str, Any]] = [
        {"artifact": "run_config", "path": str(output_dir / "run_config.json")},
        {"artifact": "dataset_manifest", "path": str(output_dir / "dataset_manifest.json")},
        {"artifact": "probe_metadata", "path": str(output_dir / "probe_metadata.json")},
        {"artifact": "selected_queries", "path": str(output_dir / "selected_queries.jsonl")},
        {"artifact": "prompt_index", "path": str(output_dir / "prompt_index.csv")},
    ]

    try:
        for condition_spec in condition_specs:
            condition_name = str(condition_spec["condition"])
            if args.skip_generation:
                condition_records = collect_condition_records(output_dir, condition_name, selected_queries)
            elif condition_name in complete_generation_conditions:
                condition_records = collect_condition_records(output_dir, condition_name, selected_queries)
                write_condition_progress(
                    output_dir=output_dir,
                    condition_spec=condition_spec,
                    records=condition_records,
                    total_queries=len(selected_queries),
                    status="complete_reused",
                )
                print(f"[reuse] {condition_name}: reused {len(condition_records)} generated records.")
            else:
                condition_records = generate_condition_records(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    output_dir=output_dir,
                    condition_spec=condition_spec,
                    selected_queries=selected_queries,
                    prompt_rows=prompt_rows,
                    steering_vector=probe_payload["steering_vector"],
                )

            submission_path = write_condition_submission(
                output_dir=output_dir,
                condition_name=condition_name,
                records=condition_records,
            )
            artifact_rows.extend(
                [
                    {"artifact": f"{condition_name}_condition_dir", "path": str(condition_dir(output_dir, condition_name))},
                    {"artifact": f"{condition_name}_submission", "path": str(submission_path)},
                ]
            )

            if not args.no_evaluate:
                existing_eval = (
                    load_existing_condition_evaluation(output_dir, condition_name)
                    if bool(args.reuse_existing_evaluation)
                    else None
                )
                if existing_eval is not None:
                    scores, by_group_df = existing_eval
                    print(f"[reuse] {condition_name}: reused existing evaluation metrics.")
                else:
                    scores, _details, by_group_df = evaluate_condition(
                        condition_name=condition_name,
                        records=condition_records,
                        query_records=selected_queries,
                        travelplanner_repo=travelplanner_repo,
                        output_dir=output_dir,
                        set_type=args.set_type,
                    )
                    scores["evaluation_result_source"] = "computed_now"
                scores.update(
                    {
                        "signed_strength": float(condition_spec["signed_strength"]),
                        "steering_applied": bool(condition_spec["steering_applied"]),
                    }
                )
                score_rows.append(scores)
                if not by_group_df.empty:
                    all_group_dfs.append(by_group_df)
                make_scores_dataframe(score_rows).to_csv(output_dir / "scores_so_far.csv", index=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_gpu_cache()

    if score_rows:
        scores_df = make_scores_dataframe(score_rows)
        scores_path = output_dir / "scores.csv"
        scores_df.to_csv(scores_path, index=False)
        artifact_rows.append({"artifact": "scores_csv", "path": str(scores_path)})
        artifact_rows.append({"artifact": "scores_so_far_csv", "path": str(output_dir / "scores_so_far.csv")})
    if all_group_dfs:
        group_scores_path = output_dir / "scores_by_level_day.csv"
        pd.concat(all_group_dfs, ignore_index=True).to_csv(group_scores_path, index=False)
        artifact_rows.append({"artifact": "scores_by_level_day_csv", "path": str(group_scores_path)})

    artifact_index_path = output_dir / "artifact_index.csv"
    pd.DataFrame(artifact_rows).to_csv(artifact_index_path, index=False)

    result = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "n_queries_selected": len(selected_queries),
        "conditions": [spec["condition"] for spec in condition_specs],
        "artifact_index_path": str(artifact_index_path),
        "scores_path": str(output_dir / "scores.csv") if score_rows else None,
    }
    write_json(output_dir / "run_summary.json", result)
    print("Saved output dir:", output_dir)
    print("Saved artifact index:", artifact_index_path)
    if score_rows:
        print("Saved scores:", output_dir / "scores.csv")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and temporal-probe-steered Qwen3-32B on TravelPlanner sole-planning mode. "
            "Defaults use the Qwen3-32B time-utility MM mean-answer-token probe vector at layer 40."
        )
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--result-root-name", type=str, default="qwen3_32b")
    parser.add_argument("--travelplanner-repo", type=str, default=DEFAULT_TRAVELPLANNER_REPO)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--set-type", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--query-start", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-indices", type=str, default=None, help="1-based indices/ranges, e.g. 1,2,10-15.")

    parser.add_argument("--artifact-file-prefix", type=str, default="qwen3_32b_question_only_probe")
    parser.add_argument("--artifact-search-roots", type=str, default=None, help="Comma-separated roots to scan.")
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--train-regime", type=str, default="explicit_train_only")
    parser.add_argument("--feature-name", type=str, default="mean_answer_tokens")
    parser.add_argument("--vector-key", type=str, default="mm_probe_vectors")
    parser.add_argument("--steering-layer", type=int, default=40)
    parser.add_argument(
        "--strengths",
        type=str,
        default=None,
        help="Comma-separated signed strengths. Default alternates extremes: -128,+128,-91,+91,-64,+64,...",
    )
    parser.add_argument("--conditions", type=str, default=None, help="Comma-separated condition names to run after expansion.")
    parser.add_argument("--no-baseline", action="store_true")
    add_bool_argument(parser, "--patch-prompt-last-only", default=True)
    add_bool_argument(parser, "--patch-generation-tokens", default=True)

    add_bool_argument(parser, "--use-chat-template", default=True)
    add_bool_argument(parser, "--disable-thinking-trace", default=True)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default=None, help="Set to auto for Accelerate model sharding, or none for .to(cuda).")
    parser.add_argument("--attn-implementation", type=str, default=None)
    add_bool_argument(parser, "--require-cuda", default=True)

    add_bool_argument(parser, "--resume-existing", default=True)
    add_bool_argument(
        parser,
        "--auto-resume-compatible-run",
        default=True,
        help_text="When --run-id is omitted, resume the latest compatible existing run if the config-derived run id is absent.",
    )
    add_bool_argument(parser, "--reuse-existing-evaluation", default=True)
    parser.add_argument(
        "--allow-run-config-mismatch",
        action="store_true",
        help="Allow writing into an output directory whose existing run_config.json does not match the current fingerprint.",
    )
    parser.add_argument("--skip-generation", action="store_true", help="Only collect/evaluate existing per-query records.")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts/config and empty plans without loading the model.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(f"TravelPlanner run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
