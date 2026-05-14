#!/usr/bin/env python3
"""Probe-guided temporal interventions across multiple causal LMs.

This script is intentionally additive: it does not modify existing probe,
validation, or notebook code. It reuses probe result CSVs to select candidate
layers, then runs causal follow-ups:

1. steering: add a temporal direction to residual-stream activations
2. activation-patching: patch clean activations into corrupted prompts
3. attribution-patching: first-order patching estimate via activation gradients
4. ablation: zero or mean-ablate residual/attention/MLP outputs

The default pair dataset uses the existing clean/corrupted temporal
classification format:

[
  {
    "clean": {"question": "... The answer is:", "answer": " short"},
    "corrupted": {"question": "... The answer is:", "answer": " long"}
  }
]
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

import pandas as pd
import torch
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - convenience for minimal analysis envs
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

MODEL_DEFAULT_PATCH_DATASETS = {
    "qwen3-4b": ROOT / "data/raw/temporal_scope_for_attribution_patching_for_qwen3.json",
    "Qwen/Qwen3-4B": ROOT / "data/raw/temporal_scope_for_attribution_patching_for_qwen3.json",
    "phi-3-mini-4k-instruct": ROOT / "data/raw/temporal_scope_for_attribution_patching_for_phi3.json",
    "microsoft/Phi-3-mini-4k-instruct": ROOT / "data/raw/temporal_scope_for_attribution_patching_for_phi3.json",
}

DEFAULT_CLASSIFICATION_DATASET = ROOT / "data/raw/temporal_scope_for_attribution_patching_for_qwen3.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/temporal_interventions"

Component = Literal["resid", "attn", "mlp"]


@dataclass(frozen=True)
class TemporalPair:
    clean_prompt: str
    clean_answer: str
    corrupted_prompt: str
    corrupted_answer: str
    category: str = "unknown"


@dataclass
class ExperimentRow:
    experiment: str
    model_alias: str
    model_name: str
    layer: int
    component: str
    pair_index: int
    metric_name: str
    baseline_clean: float
    baseline_corrupted: float
    intervention_value: float
    normalized_effect: float | None
    strength: float | None = None
    direction: str | None = None
    ablation_mode: str | None = None
    layer_source_method: str | None = None
    layer_selection: str | None = None
    selected_layers: str | None = None


def resolve_model_name(model_alias_or_name: str) -> str:
    return SUPPORTED_MODELS.get(model_alias_or_name, model_alias_or_name)


def make_model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def choose_pair_dataset(model_alias: str, requested_path: str | None) -> Path:
    if requested_path:
        return repo_path(requested_path)
    model_name = resolve_model_name(model_alias)
    return MODEL_DEFAULT_PATCH_DATASETS.get(model_alias) or MODEL_DEFAULT_PATCH_DATASETS.get(
        model_name, DEFAULT_CLASSIFICATION_DATASET
    )


def load_classification_pairs(path: Path, max_pairs: int | None = None) -> list[TemporalPair]:
    with open(path) as f:
        rows = json.load(f)

    pairs = []
    for row in rows:
        pairs.append(
            TemporalPair(
                clean_prompt=row["clean"]["question"],
                clean_answer=row["clean"]["answer"],
                corrupted_prompt=row["corrupted"]["question"],
                corrupted_answer=row["corrupted"]["answer"],
                category=row.get("category", "classification"),
            )
        )

    return pairs[:max_pairs] if max_pairs is not None else pairs


def extract_ab_letter(answer: str) -> str:
    if "(A)" in answer:
        return " A"
    if "(B)" in answer:
        return " B"
    raise ValueError(f"Could not find '(A)' or '(B)' in answer: {answer!r}")


def load_caa_pairs(path: Path, max_pairs: int | None = None) -> list[TemporalPair]:
    with open(path) as f:
        data = json.load(f)
    rows = data["pairs"] if isinstance(data, dict) and "pairs" in data else data

    pairs = []
    for row in rows:
        immediate = row["immediate"]
        long_term = row["long_term"]
        prompt = (
            f"{row['question']}\n\nChoices:\n"
            f"{immediate}\n"
            f"{long_term}\n\nAnswer:"
        )
        pairs.append(
            TemporalPair(
                clean_prompt=prompt,
                clean_answer=extract_ab_letter(immediate),
                corrupted_prompt=prompt,
                corrupted_answer=extract_ab_letter(long_term),
                category=row.get("category", "caa"),
            )
        )

    return pairs[:max_pairs] if max_pairs is not None else pairs


def load_pairs(path: Path, pair_format: str, max_pairs: int | None = None) -> list[TemporalPair]:
    if pair_format == "classification":
        return load_classification_pairs(path, max_pairs=max_pairs)
    if pair_format == "caa":
        return load_caa_pairs(path, max_pairs=max_pairs)
    raise ValueError(f"Unsupported pair format: {pair_format}")


def answer_token_id(tokenizer, answer: str) -> int:
    token_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not token_ids:
        raise ValueError(f"Answer tokenized to empty sequence: {answer!r}")
    return int(token_ids[0])


def get_model_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_decoder_blocks(model) -> list[torch.nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError(
        "Could not locate decoder blocks. Add this architecture to get_decoder_blocks()."
    )


def get_component_module(block: torch.nn.Module, component: Component) -> torch.nn.Module:
    if component == "resid":
        return block
    if component == "attn":
        for name in ("attn", "self_attn", "attention"):
            if hasattr(block, name):
                return getattr(block, name)
    if component == "mlp":
        for name in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, name):
                return getattr(block, name)
    raise ValueError(f"Could not locate component {component!r} on block {type(block).__name__}")


def get_first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_first_tensor(output, tensor: torch.Tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def load_model_and_tokenizer(
    model_name: str,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
    attn_implementation: str | None,
    device_map: str,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required to load models. Run this script from the "
            "project environment, for example `.venv/bin/python ...`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": "auto",
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if device_map == "auto":
        kwargs["device_map"] = "auto"
    elif device_map == "single" and torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif device_map != "single":
        raise ValueError(f"Unsupported device-map: {device_map}")

    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not torch.cuda.is_available() and device_map == "single":
        model.to("cpu")
    model.eval()
    model.requires_grad_(False)

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def tokenize_one(tokenizer, prompt: str, device: torch.device, max_length: int | None):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def final_token_hidden(model, tokenizer, prompt: str, layer: int, max_length: int | None) -> torch.Tensor:
    device = get_model_device(model)
    inputs = tokenize_one(tokenizer, prompt, device, max_length)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer + 1][0, -1, :].detach().float().cpu()


def mean_temporal_direction(
    model,
    tokenizer,
    pairs: list[TemporalPair],
    layer: int,
    max_length: int | None,
    direction: str,
) -> torch.Tensor:
    clean_acts = []
    corrupted_acts = []
    for pair in pairs:
        clean_text = pair.clean_prompt + pair.clean_answer
        corrupted_text = pair.corrupted_prompt + pair.corrupted_answer
        clean_acts.append(final_token_hidden(model, tokenizer, clean_text, layer, max_length))
        corrupted_acts.append(final_token_hidden(model, tokenizer, corrupted_text, layer, max_length))

    clean_mean = torch.stack(clean_acts).mean(dim=0)
    corrupted_mean = torch.stack(corrupted_acts).mean(dim=0)
    vector = corrupted_mean - clean_mean
    if direction == "clean":
        vector = -vector
    norm = vector.norm().clamp_min(1e-12)
    return vector / norm


@contextmanager
def temporary_hook(module: torch.nn.Module, hook_fn: Callable):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def make_add_hook(vector: torch.Tensor, strength: float):
    def hook(_module, _inputs, output):
        tensor = get_first_tensor(output)
        patch = vector.to(device=tensor.device, dtype=tensor.dtype) * strength
        patched = tensor.clone()
        patched[:, -1, :] = patched[:, -1, :] + patch
        return replace_first_tensor(output, patched)

    return hook


def make_patch_hook(source_activation: torch.Tensor):
    def hook(_module, _inputs, output):
        tensor = get_first_tensor(output)
        patch = source_activation.to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone()
        patched[:, -1, :] = patch
        return replace_first_tensor(output, patched)

    return hook


def make_ablation_hook(mode: str, replacement: torch.Tensor | None = None):
    def hook(_module, _inputs, output):
        tensor = get_first_tensor(output)
        patched = tensor.clone()
        if mode == "zero":
            patched[:, -1, :] = 0
        elif mode == "mean":
            if replacement is None:
                raise ValueError("mean ablation requires a replacement activation")
            patched[:, -1, :] = replacement.to(device=tensor.device, dtype=tensor.dtype)
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        return replace_first_tensor(output, patched)

    return hook


def logits_for_prompt(
    model,
    tokenizer,
    prompt: str,
    max_length: int | None,
    hook_spec: tuple[int, Component, Callable] | None = None,
) -> torch.Tensor:
    device = get_model_device(model)
    blocks = get_decoder_blocks(model)
    inputs = tokenize_one(tokenizer, prompt, device, max_length)

    if hook_spec is None:
        with torch.no_grad():
            return model(**inputs, use_cache=False).logits[0, -1, :].detach().float().cpu()

    layer, component, hook_fn = hook_spec
    module = get_component_module(blocks[layer], component)
    with torch.no_grad(), temporary_hook(module, hook_fn):
        return model(**inputs, use_cache=False).logits[0, -1, :].detach().float().cpu()


def logit_diff(logits: torch.Tensor, positive_id: int, negative_id: int) -> float:
    return float((logits[positive_id] - logits[negative_id]).item())


def normalized_effect(value: float, clean: float, corrupted: float) -> float | None:
    denom = clean - corrupted
    if math.isclose(denom, 0.0, abs_tol=1e-8):
        return None
    return (value - corrupted) / denom


def cache_activation(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    component: Component,
    max_length: int | None,
) -> torch.Tensor:
    blocks = get_decoder_blocks(model)
    module = get_component_module(blocks[layer], component)
    captured = {}

    def save_hook(_module, _inputs, output):
        tensor = get_first_tensor(output)
        captured["activation"] = tensor[:, -1, :].detach().float().cpu()
        return output

    device = get_model_device(model)
    inputs = tokenize_one(tokenizer, prompt, device, max_length)
    with torch.no_grad(), temporary_hook(module, save_hook):
        model(**inputs, use_cache=False)
    if "activation" not in captured:
        raise RuntimeError("Activation hook did not capture any tensor")
    return captured["activation"][0]


def selected_layers_from_probe_csv(
    model_name: str,
    method: str,
    layer_mode: str,
    top_k: int,
) -> list[int]:
    model_tag = make_model_tag(model_name)
    path = ROOT / "research/results" / method / f"{model_tag}_temporal_probe_{method}_implicit_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing probe CSV for layer selection: {path}")

    df = pd.read_csv(path).sort_values(["test_accuracy", "cv_accuracy_mean"], ascending=False)
    if layer_mode == "best":
        return [int(df.iloc[0].layer)]
    if layer_mode == "top-k":
        return sorted(int(layer) for layer in df.head(top_k).layer.tolist())
    raise ValueError(f"Layer mode {layer_mode!r} needs explicit parsing before CSV selection")


def parse_layers(layer_arg: str, model_name: str, method: str, n_layers: int, top_k: int) -> list[int]:
    if layer_arg == "all":
        return list(range(n_layers))
    if layer_arg in {"best", "top-k"}:
        return selected_layers_from_probe_csv(model_name, method, layer_arg, top_k)
    layers = sorted({int(part.strip()) for part in layer_arg.split(",") if part.strip()})
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range for {model_name}: {bad}; n_layers={n_layers}")
    return layers


def run_steering(
    model,
    tokenizer,
    model_alias: str,
    model_name: str,
    pairs: list[TemporalPair],
    layers: list[int],
    components: list[Component],
    strengths: list[float],
    direction_pairs: list[TemporalPair],
    max_length: int | None,
    direction: str,
) -> list[ExperimentRow]:
    rows = []
    for layer in layers:
        vector = mean_temporal_direction(
            model, tokenizer, direction_pairs, layer, max_length, direction=direction
        )
        for component in components:
            for pair_index, pair in enumerate(tqdm(pairs, desc=f"steer L{layer} {component}")):
                clean_id = answer_token_id(tokenizer, pair.clean_answer)
                corrupted_id = answer_token_id(tokenizer, pair.corrupted_answer)
                clean_logits = logits_for_prompt(model, tokenizer, pair.clean_prompt, max_length)
                corrupted_logits = logits_for_prompt(model, tokenizer, pair.corrupted_prompt, max_length)
                baseline_clean = logit_diff(clean_logits, corrupted_id, clean_id)
                baseline_corrupted = logit_diff(corrupted_logits, corrupted_id, clean_id)

                for strength in strengths:
                    hook = make_add_hook(vector, strength)
                    steered_logits = logits_for_prompt(
                        model,
                        tokenizer,
                        pair.clean_prompt,
                        max_length,
                        hook_spec=(layer, component, hook),
                    )
                    value = logit_diff(steered_logits, corrupted_id, clean_id)
                    rows.append(
                        ExperimentRow(
                            experiment="steering",
                            model_alias=model_alias,
                            model_name=model_name,
                            layer=layer,
                            component=component,
                            pair_index=pair_index,
                            metric_name="long_minus_short_logit_diff",
                            baseline_clean=baseline_clean,
                            baseline_corrupted=baseline_corrupted,
                            intervention_value=value,
                            normalized_effect=None,
                            strength=strength,
                            direction=direction,
                        )
                    )
    return rows


def run_activation_patching(
    model,
    tokenizer,
    model_alias: str,
    model_name: str,
    pairs: list[TemporalPair],
    layers: list[int],
    components: list[Component],
    max_length: int | None,
) -> list[ExperimentRow]:
    rows = []
    for layer in layers:
        for component in components:
            for pair_index, pair in enumerate(tqdm(pairs, desc=f"act-patch L{layer} {component}")):
                clean_id = answer_token_id(tokenizer, pair.clean_answer)
                corrupted_id = answer_token_id(tokenizer, pair.corrupted_answer)
                clean_logits = logits_for_prompt(model, tokenizer, pair.clean_prompt, max_length)
                corrupted_logits = logits_for_prompt(model, tokenizer, pair.corrupted_prompt, max_length)
                baseline_clean = logit_diff(clean_logits, clean_id, corrupted_id)
                baseline_corrupted = logit_diff(corrupted_logits, clean_id, corrupted_id)

                source = cache_activation(
                    model, tokenizer, pair.clean_prompt, layer, component, max_length
                )
                hook = make_patch_hook(source)
                patched_logits = logits_for_prompt(
                    model,
                    tokenizer,
                    pair.corrupted_prompt,
                    max_length,
                    hook_spec=(layer, component, hook),
                )
                value = logit_diff(patched_logits, clean_id, corrupted_id)
                rows.append(
                    ExperimentRow(
                        experiment="activation_patching",
                        model_alias=model_alias,
                        model_name=model_name,
                        layer=layer,
                        component=component,
                        pair_index=pair_index,
                        metric_name="clean_minus_corrupted_logit_diff",
                        baseline_clean=baseline_clean,
                        baseline_corrupted=baseline_corrupted,
                        intervention_value=value,
                        normalized_effect=normalized_effect(value, baseline_clean, baseline_corrupted),
                    )
                )
    return rows


def attribution_patch_value(
    model,
    tokenizer,
    pair: TemporalPair,
    layer: int,
    component: Component,
    max_length: int | None,
) -> tuple[float, float, float, float | None]:
    clean_id = answer_token_id(tokenizer, pair.clean_answer)
    corrupted_id = answer_token_id(tokenizer, pair.corrupted_answer)
    clean_logits = logits_for_prompt(model, tokenizer, pair.clean_prompt, max_length)
    corrupted_logits = logits_for_prompt(model, tokenizer, pair.corrupted_prompt, max_length)
    baseline_clean = logit_diff(clean_logits, clean_id, corrupted_id)
    baseline_corrupted = logit_diff(corrupted_logits, clean_id, corrupted_id)

    source = cache_activation(model, tokenizer, pair.clean_prompt, layer, component, max_length)
    blocks = get_decoder_blocks(model)
    module = get_component_module(blocks[layer], component)
    captured = {}

    def grad_hook(_module, _inputs, output):
        tensor = get_first_tensor(output)
        tracked = tensor.detach().requires_grad_(True)
        captured["activation"] = tracked
        return replace_first_tensor(output, tracked)

    device = get_model_device(model)
    inputs = tokenize_one(tokenizer, pair.corrupted_prompt, device, max_length)
    model.zero_grad(set_to_none=True)
    with temporary_hook(module, grad_hook):
        logits = model(**inputs, use_cache=False).logits[0, -1, :]
        metric = logits[clean_id] - logits[corrupted_id]
        metric.backward()

    target = captured["activation"][:, -1, :].detach().float().cpu()[0]
    grad = captured["activation"].grad[:, -1, :].detach().float().cpu()[0]
    value = float((grad * (source - target)).sum().item())
    del clean_logits, corrupted_logits, source, inputs, logits, metric
    model.zero_grad(set_to_none=True)
    clear_memory()
    return baseline_clean, baseline_corrupted, value, normalized_effect(
        baseline_corrupted + value, baseline_clean, baseline_corrupted
    )


def run_attribution_patching(
    model,
    tokenizer,
    model_alias: str,
    model_name: str,
    pairs: list[TemporalPair],
    layers: list[int],
    components: list[Component],
    max_length: int | None,
) -> list[ExperimentRow]:
    rows = []
    for layer in layers:
        for component in components:
            for pair_index, pair in enumerate(tqdm(pairs, desc=f"attr-patch L{layer} {component}")):
                baseline_clean, baseline_corrupted, value, effect = attribution_patch_value(
                    model, tokenizer, pair, layer, component, max_length
                )
                rows.append(
                    ExperimentRow(
                        experiment="attribution_patching",
                        model_alias=model_alias,
                        model_name=model_name,
                        layer=layer,
                        component=component,
                        pair_index=pair_index,
                        metric_name="estimated_clean_minus_corrupted_logit_diff_change",
                        baseline_clean=baseline_clean,
                        baseline_corrupted=baseline_corrupted,
                        intervention_value=value,
                        normalized_effect=effect,
                    )
                )
    return rows


def mean_replacement_activation(
    model,
    tokenizer,
    pairs: list[TemporalPair],
    layer: int,
    component: Component,
    max_length: int | None,
) -> torch.Tensor:
    activations = []
    for pair in pairs:
        activations.append(cache_activation(model, tokenizer, pair.clean_prompt, layer, component, max_length))
        activations.append(cache_activation(model, tokenizer, pair.corrupted_prompt, layer, component, max_length))
    return torch.stack(activations).mean(dim=0)


def run_ablation(
    model,
    tokenizer,
    model_alias: str,
    model_name: str,
    pairs: list[TemporalPair],
    layers: list[int],
    components: list[Component],
    max_length: int | None,
    ablation_mode: str,
) -> list[ExperimentRow]:
    rows = []
    for layer in layers:
        for component in components:
            replacement = None
            if ablation_mode == "mean":
                replacement = mean_replacement_activation(
                    model, tokenizer, pairs, layer, component, max_length
                )
            for pair_index, pair in enumerate(tqdm(pairs, desc=f"ablate L{layer} {component}")):
                clean_id = answer_token_id(tokenizer, pair.clean_answer)
                corrupted_id = answer_token_id(tokenizer, pair.corrupted_answer)
                clean_logits = logits_for_prompt(model, tokenizer, pair.clean_prompt, max_length)
                corrupted_logits = logits_for_prompt(model, tokenizer, pair.corrupted_prompt, max_length)
                baseline_clean = logit_diff(clean_logits, clean_id, corrupted_id)
                baseline_corrupted = logit_diff(corrupted_logits, clean_id, corrupted_id)

                hook = make_ablation_hook(ablation_mode, replacement=replacement)
                ablated_logits = logits_for_prompt(
                    model,
                    tokenizer,
                    pair.clean_prompt,
                    max_length,
                    hook_spec=(layer, component, hook),
                )
                value = logit_diff(ablated_logits, clean_id, corrupted_id)
                rows.append(
                    ExperimentRow(
                        experiment="ablation",
                        model_alias=model_alias,
                        model_name=model_name,
                        layer=layer,
                        component=component,
                        pair_index=pair_index,
                        metric_name="clean_minus_corrupted_logit_diff_after_ablation",
                        baseline_clean=baseline_clean,
                        baseline_corrupted=baseline_corrupted,
                        intervention_value=value,
                        normalized_effect=normalized_effect(value, baseline_clean, baseline_corrupted),
                        ablation_mode=ablation_mode,
                    )
                )
    return rows


def slug(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(",", "-")
        .replace(" ", "")
        .replace("_", "-")
    )


def write_outputs(
    rows: list[ExperimentRow],
    output_dir: Path,
    model_tag: str,
    experiment: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    layer_selection = slug(str(metadata["layer_selection"]))
    layer_source_method = slug(str(metadata["layer_source_method"]))
    stem = f"{model_tag}_{experiment}_layers-{layer_selection}_from-{layer_source_method}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    payload = {
        "metadata": {
            "experiment": experiment,
            "timestamp": timestamp,
            "n_rows": len(rows),
            **metadata,
        },
        "rows": [asdict(row) for row in rows],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    print(f"Saved {len(rows)} rows to {json_path.relative_to(ROOT)}")
    if rows:
        print(f"Saved CSV to {csv_path.relative_to(ROOT)}")


def parse_components(values: Iterable[str]) -> list[Component]:
    allowed = {"resid", "attn", "mlp"}
    components = []
    for value in values:
        if value not in allowed:
            raise ValueError(f"Unsupported component {value!r}; choose from {sorted(allowed)}")
        components.append(value)  # type: ignore[arg-type]
    return components


def run_for_model(args, model_alias: str) -> None:
    model_name = resolve_model_name(model_alias)
    model_tag = make_model_tag(model_name)
    pair_path = choose_pair_dataset(model_alias, args.dataset)
    pairs = load_pairs(pair_path, args.pair_format, max_pairs=args.max_pairs)
    direction_pairs = load_pairs(pair_path, args.pair_format, max_pairs=args.direction_max_pairs)

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"Pairs: {pair_path.relative_to(ROOT) if pair_path.is_relative_to(ROOT) else pair_path}")
    print(f"Experiment: {args.experiment}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    n_layers = len(get_decoder_blocks(model))
    layers = parse_layers(args.layers, model_name, args.layer_source_method, n_layers, args.top_k_layers)
    components = parse_components(args.components)
    print(f"Selected layers: {layers}")
    print(f"Layer source method: {args.layer_source_method}")
    print(f"Layer selection: {args.layers}")
    print(f"Components: {components}")

    experiment_names = (
        ["steering", "activation_patching", "attribution_patching", "ablation"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for experiment in experiment_names:
        if experiment == "steering":
            rows = run_steering(
                model=model,
                tokenizer=tokenizer,
                model_alias=model_alias,
                model_name=model_name,
                pairs=pairs,
                layers=layers,
                components=components,
                strengths=args.strengths,
                direction_pairs=direction_pairs,
                max_length=args.max_length,
                direction=args.steering_direction,
            )
        elif experiment == "activation_patching":
            rows = run_activation_patching(
                model, tokenizer, model_alias, model_name, pairs, layers, components, args.max_length
            )
        elif experiment == "attribution_patching":
            rows = run_attribution_patching(
                model, tokenizer, model_alias, model_name, pairs, layers, components, args.max_length
            )
        elif experiment == "ablation":
            rows = run_ablation(
                model,
                tokenizer,
                model_alias,
                model_name,
                pairs,
                layers,
                components,
                args.max_length,
                args.ablation_mode,
            )
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        selected_layers = ",".join(str(layer) for layer in layers)
        for row in rows:
            row.layer_source_method = args.layer_source_method
            row.layer_selection = args.layers
            row.selected_layers = selected_layers

        metadata = {
            "model_alias": model_alias,
            "model_name": model_name,
            "layer_source_method": args.layer_source_method,
            "layer_selection": args.layers,
            "top_k_layers": args.top_k_layers,
            "selected_layers": layers,
            "components": components,
            "pair_format": args.pair_format,
            "pair_dataset": str(pair_path.relative_to(ROOT) if pair_path.is_relative_to(ROOT) else pair_path),
            "max_pairs": args.max_pairs,
            "direction_max_pairs": args.direction_max_pairs,
            "strengths": args.strengths,
            "steering_direction": args.steering_direction,
            "ablation_mode": args.ablation_mode,
            "attn_implementation": args.attn_implementation,
            "device_map": args.device_map,
        }
        method_output_dir = repo_path(args.output_dir) / experiment / args.layer_source_method
        write_outputs(rows, method_output_dir, model_tag, experiment, metadata)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run probe-guided temporal steering, patching, attribution patching, and ablations."
    )
    parser.add_argument(
        "--experiment",
        default="steering",
        choices=["steering", "activation_patching", "attribution_patching", "ablation", "all"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "qwen3-4b", "phi-3-mini-4k-instruct", "llama-3.2-3b"],
        help="Model aliases or Hugging Face ids.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Pair dataset. Defaults to model-specific attribution-patching datasets when available.",
    )
    parser.add_argument("--pair-format", default="classification", choices=["classification", "caa"])
    parser.add_argument("--max-pairs", type=int, default=7)
    parser.add_argument("--direction-max-pairs", type=int, default=7)
    parser.add_argument(
        "--layers",
        default="best",
        help="Layer selection: best, top-k, all, or comma-separated explicit layers.",
    )
    parser.add_argument("--top-k-layers", type=int, default=3)
    parser.add_argument(
        "--layer-source-method",
        default="lr",
        choices=["lr", "dmm", "attn"],
        help="Probe CSV method used when --layers is best or top-k.",
    )
    parser.add_argument("--components", nargs="+", default=["resid"], choices=["resid", "attn", "mlp"])
    parser.add_argument("--strengths", nargs="+", type=float, default=[-3, -2, -1, 0, 1, 2, 3])
    parser.add_argument("--steering-direction", default="corrupted", choices=["clean", "corrupted"])
    parser.add_argument("--ablation-mode", default="zero", choices=["zero", "mean"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--device-map", default="single", choices=["single", "auto"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for model_alias in args.models:
        run_for_model(args, model_alias)


if __name__ == "__main__":
    main()
