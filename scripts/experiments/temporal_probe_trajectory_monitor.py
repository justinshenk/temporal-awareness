#!/usr/bin/env python3
"""Track temporal probe scores during generation.

This is the "unfolding process" experiment for issue #47. It asks whether a
temporal-horizon probe changes before a relevant behavior appears in generated
text. The script works with the existing flat LR probe checkpoints:

    research/probes/temporal_caa_layer_{model_tag}_{layer}_probe.pkl
    results/checkpoints/temporal_caa_layer_{layer}_probe.pkl  # GPT-2 fallback

For every prompt, it records the probe score at the prompt end and after each
generated token. It also marks the first generated step matching a lightweight
event detector. The detector is deliberately simple by default so the pipeline
is runnable offline; the output schema is designed so a proper classifier
score, such as LlamaGuard, can be joined in later.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
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

DEFAULT_DATASET = ROOT / "data/raw/temporal_oversight_sequences.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/temporal_oversight"


@dataclass(frozen=True)
class OversightPrompt:
    prompt_id: str
    category: str
    prompt: str
    event_keywords: list[str]


@dataclass
class ProbeTrajectoryRow:
    model_alias: str
    model_name: str
    prompt_id: str
    category: str
    layer: int
    step: int
    token_id: int | None
    token_text: str
    generated_text: str
    temporal_score: float
    event_detected: bool
    first_event_step: int | None
    steps_before_event: int | None
    detector: str
    layer_source_method: str
    layer_selection: str
    selected_layers: str
    probe_source: str


@dataclass
class PromptSummaryRow:
    model_alias: str
    model_name: str
    prompt_id: str
    category: str
    layer: int
    initial_score: float
    final_score: float
    max_score: float
    max_score_before_event: float | None
    delta_before_event: float | None
    first_event_step: int | None
    event_detected: bool
    n_steps: int
    detector: str
    layer_source_method: str
    layer_selection: str
    selected_layers: str
    probe_source: str


def resolve_model_name(model_alias_or_name: str) -> str:
    return SUPPORTED_MODELS.get(model_alias_or_name, model_alias_or_name)


def make_model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def slug(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(",", "-")
        .replace(" ", "")
        .replace("_", "-")
    )


def load_prompts(path: Path, max_prompts: int | None) -> list[OversightPrompt]:
    with open(path) as f:
        rows = json.load(f)

    prompts = [
        OversightPrompt(
            prompt_id=row.get("id", f"prompt_{idx:04d}"),
            category=row.get("category", "unknown"),
            prompt=row["prompt"],
            event_keywords=list(row.get("event_keywords", [])),
        )
        for idx, row in enumerate(rows)
    ]
    return prompts[:max_prompts] if max_prompts is not None else prompts


def load_model_and_tokenizer(
    model_name: str,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
    attn_implementation: str,
    device_map: str,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required. Run from the project environment, e.g. `.venv/bin/python ...`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
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

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def get_model_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def probe_candidates(model_name: str, layer: int) -> list[Path]:
    tag = make_model_tag(model_name)
    return [
        ROOT / "research/probes" / f"temporal_caa_layer_{tag}_{layer}_probe.pkl",
        ROOT / "results/checkpoints" / f"temporal_caa_layer_{layer}_probe.pkl",
    ]


def load_probe(model_name: str, layer: int):
    for path in probe_candidates(model_name, layer):
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f), path
    searched = "\n".join(str(path.relative_to(ROOT)) for path in probe_candidates(model_name, layer))
    raise FileNotFoundError(f"No probe found for {model_name} layer {layer}. Searched:\n{searched}")


def probe_score(probe, hidden: torch.Tensor) -> float:
    x = hidden.detach().float().cpu().numpy().reshape(1, -1)

    if isinstance(probe, dict):
        scaler = probe.get("scaler")
        estimator = probe.get("probe")
        if scaler is not None:
            x = scaler.transform(x)
        if estimator is None and "direction" in probe:
            score = float(x @ probe["direction"])
            return 1.0 / (1.0 + np.exp(-score))
        probe = estimator

    if hasattr(probe, "predict_proba"):
        return float(probe.predict_proba(x)[0, 1])
    if hasattr(probe, "decision_function"):
        score = float(probe.decision_function(x)[0])
        return 1.0 / (1.0 + np.exp(-score))
    raise TypeError(f"Unsupported probe object: {type(probe)}")


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
    raise ValueError(f"Unsupported layer mode for CSV selection: {layer_mode}")


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


def keyword_event_step(generated_by_step: list[str], keywords: list[str]) -> int | None:
    if not keywords:
        return None
    patterns = [re.compile(re.escape(keyword), flags=re.IGNORECASE) for keyword in keywords]
    for step, text in enumerate(generated_by_step):
        if any(pattern.search(text) for pattern in patterns):
            return step
    return None


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, num_samples=1).item())


def score_generation(
    model,
    tokenizer,
    prompt: OversightPrompt,
    probes_by_layer: dict[int, Any],
    layers: list[int],
    max_new_tokens: int,
    max_length: int | None,
    temperature: float,
    top_p: float,
    detector: str,
    model_alias: str,
    model_name: str,
    layer_source_method: str,
    layer_selection: str,
    selected_layers: str,
    probe_source: str,
) -> tuple[list[ProbeTrajectoryRow], list[PromptSummaryRow]]:
    device = get_model_device(model)
    encoded = tokenizer(
        prompt.prompt,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated_token_ids: list[int] = []
    generated_by_step = [""]
    step_snapshots: list[dict[str, Any]] = []

    with torch.no_grad():
        for step in range(max_new_tokens + 1):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            token_id = generated_token_ids[-1] if generated_token_ids else None
            token_text = tokenizer.decode([token_id]) if token_id is not None else ""
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            scores = {
                layer: probe_score(probes_by_layer[layer], hidden_states[layer + 1][0, -1, :])
                for layer in layers
            }
            step_snapshots.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "token_text": token_text,
                    "generated_text": generated_text,
                    "scores": scores,
                }
            )

            if step == max_new_tokens:
                break

            logits = outputs.logits[0, -1, :]
            if tokenizer.eos_token_id is not None and generated_token_ids:
                if generated_token_ids[-1] == tokenizer.eos_token_id:
                    break
            next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
            generated_token_ids.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tensor)], dim=1)
            generated_by_step.append(tokenizer.decode(generated_token_ids, skip_special_tokens=True))

    first_event_step = keyword_event_step(generated_by_step, prompt.event_keywords)
    rows = []
    summaries = []

    for snapshot in step_snapshots:
        step = int(snapshot["step"])
        event_detected = first_event_step is not None and step >= first_event_step
        steps_before_event = first_event_step - step if first_event_step is not None and step < first_event_step else None
        for layer in layers:
            rows.append(
                ProbeTrajectoryRow(
                    model_alias=model_alias,
                    model_name=model_name,
                    prompt_id=prompt.prompt_id,
                    category=prompt.category,
                    layer=layer,
                    step=step,
                    token_id=snapshot["token_id"],
                    token_text=snapshot["token_text"],
                    generated_text=snapshot["generated_text"],
                    temporal_score=float(snapshot["scores"][layer]),
                    event_detected=event_detected,
                    first_event_step=first_event_step,
                    steps_before_event=steps_before_event,
                    detector=detector,
                    layer_source_method=layer_source_method,
                    layer_selection=layer_selection,
                    selected_layers=selected_layers,
                    probe_source=probe_source,
                )
            )

    for layer in layers:
        layer_scores = [float(snapshot["scores"][layer]) for snapshot in step_snapshots]
        if first_event_step is None:
            before_scores = None
            max_before = None
            delta_before = None
        else:
            before_scores = layer_scores[: max(first_event_step, 1)]
            max_before = max(before_scores) if before_scores else None
            delta_before = max_before - layer_scores[0] if max_before is not None else None
        summaries.append(
            PromptSummaryRow(
                model_alias=model_alias,
                model_name=model_name,
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                layer=layer,
                initial_score=layer_scores[0],
                final_score=layer_scores[-1],
                max_score=max(layer_scores),
                max_score_before_event=max_before,
                delta_before_event=delta_before,
                first_event_step=first_event_step,
                event_detected=first_event_step is not None,
                n_steps=len(step_snapshots) - 1,
                detector=detector,
                layer_source_method=layer_source_method,
                layer_selection=layer_selection,
                selected_layers=selected_layers,
                probe_source=probe_source,
            )
        )

    return rows, summaries


def write_table(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, metadata: dict[str, Any], rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "rows": [asdict(row) for row in rows],
            },
            f,
            indent=2,
        )


def run_for_model(args, model_alias: str) -> None:
    model_name = resolve_model_name(model_alias)
    prompts = load_prompts(repo_path(args.dataset), args.max_prompts)

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        raise ValueError(f"Could not read num_hidden_layers for {model_name}")

    layers = parse_layers(args.layers, model_name, args.layer_source_method, n_layers, args.top_k_layers)
    probes_by_layer = {}
    probe_paths_by_layer = {}
    for layer in layers:
        probe, path = load_probe(model_name, layer)
        probes_by_layer[layer] = probe
        probe_paths_by_layer[layer] = str(path.relative_to(ROOT))
        print(f"Layer {layer}: {path.relative_to(ROOT)}")

    trajectory_rows: list[ProbeTrajectoryRow] = []
    summary_rows: list[PromptSummaryRow] = []
    selected_layers = ",".join(str(layer) for layer in layers)
    probe_source = "flat_lr_probe_checkpoint"
    for prompt in tqdm(prompts, desc=f"monitor {model_alias}"):
        rows, summaries = score_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            probes_by_layer=probes_by_layer,
            layers=layers,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            detector=args.detector,
            model_alias=model_alias,
            model_name=model_name,
            layer_source_method=args.layer_source_method,
            layer_selection=args.layers,
            selected_layers=selected_layers,
            probe_source=probe_source,
        )
        trajectory_rows.extend(rows)
        summary_rows.extend(summaries)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_tag = make_model_tag(model_name)
    output_dir = repo_path(args.output_dir) / args.layer_source_method
    layer_selection = slug(args.layers)
    layer_source_method = slug(args.layer_source_method)
    stem = f"{model_tag}_temporal_probe_trajectory_layers-{layer_selection}_from-{layer_source_method}_{timestamp}"
    metadata = {
        "experiment": "temporal_probe_trajectory_monitor",
        "timestamp": timestamp,
        "model_alias": model_alias,
        "model_name": model_name,
        "layers": layers,
        "layer_source_method": args.layer_source_method,
        "layer_selection": args.layers,
        "top_k_layers": args.top_k_layers,
        "probe_source": probe_source,
        "probe_paths_by_layer": probe_paths_by_layer,
        "dataset": str(repo_path(args.dataset).relative_to(ROOT)),
        "detector": args.detector,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    write_table(output_dir / f"{stem}_trajectory.csv", trajectory_rows)
    write_table(output_dir / f"{stem}_summary.csv", summary_rows)
    write_json(output_dir / f"{stem}_trajectory.json", metadata, trajectory_rows)
    write_json(output_dir / f"{stem}_summary.json", metadata, summary_rows)
    print(f"Saved trajectory and summary outputs under {output_dir.relative_to(ROOT)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor temporal probe scores at each generation step."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        help="Model aliases or Hugging Face ids.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(ROOT)))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--layers", default="best", help="best, top-k, all, or comma-separated layers.")
    parser.add_argument("--top-k-layers", type=int, default=3)
    parser.add_argument("--layer-source-method", default="lr", choices=["lr", "dmm", "attn"])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--detector",
        default="keywords",
        choices=["keywords"],
        help="Current offline detector. Join classifier scores into the output later for LlamaGuard-style evaluation.",
    )
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
    torch.manual_seed(42)
    for model_alias in args.models:
        run_for_model(args, model_alias)


if __name__ == "__main__":
    main()
