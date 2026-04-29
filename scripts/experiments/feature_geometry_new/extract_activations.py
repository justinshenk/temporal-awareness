"""Extract selected-node activations for generated feature-geometry completions.

The input is a JSONL file produced by ``generate_completions.py``. Each record is
expected to contain ``full_text`` plus prompt metadata. Activations are averaged
over the generated ``Strategy:`` section up to, but not including, the generated
``Steps:`` marker, then cached in batch ``.pt`` files with the same selected-node
layout used by the old ``feature_geometry`` activation cache scripts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    """Find the repository root by walking upward until src/ is present."""
    for path in (start, *start.parents):
        if (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find project root containing src/ from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "model_completions" / "completions_256.jsonl"
)
DEFAULT_NODES_PATH = PROJECT_ROOT / "data" / "selected_nodes" / "final_node_list.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_geometry_new_activations"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

SelectedNode = tuple[tuple[int, str], int]
SelectedNodeGroups = dict[str, list[SelectedNode]]


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle only primitive containers used by selected-node files."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def load_selected_node_groups(nodes_path: Path) -> SelectedNodeGroups:
    """Load selected nodes as group -> [((layer, component), node_index), ...]."""
    with nodes_path.open("rb") as f:
        raw_nodes = RestrictedUnpickler(f).load()

    if not isinstance(raw_nodes, dict):
        raise ValueError(f"Expected selected nodes to be a dict, got {type(raw_nodes)}")

    selected_node_groups: SelectedNodeGroups = {}
    for group_name, group_nodes in raw_nodes.items():
        if not isinstance(group_name, str):
            raise ValueError(
                f"Selected node group names must be strings: {group_name!r}"
            )

        selected_node_groups[group_name] = []
        for raw_node in sorted(group_nodes):
            if (
                not isinstance(raw_node, tuple)
                or len(raw_node) != 2
                or not isinstance(raw_node[0], str)
                or not isinstance(raw_node[1], int)
            ):
                raise ValueError(f"Invalid selected node entry: {raw_node!r}")

            component_layer, node_index = raw_node
            if "/" not in component_layer:
                raise ValueError(
                    f"Expected component/layer entry, got {component_layer!r}"
                )
            component, layer_text = component_layer.split("/", maxsplit=1)
            selected_node_groups[group_name].append(
                ((int(layer_text), component), node_index)
            )

    return selected_node_groups


def get_unique_layer_components(
    selected_node_groups: SelectedNodeGroups,
) -> list[tuple[int, str]]:
    """Return sorted unique layer/component pairs required by selected nodes."""
    return sorted(
        {
            layer_component
            for group_nodes in selected_node_groups.values()
            for layer_component, _ in group_nodes
        }
    )


def group_node_indices(
    selected_node_groups: SelectedNodeGroups,
) -> dict[tuple[int, str], list[int]]:
    """Return unique selected node indices grouped by layer/component."""
    grouped: dict[tuple[int, str], set[int]] = {}
    for group_nodes in selected_node_groups.values():
        for layer_component, node_index in group_nodes:
            grouped.setdefault(layer_component, set()).add(node_index)

    return {
        layer_component: sorted(node_indices)
        for layer_component, node_indices in grouped.items()
    }


def selected_nodes_include_attention_heads(
    selected_node_groups: SelectedNodeGroups,
) -> bool:
    """Return whether selected nodes include z attention-head entries."""
    return any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    )


def maybe_average_positions(activations: Any, average_positions: bool) -> Any:
    """Optionally average activations across valid token positions."""
    if not average_positions:
        return activations
    return activations.apply(torch.nanmean, dim=1, mask_aware=True)


def extract_selected_activations(
    activations: Any,
    selected_node_groups: SelectedNodeGroups,
) -> dict[str, dict[str, list[int] | torch.Tensor]]:
    """Extract selected nodes once per layer/component from an ActivationDict."""
    selected: dict[str, dict[str, list[int] | torch.Tensor]] = {}
    for (layer, component), node_indices in group_node_indices(
        selected_node_groups
    ).items():
        activation = activations[(layer, component)]
        index = torch.as_tensor(node_indices, device=activation.device)
        if component == "z":
            node_dim = 2 if activation.ndim == 4 else 1
        else:
            node_dim = 2 if activation.ndim == 3 else 1

        selected[f"{component}/{layer}"] = {
            "node_indices": node_indices,
            "values": activation.index_select(node_dim, index).detach().cpu(),
        }
    return selected


def load_completion_records(
    input_path: Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load completion JSONL records, preserving per-record metadata."""
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            full_text = record.get("full_text")
            if not isinstance(full_text, str) or not full_text:
                raise ValueError(
                    f"Record {line_number} in {input_path} is missing full_text"
                )
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def iter_batches(
    records: list[dict[str, Any]],
    batch_size: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield indexed record batches."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    for batch_idx, start in enumerate(range(0, len(records), batch_size)):
        yield batch_idx, records[start : start + batch_size]


def build_metadata(
    batch: list[dict[str, Any]], start_index: int
) -> list[dict[str, Any]]:
    """Build serializable metadata for a completion batch."""
    metadata = []
    for offset, record in enumerate(batch):
        metadata.append(
            {
                "sample_index": start_index + offset,
                "prompt": record.get("prompt"),
                "prompt_metadata": record.get("prompt_metadata", {}),
                "completion_model_name": record.get("model_name"),
            }
        )
    return metadata


def get_underlying_tokenizer(tokenizer: Any) -> Any:
    """Return the HF tokenizer wrapped by mech_interp_toolkit when present."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def find_strategy_steps_span(full_text: str) -> tuple[int, int, str] | None:
    """Return the char span for generated Strategy text before Steps."""
    assistant_marker = "assistant\n"
    assistant_start = full_text.rfind(assistant_marker)
    if assistant_start == -1:
        return None
    search_start = assistant_start + len(assistant_marker)

    strategy_start = full_text.find("Strategy:", search_start)
    if strategy_start == -1:
        return None

    steps_start = full_text.find("Steps:", strategy_start)
    if steps_start == -1:
        return None

    span_end = steps_start
    while span_end > strategy_start and full_text[span_end - 1].isspace():
        span_end -= 1
    if span_end <= strategy_start:
        return None

    return strategy_start, span_end, full_text[strategy_start:span_end]


def token_positions_for_char_span(
    tokenizer: Any,
    text: str,
    start_char: int,
    end_char: int,
) -> list[int]:
    """Map a character span to token positions using tokenizer offsets."""
    hf_tokenizer = get_underlying_tokenizer(tokenizer)
    encoding = hf_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    positions = [
        token_idx
        for token_idx, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start_char and token_start < end_char
    ]
    if not positions:
        raise ValueError(
            f"Character span {start_char}:{end_char} did not map to any tokens"
        )
    return positions


def build_strategy_steps_metadata(
    record: dict[str, Any],
    sample_index: int,
    char_span: tuple[int, int],
    token_positions: list[int],
    activation_text: str,
) -> list[dict[str, Any]]:
    """Build metadata for one averaged Strategy/Steps activation sample."""
    return [
        {
            "sample_index": sample_index,
            "prompt": record.get("prompt"),
            "prompt_metadata": record.get("prompt_metadata", {}),
            "completion_model_name": record.get("model_name"),
            "activation_section": "strategy_steps_generation",
            "activation_char_span": list(char_span),
            "activation_token_positions": token_positions,
            "activation_text": activation_text,
        }
    ]


def cache_completion_activations(
    *,
    input_path: Path,
    model_name: str,
    nodes_path: Path,
    output_dir: Path,
    batch_size: int,
    dtype: str | None,
    device: str | None,
    attn_type: str,
    max_samples: int | None,
    overwrite: bool,
) -> None:
    """Load completions, average Strategy/Steps activations, and save caches."""
    from mech_interp_toolkit.activation_utils import get_activations
    from mech_interp_toolkit.utils import load_model_tokenizer_config

    if batch_size != 1:
        raise ValueError(
            "Strategy/Steps spans have different token positions per prompt; "
            "use batch_size=1."
        )

    selected_node_groups = load_selected_node_groups(nodes_path)
    layer_components = get_unique_layer_components(selected_node_groups)
    records = load_completion_records(input_path, max_samples=max_samples)

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token  # type:ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    total_batches = (len(records) + batch_size - 1) // batch_size
    skipped_spans: list[dict[str, Any]] = []

    for batch_idx, batch in tqdm(
        iter_batches(records, batch_size),
        total=total_batches,
        desc="Caching completion activation batches",
    ):
        output_file = output_dir / f"activations_batch_{batch_idx:05d}.pt"
        if output_file.exists() and not overwrite:
            continue

        record = batch[0]
        text = record["full_text"]
        start_index = batch_idx * batch_size
        span = find_strategy_steps_span(text)
        if span is None:
            skipped_spans.append(
                {
                    "sample_index": start_index,
                    "error": "Could not find non-empty Strategy/Steps span",
                    "prompt": record.get("prompt"),
                    "prompt_metadata": record.get("prompt_metadata", {}),
                }
            )
            continue

        start_char, end_char, activation_text = span
        token_positions = token_positions_for_char_span(
            tokenizer,
            text,
            start_char,
            end_char,
        )

        texts = [text]
        tokenized_batch = tokenizer(texts)
        activations, logits = get_activations(
            model,
            tokenized_batch,
            layer_components,
            positions=token_positions,
            return_logits=False,
            clone_tensors=True,
        )

        if selected_nodes_include_attention_heads(selected_node_groups):
            activations = activations.split_heads()
        activations = maybe_average_positions(activations, average_positions=True)
        cache_payload: dict[str, Any] = {
            "input_path": str(input_path),
            "model_name": model_name,
            "nodes_path": str(nodes_path),
            "layer_components": layer_components,
            "metadata": build_strategy_steps_metadata(
                record=record,
                sample_index=start_index,
                char_span=(start_char, end_char),
                token_positions=token_positions,
                activation_text=activation_text,
            ),
            "positions": token_positions,
            "average_positions": True,
            "activation_section": "strategy_steps_generation",
            "activations": extract_selected_activations(
                activations,
                selected_node_groups,
            ),
        }
        if logits is not None:
            cache_payload["logits"] = logits.detach().cpu()

        torch.save(cache_payload, output_file)

    if skipped_spans:
        skipped_path = output_dir / "skipped_strategy_steps_spans.jsonl"
        with skipped_path.open("w", encoding="utf-8") as f:
            for skipped in skipped_spans:
                f.write(json.dumps(skipped, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache selected-node activations for generated completions."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Completion JSONL input. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model used to extract activations.",
    )
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=DEFAULT_NODES_PATH,
        help="Pickle file containing selected nodes to cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Activation cache output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Must be 1 because Strategy/Steps token spans differ by sample.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-type", default="sdpa")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute batches even when their output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_completion_activations(
        input_path=resolve_path(args.input_path),
        model_name=args.model_name,
        nodes_path=resolve_path(args.nodes_path),
        output_dir=resolve_path(args.output_dir),
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        attn_type=args.attn_type,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
