"""Cache model activations for intertemporal prompt datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
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
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.data.default_datasets import (
    FULL_EXPERIMENT_DATASET_CONFIG,
    GEO_VIZ_CFG,
    MINIMAL_EXPERIMENT_DATASET_CONFIG,
    MULTILABEL_EXPERIMENT_DATASET_CONFIG,
)
from src.intertemporal.prompt import PromptDataset, PromptDatasetConfig  # type: ignore
from src.intertemporal.prompt.prompt_dataset_generator import PromptDatasetGenerator

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "geo_viz": GEO_VIZ_CFG,
    "full": FULL_EXPERIMENT_DATASET_CONFIG,
    "minimal": MINIMAL_EXPERIMENT_DATASET_CONFIG,
    "multilabel": MULTILABEL_EXPERIMENT_DATASET_CONFIG,
}

DEFAULT_NODES_PATH = PROJECT_ROOT / "data" / "selected_nodes" / "final_node_list.pkl"
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent
    / "config"
    / "final_configs"
    / "Q_A"
    / "(1)FM_4B_explicit_GC.yaml"
)
HF_REPO_ID = "Temporal_Awareness_Node_Scores"

SelectedNode = tuple[tuple[int, str], int]
SelectedNodeGroups = dict[str, list[SelectedNode]]


def load_prompt_dataset(dataset: str = "geo_viz") -> PromptDataset:
    """Load a saved PromptDataset JSON or generate a named default dataset."""
    dataset_path = Path(dataset)
    if dataset_path.exists():
        return PromptDataset.from_json(str(dataset_path))

    if dataset not in DATASET_CONFIGS:
        known = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Use one of {known}, or pass a JSON path."
        )

    config = PromptDatasetConfig.from_dict(DATASET_CONFIGS[dataset])
    return PromptDatasetGenerator(config).generate()


def get_time_horizon_months(sample: Any) -> float:
    """Return the sample horizon in months; use 60 months for no horizon."""
    if sample.prompt.time_horizon is None:
        return 60.0
    return sample.prompt.time_horizon.to_months()


def get_time_horizon_label(sample: Any) -> str:
    """Return a stable label for the raw horizon value."""
    if sample.prompt.time_horizon is None:
        return "none"
    return str(sample.prompt.time_horizon)


def chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the EAP-IG YAML config."""
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_settings(config_path: Path) -> tuple[str, int, str | None]:
    """Read only setup.model, setup.batch_size, and setup.dtype from config."""
    config = load_config(config_path)
    setup = config["setup"]
    return setup["model"], setup["batch_size"], setup.get("dtype")


def resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths from config values."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle only primitive containers used by selected-node files."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


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


def extract_selected_activations(
    activations: Any,
    selected_node_groups: SelectedNodeGroups,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract only selected neurons/attention heads from an ActivationDict."""
    if any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    ):
        activations = activations.split_heads()

    selected: dict[str, dict[str, torch.Tensor]] = {}
    for group_name, group_nodes in selected_node_groups.items():
        selected[group_name] = {}
        for (layer, component), node_index in group_nodes:
            activation = activations[(layer, component)]
            if component == "z":
                node_activation = activation[:, :, node_index, :]
            else:
                node_activation = activation[:, :, node_index]

            selected[group_name][f"{component}/{layer}__{node_index}"] = (
                node_activation.detach().cpu()
            )

    return selected


def cache_prompt_activations(
    *,
    dataset: str,
    model_name: str,
    nodes_path: Path,
    output_dir: Path,
    batch_size: int,
    dtype: str | None,
    device: str | None,
    attn_type: str,
    max_samples: int | None,
    save_to_hf: bool,
    hf_repo_id: str,
    hf_repo_type: str,
) -> None:
    """Load a model with mech_interp_toolkit and cache activations for all positions."""
    from mech_interp_toolkit.activation_utils import get_activations
    from mech_interp_toolkit.utils import load_model_tokenizer_config

    try:
        from .eap_ig_qanda_upload import maybe_start_upload_worker
    except ImportError:
        from eap_ig_qanda_upload import maybe_start_upload_worker

    selected_node_groups = load_selected_node_groups(nodes_path)
    layer_components = get_unique_layer_components(selected_node_groups)
    upload_queue, upload_thread, enqueue_upload = maybe_start_upload_worker(
        save_to_hf=save_to_hf,
        hf_repo_id=hf_repo_id,
        hf_repo_type=hf_repo_type,
    )

    prompt_dataset = load_prompt_dataset(dataset)
    samples = prompt_dataset.samples
    if max_samples is not None:
        samples = samples[:max_samples]

    prompts = [sample.prompt.text for sample in samples]
    metadata = [
        {
            "sample_index": idx,
            "time_horizon_months": get_time_horizon_months(sample),
            "time_horizon_label": get_time_horizon_label(sample),
        }
        for idx, sample in enumerate(samples)
    ]

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_batches = chunk_list(prompts, batch_size)
    metadata_batches = chunk_list(metadata, batch_size)

    for batch_idx, prompt_batch in enumerate(
        tqdm(prompt_batches, desc="Caching activation batches")
    ):
        tokenized_batch = tokenizer(prompt_batch)
        activations, logits = get_activations(
            model,
            tokenized_batch,
            layer_components,
            positions=None,
            return_logits=False,
            clone_tensors=True,
        )

        cache_payload = {
            "model_name": model_name,
            "nodes_path": str(nodes_path),
            "layer_components": layer_components,
            "selected_node_groups": selected_node_groups,
            "prompts": prompt_batch,
            "formatted_prompts": list(tokenizer.structured_prompt or []),
            "metadata": metadata_batches[batch_idx],
            "input_ids": tokenized_batch["input_ids"].detach().cpu(),
            "attention_mask": tokenized_batch["attention_mask"].detach().cpu(),
            "activations": extract_selected_activations(
                activations,
                selected_node_groups,
            ),
        }
        if logits is not None:
            cache_payload["logits"] = logits.detach().cpu()

        output_file = output_dir / f"activations_batch_{batch_idx:05d}.pt"
        torch.save(cache_payload, output_file)

        output_file_abs = output_file.resolve()
        try:
            path_in_repo = output_file_abs.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            path_in_repo = output_file.name
        enqueue_upload(output_file_abs, path_in_repo)

    if upload_queue is not None and upload_thread is not None:
        upload_queue.put(None)
        upload_thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache chat-templated model activations for an intertemporal dataset."
    )
    parser.add_argument(
        "--dataset",
        default="geo_viz",
        help=(
            "Dataset source: one of geo_viz, full, minimal, multilabel, or a "
            "PromptDataset JSON path. Default: geo_viz."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config to read setup.model, setup.batch_size, and setup.dtype from.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional override for setup.model from the config.",
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
        default=None,
        help="Optional output directory override. Defaults to results/activation_caches.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-type", default="sdpa")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--save-to-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload cached activation files to Hugging Face Hub.",
    )
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-repo-type", default="dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    config_model, config_batch_size, config_dtype = load_model_settings(config_path)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "results" / "activation_caches"
    )
    nodes_path = resolve_path(args.nodes_path)

    cache_prompt_activations(
        dataset=args.dataset,
        model_name=args.model or config_model,
        nodes_path=nodes_path,
        output_dir=output_dir,
        batch_size=args.batch_size or config_batch_size,
        dtype=args.dtype if args.dtype is not None else config_dtype,
        device=args.device,
        attn_type=args.attn_type,
        max_samples=args.max_samples,
        save_to_hf=args.save_to_hf,
        hf_repo_id=args.hf_repo_id,
        hf_repo_type=args.hf_repo_type,
    )


if __name__ == "__main__":
    main()
