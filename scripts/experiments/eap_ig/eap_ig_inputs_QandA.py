"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
import gc
import json
import os
import queue
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm

warnings.filterwarnings("ignore")

load_dotenv()
CONFIG_PATH = Path(__file__).parent / "config"
torch.set_grad_enabled(False)
HF_REPO_ID = os.getenv("HF_REPO_ID", "Temporal_Awareness_EAP_IG")
SUPPORTED_QUADRATURES = {
    "gauss-chebyshev",
    "gauss-legendre",
    "riemann-midpoint",
}
QUADRATURE_ALIASES = {
    "midpoint": "riemann-midpoint",
}
DOT_CONFIG_SYMBOLS = {"●", "■"}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, normalizing unsupported dtypes."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor.numpy()


def load_and_merge_pairs(
    input_file: Path,
    template: str,
    option_keys: list[str],
    text_order: list[str],
) -> tuple[list[str], list[str]]:
    """Load pairs from ``input_file`` and return both clean and swapped prompts.

    Args:
        input_file: Path to JSON file containing question pairs
        template: Template string for formatting prompts
        option_keys: List of option keys to use
        text_order: List of keys specifying the order in which to extract fields
            from each pair dict (e.g. ``["question", "immediate", "long_term"]``)

    Returns:
        Tuple of (clean_prompts, swapped_prompts)
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts = []
    swapped_prompts = []
    option_a, option_b = option_keys
    pairs = data.get("pairs", [])

    # The regex is robust for cases where one option might be a substring of another.
    for pair in pairs:
        if isinstance(pair, str):
            prompt = pair
        elif isinstance(pair, dict):
            prompt = template.format(
                pair.get(text_order[0], ""),
                pair.get(text_order[1], ""),
                pair.get(text_order[2], ""),
            )
        else:
            raise RuntimeError("Incorrect type for pairs")

        prompt = prompt.replace("(A)", option_a)
        prompt = prompt.replace("(B)", option_b)

        clean_prompts.append(prompt)

        swapped_prompt = re.sub(
            f"{re.escape(option_a)}|{re.escape(option_b)}",
            lambda m: option_b if m.group(0) == option_a else option_a,
            prompt,
        )
        swapped_prompts.append(swapped_prompt)

    return clean_prompts, swapped_prompts


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary with configuration values
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_mech_interp_toolkit_installed() -> None:
    """Raise a clear error when ``mech_interp_toolkit`` is unavailable."""
    try:
        import mech_interp_toolkit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mech_interp_toolkit is required for the EAP-IG workflow. "
            'Install the pinned dependency with `pip install -e ".[eap_ig]"` '
            "before running this script."
        ) from exc


def extract_alnum(s: str) -> str:
    """Extract the semantic token string from an option marker.

    Args:
        s: Input string to search

    Returns:
        Concatenated alphanumeric content when present, otherwise the first
        non-wrapper symbol for symbolic markers like ``(●)``.

    Raises:
        ValueError: If no supported token content is found
    """
    out = []
    for c in s:
        if c.isalnum() or c in DOT_CONFIG_SYMBOLS:
            out.append(c)
    if out:
        return "".join(out)

    raise ValueError(f"malformed option string {s}")


def resolve_hf_repo_id(hf_api: HfApi, repo_id: str) -> str:
    """Return a fully qualified Hub repo id."""
    if "/" in repo_id:
        return repo_id

    whoami = hf_api.whoami()
    username = whoami.get("name")
    if not username:
        raise ValueError(
            "HF repo id must include a namespace like 'username/repo', or the "
            "HF token must expose an account name so one can be inferred."
        )
    return f"{username}/{repo_id}"


def resolve_quadrature(config: dict) -> str:
    """Resolve config quadrature into a mech-interp-toolkit-supported value."""
    raw_quadrature = config["setup"].get("quadrature")
    if raw_quadrature is None:
        raw_quadrature = config["parameters"].get("quadrature", "riemann-midpoint")

    quadrature = QUADRATURE_ALIASES.get(raw_quadrature, raw_quadrature)
    if quadrature not in SUPPORTED_QUADRATURES:
        supported = ", ".join(sorted(SUPPORTED_QUADRATURES))
        raise ValueError(
            f"Unsupported quadrature '{raw_quadrature}'. Use one of: {supported}."
        )
    return quadrature


def resolve_config_path(config_path: Path) -> Path:
    """Resolve config paths relative to the local config directory by default."""
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return CONFIG_PATH / config_path


def build_layer_components(
    n_layers: int,
    granularity: str,
    layer_components: list[list[Any]] | list[tuple[Any, ...]] | None,
) -> list[tuple[int, str]]:
    """Build or normalize the requested layer/component list."""
    if layer_components is None:
        if granularity == "coarse":
            return [
                (layer, component)
                for layer in range(n_layers)
                for component in ("attn", "mlp")
            ]
        if granularity == "fine":
            return [
                (layer, component)
                for layer in range(n_layers)
                for component in ("z", "mlp_hidden")
            ]
        raise ValueError(f"Invalid granularity: {granularity}")

    return [tuple(lc) for lc in layer_components]  # type: ignore[misc]


def maybe_start_upload_worker(
    save_to_hf: bool,
    hf_repo_id: str,
    hf_repo_type: str,
) -> tuple[
    queue.Queue[tuple[Path, str] | None] | None,
    threading.Thread | None,
    Callable[[Path, str], None],
]:
    """Start the background Hub uploader when enabled."""
    if not save_to_hf:
        return None, None, lambda _local_file, _path_in_repo: None

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for Hub uploads.")

    hf_api = HfApi(token=hf_token)
    resolved_repo_id = resolve_hf_repo_id(hf_api, hf_repo_id)
    hf_api.create_repo(repo_id=resolved_repo_id, repo_type=hf_repo_type, exist_ok=True)
    print(
        f"[HF upload] Using repo_id={resolved_repo_id} repo_type={hf_repo_type}",
        flush=True,
    )

    upload_queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()

    def _upload_worker() -> None:
        while True:
            item = upload_queue.get()
            if item is None:
                upload_queue.task_done()
                break
            local_file, path_in_repo = item
            file_size = local_file.stat().st_size
            print(
                f"[HF upload] Starting file={local_file.name} "
                f"local_path={local_file} repo_path={path_in_repo} "
                f"size_gb={file_size / (1024**3):.2f}",
                flush=True,
            )
            hf_api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=path_in_repo,
                repo_id=resolved_repo_id,
                repo_type=hf_repo_type,
                commit_message=f"Upload {path_in_repo}",
            )
            print(
                f"[HF upload] Completed file={local_file.name}",
                flush=True,
            )
            upload_queue.task_done()

    upload_thread = threading.Thread(target=_upload_worker, daemon=True)
    upload_thread.start()

    def _enqueue_upload(local_file: Path, path_in_repo: str) -> None:
        upload_queue.put((local_file, path_in_repo))

    return upload_queue, upload_thread, _enqueue_upload


def run_eap_ig(
    config_path: Path,
    model=None,
    tokenizer=None,
    *,
    save_to_hf: bool = True,
) -> tuple[Any, Any]:
    """Run Q&A EAP-IG from Python or notebooks.

    Args:
        config_path: Config path, either absolute or relative to ``CONFIG_PATH``.
        model: Optional pre-loaded model to reuse across calls.
        tokenizer: Optional pre-loaded tokenizer to reuse across calls.
        save_to_hf: Whether to upload generated NPZ files to the Hub.

    Returns:
        Tuple of (model, tokenizer) so the caller can reuse them.

    """
    ensure_mech_interp_toolkit_installed()
    config = load_config(resolve_config_path(config_path))

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    layer_components = config["setup"].get("layer_components", None)
    granularity = config["setup"].get("granularity", "coarse")
    quadrature = resolve_quadrature(config)

    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    template = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    filename: str = config["output"]["filename"]
    hf_repo_id: str = HF_REPO_ID
    hf_repo_type: str = config["output"].get("hf_repo_type", "dataset")

    system_prompt: str = config["parameters"]["system_prompt"]
    metric_type: str = config["parameters"]["metric_type"]
    steps: list[int] = config["parameters"]["steps"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)
    upload_queue, upload_thread, enqueue_upload = maybe_start_upload_worker(
        save_to_hf=save_to_hf,
        hf_repo_id=hf_repo_id,
        hf_repo_type=hf_repo_type,
    )

    from mech_interp_toolkit.activation_dict import expand_mask
    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)

    if model is None or tokenizer is None:
        # Suffix prompts the model to complete with option character
        model, tokenizer, _ = load_model_tokenizer_config(
            model_name=model_name,
            suffix=prompt_suffix,
            system_prompt=system_prompt,
            attn_type="eager",
            dtype=dtype,
        )

    n_layers = model.config.num_hidden_layers

    layer_components = build_layer_components(
        n_layers=n_layers,
        granularity=granularity,
        layer_components=layer_components,
    )

    system_prompt_length = (
        len(tokenizer.tokenizer.encode(system_prompt, add_special_tokens=False)) + 1
    )  # For <|im_end|>

    token_a_str = extract_alnum(option_keys[0])
    token_b_str = extract_alnum(option_keys[1])

    token_a = tokenizer.tokenizer.encode(token_a_str, add_special_tokens=False)
    token_b = tokenizer.tokenizer.encode(token_b_str, add_special_tokens=False)

    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError(
            f"Token A tokenizes to {token_a}\nToken B tokenizes to {token_b}\nOptionkeys must tokenize to single token each."
        )
    token_a = token_a[0]
    token_b = token_b[0]

    metrics = {
        "logit_A": lambda logits: logits[:, -1, token_a].mean(),
        "logit_B": lambda logits: logits[:, -1, token_b].mean(),
    }

    def chunk_list(prompt_list: list[str]) -> list[list[str]]:
        """Split a list into chunks of size batch_size."""
        return [
            prompt_list[i : i + batch_size]
            for i in range(0, len(prompt_list), batch_size)
        ]

    # Load the data file once and generate both clean and swapped prompts.
    # "normal" order: short-term option first (option_keys as-is).
    # "swapped" order: long-term option first (option_keys reversed).
    all_clean_prompts, all_corrupted_prompts = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "immediate", "long_term"],
    )

    # Reverse order is an artifact of data syntax
    all_corrupted_prompts_swapped, all_clean_prompts_swapped = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "long_term", "immediate"],
    )

    option_orders = [
        (
            "short_first",
            chunk_list(all_clean_prompts),
            chunk_list(all_corrupted_prompts),
        ),
        (
            "long_first",
            chunk_list(all_clean_prompts_swapped),
            chunk_list(all_corrupted_prompts_swapped),
        ),
    ]

    for (
        order_label,
        chunked_clean_prompts,
        chunked_corrupted_prompts,
    ) in option_orders:
        # Pre-tokenize all batches once per order to avoid redundant work across num_steps iterations
        tokenized_clean = [tokenizer(b) for b in chunked_clean_prompts]
        tokenized_corrupted = [tokenizer(b) for b in chunked_corrupted_prompts]

        for metric_label, metric_fn in metrics.items():
            for i in tqdm(
                range(len(tokenized_clean)),
                desc=f"[{order_label}/{metric_label}] Batches",
            ):
                batch_output: dict[str, np.ndarray] = {}
                batch_output["metadata__config_json"] = np.array(
                    json.dumps(config), dtype=np.str_
                )
                batch_output["metadata__option_order"] = np.array(
                    order_label, dtype=np.str_
                )
                batch_output["metadata__metric_type"] = np.array(
                    metric_type, dtype=np.str_
                )

                for num_steps in tqdm(
                    steps,
                    desc=f"Step counts (batch={i})",
                    leave=False,
                ):
                    # Deep-copy tensors to prevent get_embeddings_dict from mutating the
                    # pre-tokenized dicts (it pops input_ids and injects inputs_embeds
                    # in-place, causing stale GPU tensors with live graphs to accumulate).
                    clean_inputs = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in tokenized_clean[i].items()
                    }
                    corrupted_inputs = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in tokenized_corrupted[i].items()
                    }

                    eap_ig_scores, (clean_logits, corrupted_logits) = (
                        eap_integrated_gradients(  # (batch, pos) | (batch, pos, n_head) | (batch, pos, neuron)
                            model,  # type: ignore
                            clean_inputs,
                            corrupted_inputs,
                            metric_fn,
                            layer_components,
                            steps=num_steps,
                            include_block_outputs=True,
                            quadrature=quadrature,
                        )
                    )

                    clean_logits_cpu = (
                        clean_logits[:, -1, [token_a, token_b]].detach().cpu()  # type: ignore[index]
                    )
                    corrupted_logits_cpu = (
                        corrupted_logits[:, -1, [token_a, token_b]].detach().cpu()  # type: ignore[index]
                    )
                    del clean_logits, corrupted_logits

                    eap_ig_scores.attention_mask = expand_mask(
                        eap_ig_scores.attention_mask, system_prompt_length
                    )
                    token_position_counts = (
                        eap_ig_scores.attention_mask.sum(dim=1).detach().cpu()
                    )
                    eap_ig_scores = eap_ig_scores.apply(
                        torch.nansum, dim=1, mask_aware=True
                    )  # (batch,) | (batch, n_head) | (batch, neuron)
                    eap_ig_scores = eap_ig_scores.apply(lambda x: x.detach().cpu())

                    for key, value in eap_ig_scores.items():
                        batch_output[f"step_{num_steps}__{key[1]}__{key[0]}"] = (
                            tensor_to_numpy(value)
                        )

                    batch_output[f"step_{num_steps}__clean_logits"] = (
                        clean_logits_cpu.float().numpy()
                    )
                    batch_output[f"step_{num_steps}__corrupted_logits"] = (
                        corrupted_logits_cpu.float().numpy()
                    )
                    batch_output[f"step_{num_steps}__token_positions_considered"] = (
                        tensor_to_numpy(token_position_counts)
                    )

                    # Delete temporary objects to free memory
                    del (
                        eap_ig_scores,
                        token_position_counts,
                        clean_inputs,
                        corrupted_inputs,
                        clean_logits_cpu,
                        corrupted_logits_cpu,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                output_file = save_loc / (
                    f"{filename}_{order_label}_{metric_label}_batch_{i:05d}.npz"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_file, **batch_output)
                output_file_abs = output_file.resolve()
                try:
                    path_in_repo = output_file_abs.relative_to(
                        Path.cwd().resolve()
                    ).as_posix()
                except ValueError:
                    path_in_repo = output_file.name
                enqueue_upload(output_file_abs, path_in_repo)

    if upload_queue is not None and upload_thread is not None:
        upload_queue.put(None)
        upload_thread.join()

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EAP Integrated Gradients on clean vs corrupted prompts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file (e.g., step_numbers.yaml)",
    )
    parser.add_argument(
        "--save-to-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload generated NPZ files to the configured Hugging Face dataset repo.",
    )
    args = parser.parse_args()
    run_eap_ig(args.config, save_to_hf=args.save_to_hf)


if __name__ == "__main__":
    main()
