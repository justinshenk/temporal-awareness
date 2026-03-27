"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm

warnings.filterwarnings("ignore")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/SD-interp/mech-interp-toolkit.git",
    ],
    check=True,
)

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


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, normalizing unsupported dtypes."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor.numpy()


def load_prompts(
    input_file: Path,
) -> list[str]:
    """Load prompts from ``input_file`` and return a list of clean prompts.

    Args:
        input_file: Path to a JSON file containing a ``pairs`` list. Each item
            may be either a prompt string or a dict with a ``question`` field.

    Returns:
        List of prompt strings extracted from the input file.
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts = []
    pairs = data.get("data", [])

    # Accept either raw prompt strings or dict records containing a question field.
    for pair in pairs:
        if isinstance(pair, str):
            prompt = pair
        elif isinstance(pair, dict):
            prompt = "{}".format(
                pair.get("question", ""),
            )
        else:
            raise RuntimeError("Incorrect type for pairs")

        clean_prompts.append(prompt)
    return clean_prompts


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary with configuration values
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_alnum(s: str) -> str:
    """Extract the first alphanumeric character from a string.

    Args:
        s: Input string to search

    Returns:
        First alphanumeric character found

    Raises:
        ValueError: If no alphanumeric character is found
    """
    for c in s:
        if c.isalnum():
            return c
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


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    """Return the start index of ``subsequence`` in ``sequence``."""
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    raise ValueError("Subsequence not found")


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

    args = parser.parse_args()

    config = load_config(CONFIG_PATH / args.config)

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    layer_components = config["setup"].get("layer_components", None)
    granularity = config["setup"].get("granularity", "coarse")
    quadrature = resolve_quadrature(config)
    task = config["setup"].get("task", None)

    if task != "classification":
        raise ValueError("task must be classification")

    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    filename: str = config["output"]["filename"]
    hf_repo_id: str = HF_REPO_ID
    hf_repo_type: str = config["output"].get("hf_repo_type", "dataset")

    system_prompt: str = config["parameters"]["system_prompt"]
    steps: list[int] = config["parameters"]["steps"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for Hub uploads.")
    hf_api = HfApi(token=hf_token)
    hf_repo_id = resolve_hf_repo_id(hf_api, hf_repo_id)
    hf_api.create_repo(repo_id=hf_repo_id, repo_type=hf_repo_type, exist_ok=True)
    print(
        f"[HF upload] Using repo_id={hf_repo_id} repo_type={hf_repo_type}",
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
                repo_id=hf_repo_id,
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

    from mech_interp_toolkit.activation_dict import expand_mask
    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)

    # Suffix prompts the model to complete with option character
    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        suffix=prompt_suffix,
        system_prompt=system_prompt,
        attn_type="eager",
        dtype=dtype,
    )

    baseline_token = tokenizer.eos_token_id
    user_tag_ids = tokenizer.tokenizer.encode(
        "<|im_start|>user", add_special_tokens=False
    )
    assistant_tag_ids = tokenizer.tokenizer.encode(
        "<|im_start|>assistant", add_special_tokens=False
    )

    n_layers = model.config.num_hidden_layers

    if layer_components is None:
        if granularity == "coarse":
            layer_components = [
                (layer, component)
                for layer in range(n_layers)
                for component in ("attn", "mlp")
            ]
        elif granularity == "fine":
            layer_components = [
                (layer, component)
                for layer in range(n_layers)
                for component in ("z", "mlp_hidden")
            ]
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
    else:
        layer_components = [tuple(lc) for lc in layer_components]

    system_prompt_length = (
        len(tokenizer.tokenizer.encode(system_prompt, add_special_tokens=False)) + 1
    )  # For <|im_end|>

    token_id_short = tokenizer.tokenizer.encode(
        option_keys[0], add_special_tokens=False
    )[0]
    token_id_long = tokenizer.tokenizer.encode(
        option_keys[1], add_special_tokens=False
    )[0]

    metrics = {
        "logit_short": lambda logits: logits[:, -1, token_id_short].mean(),
        "logit_long": lambda logits: logits[:, -1, token_id_long].mean(),
    }

    def chunk_list(prompt_list: list[str]) -> list[list[str]]:
        """Split a list into chunks of size batch_size."""
        return [
            prompt_list[i : i + batch_size]
            for i in range(0, len(prompt_list), batch_size)
        ]

    all_clean_prompts = load_prompts(
        input_file_path,
    )
    all_clean_prompts = chunk_list(all_clean_prompts)

    # Pre-tokenize all batches once per order to avoid redundant work across num_steps iterations
    tokenized_clean = [tokenizer(b) for b in all_clean_prompts]

    for metric_label, metric_fn in metrics.items():
        batch_outputs: list[dict[str, np.ndarray]] = []
        for _ in range(len(tokenized_clean)):
            batch_output: dict[str, np.ndarray] = {}
            batch_output["metadata__config_json"] = np.array(
                json.dumps(config), dtype=np.str_
            )

            batch_outputs.append(batch_output)

        for num_steps in tqdm(steps, desc=f"[{metric_label}] Processing step counts"):
            for i in tqdm(
                range(len(tokenized_clean)),
                desc=f"Batches (steps={num_steps})",
                leave=False,
            ):
                clean_inputs = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_clean[i].items()
                }
                corrupted_inputs = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_clean[i].items()
                }
                corrupted_input_ids = corrupted_inputs["input_ids"]
                for row_idx in range(corrupted_input_ids.shape[0]):
                    row_tokens = corrupted_input_ids[row_idx].tolist()
                    user_tag_start = find_subsequence(row_tokens, user_tag_ids)
                    assistant_tag_start = find_subsequence(
                        row_tokens, assistant_tag_ids
                    )
                    content_start = user_tag_start + len(user_tag_ids) + 1  # skip \n after user tag
                    content_end = assistant_tag_start - 2  # skip <|im_end|>\n before assistant tag
                    corrupted_input_ids[row_idx, content_start:content_end] = (
                        baseline_token
                    )

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
                    clean_logits[:, -1, [token_id_short, token_id_long]].detach().cpu()  # type: ignore[index]
                )
                corrupted_logits_cpu = (
                    corrupted_logits[:, -1, [token_id_short, token_id_long]]  # type: ignore[index]
                    .detach()
                    .cpu()
                )
                del clean_logits, corrupted_logits

                eap_ig_scores.attention_mask = expand_mask(
                    eap_ig_scores.attention_mask, system_prompt_length
                )
                eap_ig_scores = eap_ig_scores.apply(
                    torch.nanmean, dim=1, mask_aware=True
                )  # (batch,) | (batch, n_head) | (batch, neuron)
                eap_ig_scores = eap_ig_scores.apply(lambda x: x.detach().cpu())

                for key, value in eap_ig_scores.items():
                    batch_outputs[i][f"step_{num_steps}__{key[1]}__{key[0]}"] = (
                        tensor_to_numpy(value)
                    )

                batch_outputs[i][f"step_{num_steps}__clean_logits"] = (
                    clean_logits_cpu.float().numpy()
                )
                batch_outputs[i][f"step_{num_steps}__corrupted_logits"] = (
                    corrupted_logits_cpu.float().numpy()
                )

                # Delete temporary objects to free memory
                del (
                    eap_ig_scores,
                    clean_inputs,
                    corrupted_inputs,
                    clean_logits_cpu,
                    corrupted_logits_cpu,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        for batch_idx, output_arrays in enumerate(batch_outputs):
            output_file = save_loc / (
                f"{filename}_{metric_label}_batch_{batch_idx:05d}.npz"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, **output_arrays)
            output_file_abs = output_file.resolve()
            try:
                path_in_repo = output_file_abs.relative_to(
                    Path.cwd().resolve()
                ).as_posix()
            except ValueError:
                path_in_repo = output_file.name
            _enqueue_upload(output_file_abs, path_in_repo)

    upload_queue.put(None)
    upload_thread.join()


if __name__ == "__main__":
    main()
