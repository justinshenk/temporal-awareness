"""
Execution pipeline for the Q&A EAP-IG workflow.
"""

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

try:
    from .eap_ig_qanda_common import (
        HF_REPO_ID,
        build_layer_components,
        chunk_list,
        ensure_mech_interp_toolkit_installed,
        extract_alnum,
        load_and_merge_pairs,
        load_config,
        resolve_config_path,
        resolve_quadrature,
        tensor_to_numpy,
    )
    from .eap_ig_qanda_upload import maybe_start_upload_worker
except ImportError:
    from eap_ig_qanda_common import (
        HF_REPO_ID,
        build_layer_components,
        chunk_list,
        ensure_mech_interp_toolkit_installed,
        extract_alnum,
        load_and_merge_pairs,
        load_config,
        resolve_config_path,
        resolve_quadrature,
        tensor_to_numpy,
    )
    from eap_ig_qanda_upload import maybe_start_upload_worker


def build_metrics(token_a: int, token_b: int) -> dict[str, Any]:
    """Build metric functions from the resolved option tokens."""
    return {
        "logit_A": lambda logits: logits[:, -1, token_a].mean(),
        "logit_B": lambda logits: logits[:, -1, token_b].mean(),
    }


def resolve_option_tokens(tokenizer: Any, option_keys: list[str]) -> tuple[int, int]:
    """Resolve option keys into single-token ids."""
    token_a_str = extract_alnum(option_keys[0])
    token_b_str = extract_alnum(option_keys[1])

    token_a = tokenizer.tokenizer.encode(token_a_str, add_special_tokens=False)
    token_b = tokenizer.tokenizer.encode(token_b_str, add_special_tokens=False)

    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError(
            f"Token A tokenizes to {token_a}\nToken B tokenizes to {token_b}\n"
            "Optionkeys must tokenize to single token each."
        )
    return token_a[0], token_b[0]


def build_option_orders(
    input_file_path: Path,
    template: str,
    option_keys: list[str],
    batch_size: int,
) -> list[tuple[str, list[list[str]], list[list[str]]]]:
    """Build prompt batches for both option orders."""
    all_clean_prompts, all_corrupted_prompts = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "immediate", "long_term"],
    )

    all_corrupted_prompts_swapped, all_clean_prompts_swapped = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "long_term", "immediate"],
    )

    return [
        (
            "short_first",
            chunk_list(all_clean_prompts, batch_size),
            chunk_list(all_corrupted_prompts, batch_size),
        ),
        (
            "long_first",
            chunk_list(all_clean_prompts_swapped, batch_size),
            chunk_list(all_corrupted_prompts_swapped, batch_size),
        ),
    ]


def process_batch(
    *,
    model: Any,
    expand_mask: Any,
    eap_integrated_gradients: Any,
    tokenized_clean_batch: dict[str, Any],
    tokenized_corrupted_batch: dict[str, Any],
    metric_fn: Any,
    layer_components: list[tuple[int, str]],
    steps: list[int],
    quadrature: str,
    system_prompt_length: int,
    token_a: int,
    token_b: int,
    config: dict,
    order_label: str,
    metric_type: str,
) -> dict[str, np.ndarray]:
    """Run all configured step counts for one tokenized prompt batch."""
    batch_output: dict[str, np.ndarray] = {}
    batch_output["metadata__config_json"] = np.array(json.dumps(config), dtype=np.str_)
    batch_output["metadata__option_order"] = np.array(order_label, dtype=np.str_)
    batch_output["metadata__metric_type"] = np.array(metric_type, dtype=np.str_)

    for num_steps in tqdm(
        steps,
        desc="Step counts",
        leave=False,
    ):
        clean_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_clean_batch.items()
        }
        corrupted_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_corrupted_batch.items()
        }

        eap_ig_scores, (clean_logits, corrupted_logits) = eap_integrated_gradients(
            model,
            clean_inputs,
            corrupted_inputs,
            metric_fn,
            layer_components,
            steps=num_steps,
            include_block_outputs=True,
            quadrature=quadrature,
        )

        clean_logits_cpu = clean_logits[:, -1, [token_a, token_b]].detach().cpu()
        corrupted_logits_cpu = corrupted_logits[:, -1, [token_a, token_b]].detach().cpu()
        del clean_logits, corrupted_logits

        eap_ig_scores.attention_mask = expand_mask(
            eap_ig_scores.attention_mask, system_prompt_length
        )
        token_position_counts = eap_ig_scores.attention_mask.sum(dim=1).detach().cpu()
        eap_ig_scores = eap_ig_scores.apply(torch.nansum, dim=1, mask_aware=True)
        eap_ig_scores = eap_ig_scores.apply(lambda x: x.detach().cpu())

        for key, value in eap_ig_scores.items():
            batch_output[f"step_{num_steps}__{key[1]}__{key[0]}"] = tensor_to_numpy(
                value
            )

        batch_output[f"step_{num_steps}__clean_logits"] = clean_logits_cpu.float().numpy()
        batch_output[f"step_{num_steps}__corrupted_logits"] = (
            corrupted_logits_cpu.float().numpy()
        )
        batch_output[f"step_{num_steps}__token_positions_considered"] = (
            tensor_to_numpy(token_position_counts)
        )

        del (
            eap_ig_scores,
            token_position_counts,
            clean_inputs,
            corrupted_inputs,
            clean_logits_cpu,
            corrupted_logits_cpu,
        )

    return batch_output


def run_eap_ig(
    config_path: Path,
    model=None,
    tokenizer=None,
    *,
    save_to_hf: bool = True,
) -> tuple[Any, Any]:
    """Run Q&A EAP-IG from Python or notebooks."""
    ensure_mech_interp_toolkit_installed()
    config = load_config(resolve_config_path(config_path))

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    layer_components = config["setup"].get("layer_components", None)
    granularity = config["setup"].get("granularity", "coarse")
    quadrature = resolve_quadrature(config)

    dtype = config["setup"].get("dtype", None)

    data_loc = Path(config["paths"]["data_loc"])
    save_loc = Path(config["paths"]["save_loc"])

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
    )
    token_a, token_b = resolve_option_tokens(tokenizer, option_keys)
    metrics = build_metrics(token_a, token_b)
    option_orders = build_option_orders(
        input_file_path=input_file_path,
        template=template,
        option_keys=option_keys,
        batch_size=batch_size,
    )

    for order_label, chunked_clean_prompts, chunked_corrupted_prompts in option_orders:
        tokenized_clean = [tokenizer(batch) for batch in chunked_clean_prompts]
        tokenized_corrupted = [tokenizer(batch) for batch in chunked_corrupted_prompts]

        for metric_label, metric_fn in metrics.items():
            for i in tqdm(
                range(len(tokenized_clean)),
                desc=f"[{order_label}/{metric_label}] Batches",
            ):
                batch_output = process_batch(
                    model=model,
                    expand_mask=expand_mask,
                    eap_integrated_gradients=eap_integrated_gradients,
                    tokenized_clean_batch=tokenized_clean[i],
                    tokenized_corrupted_batch=tokenized_corrupted[i],
                    metric_fn=metric_fn,
                    layer_components=layer_components,
                    steps=steps,
                    quadrature=quadrature,
                    system_prompt_length=system_prompt_length,
                    token_a=token_a,
                    token_b=token_b,
                    config=config,
                    order_label=order_label,
                    metric_type=metric_type,
                )
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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    if upload_queue is not None and upload_thread is not None:
        upload_queue.put(None)
        upload_thread.join()

    return model, tokenizer
