"""CLI entry point for selected-node activation caching."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .cache_activations_config import load_model_settings
    from .cache_activations_paths import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_NODES_PATH,
        HF_REPO_ID,
        PROJECT_ROOT,
        resolve_path,
    )
    from .cache_activations_runner import cache_prompt_activations
except ImportError:
    from cache_activations_config import load_model_settings
    from cache_activations_paths import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_NODES_PATH,
        HF_REPO_ID,
        PROJECT_ROOT,
        resolve_path,
    )
    from cache_activations_runner import cache_prompt_activations


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
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help="Optional token position(s) to cache. Defaults to all positions.",
    )
    parser.add_argument(
        "--average-positions",
        action="store_true",
        help="Save activations averaged across valid token positions.",
    )
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

    cache_prompt_activations(
        dataset=args.dataset,
        model_name=args.model or config_model,
        nodes_path=resolve_path(args.nodes_path),
        output_dir=output_dir,
        batch_size=args.batch_size or config_batch_size,
        dtype=args.dtype if args.dtype is not None else config_dtype,
        device=args.device,
        attn_type=args.attn_type,
        max_samples=args.max_samples,
        positions=args.positions,
        save_to_hf=args.save_to_hf,
        hf_repo_id=args.hf_repo_id,
        hf_repo_type=args.hf_repo_type,
        average_positions=args.average_positions,
    )


if __name__ == "__main__":
    main()
