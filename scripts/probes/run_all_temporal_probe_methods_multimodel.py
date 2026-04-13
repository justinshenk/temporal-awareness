#!/usr/bin/env python3
"""Run LR, DMM, and AttnProbe sequentially across the supported multimodel set.

For each probe method and model, this script trains on the implicit
AB-randomized dataset and then validates on the explicit AB-randomized dataset.
Every subprocess is run sequentially so GPU memory is released between runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "probes" / "train_temporal_probes_caa_multimodel.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "probes" / "validate_temporal_probes_multimodel.py"
IMPLICIT_DATASET = ROOT / "data" / "raw" / "temporal_scope_AB_randomized" / "temporal_scope_implicit.json"
DEFAULT_MODELS = ["gpt2", "qwen3-4b", "phi-3-mini-4k-instruct", "llama-3.2-3b"]
DEFAULT_METHODS = ["lr", "dmm", "attn"]
MIN_CUDA_COMPUTE_CAPABILITY = 7.0


def detect_supported_cuda_visible_devices() -> str | None:
    """Return compatible CUDA ordinals in PyTorch's current device order."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    supported = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        compute_cap = float(f"{major}.{minor}")
        if compute_cap >= MIN_CUDA_COMPUTE_CAPABILITY:
            supported.append(str(index))

    return ",".join(supported) if supported else None


def build_subprocess_env(cuda_visible_devices: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(cmd))
    if "CUDA_VISIBLE_DEVICES" in env:
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print("=" * 90)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def append_common_loader_args(cmd: list[str], args) -> list[str]:
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.attn_implementation is not None:
        cmd.extend(["--attn-implementation", args.attn_implementation])
    cmd.extend(["--device-map", args.device_map])
    return cmd


def method_batch_size(method: str, default_batch_size: int, attn_batch_size: int) -> int:
    if method == "attn":
        return attn_batch_size
    return default_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially train and validate LR, DMM, and AttnProbe across all multimodel targets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model aliases or Hugging Face ids to run sequentially",
    )
    parser.add_argument(
        "--probe-methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Probe methods to run sequentially",
    )
    parser.add_argument(
        "--dataset",
        default=str(IMPLICIT_DATASET.relative_to(ROOT)),
        help="Implicit AB-randomized training dataset",
    )
    parser.add_argument(
        "--output",
        default="research/probes",
        help="Root output directory for saved probes",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Training feature-extraction batch size for LR and DMM",
    )
    parser.add_argument(
        "--validate-batch-size",
        type=int,
        default=2,
        help="Validation feature-extraction batch size for LR and DMM",
    )
    parser.add_argument(
        "--attn-batch-size",
        type=int,
        default=1,
        help="Feature-extraction batch size for AttnProbe training and validation",
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=None,
        help="Optional tokenizer max length for training prompts",
    )
    parser.add_argument(
        "--validate-max-length",
        type=int,
        default=256,
        help="Optional tokenizer max length for validation prompts",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention backend to request from Transformers",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders if needed",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache",
    )
    parser.add_argument(
        "--device-map",
        default="single",
        choices=["single", "auto"],
        help=(
            "Device placement for model loading. 'single' keeps each model on subprocess cuda:0; "
            "'auto' allows Accelerate to shard across visible devices."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help=(
            "Comma-separated PyTorch CUDA ordinals for subprocesses. If omitted, the runner "
            "auto-masks GPUs with compute capability below 7.0 in PyTorch's device order."
        ),
    )
    parser.add_argument(
        "--no-auto-cuda-mask",
        action="store_true",
        help="Do not auto-mask unsupported GPUs; inherit CUDA_VISIBLE_DEVICES unchanged",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Only train probes; do not run explicit cross-dataset validation",
    )
    args = parser.parse_args()

    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None and not args.no_auto_cuda_mask:
        cuda_visible_devices = detect_supported_cuda_visible_devices()
        if cuda_visible_devices:
            print(
                "Auto-selected CUDA_VISIBLE_DEVICES="
                f"{cuda_visible_devices} in PyTorch CUDA order for GPUs with compute capability >= "
                f"{MIN_CUDA_COMPUTE_CAPABILITY:.1f}"
            )
        else:
            print("No compatible CUDA device mask auto-detected; inheriting current environment")

    subprocess_env = build_subprocess_env(cuda_visible_devices)

    for method in args.probe_methods:
        for model in args.models:
            train_batch_size = method_batch_size(method, args.train_batch_size, args.attn_batch_size)
            validate_batch_size = method_batch_size(method, args.validate_batch_size, args.attn_batch_size)

            train_cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--probe-method",
                method,
                "--model",
                model,
                "--dataset",
                args.dataset,
                "--output",
                args.output,
                "--batch-size",
                str(train_batch_size),
            ]
            if args.train_max_length is not None:
                train_cmd.extend(["--max-length", str(args.train_max_length)])
            append_common_loader_args(train_cmd, args)
            run_command(train_cmd, subprocess_env)

            if args.skip_validation:
                continue

            validate_cmd = [
                sys.executable,
                str(VALIDATE_SCRIPT),
                "--probe-method",
                method,
                "--models",
                model,
                "--batch-size",
                str(validate_batch_size),
                "--max-length",
                str(args.validate_max_length),
            ]
            append_common_loader_args(validate_cmd, args)
            run_command(validate_cmd, subprocess_env)

    print("\nAll requested probe runs completed successfully.")


if __name__ == "__main__":
    main()
