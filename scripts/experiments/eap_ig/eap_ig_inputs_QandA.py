"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
from pathlib import Path

import torch

try:
    from .eap_ig_qanda_pipeline import run_eap_ig
except ImportError:
    from eap_ig_qanda_pipeline import run_eap_ig

torch.set_grad_enabled(False)


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
