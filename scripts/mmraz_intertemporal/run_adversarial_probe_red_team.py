#!/usr/bin/env python3
"""Run MMRAZ temporal probe adversarial red-teaming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmraz_intertemporal import RedTeamConfig, run_red_teaming


def build_parser() -> argparse.ArgumentParser:
    defaults = RedTeamConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run a cold-start red-teaming loop against a GPT-2 layer-6 temporal MM probe using an "
            "Anthropic Claude Sonnet 4 attacker model over prompt+completion pairs."
        )
    )
    parser.add_argument("--explicit-dataset-path", type=str, default=None, help="Path to explicit-expanded dataset JSON.")
    parser.add_argument("--output-root", type=str, default=None, help="Output root for run artifacts.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a prior run. Uses --run-id when provided, otherwise resumes the latest run under --output-root.",
    )
    parser.add_argument("--attacker-model", type=str, default=defaults.attacker_model, help="Anthropic attacker model.")
    parser.add_argument("--attacker-max-output-tokens", type=int, default=defaults.attacker_max_output_tokens, help="Max attacker output tokens.")
    parser.add_argument("--num-rounds", type=int, default=defaults.num_rounds, help="Number of cold-start attack rounds.")
    parser.add_argument("--candidates-per-round", type=int, default=defaults.candidates_per_round, help="Candidates per round.")
    parser.add_argument("--attacker-max-retries", type=int, default=defaults.attacker_max_retries, help="Retries per round if generation/parsing fails.")
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed, help="Random seed for reproducibility metadata.")
    parser.add_argument("--mm-probe-layer", type=int, default=defaults.mm_probe_layer, help="GPT-2 layer index for MM probe.")
    parser.add_argument("--probe-batch-size", type=int, default=defaults.probe_batch_size, help="Batch size for GPT-2 activation extraction.")
    parser.add_argument("--probe-train-test-split", type=float, default=defaults.probe_train_test_split, help="Prompt-level split used to train the MM probe.")
    parser.add_argument("--probe-random-state", type=int, default=defaults.probe_random_state, help="Random state for probe split.")
    parser.add_argument("--attacker-timeout-seconds", type=float, default=defaults.attacker_timeout_seconds, help="Anthropic API timeout.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=defaults.show_progress, help="Show tqdm progress output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = RedTeamConfig(
        explicit_dataset_path=args.explicit_dataset_path or RedTeamConfig().explicit_dataset_path,
        output_root=args.output_root or RedTeamConfig().output_root,
        run_id=args.run_id,
        resume=args.resume,
        attacker_model=args.attacker_model,
        attacker_max_output_tokens=args.attacker_max_output_tokens,
        num_rounds=args.num_rounds,
        candidates_per_round=args.candidates_per_round,
        attacker_max_retries=args.attacker_max_retries,
        random_seed=args.random_seed,
        mm_probe_layer=args.mm_probe_layer,
        probe_batch_size=args.probe_batch_size,
        probe_train_test_split=args.probe_train_test_split,
        probe_random_state=args.probe_random_state,
        attacker_timeout_seconds=args.attacker_timeout_seconds,
        show_progress=args.progress,
    )

    summary = run_red_teaming(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
