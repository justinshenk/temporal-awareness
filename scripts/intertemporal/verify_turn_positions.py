#!/usr/bin/env python3
"""Prove that a geometry run's saved activations sit at the positions they claim.

The defect this gate exists for was silent. `run_with_cache` re-applied the chat
template to prompt+response, so the cached sequence had the response inside the
user turn and the chat suffix after it, while the position mapping indexes the
trajectory, where the suffix comes first. Both orderings have the same length, so
nothing ever errored and every position labelled `chat_suffix` held a response
token.

Length checks cannot catch that. Embedding ground truth can. `hook_embed` at
position i is exactly `W_E[token_ids[i]]`, so caching that hook and comparing it
row by row against the trajectory's own token ids proves the cache and the
mapping index the same sequence. This runs the real pipeline (PreferenceQuerier
plus SamplePositionMapping), not a reconstruction of it.

The gate also decodes the tokens the mapping calls `chat_suffix` and
`chat_suffix_tail` and requires the model's real turn tokens among them, so a run
whose suffix detection silently found nothing fails here rather than producing an
empty dataset.

Usage:
    uv run python scripts/intertemporal/verify_turn_positions.py \\
        --config data/intertemporal/health/health_geometry.json \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --expect '<|eot_id|>' --expect '<|start_header_id|>' \\
        --expect 'assistant' --expect '<|end_header_id|>'

Exit code 0 means every named position matched embedding ground truth and every
expected turn token was found. Any other code means do not run the extraction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.binary_choice.binary_choice_runner import BinaryChoiceRunner
from src.common.file_io import parse_file_path
from src.inference.backends.model_backend import ModelBackend
from src.intertemporal.common.project_paths import get_prompt_dataset_configs_dir
from src.intertemporal.common.sample_position_mapping import SamplePositionMapping
from src.intertemporal.formatting.prompt_formats import find_prompt_format_config
from src.intertemporal.preference import PreferenceQuerier, PreferenceQueryConfig
from src.intertemporal.prompt import PromptDatasetConfig, PromptDatasetGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EMBED_HOOK = "hook_embed"
TURN_POSITIONS = ("chat_suffix", "chat_suffix_tail")


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify cached activations against embedding ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Dataset config path or name")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples to verify (default: 1)",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=25,
        help="Samples to attempt before giving up on finding valid ones",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4, help="Absolute tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="TOKEN",
        help="A decoded token that must appear in the turn window. Repeatable.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=[b.value for b in ModelBackend],
        help="Force a backend. Default: whatever the extraction pipeline picks "
        "on this machine, which is the only setting the gate actually certifies.",
    )
    parser.add_argument(
        "--out", default=None, help="Write the gate report to this JSON path"
    )
    return parser.parse_args(argv)


def load_dataset_config(spec: str) -> PromptDatasetConfig:
    path = parse_file_path(
        spec, default_dir_path=str(get_prompt_dataset_configs_dir()), default_ext=".json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return PromptDatasetConfig.from_json(path)


def check_sample(sample, runner, querier, atol: float) -> dict:
    """Verify one sample. Returns a report; report["ok"] is the verdict."""
    prompt_format = find_prompt_format_config(sample.formatting_id)
    choice_prefix = prompt_format.get_response_prefix_before_choice()

    pref = querier.query_sample(
        sample, runner, choice_prefix, activation_names=[EMBED_HOOK]
    )
    if pref.chosen_traj is None:
        return {"ok": False, "reason": "no_chosen_traj"}
    if pref.internals is None or EMBED_HOOK not in pref.internals.activations:
        return {"ok": False, "reason": "no_embed_capture"}

    try:
        mapping = SamplePositionMapping.build(sample, runner, pref=pref)
    except AssertionError as exc:
        return {"ok": False, "reason": f"mapping_invalid:{str(exc).splitlines()[0]}"}

    token_ids = list(pref.chosen_traj.token_ids)
    embed = pref.internals.activations[EMBED_HOOK].float()
    if embed.shape[0] != len(token_ids):
        return {
            "ok": False,
            "reason": f"length_mismatch:cache={embed.shape[0]}:traj={len(token_ids)}",
        }

    w_e = runner.W_E.detach().float().cpu()
    tokenizer = runner._tokenizer

    checked = 0
    mismatches: list[dict] = []
    max_abs_diff = 0.0
    for name, abs_positions in sorted(mapping.named_positions.items()):
        for abs_pos in abs_positions:
            expected = w_e[token_ids[abs_pos]]
            actual = embed[abs_pos]
            diff = (expected - actual).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            checked += 1
            if not torch.allclose(expected, actual, atol=atol, rtol=0.0):
                mismatches.append(
                    {
                        "position": name,
                        "abs_pos": abs_pos,
                        "labelled_token": tokenizer.decode([token_ids[abs_pos]]),
                        "max_abs_diff": diff,
                    }
                )

    turn_tokens: dict[str, list[str]] = {}
    for name in TURN_POSITIONS:
        turn_tokens[name] = [
            tokenizer.decode([token_ids[p]])
            for p in mapping.named_positions.get(name, [])
        ]

    return {
        "ok": not mismatches,
        "sample_idx": sample.sample_idx,
        "n_tokens": len(token_ids),
        "n_named_positions": checked,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "turn_tokens": turn_tokens,
        "prompt_len": mapping.prompt_len,
        "traj_tail": [tokenizer.decode([t]) for t in token_ids[-12:]],
    }


def main() -> int:
    args = get_args()

    dataset_config = load_dataset_config(args.config)
    dataset = PromptDatasetGenerator(dataset_config).generate()
    logger.info("Generated %d prompt samples", len(dataset.samples))

    querier = PreferenceQuerier(PreferenceQueryConfig(skip_generation=True))
    if args.backend:
        querier._runner = BinaryChoiceRunner(
            model_name=args.model, backend=ModelBackend(args.backend)
        )
    runner = querier._load_model(args.model)
    backend_name = type(runner._backend).__name__
    logger.info("backend: %s dtype: %s", backend_name, runner.dtype)

    reports: list[dict] = []
    attempts = 0
    for sample in dataset.samples:
        if len(reports) >= args.n_samples or attempts >= args.max_tries:
            break
        attempts += 1
        report = check_sample(sample, runner, querier, args.atol)
        if "n_named_positions" not in report:
            logger.warning("Sample skipped: %s", report.get("reason"))
            continue
        reports.append(report)

    verdict = {
        "model": args.model,
        "config": args.config,
        "backend": backend_name,
        "dtype": str(runner.dtype),
        "samples_verified": len(reports),
        "samples_attempted": attempts,
        "reports": reports,
    }

    if not reports:
        verdict["result"] = "FAIL"
        verdict["reason"] = "no_verifiable_sample"
        print(json.dumps(verdict, indent=2))
        print("GATE FAIL: no sample produced a trajectory and a valid mapping.")
        return 2

    print("=" * 70)
    print(f"TURN-POSITION GATE  model={args.model}  config={args.config}")
    print("=" * 70)
    for report in reports:
        print(f"\nsample_idx={report['sample_idx']}  tokens={report['n_tokens']}  "
              f"prompt_len={report['prompt_len']}")
        print(f"  trajectory tail: {report['traj_tail']}")
        for name in TURN_POSITIONS:
            positions = report["turn_tokens"][name]
            print(f"  {name}: {positions}")
        print(f"  named positions checked: {report['n_named_positions']}")
        print(f"  mismatches: {report['n_mismatches']}  "
              f"max|W_E[tok] - hook_embed| = {report['max_abs_diff']:.3e} "
              f"(atol {report['atol']:.0e})")
        for bad in report["mismatches"]:
            print(f"    MISMATCH {bad['position']}[{bad['abs_pos']}] "
                  f"token={bad['labelled_token']!r} diff={bad['max_abs_diff']:.3e}")

    aligned = all(r["ok"] for r in reports)

    observed: set[str] = set()
    for report in reports:
        for name in TURN_POSITIONS:
            observed.update(report["turn_tokens"][name])
    stripped = {t.strip() for t in observed}
    missing = [
        t for t in args.expect if t not in observed and t.strip() not in stripped
    ]
    empty_window = any(
        not report["turn_tokens"]["chat_suffix"]
        and not report["turn_tokens"]["chat_suffix_tail"]
        for report in reports
    )

    verdict["all_positions_aligned"] = aligned
    verdict["expected_tokens"] = args.expect
    verdict["missing_expected_tokens"] = missing
    verdict["empty_turn_window"] = empty_window
    verdict["result"] = (
        "PASS" if aligned and not missing and not empty_window else "FAIL"
    )

    print()
    print(f"embedding ground truth : {'ALIGNED' if aligned else 'MISALIGNED'}")
    print(f"expected turn tokens   : {args.expect or '(none requested)'}")
    print(f"missing                : {missing or 'none'}")
    print(f"turn window empty      : {empty_window}")
    print(f"GATE {verdict['result']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(verdict, handle, indent=2)
        print(f"report written to {out_path}")

    return 0 if verdict["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
