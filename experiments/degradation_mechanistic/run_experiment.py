#!/usr/bin/env python3
"""
Single entry-point for the degradation mechanistic interpretability experiment suite.

This script runs any combination of Phase 3/4/5 experiments for the paper:
"Temporal Awareness in Language Models: Mechanistic Interpretability of
Patience Degradation Under Repetitive Tasks"

Usage:
    # Run all experiments for one model:
    python experiments/degradation_mechanistic/run_experiment.py --model Llama-3.1-8B-Instruct

    # Run a specific phase:
    python experiments/degradation_mechanistic/run_experiment.py --model Qwen3-8B --phase 3

    # Run a specific experiment:
    python experiments/degradation_mechanistic/run_experiment.py --model Qwen3-8B --experiment refusal

    # Run all experiments for all models (full suite):
    python experiments/degradation_mechanistic/run_experiment.py --all-models --all-phases

    # Quick mode (fewer samples, for testing):
    python experiments/degradation_mechanistic/run_experiment.py --model Qwen3-8B --quick

    # On Sherlock (submits SLURM jobs instead of running locally):
    python experiments/degradation_mechanistic/run_experiment.py --all-models --all-phases --slurm

See experiments/degradation_mechanistic/README.md for full documentation.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# Experiment registry: name -> (module_path, phase, description, depends_on)
EXPERIMENTS = {
    # Phase 3: Mechanistic analysis
    "refusal": (
        "scripts.experiments.phase3_refusal_direction",
        3,
        "Refusal vs degradation direction comparison",
        [],
    ),
    "confound": (
        "scripts.experiments.phase3_context_confound",
        3,
        "Context length confound control",
        [],
    ),
    "trajectory": (
        "scripts.experiments.phase3_trajectory_geometry",
        3,
        "Degradation trajectory PCA geometry",
        [],
    ),
    "early_detection": (
        "scripts.experiments.phase3_early_detection",
        3,
        "Early detection probe analysis",
        [],
    ),
    "attention": (
        "scripts.experiments.phase3_attention_analysis",
        3,
        "Attention head degradation analysis",
        [],
    ),
    "prompt_dimensions": (
        "scripts.experiments.phase3_prompt_dimensions",
        3,
        "6-dimension prompt cartography",
        [],
    ),
    "implicit": (
        "scripts.experiments.phase3_implicit_repetition",
        3,
        "Implicit multi-turn repetition transfer",
        [],
    ),
    "reasoning": (
        "scripts.experiments.phase3_reasoning_degradation",
        3,
        "Reasoning-phase degradation (DeepSeek-R1)",
        [],
    ),
    "patching": (
        "scripts.experiments.phase3_causal_patching",
        3,
        "Causal activation patching",
        ["refusal"],
    ),
    "cross_model": (
        "scripts.experiments.phase3_cross_model_transfer",
        3,
        "Cross-model direction transfer",
        [],
    ),
    # Phase 4: Causal bridge
    "causal_bridge": (
        "scripts.experiments.phase4_causal_bridge",
        4,
        "Observational-to-causal bridge (r_obs vs r_causal)",
        [],
    ),
    "steering": (
        "scripts.experiments.phase4_intervention_steering",
        4,
        "Activation steering dose-response",
        ["refusal"],
    ),
    # Phase 5: Safety evaluations
    "safety": (
        "scripts.experiments.phase5_safety_evaluations",
        5,
        "Deployment robustness + alignment stability + intervention efficacy",
        [],
    ),
}

# Model registry
MODELS_8B = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B",
    "Qwen3-8B",
    "Qwen3-4B-Instruct-2507",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Ouro-2.6B",
]
MODEL_30B = "Qwen3-30B-A3B"
ALL_MODELS = MODELS_8B + [MODEL_30B]

# Wave ordering (for dependency-aware execution)
WAVE_1 = ["refusal", "confound", "trajectory", "early_detection", "attention", "prompt_dimensions"]
WAVE_2 = ["patching", "steering", "cross_model", "causal_bridge", "implicit", "reasoning"]
WAVE_3 = ["safety"]


def list_experiments():
    """Print available experiments grouped by phase."""
    print("\nAvailable experiments:")
    print("=" * 70)
    for phase in [3, 4, 5]:
        phase_exps = {k: v for k, v in EXPERIMENTS.items() if v[1] == phase}
        if phase_exps:
            print(f"\n  Phase {phase}:")
            for name, (_, _, desc, deps) in sorted(phase_exps.items()):
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                print(f"    {name:20s} — {desc}{dep_str}")
    print(f"\n  Models: {', '.join(ALL_MODELS)}")
    print()


def run_experiment_local(experiment_name, model, quick=False, backend="pytorch", device="cuda"):
    """Run a single experiment locally by importing and calling its main()."""
    module_path, phase, desc, deps = EXPERIMENTS[experiment_name]
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name} (Phase {phase})")
    print(f"Model:   {model}")
    print(f"Desc:    {desc}")
    print(f"{'='*60}\n")

    # Build sys.argv for the experiment script's argparse
    sys.argv = [
        module_path,
        "--model", model,
        "--backend", backend,
        "--device", device,
    ]
    if quick:
        sys.argv.append("--quick")

    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "main"):
            module.main()
        else:
            # Fall back to subprocess if no main() function
            script_path = module_path.replace(".", "/") + ".py"
            cmd = [sys.executable, script_path, "--model", model, "--backend", backend, "--device", device]
            if quick:
                cmd.append("--quick")
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"ERROR in {experiment_name}: {e}")
        return False
    return True


def submit_slurm(experiment_name, model, quick=False):
    """Submit experiment as a SLURM job on Sherlock."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    submit_all = repo_root / "scripts" / "experiments" / "submit_all_models.sh"

    if not submit_all.exists():
        print(f"ERROR: SLURM submit script not found at {submit_all}")
        return False

    cmd = ["bash", str(submit_all), experiment_name]
    print(f"Submitting SLURM: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run degradation mechanistic interpretability experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model Qwen3-8B --experiment refusal
  %(prog)s --model Llama-3.1-8B-Instruct --phase 3
  %(prog)s --all-models --all-phases
  %(prog)s --all-models --all-phases --slurm
  %(prog)s --list
        """,
    )
    parser.add_argument("--model", type=str, help="Model to run experiments on")
    parser.add_argument("--all-models", action="store_true", help="Run on all 7 models")
    parser.add_argument("--experiment", type=str, choices=list(EXPERIMENTS.keys()), help="Run a specific experiment")
    parser.add_argument("--phase", type=int, choices=[3, 4, 5], help="Run all experiments in a phase")
    parser.add_argument("--all-phases", action="store_true", help="Run all experiments (Phases 3-5)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "transformer_lens", "auto"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM jobs on Sherlock")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], help="Run experiments in a specific wave (respects dependencies)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_experiments()
        return

    # Determine models
    if args.all_models:
        models = ALL_MODELS
    elif args.model:
        models = [args.model]
    else:
        print("ERROR: Specify --model MODEL or --all-models")
        list_experiments()
        sys.exit(1)

    # Determine experiments
    if args.experiment:
        experiments = [args.experiment]
    elif args.phase:
        experiments = [name for name, (_, phase, _, _) in EXPERIMENTS.items() if phase == args.phase]
    elif args.wave:
        wave_map = {1: WAVE_1, 2: WAVE_2, 3: WAVE_3}
        experiments = wave_map[args.wave]
    elif args.all_phases:
        experiments = WAVE_1 + WAVE_2 + WAVE_3
    else:
        print("ERROR: Specify --experiment, --phase, --all-phases, or --wave")
        list_experiments()
        sys.exit(1)

    print(f"\n{'#'*60}")
    print("Degradation Mechanistic Interpretability Experiment Suite")
    print(f"Models:      {', '.join(models)}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Mode:        {'SLURM' if args.slurm else 'local'} ({'quick' if args.quick else 'full'})")
    print(f"{'#'*60}\n")

    if args.slurm:
        # SLURM mode: submit_all_models.sh handles per-model submission
        for exp in experiments:
            submit_slurm(exp, models[0], quick=args.quick)
    else:
        # Local mode: run each (experiment, model) pair sequentially
        total = len(experiments) * len(models)
        completed = 0
        failed = []

        for exp in experiments:
            for model in models:
                completed += 1
                print(f"\n[{completed}/{total}] {exp} × {model}")
                success = run_experiment_local(exp, model, quick=args.quick, backend=args.backend, device=args.device)
                if not success:
                    failed.append(f"{exp} × {model}")

        print(f"\n{'='*60}")
        print(f"COMPLETE: {completed - len(failed)}/{total} succeeded")
        if failed:
            print(f"FAILED ({len(failed)}):")
            for f in failed:
                print(f"  - {f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
