#!/usr/bin/env python3
"""launch_partition.py — partitioned launcher for the 4-GPU EMNLP run.

ARCHITECTURE
------------
We rent four Vast.ai instances in parallel. Each instance is one GPU
running ONE partition of (model, domain) jobs. Each job is one call to
`run_staircase_v2.py`. Models are loaded once per partition and reused
across the four domains they touch.

Partitions (matches the locked plan):
    A  — Large models       (1× A100/H100 80GB):  27B / 32B / 70B variants
    B  — Medium models      (1× RTX 6000 Ada 48GB): 8B-14B variants
    C  — Small + code       (1× RTX 6000 Ada 48GB): 1B-4B + workshop 11
    D  — Training dynamics  (1× A6000 48GB):       Pythia checkpoint sweeps

USAGE (on each Vast.ai instance)
---------------------------------
    # First do the env setup (clone, pip install, set MAAR_DATA_ROOT)
    # then:
    python scripts/lookahead/experiments/launch_partition.py \\
        --partition A \\
        --output_dir results/v2

To debug locally before launching:
    python scripts/lookahead/experiments/launch_partition.py \\
        --partition C --dry_run

Resume is automatic — finished outputs are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Job spec
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    model: str
    domain: str
    quantization: str = "bf16"
    layer_mode: str = "maar_range"  # Maar's domain → maar_range; code → workshop_6
    probe_types: str = "linear"     # add 'mlp' for Block 3 rigor pass
    max_examples: int = 0           # 0 = use all
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# Model lists per partition (matches the GPU rental plan)
# ──────────────────────────────────────────────────────────────────────

PARTITION_A_MODELS = [   # 80GB GPU
    "google/gemma-2-27b-it",
    "google/gemma-2-27b",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-Base",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.3-70B",
]

PARTITION_B_MODELS = [   # 48GB GPU
    "google/gemma-2-9b-it",
    "google/gemma-2-9b",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-Base",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
]

PARTITION_C_MAAR_SMALL = [  # 48GB GPU — small Maar models
    "google/gemma-3-1b-it",
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-3B",
]

PARTITION_C_WORKSHOP = [   # 48GB GPU — workshop 11 (code domain only)
    "gpt2",
    "gpt2-medium",
    "gpt2-xl",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "bigcode/santacoder",
    "codellama/CodeLlama-7b-Python-hf",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
]

PARTITION_D_PYTHIA = [   # 48GB GPU — training dynamics
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
]


# ──────────────────────────────────────────────────────────────────────
# Per-partition job construction
# ──────────────────────────────────────────────────────────────────────

# Maar's 4 main domains for any model in their list
MAAR_DOMAINS = ("rhyme", "qa_suggestive", "qa_neutral", "trivia")

# Workshop's code domain (and re-run for full progression)
CODE_DOMAIN = ("code",)


def jobs_partition_A(probe_types: str) -> list[Job]:
    out: list[Job] = []
    for m in PARTITION_A_MODELS:
        # 70B needs INT8 to fit a single 80GB GPU
        quant = "int8" if "70B" in m else "bf16"
        for d in MAAR_DOMAINS:
            out.append(Job(
                model=m, domain=d,
                quantization=quant,
                layer_mode="maar_range",
                probe_types=probe_types,
                notes="partition_A_large",
            ))
    return out


def jobs_partition_B(probe_types: str) -> list[Job]:
    out: list[Job] = []
    for m in PARTITION_B_MODELS:
        for d in MAAR_DOMAINS:
            out.append(Job(
                model=m, domain=d,
                quantization="bf16",
                layer_mode="maar_range",
                probe_types=probe_types,
                notes="partition_B_medium",
            ))
    return out


def jobs_partition_C(probe_types: str) -> list[Job]:
    out: list[Job] = []
    # Maar small models on all 4 Maar domains
    for m in PARTITION_C_MAAR_SMALL:
        for d in MAAR_DOMAINS:
            out.append(Job(
                model=m, domain=d,
                quantization="bf16",
                layer_mode="maar_range",
                probe_types=probe_types,
                notes="partition_C_maar_small",
            ))
    # Workshop 11 models on code (re-run with new staircase format)
    for m in PARTITION_C_WORKSHOP:
        out.append(Job(
            model=m, domain="code",
            quantization="bf16",
            layer_mode="workshop_6",
            probe_types=probe_types,
            notes="partition_C_workshop_code",
        ))
    return out


def jobs_partition_D(probe_types: str) -> list[Job]:
    """Partition D is training-dynamics work, mostly rerunning Pythia
    checkpoints with new layer-by-layer K-decay analysis. Those scripts
    are separate from the staircase pipeline. For now we just rerun the
    staircase on the final-checkpoint Pythia models for cross-partition
    consistency, and leave checkpoint-sweep work to the existing
    `run_rq4_critfixes.py` / `run_remaining_fixes.py` lineage.
    """
    out: list[Job] = []
    for m in PARTITION_D_PYTHIA:
        for d in ("code", "trivia"):  # Pythia base models on negative-control domains
            out.append(Job(
                model=m, domain=d,
                quantization="bf16",
                layer_mode="workshop_6",
                probe_types=probe_types,
                notes="partition_D_pythia",
            ))
    return out


PARTITIONS = {
    "A": jobs_partition_A,
    "B": jobs_partition_B,
    "C": jobs_partition_C,
    "D": jobs_partition_D,
}


# ──────────────────────────────────────────────────────────────────────
# Job runner
# ──────────────────────────────────────────────────────────────────────

RUNNER = "scripts/lookahead/experiments/run_staircase_v2.py"


def make_cmd(job: Job, output_dir: str, overwrite: bool, max_examples: int,
             ablation: str, n_boot: int) -> list[str]:
    cmd = [
        sys.executable, RUNNER,
        "--model", job.model,
        "--domain", job.domain,
        "--output_dir", output_dir,
        "--quantization", job.quantization,
        "--layer_mode", job.layer_mode,
        "--probe_types", job.probe_types,
        "--n_boot", str(n_boot),
    ]
    if ablation:
        cmd.extend(["--ablation", ablation])
    if overwrite:
        cmd.append("--overwrite")
    if max_examples > 0:
        cmd.extend(["--max_examples", str(max_examples)])
    return cmd


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def output_path_for(job: Job, output_dir: str) -> Path:
    return Path(output_dir) / f"{model_slug(job.model)}__{job.domain}__staircase.json"


def run_partition(jobs: list[Job], output_dir: str, overwrite: bool,
                  dry_run: bool, max_examples: int, log_dir: str,
                  ablation: str, n_boot: int):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    n_total = len(jobs)
    n_done = n_skip = n_fail = 0
    t_start = time.time()

    for i, job in enumerate(jobs, start=1):
        out_path = output_path_for(job, output_dir)
        if out_path.exists() and not overwrite:
            print(f"[{i:3d}/{n_total}] SKIP (exists): {out_path}")
            n_skip += 1
            continue

        cmd = make_cmd(job, output_dir, overwrite, max_examples, ablation, n_boot)
        print(f"[{i:3d}/{n_total}] RUN: {job.model} × {job.domain}  ({job.quantization})")
        print(f"   $ {' '.join(cmd)}")
        if dry_run:
            n_done += 1
            continue

        log_path = Path(log_dir) / f"{model_slug(job.model)}__{job.domain}.log"
        t0 = time.time()
        try:
            with open(log_path, "w") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            if proc.returncode == 0:
                print(f"   ✓ done in {elapsed:.0f}s   [log: {log_path}]")
                n_done += 1
            else:
                print(f"   ✗ FAILED rc={proc.returncode} after {elapsed:.0f}s   [log: {log_path}]")
                n_fail += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            raise

    total_elapsed = time.time() - t_start
    print()
    print(f"Summary: done={n_done} skip={n_skip} fail={n_fail}  total_wall={total_elapsed:.0f}s")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Partitioned launcher for staircase v2")
    ap.add_argument("--partition", required=True, choices=sorted(PARTITIONS.keys()),
                    help="Which GPU partition's job list to run")
    ap.add_argument("--output_dir", default="results/v2",
                    help="Where each job writes its JSON output")
    ap.add_argument("--log_dir", default="logs/v2",
                    help="Per-job stdout/stderr logs")
    ap.add_argument("--probe_types", default="linear",
                    help="Comma-separated: 'linear' (default), 'linear,mlp' for rigor pass")
    ap.add_argument("--ablation", default="zero,mean",
                    help="Comma-separated subset of {zero, mean}; empty disables ablation")
    ap.add_argument("--n_boot", type=int, default=500,
                    help="Bootstrap iterations per (probe_type, layer)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if output JSON exists")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="Cap examples (for quick smoke tests)")
    args = ap.parse_args()

    jobs = PARTITIONS[args.partition](probe_types=args.probe_types)
    print(f"Partition {args.partition}: {len(jobs)} jobs")
    print()

    run_partition(
        jobs=jobs,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_examples=args.max_examples,
        log_dir=args.log_dir,
        ablation=args.ablation,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
