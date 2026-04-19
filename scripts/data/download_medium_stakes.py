#!/usr/bin/env python3
"""
Download medium-stakes datasets for temporal awareness experiments:
1. TRAM Temporal Arithmetic (from TRAM Benchmark, ACL 2024)
2. MBPP (Mostly Basic Python Problems, from Google Research)

Saves to data/filtered_temporal/ in consistent JSON format.

Usage (make sure venv is active):
    source .venv/bin/activate
    python scripts/data/download_medium_stakes.py
"""

import json
import csv
import io
import ssl
import zipfile
import urllib.request
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SSL context for macOS
ctx = ssl.create_default_context()


def download_tram_arithmetic():
    """Download and process TRAM temporal arithmetic dataset."""
    print("=" * 60)
    print("1/2  TRAM Arithmetic")
    print("=" * 60)

    url = "https://raw.githubusercontent.com/EternityYW/TRAM-Benchmark/main/datasets/arithmetic.zip"
    print("  Downloading from GitHub... ", end="", flush=True)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req, timeout=30, context=ctx)
    zip_bytes = response.read()
    print(f"{len(zip_bytes)} bytes")

    examples = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = zf.namelist()
        print(f"  Files in zip: {file_list}")

        for fname in file_list:
            if fname.endswith('.csv') and 'shots' not in fname:
                print(f"  Processing: {fname}")
                with zf.open(fname) as f:
                    content = f.read().decode('utf-8')
                    reader = csv.reader(io.StringIO(content))
                    header = next(reader)
                    print(f"  Header: {header}")

                    for i, row in enumerate(reader):
                        if len(row) < 3:
                            continue
                        example = {"original_index": i, "source_file": fname}
                        for j, col in enumerate(header):
                            if j < len(row):
                                example[col.strip()] = row[j].strip()
                        examples.append(example)

    print(f"  => {len(examples)} examples")

    output = {
        "source_dataset": "TRAM-Benchmark/arithmetic (ACL 2024)",
        "source_url": "https://github.com/EternityYW/TRAM-Benchmark",
        "description": "Temporal arithmetic reasoning - computing time differences, durations, and temporal calculations",
        "total_examples": len(examples),
        "stake_level": "medium",
        "examples": examples,
    }

    out_path = DATA_DIR / "tram_arithmetic.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


def download_mbpp():
    """Download MBPP dataset via HuggingFace datasets library."""
    print("\n" + "=" * 60)
    print("2/2  MBPP")
    print("=" * 60)

    print("  Loading from HuggingFace... ", end="", flush=True)
    ds = load_dataset("google-research-datasets/mbpp", "full")
    print("done")

    examples = []
    for split_name in ds:
        for i, item in enumerate(ds[split_name]):
            examples.append({
                "original_index": i,
                "split": split_name,
                "task_id": item.get("task_id", i),
                "text": item["text"],
                "code": item["code"],
                "test_list": item.get("test_list", []),
            })

    print(f"  => {len(examples)} examples across splits: {list(ds.keys())}")

    output = {
        "source_dataset": "google-research-datasets/mbpp",
        "source_url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
        "description": "Mostly Basic Python Problems - code generation benchmark for repeated code review tasks",
        "total_examples": len(examples),
        "stake_level": "medium",
        "examples": examples,
    }

    out_path = DATA_DIR / "mbpp_code_review.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


if __name__ == "__main__":
    print("Downloading medium-stakes datasets\n")

    n_tram = download_tram_arithmetic()
    n_mbpp = download_mbpp()

    print("\n" + "=" * 60)
    print(f"DONE — TRAM: {n_tram}, MBPP: {n_mbpp}")
    print(f"Saved to {DATA_DIR}/")
    print("=" * 60)
