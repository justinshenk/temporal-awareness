"""Download the LLM-Adapters commonsense data used by the LoReFT reproduction.

Fetches the commonsense-170k training file and the eval test splits into ``data/commonsense/``
(train file keeps its name; eval splits land as ``{dataset}_test.json``). Idempotent — existing
files are skipped.

    uv run python -m scripts.attribution.download_commonsense_data
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

LLM_ADAPTERS_RAW = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main"
TRAIN_FILE = "commonsense_170k.json"
EVAL_DATASETS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
                 "ARC-Easy", "ARC-Challenge", "openbookqa"]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists, skipping: {dest}", flush=True)
        return
    print(f"  {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/commonsense")
    ap.add_argument("--datasets", default="boolq,piqa,ARC-Challenge",
                    help=f"comma list from {EVAL_DATASETS}, or 'all'")
    args = ap.parse_args()

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = EVAL_DATASETS if args.datasets == "all" else args.datasets.split(",")

    fetch(f"{LLM_ADAPTERS_RAW}/ft-training_set/{TRAIN_FILE}", out / TRAIN_FILE)
    for name in datasets:
        fetch(f"{LLM_ADAPTERS_RAW}/dataset/{name}/test.json", out / f"{name}_test.json")
    print("done", flush=True)


if __name__ == "__main__":
    main()
