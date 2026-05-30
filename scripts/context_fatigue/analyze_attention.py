"""Re-run the context-rot attention analysis on a saved attention_stats.csv
(produced incrementally by run_olmo_attention.py) without reloading the model."""

import argparse
from pathlib import Path

import pandas as pd

from run_olmo_attention import analyze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--model", default=None)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    layers = [int(x) for x in args.layers.split(",") if int(x) in set(df["layer"].unique())]
    n_cases = df.groupby(["session", "case"]).ngroups if "session" in df.columns else df["case"].nunique()
    print(f"{len(df)} records, {n_cases} pooled cases, layers {sorted(df['layer'].unique())}")
    analyze(df, layers, args.model or Path(args.csv).parent.name, Path(args.csv).parent)


if __name__ == "__main__":
    main()
