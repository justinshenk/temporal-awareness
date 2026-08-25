"""Render the figures for the context-fatigue paper.

Usage:
    uv run python scripts/context_fatigue/make_paper_figures.py [--only main|appendix|all]
Figures are written to context_fatigue_paper/figures/ as PDFs. The main set
needs the raw OLMo run directories under results/; the appendix set renders
from documented constants and the retained Qwen artifacts.
"""

import argparse
from pathlib import Path

from src.probes.context_fatigue.paper_figures import make_appendix_figures, make_figures

OUTDIR = Path(__file__).resolve().parents[2] / "context_fatigue_paper" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["main", "appendix", "all"], default="all")
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    if args.only in ("main", "all"):
        paths.update(make_figures(OUTDIR))
    if args.only in ("appendix", "all"):
        paths.update(make_appendix_figures(OUTDIR))
    for name, path in paths.items():
        print(f"wrote {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
