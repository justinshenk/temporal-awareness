"""Render the four headline figures for the context-fatigue extended abstract.

Usage:
    uv run python scripts/context_fatigue/make_paper_figures.py
Figures are written to context_fatigue_paper/figures/ as PDFs.
"""

from pathlib import Path

from src.probes.context_fatigue.paper_figures import make_figures

OUTDIR = Path(__file__).resolve().parents[2] / "context_fatigue_paper" / "figures"


def main() -> None:
    paths = make_figures(OUTDIR)
    for name, path in paths.items():
        print(f"wrote {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
