"""Precompute all embeddings for the geoapp visualization.

Run this script once after generating new activation data to cache all
PCA, UMAP, and t-SNE embeddings to disk. This makes the UI load instantly
when switching between layers/positions.

Usage:
    uv run python -m src.intertemporal.geoapp.precompute_embeddings --data-dir /path/to/geo_viz_output

Options:
    --data-dir PATH    Path to geo_viz output directory
    --methods LIST     Comma-separated methods to compute (default: pca,umap,tsne)
    --layers LIST      Comma-separated layers to compute (default: all)
    --positions LIST   Comma-separated positions to compute (default: all)
    --components LIST  Comma-separated components (default: resid_pre)
"""

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute geoapp embeddings")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/geo_viz_output",
        help="Path to geo_viz output directory",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,umap,tsne",
        help="Comma-separated methods to compute",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to compute (default: all)",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated positions to compute (default: all)",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="resid_pre",
        help="Comma-separated components to compute",
    )
    args = parser.parse_args()

    console = Console()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        return 1

    # Import here to avoid slow startup
    from .data_loader import GeometryDataLoader

    console.print(f"[bold]Loading data from:[/bold] {data_dir}")
    loader = GeometryDataLoader(data_dir)

    # Parse arguments
    methods = [m.strip() for m in args.methods.split(",")]
    components = [c.strip() for c in args.components.split(",")]

    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = loader.get_layers()

    if args.positions:
        positions = [p.strip() for p in args.positions.split(",")]
    else:
        positions = loader.get_positions()

    console.print(f"[bold]Layers:[/bold] {len(layers)} ({min(layers)}-{max(layers)})")
    console.print(f"[bold]Positions:[/bold] {len(positions)}")
    console.print(f"[bold]Components:[/bold] {components}")
    console.print(f"[bold]Methods:[/bold] {methods}")

    # Calculate total work
    total_tasks = len(methods) * len(layers) * len(components) * len(positions)
    console.print(f"\n[bold]Total embeddings to compute:[/bold] {total_tasks}")

    # Check what's already cached
    console.print("\n[dim]Checking existing cache...[/dim]")
    already_cached = 0
    to_compute = []

    for method in methods:
        for layer in layers:
            for component in components:
                for position in positions:
                    cache_path = loader._get_cache_path(method, layer, component, position)
                    if cache_path.exists():
                        already_cached += 1
                    else:
                        to_compute.append((method, layer, component, position))

    console.print(f"[green]Already cached:[/green] {already_cached}")
    console.print(f"[yellow]Need to compute:[/yellow] {len(to_compute)}")

    if not to_compute:
        console.print("\n[bold green]All embeddings already cached![/bold green]")
        return 0

    # Compute missing embeddings with progress bar
    console.print()
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Computing embeddings...",
            total=len(to_compute),
        )

        computed = 0
        failed = 0

        for method, layer, component, position in to_compute:
            progress.update(
                task,
                description=f"[cyan]{method.upper()} L{layer} {component} @ {position}",
            )

            try:
                result = None
                if method == "pca":
                    result = loader.load_pca(layer, component, position)
                elif method == "umap":
                    result = loader.load_umap(layer, component, position)
                elif method == "tsne":
                    result = loader.load_tsne(layer, component, position)

                if result is not None:
                    computed += 1
                else:
                    failed += 1
            except Exception as e:
                console.print(f"[red]Error computing {method} L{layer} {position}: {e}[/red]")
                failed += 1

            progress.advance(task)

    elapsed = time.time() - start_time
    console.print(f"\n[bold]Completed in {elapsed:.1f}s[/bold]")
    console.print(f"[green]Successfully computed:[/green] {computed}")
    if failed:
        console.print(f"[yellow]Failed (no data):[/yellow] {failed}")

    # Show cache directory size
    cache_dir = data_dir / "cache"
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*.npy"))
        console.print(f"\n[dim]Cache size: {total_size / 1024 / 1024:.1f} MB[/dim]")

    return 0


if __name__ == "__main__":
    exit(main())
