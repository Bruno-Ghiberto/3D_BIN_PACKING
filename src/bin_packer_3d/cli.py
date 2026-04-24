"""Command-line interface for bin_packer_3d.

This module provides a professional CLI using Click with commands for
packing, visualization, and configuration management.

Usage:
    bin-packer pack input.csv --strategy ffd --visualize
    bin-packer info
    bin-packer init
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from bin_packer_3d import __version__
from bin_packer_3d.algorithms import ALGORITHMS
from bin_packer_3d.config import PackerConfig, Settings

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="bin-packer")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """3D Bin Packing solver with multiple algorithms and visualization.

    A professional tool for solving the 3D Bin Packing Problem (3D-BPP)
    using heuristic algorithms with interactive 3D visualization.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["settings"] = Settings(debug=debug)


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(sorted(ALGORITHMS)),
    default="ffd",
    help="Packing strategy to use (sourced from ALGORITHMS registry)",
)
@click.option(
    "--bin-length",
    "-l",
    type=float,
    default=860.0,
    help="Bin length in mm",
)
@click.option(
    "--bin-width",
    "-w",
    type=float,
    default=890.0,
    help="Bin width in mm",
)
@click.option(
    "--bin-height",
    "-h",
    type=float,
    default=1040.0,
    help="Bin height in mm",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory for results",
)
@click.option(
    "--visualize/--no-visualize",
    default=True,
    help="Generate 3D visualizations",
)
@click.pass_context
def pack(
    ctx: click.Context,
    input_file: Path,
    strategy: str,
    bin_length: float,
    bin_width: float,
    bin_height: float,
    output_dir: Path,
    visualize: bool,
) -> None:
    """Pack boxes from INPUT_FILE into bins.

    Reads box dimensions from CSV/Excel and applies the selected packing
    algorithm. Outputs placement results and optional 3D visualizations.

    Example:
        bin-packer pack boxes.csv --strategy ffd --visualize
    """
    from bin_packer_3d.data.loaders import load_boxes_from_csv, save_placements_to_csv
    from bin_packer_3d.utils.metrics import calculate_metrics
    from bin_packer_3d.visualization.plotter import Plotter3D

    settings: Settings = ctx.obj["settings"]

    console.print(f"\n[bold blue]3D Bin Packer v{__version__}[/bold blue]\n")

    # Load boxes
    console.print(f"[yellow]Loading boxes from:[/yellow] {input_file}")

    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        report = load_boxes_from_csv(input_file, config=settings.data)
    elif suffix in (".xlsx", ".xls"):
        from bin_packer_3d.data.loaders import load_boxes_from_excel

        report = load_boxes_from_excel(input_file, config=settings.data)
    else:
        raise click.ClickException(f"Unsupported file format: {suffix}")

    boxes = report.boxes
    if report.rejected_rows:
        console.print(
            f"[yellow]Warning: {len(report.rejected_rows)} row(s) rejected during load[/yellow]"
        )
    console.print(f"[green]Loaded {len(boxes)} boxes[/green]\n")

    # Configure packer
    config = PackerConfig(
        bin_length=bin_length,
        bin_width=bin_width,
        bin_height=bin_height,
        strategy=strategy,
    )

    console.print("[yellow]Bin dimensions:[/yellow]")
    console.print(f"  Length: {bin_length} mm")
    console.print(f"  Width:  {bin_width} mm")
    console.print(f"  Height: {bin_height} mm\n")

    # Select algorithm from the registry — no hardcoded branches.
    packer_cls = ALGORITHMS[strategy]
    packer = packer_cls(config)

    console.print(f"[yellow]Algorithm:[/yellow] {packer.name}\n")

    # Pack
    with console.status("[bold green]Packing boxes..."):
        result = packer.pack(boxes)

    # Calculate and display metrics
    metrics = calculate_metrics(result)

    console.print("\n[bold green]Results:[/bold green]")
    _display_metrics_table(metrics)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "placements.csv"
    save_placements_to_csv(result, csv_path)
    console.print(f"\n[green]Saved placements to:[/green] {csv_path}")

    # Generate visualizations
    if visualize and result.bins:
        console.print("\n[yellow]Generating visualizations...[/yellow]")
        plotter = Plotter3D()
        files = plotter.plot_result(result, output_dir)
        console.print(f"[green]Created {len(files)} visualization files[/green]")

    console.print("\n[bold green]Done![/bold green]\n")


def _display_metrics_table(metrics) -> None:
    """Display metrics in a formatted table."""
    table = Table(title="Packing Metrics")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Algorithm", metrics.algorithm)
    table.add_row("Boxes Placed", f"{metrics.placed_boxes}/{metrics.total_boxes}")
    table.add_row("Success Rate", f"{metrics.success_rate:.1f}%")
    table.add_row("Bins Used", str(metrics.bins_used))
    table.add_row("Overall Utilization", f"{metrics.utilization_percent:.1f}%")
    table.add_row("Avg Bin Utilization", f"{metrics.avg_bin_utilization:.1f}%")
    table.add_row("Time", f"{metrics.elapsed_time_ms:.2f}ms")

    console.print(table)


@main.command()
def info() -> None:
    """Display information about registered algorithms and default configuration.

    The algorithm listing is sourced from the ALGORITHMS registry at
    runtime; adding a new algorithm (i.e. decorating its class with
    @register) extends this output automatically with no edit to
    this command (FR-003, ADR-0001).
    """
    console.print(f"\n[bold blue]3D Bin Packer v{__version__}[/bold blue]\n")

    table = Table(title="Available Algorithms")
    table.add_column("Strategy", style="cyan")
    table.add_column("Complexity", style="yellow")
    table.add_column("Description", style="green")

    for name, cls in sorted(ALGORITHMS.items()):
        table.add_row(name, cls.complexity, cls.description)

    console.print(table)

    console.print("\n[bold]Default Configuration:[/bold]")
    settings = Settings()
    console.print(
        f"  Bin dimensions: {settings.packer.bin_length} x "
        f"{settings.packer.bin_width} x {settings.packer.bin_height} mm"
    )
    console.print(f"  Allow rotation: {settings.packer.allow_rotation}")


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path(".env"),
    help="Output path for configuration file",
)
def init(output: Path) -> None:
    """Initialize a configuration file with default settings."""
    content = """# 3D Bin Packer Configuration
# Uncomment and modify values as needed

# Bin dimensions (mm)
# BIN_PACKER_PACKER__BIN_LENGTH=860
# BIN_PACKER_PACKER__BIN_WIDTH=890
# BIN_PACKER_PACKER__BIN_HEIGHT=1040

# Packing strategy (ffd, shelf)
# BIN_PACKER_PACKER__STRATEGY=ffd

# Allow box rotation
# BIN_PACKER_PACKER__ALLOW_ROTATION=true

# Visualization settings
# BIN_PACKER_VISUALIZATION__OPACITY=0.3
# BIN_PACKER_VISUALIZATION__SHOW_WIREFRAME=true

# Data column mappings
# BIN_PACKER_DATA__ITEM_COLUMN=ITEM
# BIN_PACKER_DATA__WIDTH_COLUMN=W
# BIN_PACKER_DATA__HEIGHT_COLUMN=H
# BIN_PACKER_DATA__LENGTH_COLUMN=L

# Debug mode
# BIN_PACKER_DEBUG=false
"""

    output.write_text(content)
    console.print(f"[green]Created configuration file:[/green] {output}")
    console.print("\nEdit this file to customize settings.")


if __name__ == "__main__":
    main()
