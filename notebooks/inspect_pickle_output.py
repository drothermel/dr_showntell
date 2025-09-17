from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from dr_showntell.fancy_table import FancyTable

console = Console()


def inspect_latest_pickle() -> None:
    console.print(f"[bold blue]🔍 Inspecting Latest Pickle Output Format[/bold blue]")

    pickle_files = sorted(Path('data').glob('*_modernized_run_data.pkl'))
    if not pickle_files:
        console.print("[red]No modernized pickle files found in data/ directory[/red]")
        return

    latest_file = pickle_files[-1]
    console.print(f"Inspecting: [cyan]{latest_file}[/cyan]")
    console.print(f"File size: [yellow]{latest_file.stat().st_size / 1024 / 1024:.2f} MB[/yellow]")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    console.print(f"\n[bold green]📋 Top-Level Structure:[/bold green]")

    structure_table = FancyTable(
        title="Pickle File Contents",
        show_header=True,
        header_style="bold blue"
    )

    structure_table.add_column("Key", style="cyan")
    structure_table.add_column("Type", style="yellow")
    structure_table.add_column("Description", style="dim")

    for key, value in data.items():
        if isinstance(value, list):
            desc = f"{len(value)} items"
        elif isinstance(value, dict):
            desc = f"{len(value)} keys: {list(value.keys())}"
        else:
            desc = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)

        structure_table.add_row(key, type(value).__name__, desc)

    console.print(structure_table)

    if 'processed_runs' in data:
        processed_runs = data['processed_runs']
        console.print(f"\n[bold green]📊 Processed Runs Analysis ({len(processed_runs)} total):[/bold green]")

        run_types = {}
        sample_run = None

        for run in processed_runs:
            run_type = run.get('_run_type', 'unknown')
            if run_type not in run_types:
                run_types[run_type] = []
            run_types[run_type].append(run)

            if sample_run is None:
                sample_run = run

        type_table = FancyTable(
            title="Run Types in Processed Data",
            show_header=True,
            header_style="bold green"
        )

        type_table.add_column("Run Type", style="cyan")
        type_table.add_column("Count", justify="right", style="yellow")
        type_table.add_column("Sample Fields", style="dim")

        for run_type in sorted(run_types.keys(), key=lambda x: len(run_types[x]), reverse=True):
            runs = run_types[run_type]
            count = len(runs)

            sample_fields = list(runs[0].keys())[:5]
            fields_str = ", ".join(sample_fields) + f" + {len(runs[0]) - 5} more" if len(runs[0]) > 5 else ", ".join(sample_fields)

            type_table.add_row(run_type, f"{count:,}", fields_str)

        console.print(type_table)

        if sample_run:
            console.print(f"\n[bold green]🔬 Sample Run Structure:[/bold green]")
            console.print(f"Run Type: [cyan]{sample_run.get('_run_type', 'N/A')}[/cyan]")
            console.print(f"Run ID: [yellow]{sample_run.get('run_id', 'N/A')}[/yellow]")
            console.print(f"Total fields: [magenta]{len(sample_run)}[/magenta]")

            fields_table = FancyTable(
                title="Sample Run Fields",
                show_header=True,
                header_style="bold cyan"
            )

            fields_table.add_column("Field", style="cyan")
            fields_table.add_column("Value", style="yellow")
            fields_table.add_column("Type", style="dim")

            for key in sorted(sample_run.keys()):
                value = sample_run[key]
                if pd.isna(value):
                    value_str = "N/A"
                elif isinstance(value, (int, float)):
                    if key.endswith('_tokens') or key.endswith('_tokens_real'):
                        value_str = f"{value/1_000_000:.1f}M" if value > 1_000_000 else f"{value:,}"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)

                fields_table.add_row(key, value_str, type(value).__name__)

            console.print(fields_table)

    if 'processed_dataframes' in data:
        processed_dfs = data['processed_dataframes']
        console.print(f"\n[bold green]📈 Processed DataFrames Structure:[/bold green]")

        df_table = FancyTable(
            title="DataFrame Contents by Run Type",
            show_header=True,
            header_style="bold blue"
        )

        df_table.add_column("Run Type", style="cyan")
        df_table.add_column("Rows", justify="right", style="yellow")
        df_table.add_column("Columns", justify="right", style="magenta")
        df_table.add_column("Sample Columns", style="dim")

        for run_type in sorted(processed_dfs.keys()):
            df = processed_dfs[run_type]
            sample_cols = list(df.columns)[:3]
            cols_str = ", ".join(sample_cols) + f" + {len(df.columns) - 3} more" if len(df.columns) > 3 else ", ".join(sample_cols)

            df_table.add_row(
                run_type,
                f"{len(df):,}",
                f"{len(df.columns)}",
                cols_str
            )

        console.print(df_table)

    if 'metadata' in data:
        metadata = data['metadata']
        console.print(f"\n[bold green]ℹ️ Extraction Metadata:[/bold green]")

        for key, value in metadata.items():
            console.print(f"  • {key}: [yellow]{value}[/yellow]")

    console.print(f"\n[bold blue]💡 Usage Examples:[/bold blue]")
    console.print(f"```python")
    console.print(f"import pickle")
    console.print(f"with open('{latest_file}', 'rb') as f:")
    console.print(f"    data = pickle.load(f)")
    console.print(f"")
    console.print(f"# Access all processed runs")
    console.print(f"all_runs = data['processed_runs']")
    console.print(f"")
    console.print(f"# Access by run type")
    console.print(f"matched_runs = data['processed_dataframes']['matched']")
    console.print(f"ft_runs = data['processed_dataframes']['simple_ft_vary_tokens']")
    console.print(f"")
    console.print(f"# Get metadata")
    console.print(f"extraction_info = data['metadata']")
    console.print(f"```")


if __name__ == "__main__":
    inspect_latest_pickle()