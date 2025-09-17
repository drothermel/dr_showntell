from __future__ import annotations

import json
from typing import Any

import pandas as pd
from rich.console import Console
from rich.tree import Tree

from dr_showntell.datadec_utils import load_data
from dr_showntell.fancy_table import FancyTable

console = Console()


def analyze_json_column(df: pd.DataFrame, column_name: str) -> dict[str, Any]:
    json_structures = {}
    sample_count = 0
    max_samples = 5

    for idx, value in df[column_name].items():
        if pd.isna(value) or sample_count >= max_samples:
            continue

        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                for key, val in parsed.items():
                    if key not in json_structures:
                        json_structures[key] = {
                            'type': type(val).__name__,
                            'sample_value': str(val)[:100] + '...' if len(str(val)) > 100 else str(val),
                            'count': 0
                        }
                    json_structures[key]['count'] += 1
                sample_count += 1
        except (json.JSONDecodeError, TypeError):
            continue

    return json_structures


def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    console.print(f"\n[bold blue]{name} Analysis[/bold blue]")
    console.print(f"Shape: [cyan]{df.shape[0]:,} rows × {df.shape[1]} columns[/cyan]")

    table = FancyTable(
        title=f"{name} Columns",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Column", style="bold")
    table.add_column("Type", justify="center")
    table.add_column("Non-Null", justify="right", style="cyan")
    table.add_column("Sample Value", style="dim")

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()

        sample_value = "N/A"
        first_non_null = df[col].dropna().iloc[0] if non_null_count > 0 else None
        if first_non_null is not None:
            sample_str = str(first_non_null)
            sample_value = sample_str[:50] + "..." if len(sample_str) > 50 else sample_str

        table.add_row(
            col,
            dtype,
            f"{non_null_count:,}",
            sample_value
        )

    console.print(table)

    json_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            first_non_null = df[col].dropna().iloc[0] if df[col].count() > 0 else None
            if first_non_null is not None:
                try:
                    json.loads(str(first_non_null))
                    json_columns.append(col)
                except (json.JSONDecodeError, TypeError):
                    continue

    if json_columns:
        console.print(f"\n[bold yellow]JSON Columns Found in {name}:[/bold yellow]")
        for col in json_columns:
            console.print(f"\n[green]Column: {col}[/green]")
            json_structure = analyze_json_column(df, col)

            if json_structure:
                tree = Tree(f"[bold]{col}[/bold] JSON Structure")

                for key, info in json_structure.items():
                    key_node = tree.add(f"[cyan]{key}[/cyan] ({info['type']})")
                    key_node.add(f"Sample: [dim]{info['sample_value']}[/dim]")
                    key_node.add(f"Found in: [yellow]{info['count']} records[/yellow]")

                console.print(tree)
            else:
                console.print("[dim]No valid JSON structures found[/dim]")


def main() -> None:
    console.print("[bold green]Loading data...[/bold green]")
    pretrain_df, runs_df, history_df = load_data()

    inspect_dataframe(runs_df, "runs_df")
    inspect_dataframe(history_df, "history_df")
    inspect_dataframe(pretrain_df, "pretrain_df")


if __name__ == "__main__":
    main()