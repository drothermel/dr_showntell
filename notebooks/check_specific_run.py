from __future__ import annotations

import json
from typing import Any

import pandas as pd
from rich.console import Console
from rich.tree import Tree

from dr_showntell.datadec_utils import load_data
from dr_showntell.fancy_table import FancyTable

console = Console()


def extract_all_fields_containing_seed(json_str: str) -> dict[str, Any]:
    try:
        data = json.loads(json_str)
        seed_fields = {}

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if "seed" in key.lower():
                        seed_fields[new_key] = value
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, new_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}[{i}]"
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, new_key)

        extract_recursive(data)
        return seed_fields
    except (json.JSONDecodeError, TypeError):
        return {}


def check_run_for_seed(target_run_id: str) -> None:
    console.print(f"[bold green]Searching for seed values in run: {target_run_id}[/bold green]")

    pretrain_df, runs_df, history_df = load_data()

    console.print(f"\n[bold blue]Checking runs_df...[/bold blue]")
    run_row = runs_df[runs_df['run_id'] == target_run_id]

    if run_row.empty:
        console.print(f"[red]Run ID not found in runs_df[/red]")
        return

    console.print(f"[green]Found run in runs_df[/green]")
    row_data = run_row.iloc[0]

    all_seed_fields = {}

    for col in runs_df.columns:
        if pd.notna(row_data[col]):
            value = row_data[col]

            if "seed" in col.lower():
                all_seed_fields[f"column.{col}"] = value

            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                seed_fields = extract_all_fields_containing_seed(value)
                for field_name, field_value in seed_fields.items():
                    all_seed_fields[f"{col}.{field_name}"] = field_value

    if all_seed_fields:
        table = FancyTable(
            title="Seed Fields Found in runs_df",
            show_header=True,
            header_style="bold green"
        )

        table.add_column("Field", style="cyan")
        table.add_column("Value", style="yellow")

        for field, value in all_seed_fields.items():
            table.add_row(field, str(value))

        console.print(table)
    else:
        console.print("[dim]No seed-related fields found in runs_df[/dim]")

    console.print(f"\n[bold blue]Checking history_df...[/bold blue]")
    history_rows = history_df[history_df['run_id'] == target_run_id]

    if history_rows.empty:
        console.print(f"[red]Run ID not found in history_df[/red]")
        return

    console.print(f"[green]Found {len(history_rows)} history records[/green]")

    history_seed_fields = {}

    for col in history_df.columns:
        if "seed" in col.lower():
            unique_values = history_rows[col].dropna().unique()
            if len(unique_values) > 0:
                history_seed_fields[f"column.{col}"] = list(unique_values)

    for _, history_row in history_rows.iterrows():
        for col in history_df.columns:
            if pd.notna(history_row[col]):
                value = history_row[col]

                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    seed_fields = extract_all_fields_containing_seed(value)
                    for field_name, field_value in seed_fields.items():
                        field_key = f"{col}.{field_name}"
                        if field_key not in history_seed_fields:
                            history_seed_fields[field_key] = []
                        if field_value not in history_seed_fields[field_key]:
                            history_seed_fields[field_key].append(field_value)

    if history_seed_fields:
        history_table = FancyTable(
            title="Seed Fields Found in history_df",
            show_header=True,
            header_style="bold blue"
        )

        history_table.add_column("Field", style="cyan")
        history_table.add_column("Unique Values", style="yellow")

        for field, values in history_seed_fields.items():
            if isinstance(values, list):
                value_str = str(values) if len(values) <= 3 else f"{values[:3]}... ({len(values)} total)"
            else:
                value_str = str(values)
            history_table.add_row(field, value_str)

        console.print(history_table)
    else:
        console.print("[dim]No seed-related fields found in history_df[/dim]")

    console.print(f"\n[bold yellow]Summary for run {target_run_id}:[/bold yellow]")
    total_seed_sources = len(all_seed_fields) + len(history_seed_fields)
    if total_seed_sources > 0:
        console.print(f"[green]Found seed information in {total_seed_sources} field(s)[/green]")
    else:
        console.print(f"[red]No seed information found in either dataframe[/red]")


if __name__ == "__main__":
    target_run_id = "250901-155734_test_finetune_DD-dclm-baseline-qc-fw-3p-150M_Ft_learning_rate=5e-05"
    check_run_for_seed(target_run_id)