from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.fancy_table import FancyTable
from dr_showntell.datadec_utils import load_data
from dr_showntell.run_id_parsing import parse_and_group_run_ids, convert_groups_to_dataframes, apply_processing

console = Console()


def load_runs_and_matched_data() -> tuple[pd.DataFrame, list[dict], str]:
    pretrain_df, runs_df, history_df = load_data()

    pickle_files = sorted(Path('data').glob('*_matched_run_data.pkl'))
    assert pickle_files, "No matched run data pickle files found in data/ directory"

    latest_file = pickle_files[-1]
    console.print(f"Loading most recent pickle: [cyan]{latest_file}[/cyan]")

    with open(latest_file, 'rb') as f:
        matched_data = pickle.load(f)

    return runs_df, matched_data, str(latest_file)


def safe_truncate(value: str | None, max_len: int) -> str:
    if value is None:
        return "N/A"
    return value[:max_len] + "..." if len(value) > max_len else value


def analyze_run_types(runs_df: pd.DataFrame) -> None:
    console.print(f"\n[bold blue]Run ID Type Classification Analysis[/bold blue]")

    grouped_data = parse_and_group_run_ids(runs_df)
    type_dataframes = convert_groups_to_dataframes(grouped_data)
    processed_dataframes = apply_processing(type_dataframes, defaults={}, column_map={})

    all_run_ids = runs_df['run_id'].tolist()
    console.print(f"Total runs analyzed: [cyan]{len(all_run_ids):,}[/cyan]")
    console.print(f"Run types found: [yellow]{len(grouped_data)}[/yellow]")

    type_colors = {
        'matched': 'green',
        'simple_ft_vary_tokens': 'magenta',
        'simple_ft': 'blue',
        'dpo': 'bright_red',
        'old': 'bright_white',
        'reduce_type': 'white',
        'other': 'dim'
    }

    type_counts = {}
    for run_type, data_list in grouped_data.items():
        if run_type == "old":
            from dr_showntell.run_id_parsing import classify_run_id_type_and_extract
            old_count = sum(1 for run_id in all_run_ids
                          if classify_run_id_type_and_extract(run_id)[0] == "old")
            type_counts[run_type] = old_count
        else:
            type_counts[run_type] = len(data_list)

    table = FancyTable(
        title="Run ID Type Summary",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Run Type", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Example Run ID", style="dim")

    for run_type in sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True):
        count = type_counts[run_type]
        color = type_colors.get(run_type, 'white')

        if run_type == "old":
            from dr_showntell.run_id_parsing import classify_run_id_type_and_extract
            example_run_id = next(run_id for run_id in all_run_ids
                                if classify_run_id_type_and_extract(run_id)[0] == "old")
        elif run_type in grouped_data and grouped_data[run_type]:
            example_run_id = grouped_data[run_type][0]['run_id']
        else:
            example_run_id = "N/A"

        table.add_row(
            f"[{color}]{run_type}[/{color}]",
            f"{count:,}",
            example_run_id
        )

    console.print(table)

    if "other" in grouped_data:
        other_run_ids = [item['run_id'] for item in grouped_data["other"]]
        color = type_colors.get("other", 'dim')

        console.print(f"\n[bold {color}]OTHER Run IDs ({len(other_run_ids)} total):[/bold {color}]")

        detail_table = FancyTable(
            title=f"OTHER Run IDs",
            show_header=True,
            header_style=f"bold {color}"
        )

        detail_table.add_column("Run ID", style=color)

        for run_id in sorted(other_run_ids):
            detail_table.add_row(run_id)

        console.print(detail_table)

    for run_type in sorted(processed_dataframes.keys(), key=lambda x: len(processed_dataframes[x]), reverse=True):
        df = processed_dataframes[run_type]
        color = type_colors.get(run_type, 'white')

        console.print(f"\n[bold {color}]{run_type.upper()} Detailed Analysis ({len(df)} runs):[/bold {color}]")

        detail_table = FancyTable(
            title=f"{run_type.upper()} - Extracted Components",
            show_header=True,
            header_style=f"bold {color}"
        )

        for col in df.columns:
            detail_table.add_column(col, style="dim" if col == "run_id" else "")

        for _, row in df.iterrows():
            detail_table.add_row(*[str(row[col]) if pd.notna(row[col]) else "N/A" for col in df.columns])

def analyze_run_matching(runs_df: pd.DataFrame, matched_data: list[dict], pickle_filepath: str) -> None:
    console.print(f"\n[bold blue]Run Matching Analysis[/bold blue]")

    all_run_ids = set(runs_df['run_id'].tolist())
    matched_run_ids = {entry['run_id'] for entry in matched_data}
    unmatched_run_ids = all_run_ids - matched_run_ids

    console.print(f"Data source: [dim]{pickle_filepath}[/dim]")
    console.print(f"Total runs in dataset: [cyan]{len(all_run_ids):,}[/cyan]")
    console.print(f"Matched runs (with history): [green]{len(matched_run_ids):,}[/green]")
    console.print(f"Unmatched runs: [red]{len(unmatched_run_ids):,}[/red]")
    console.print(f"Match rate: [yellow]{len(matched_run_ids)/len(all_run_ids)*100:.1f}%[/yellow]")

    if unmatched_run_ids:
        console.print(f"\n[bold red]Unmatched Run IDs:[/bold red]")

        unmatched_list = sorted(list(unmatched_run_ids))

        table = FancyTable(
            title=f"Unmatched Run IDs ({len(unmatched_list)} total)",
            show_header=True,
            header_style="bold red"
        )

        table.add_column("Run ID", style="cyan")

        for i, run_id in enumerate(unmatched_list):
            row_style = "dim" if i % 2 == 1 else ""
            table.add_row(run_id, style=row_style)

        console.print(table)
    else:
        console.print(f"\n[green]All run IDs were successfully matched![/green]")

    if matched_run_ids:
        console.print(f"\n[bold green]Matched Run IDs:[/bold green]")

        matched_list = sorted(list(matched_run_ids))

        matched_table = FancyTable(
            title=f"Matched Run IDs ({len(matched_list)} total)",
            show_header=True,
            header_style="bold green"
        )

        matched_table.add_column("Run ID", style="green")

        for i, run_id in enumerate(matched_list):
            row_style = "dim" if i % 2 == 1 else ""
            matched_table.add_row(run_id, style=row_style)

        console.print(matched_table)


def main() -> None:
    runs_df, matched_data, pickle_filepath = load_runs_and_matched_data()
    analyze_run_types(runs_df)


if __name__ == "__main__":
    main()