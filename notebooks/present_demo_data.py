from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.fancy_table import FancyTable
from dr_showntell.datadec_utils import load_data, parse_run_id_components, classify_run_id_type_and_extract

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

    all_run_ids = runs_df['run_id'].tolist()

    type_groups = {}
    type_data = {}

    for run_id in all_run_ids:
        run_type, extracted_data = classify_run_id_type_and_extract(run_id)

        if run_type not in type_groups:
            type_groups[run_type] = []
            type_data[run_type] = []

        type_groups[run_type].append(run_id)

        if run_type != "old":
            extracted_data['run_id'] = run_id
            type_data[run_type].append(extracted_data)

    console.print(f"Total runs analyzed: [cyan]{len(all_run_ids):,}[/cyan]")
    console.print(f"Run types found: [yellow]{len(type_groups)}[/yellow]")

    type_colors = {
        'matched': 'green',
        'simple_ft_vary_tokens': 'magenta',
        'simple_ft': 'blue',
        'dpo': 'bright_red',
        'old': 'bright_white',
        'reduce_type': 'white',
        'other': 'dim'
    }

    table = FancyTable(
        title="Run ID Type Summary",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Run Type", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Example Run ID", style="dim")

    for run_type in sorted(type_groups.keys(), key=lambda x: len(type_groups[x]), reverse=True):
        run_ids = sorted(type_groups[run_type])
        color = type_colors.get(run_type, 'white')
        example_run_id = run_ids[0]

        table.add_row(
            f"[{color}]{run_type}[/{color}]",
            f"{len(run_ids):,}",
            example_run_id
        )

    console.print(table)

    if "other" in type_groups:
        run_ids = sorted(type_groups["other"])
        color = type_colors.get("other", 'dim')

        console.print(f"\n[bold {color}]OTHER Run IDs ({len(run_ids)} total):[/bold {color}]")

        detail_table = FancyTable(
            title=f"OTHER Run IDs",
            show_header=True,
            header_style=f"bold {color}"
        )

        detail_table.add_column("Run ID", style=color)

        for run_id in run_ids:
            detail_table.add_row(run_id)

        console.print(detail_table)

    for run_type in sorted(type_data.keys(), key=lambda x: len(type_data[x]), reverse=True):
        if run_type != "old" and type_data[run_type]:
            df = pd.DataFrame(type_data[run_type])

            if 'pattern_name' in df.columns:
                df = df.sort_values('pattern_name')

            columns = ['run_id'] + [col for col in df.columns if col != 'run_id']
            df = df[columns]

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

            #console.print(detail_table)

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