from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.fancy_table import FancyTable
from dr_showntell.datadec_utils import load_data, parse_run_id_components

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


def analyze_run_matching(runs_df: pd.DataFrame, matched_data: list[dict], pickle_filepath: str) -> None:
    console.print(f"\n[bold blue]PPP Run Matching Analysis PPP[/bold blue]")

    all_run_ids = set(runs_df['run_id'].tolist())
    matched_run_ids = {entry['run_id'] for entry in matched_data}
    unmatched_run_ids = all_run_ids - matched_run_ids

    console.print(f"=Á Data source: [dim]{pickle_filepath}[/dim]")
    console.print(f"=Ê Total runs in dataset: [cyan]{len(all_run_ids):,}[/cyan]")
    console.print(f" Matched runs (with history): [green]{len(matched_run_ids):,}[/green]")
    console.print(f"L Unmatched runs: [red]{len(unmatched_run_ids):,}[/red]")
    console.print(f"=È Match rate: [yellow]{len(matched_run_ids)/len(all_run_ids)*100:.1f}%[/yellow]")

    if unmatched_run_ids:
        console.print(f"\n[bold red]Unmatched Run IDs:[/bold red]")

        unmatched_list = sorted(list(unmatched_run_ids))

        table = FancyTable(
            title=f"Unmatched Run IDs ({len(unmatched_list)} total)",
            show_header=True,
            header_style="bold red"
        )

        table.add_column("Run ID", style="cyan", width=30)
        table.add_column("DateTime", style="yellow", width=19)
        table.add_column("Experiment", style="magenta", width=18)
        table.add_column("Type", style="green", width=8)
        table.add_column("Comp Size", style="blue", width=8)
        table.add_column("Init Size", style="red", width=9)
        table.add_column("Steps", style="bright_yellow", width=7)
        table.add_column("LR", style="bright_cyan", width=10)

        for i, run_id in enumerate(unmatched_list):
            components = parse_run_id_components(run_id)
            row_style = "dim" if i % 2 == 1 else ""

            table.add_row(
                safe_truncate(components['full_id'], 25),
                safe_truncate(components['datetime'], 16),
                safe_truncate(components['exp_name'], 15),
                components['exp_type'] or "N/A",
                components['comparison_model_size'] or "N/A",
                components['initial_checkpoint_size'] or "N/A",
                components['initial_checkpoint_steps'] or "N/A",
                components['lr'] or "N/A",
                style=row_style
            )

        console.print(table)
    else:
        console.print(f"\n[green]<‰ All run IDs were successfully matched![/green]")


def main() -> None:
    runs_df, matched_data, pickle_filepath = load_runs_and_matched_data()
    analyze_run_matching(runs_df, matched_data, pickle_filepath)


if __name__ == "__main__":
    main()