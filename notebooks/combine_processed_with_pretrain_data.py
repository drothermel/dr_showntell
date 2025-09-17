from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.datadec_utils import load_data, filter_by_run_id
from dr_showntell.console_components import dataframe_to_fancy_tables
from dr_showntell.combine_pt_ft_utils import (
    resolve_main_checkpoint_steps,
    clean_dataframe_for_plotting,
    rename_columns_for_plotting,
    extract_eval_metrics,
    add_pretraining_data,
)
MAX_RUNS = 5
MAX_TABLES = 40
TABLE_COL_SPLIT = 15

console = Console()


def load_datasets() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pickle_files = sorted(Path('data').glob('*_modernized_run_data.pkl'))
    assert pickle_files, "No modernized pickle files found in data/ directory"

    latest_file = pickle_files[-1]
    console.print(f"Loading processed data from: [cyan]{latest_file}[/cyan]")

    with open(latest_file, 'rb') as f:
        processed_data = pickle.load(f)

    pretrain_df, runs_df, history_df = load_data()
    console.print(f"Loaded raw data: pretrain({len(pretrain_df):,}), runs({len(runs_df):,}), history({len(history_df):,})")

    return processed_data, pretrain_df, runs_df, history_df


def extract_finished_runs_with_history(
    processed_runs: list[dict], pretrain_df: pd.DataFrame, runs_df: pd.DataFrame, history_df: pd.DataFrame
) -> list[dict]:
    target_run_types = {'matched', 'simple_ft_vary_tokens', 'simple_ft'}

    filtered_runs = [
        run for run in processed_runs
        if run.get('_run_type') in target_run_types
    ]
    console.print(f"Filtered to {len(filtered_runs):,} runs of target types: {target_run_types}")

    finished_runs = []
    for run in filtered_runs:
        run_id = run['run_id']
        runs_row = filter_by_run_id(runs_df, run_id)
        if not runs_row.empty:
            run_state = runs_row.iloc[0].get('state', 'N/A')
            if run_state == 'finished':
                finished_runs.append(run)

    console.print(f"Filtered to {len(finished_runs):,} finished runs")

    analysis_results = []

    for i, run in enumerate(finished_runs[:MAX_RUNS]):
        console.print(f"  Processing run {i+1}/{MAX_RUNS}: {run['run_id']}")

        run_data = dict(run)
        run_data['run_state'] = 'finished'

        history_data = filter_by_run_id(history_df, run['run_id'])

        if not history_data.empty:
            history_sorted = history_data.sort_values('step')

            run_data['steps_list'] = history_sorted['step'].tolist()
            run_data['learning_rate_list'] = history_sorted['learning_rate'].tolist()
            run_data['total_tokens_list'] = history_sorted['total_tokens'].dropna().tolist()
            run_data['train_loss_list'] = history_sorted['train_loss'].tolist()
        else:
            run_data['steps_list'] = []
            run_data['learning_rate_list'] = []
            run_data['total_tokens_list'] = []
            run_data['train_loss_list'] = []

        analysis_results.append(run_data)

    console.print(f"[green]✓ Analysis complete for {len(analysis_results)} finished runs[/green]")
    return analysis_results




def display_sample_results(combined_df: pd.DataFrame) -> pd.DataFrame:
    console.print(f"\n[bold blue]📊 Final Plotting DataFrame (5 runs)[/bold blue]")
    console.print(f"DataFrame with shape: {combined_df.shape}")

    tables = dataframe_to_fancy_tables(
        combined_df,
        max_cols_per_table_split=TABLE_COL_SPLIT,
        title="Final Plotting DataFrame - First 5 Finished Runs"
    )

    for table in tables[:MAX_TABLES]:
        console.print(table)

    return combined_df


def combine_processed_with_pretrain_data() -> None:
    console.print(f"[bold blue]🔄 Combining Processed Runs with Pretraining Data[/bold blue]")

    processed_data, pretrain_df, runs_df, history_df = load_datasets()
    processed_runs = processed_data['processed_runs']

    analysis_results = extract_finished_runs_with_history(processed_runs, pretrain_df, runs_df, history_df)

    console.print(f"\n[bold blue]🧹 Cleaning DataFrame for plotting...[/bold blue]")
    raw_df = pd.DataFrame(analysis_results)
    cleaned_df = clean_dataframe_for_plotting(raw_df)

    console.print(f"\n[bold blue]🏷️ Renaming columns for plotting...[/bold blue]")
    plotting_df = rename_columns_for_plotting(cleaned_df)

    console.print(f"\n[bold blue]🔧 Resolving 'main' checkpoint steps...[/bold blue]")
    plotting_df = resolve_main_checkpoint_steps(plotting_df, pretrain_df)

    console.print(f"\n[bold blue]📊 Extracting evaluation metrics...[/bold blue]")
    plotting_df = extract_eval_metrics(runs_df, plotting_df)

    console.print(f"\n[bold blue]🔗 Adding pretraining data...[/bold blue]")
    plotting_df = add_pretraining_data(plotting_df, pretrain_df)

    display_sample_results(plotting_df)

    console.print(f"\n[green]✓ Created final plotting DataFrame with {len(plotting_df)} rows and {len(plotting_df.columns)} columns[/green]")


if __name__ == "__main__":
    combine_processed_with_pretrain_data()