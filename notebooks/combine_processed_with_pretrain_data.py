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
    has_ft_evaluations,
    add_plotting_helper_columns,
)
MAX_DISPLAY_ROWS = 5
MAX_TABLES = 50
TABLE_COL_SPLIT = 20

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
    processed_runs: list[dict], pretrain_df: pd.DataFrame, runs_df: pd.DataFrame, history_df: pd.DataFrame, run_type: str, require_ft_evaluations: bool = False
) -> list[dict]:
    filtered_runs = [
        run for run in processed_runs
        if run.get('_run_type') == run_type
    ]
    console.print(f"Filtered to {len(filtered_runs):,} runs of type: {run_type}")

    finished_runs = []
    for run in filtered_runs:
        run_id = run['run_id']
        runs_row = filter_by_run_id(runs_df, run_id)
        if not runs_row.empty:
            run_state = runs_row.iloc[0].get('state', 'N/A')
            if run_state == 'finished':
                if require_ft_evaluations:
                    if has_ft_evaluations(runs_df, run_id):
                        finished_runs.append(run)
                else:
                    finished_runs.append(run)

    eval_filter_msg = " with ft evaluations" if require_ft_evaluations else ""
    console.print(f"Filtered to {len(finished_runs):,} finished runs for {run_type}{eval_filter_msg}")

    analysis_results = []

    for i, run in enumerate(finished_runs):
        console.print(f"  Processing run {i+1}/{len(finished_runs)}: {run['run_id']} ({run_type})")

        run_data = dict(run)
        run_data['run_state'] = 'finished'

        history_data = filter_by_run_id(history_df, run['run_id'])
        if not history_data.empty:
            history_sorted = history_data.sort_values('step')




            # Fix the token count issue
            history_tok_list = history_sorted['total_tokens'].tolist()
            prev_curr = zip(history_tok_list[:-1], history_tok_list[1:])
            output_vals = [history_tok_list[0]]
            base_val = 0
            for prev, curr in prev_curr:
                if curr < prev:
                    base_val = prev
                output_vals.append(curr + base_val)
            if base_val > 0:
                tstrs = []
                for start, end in zip(history_tok_list, output_vals):
                    tstrs.append(f"{start/1_000_000:.1f}M = {end/1_000_000:.1f}M")
                history_tok_list = output_vals
                run_data['num_finetuned_tokens_real'] = output_vals[-1]


            run_data['steps_list'] = history_sorted['step'].tolist()
            run_data['learning_rate_list'] = history_sorted['learning_rate'].tolist()
            run_data['total_tokens_list'] = history_tok_list
            run_data['train_loss_list'] = history_sorted['train_loss'].tolist()


        else:
            run_data['steps_list'] = []
            run_data['learning_rate_list'] = []
            run_data['total_tokens_list'] = []
            run_data['train_loss_list'] = []

        analysis_results.append(run_data)

    console.print(f"[green]✓ Analysis complete for {len(analysis_results)} finished runs of type {run_type}[/green]")
    return analysis_results




def process_single_run_type(
    run_type: str, processed_runs: list[dict], pretrain_df: pd.DataFrame, runs_df: pd.DataFrame, history_df: pd.DataFrame, require_ft_evaluations: bool = False
) -> pd.DataFrame:
    console.print(f"\n[bold cyan]🔄 Processing run type: {run_type}[/bold cyan]")

    analysis_results = extract_finished_runs_with_history(processed_runs, pretrain_df, runs_df, history_df, run_type, require_ft_evaluations)

    if not analysis_results:
        console.print(f"[yellow]No finished runs found for {run_type}, skipping...[/yellow]")
        return pd.DataFrame()

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

    console.print(f"\n[bold blue]📈 Adding plotting helper columns...[/bold blue]")
    plotting_df = add_plotting_helper_columns(plotting_df)

    return plotting_df


def save_run_type_data(plotting_df: pd.DataFrame, run_type: str) -> None:
    if plotting_df.empty:
        console.print(f"[yellow]No data to save for {run_type}[/yellow]")
        return

    output_path = Path(f"data/combined_plotting_data_{run_type}.pkl")

    with open(output_path, 'wb') as f:
        pickle.dump(plotting_df, f)

    console.print(f"[green]✓ Saved {run_type} data to {output_path} ({len(plotting_df)} rows, {len(plotting_df.columns)} columns)[/green]")


def display_sample_results(combined_df: pd.DataFrame, run_type: str) -> pd.DataFrame:
    console.print(f"\n[bold blue]📊 Final Plotting DataFrame for {run_type} ({len(combined_df)} runs)[/bold blue]")
    console.print(f"DataFrame with shape: {combined_df.shape}")

    if not combined_df.empty:
        display_df = combined_df.head(MAX_DISPLAY_ROWS)
        console.print(f"[dim]Showing first {len(display_df)} of {len(combined_df)} rows[/dim]")

        tables = dataframe_to_fancy_tables(
            display_df,
            max_cols_per_table_split=TABLE_COL_SPLIT,
            title=f"Final Plotting DataFrame - {run_type.title()} Runs (Sample)"
        )

        for table in tables[:MAX_TABLES]:
            console.print(table)

    return combined_df


def combine_processed_with_pretrain_data(require_ft_evaluations: bool = True) -> None:
    console.print(f"[bold blue]🔄 Combining Processed Runs with Pretraining Data[/bold blue]")

    eval_filter_msg = " (requiring ft evaluations)" if require_ft_evaluations else " (including runs without ft evaluations)"
    console.print(f"[dim]Filter mode: {eval_filter_msg}[/dim]")

    processed_data, pretrain_df, runs_df, history_df = load_datasets()
    processed_runs = processed_data['processed_runs']

    #target_run_types = ['matched', 'simple_ft_vary_tokens', 'simple_ft']
    target_run_types = ['matched']

    for run_type in target_run_types:
        plotting_df = process_single_run_type(run_type, processed_runs, pretrain_df, runs_df, history_df, require_ft_evaluations)

        if not plotting_df.empty:
            display_sample_results(plotting_df, run_type)
            save_run_type_data(plotting_df, run_type)

        console.print(f"\n{'-' * 80}\n")

    console.print(f"[green]✓ Processing complete for all run types: {target_run_types}[/green]")


if __name__ == "__main__":
    combine_processed_with_pretrain_data()