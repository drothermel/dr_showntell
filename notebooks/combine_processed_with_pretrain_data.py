from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.datadec_utils import load_data, filter_by_model_size, filter_by_recipe, filter_by_step, filter_by_run_id
from dr_showntell.console_components import dataframe_to_fancy_tables

console = Console()

TASK_NAME_MAPPING = {
    "core_9mcqa_rc__olmes": "core9",
}

METRIC_NAME_MAPPING = {
}

EXCLUDED_METRICS = [
    "no_answer",
    "full_predicted_index_per_char_micro",
    "full_predicted_index_per_char_macro",
    "full_predicted_index_per_token_micro",
    "full_predicted_index_per_token_macro",
]


def get_pretrained_data_counts(pretrain_df: pd.DataFrame, model_size: str | None, recipe: str | None, step: int | None = None) -> int:
    if not model_size or not recipe:
        return 0

    filtered_df = filter_by_model_size(pretrain_df, model_size)
    if filtered_df.empty:
        return 0

    filtered_df = filter_by_recipe(filtered_df, recipe)
    if filtered_df.empty:
        return 0

    if step is not None:
        filtered_df = filter_by_step(filtered_df, step)

    return len(filtered_df)


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


def analyze_run_data_availability(
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

    for i, run in enumerate(finished_runs[:5]):
        console.print(f"  Processing run {i+1}/5: {run['run_id']}")

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


def clean_dataframe_for_plotting(combined_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['pattern_name', 'abs_difference_ft_tokens', '_run_type', 'run_state', 'exp_name']

    existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
    cleaned_df = combined_df.drop(columns=existing_columns_to_drop)

    console.print(f"Dropped columns: {existing_columns_to_drop}")
    console.print(f"Cleaned DataFrame shape: {cleaned_df.shape}")

    return cleaned_df


def rename_columns_for_plotting(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    column_renames = {
        'initial_checkpoint_recipe': 'ckpt_data',
        'initial_checkpoint_size': 'ckpt_params',
        'initial_checkpoint_steps': 'ckpt_steps',
        'num_finetune_tokens_per_epoch': 'ft_tok_per_epoch',
        'num_finetune_epochs': 'ft_epochs',
        'num_finetune_tokens': 'ft_tok',
        'num_finetuned_tokens_real': 'ft_tok_real',
        'steps_list': 'ft_steps_list',
        'learning_rate_list': 'ft_lrs_list',
        'total_tokens_list': 'ft_toks_list',
        "train_loss_list": "ft_loss_list",
    }

    existing_renames = {old: new for old, new in column_renames.items() if old in cleaned_df.columns}
    renamed_df = cleaned_df.rename(columns=existing_renames)

    console.print(f"Renamed columns: {existing_renames}")
    console.print(f"Final DataFrame shape: {renamed_df.shape}")

    return renamed_df


def extract_eval_metrics(runs_df: pd.DataFrame, plotting_df: pd.DataFrame) -> pd.DataFrame:
    console.print(f"Extracting evaluation metrics from runs_df...")

    all_eval_columns = {}

    for _, row in plotting_df.iterrows():
        run_id = row['run_id']

        run_row = runs_df[runs_df['run_id'] == run_id]
        if run_row.empty or pd.isna(run_row.iloc[0]['summary']):
            continue

        try:
            summary = json.loads(run_row.iloc[0]['summary'])

            for key, value in summary.items():
                if key.startswith('oe_eval_metrics/') and not key.endswith('task_config'):
                    if isinstance(value, (int, float)) and not any(excluded in key for excluded in EXCLUDED_METRICS):
                        clean_key = key.replace('oe_eval_metrics/', '').replace(':', '_').replace('/', '_')

                        for old_task, new_task in TASK_NAME_MAPPING.items():
                            if old_task in clean_key:
                                clean_key = clean_key.replace(old_task, new_task)

                        for old_metric, new_metric in METRIC_NAME_MAPPING.items():
                            if old_metric in clean_key:
                                clean_key = clean_key.replace(old_metric, new_metric)

                        all_eval_columns[clean_key] = None

        except (json.JSONDecodeError, TypeError):
            continue

    console.print(f"Found {len(all_eval_columns)} unique evaluation metrics")

    for col_name in all_eval_columns.keys():
        all_eval_columns[col_name] = []

    for _, row in plotting_df.iterrows():
        run_id = row['run_id']

        run_row = runs_df[runs_df['run_id'] == run_id]
        if run_row.empty or pd.isna(run_row.iloc[0]['summary']):
            for col_name in all_eval_columns.keys():
                all_eval_columns[col_name].append(None)
            continue

        try:
            summary = json.loads(run_row.iloc[0]['summary'])

            for col_name in all_eval_columns.keys():
                original_key = 'oe_eval_metrics/' + col_name.replace('_', '/')
                original_key = original_key.replace('_rc_', ':rc::').replace('_olmes_', '::olmes:')

                if original_key in summary and isinstance(summary[original_key], (int, float)):
                    all_eval_columns[col_name].append(summary[original_key])
                else:
                    found_value = None
                    for key, value in summary.items():
                        if key.startswith('oe_eval_metrics/') and not any(excluded in key for excluded in EXCLUDED_METRICS):
                            clean_summary_key = key.replace('oe_eval_metrics/', '').replace(':', '_').replace('/', '_')

                            for old_task, new_task in TASK_NAME_MAPPING.items():
                                if old_task in clean_summary_key:
                                    clean_summary_key = clean_summary_key.replace(old_task, new_task)

                            for old_metric, new_metric in METRIC_NAME_MAPPING.items():
                                if old_metric in clean_summary_key:
                                    clean_summary_key = clean_summary_key.replace(old_metric, new_metric)

                            if clean_summary_key == col_name and isinstance(value, (int, float)):
                                found_value = value
                                break
                    all_eval_columns[col_name].append(found_value)

        except (json.JSONDecodeError, TypeError):
            for col_name in all_eval_columns.keys():
                all_eval_columns[col_name].append(None)

    eval_columns_df = pd.DataFrame({f'ft_{col_name}': values for col_name, values in all_eval_columns.items()})
    plotting_df = pd.concat([plotting_df, eval_columns_df], axis=1)

    return plotting_df


def display_sample_results(combined_df: pd.DataFrame) -> pd.DataFrame:
    console.print(f"\n[bold blue]📊 Final Plotting DataFrame (5 runs)[/bold blue]")
    console.print(f"DataFrame with shape: {combined_df.shape}")

    tables = dataframe_to_fancy_tables(
        combined_df,
        max_cols_per_table_split=10,
        title="Final Plotting DataFrame - First 5 Finished Runs"
    )

    for table in tables:
        console.print(table)

    return combined_df


def combine_processed_with_pretrain_data() -> None:
    console.print(f"[bold blue]🔄 Combining Processed Runs with Pretraining Data[/bold blue]")

    processed_data, pretrain_df, runs_df, history_df = load_datasets()
    processed_runs = processed_data['processed_runs']

    analysis_results = analyze_run_data_availability(processed_runs, pretrain_df, runs_df, history_df)

    console.print(f"\n[bold blue]🧹 Cleaning DataFrame for plotting...[/bold blue]")
    raw_df = pd.DataFrame(analysis_results)
    cleaned_df = clean_dataframe_for_plotting(raw_df)

    console.print(f"\n[bold blue]🏷️ Renaming columns for plotting...[/bold blue]")
    plotting_df = rename_columns_for_plotting(cleaned_df)

    console.print(f"\n[bold blue]📊 Extracting evaluation metrics...[/bold blue]")
    plotting_df = extract_eval_metrics(runs_df, plotting_df)

    display_sample_results(plotting_df)

    console.print(f"\n[green]✓ Created final plotting DataFrame with {len(plotting_df)} rows and {len(plotting_df.columns)} columns[/green]")


if __name__ == "__main__":
    combine_processed_with_pretrain_data()