from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

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


def get_matching_pretrained_data(
    pretrain_df: pd.DataFrame,
    ckpt_params: str,
    ckpt_data: str
) -> pd.DataFrame:
    assert "params" in pretrain_df.columns, "params column not found in pretrain_df"
    assert "data" in pretrain_df.columns, "data column not found in pretrain_df"
    assert "seed" in pretrain_df.columns, "seed column not found in pretrain_df"
    assert "step" in pretrain_df.columns, "step column not found in pretrain_df"

    matching_data = pretrain_df[
        (pretrain_df["params"] == ckpt_params) &
        (pretrain_df["data"] == ckpt_data) &
        (pretrain_df["seed"] == 0)
    ]

    return matching_data


def resolve_main_checkpoint_steps(plotting_df: pd.DataFrame, pretrain_df: pd.DataFrame) -> pd.DataFrame:
    plotting_df = plotting_df.copy()

    for idx, row in plotting_df.iterrows():
        if row["ckpt_steps"] == "main":
            matching_pretrained = get_matching_pretrained_data(
                pretrain_df,
                row["ckpt_params"],
                row["ckpt_data"]
            )

            if not matching_pretrained.empty:
                max_step = matching_pretrained["step"].max()
                plotting_df.loc[idx, "ckpt_steps"] = max_step
            else:
                console.print(f"[yellow]Warning: No pretrained data found for {row['ckpt_params']} {row['ckpt_data']}[/yellow]")

    return plotting_df


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