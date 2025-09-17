from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

console = Console()

TASK_NAME_MAPPING = {
    "core_9mcqa_rc__olmes": "core9",
}

METRIC_NAME_MAPPING = {}

EXCLUDED_METRICS = [
    "no_answer",
    "full_predicted_index_per_char_micro",
    "full_predicted_index_per_char_macro",
    "full_predicted_index_per_token_micro",
    "full_predicted_index_per_token_macro",
]

OLMES_TASKS = [
    "mmlu_average",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
]

METRIC_NAMES = [
    "correct_choice",
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "acc_uncond",
    "no_answer",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
    "uncond_correct_prob",
    "uncond_correct_prob_per_token",
    "uncond_correct_prob_per_char",
    "uncond_total_prob",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
    "primary_metric",
]

PPL_TYPES = [
    "wikitext_103-valppl",
    "pile-valppl",
    "c4_en-valppl",
    "m2d2_s2orc-valppl",
    "ice-valppl",
    "dolma_wiki-valppl",
    "dolma_stack-valppl",
    "dolma_reddit-valppl",
    "dolma_pes2o-valppl",
    "dolma_common-crawl-valppl",
    "dolma_books-valppl",
]

CORE_METRICS = ["step", "tokens", "compute", "cumulative_lr"]


def get_matching_pretrained_data(
    pretrain_df: pd.DataFrame, ckpt_params: str, ckpt_data: str
) -> pd.DataFrame:
    assert "params" in pretrain_df.columns, "params column not found in pretrain_df"
    assert "data" in pretrain_df.columns, "data column not found in pretrain_df"
    assert "seed" in pretrain_df.columns, "seed column not found in pretrain_df"
    assert "step" in pretrain_df.columns, "step column not found in pretrain_df"

    matching_data = pretrain_df[
        (pretrain_df["params"] == ckpt_params)
        & (pretrain_df["data"] == ckpt_data)
        & (pretrain_df["seed"] == 0)
    ]

    return matching_data


def resolve_main_checkpoint_steps(
    plotting_df: pd.DataFrame, pretrain_df: pd.DataFrame
) -> pd.DataFrame:
    plotting_df = plotting_df.copy()

    for idx, row in plotting_df.iterrows():
        if row["ckpt_steps"] == "main":
            matching_pretrained = get_matching_pretrained_data(
                pretrain_df, row["ckpt_params"], row["ckpt_data"]
            )

            if not matching_pretrained.empty:
                max_step = matching_pretrained["step"].max()
                plotting_df.loc[idx, "ckpt_steps"] = max_step
            else:
                console.print(
                    f"[yellow]Warning: No pretrained data found for {row['ckpt_params']} {row['ckpt_data']}[/yellow]"
                )

    return plotting_df


def clean_dataframe_for_plotting(combined_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "pattern_name",
        "abs_difference_ft_tokens",
        "_run_type",
        "run_state",
        "exp_name",
    ]

    existing_columns_to_drop = [
        col for col in columns_to_drop if col in combined_df.columns
    ]
    cleaned_df = combined_df.drop(columns=existing_columns_to_drop)

    console.print(f"Dropped columns: {existing_columns_to_drop}")
    console.print(f"Cleaned DataFrame shape: {cleaned_df.shape}")

    return cleaned_df


def rename_columns_for_plotting(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    column_renames = {
        "initial_checkpoint_recipe": "ckpt_data",
        "initial_checkpoint_size": "ckpt_params",
        "initial_checkpoint_steps": "ckpt_steps",
        "num_finetune_tokens_per_epoch": "ft_tok_per_epoch",
        "num_finetune_epochs": "ft_epochs",
        "num_finetune_tokens": "ft_tok",
        "num_finetuned_tokens_real": "ft_tok_real",
        "steps_list": "ft_steps_list",
        "learning_rate_list": "ft_lrs_list",
        "total_tokens_list": "ft_toks_list",
        "train_loss_list": "ft_loss_list",
    }

    existing_renames = {
        old: new for old, new in column_renames.items() if old in cleaned_df.columns
    }
    renamed_df = cleaned_df.rename(columns=existing_renames)

    console.print(f"Renamed columns: {existing_renames}")
    console.print(f"Final DataFrame shape: {renamed_df.shape}")

    return renamed_df


def has_ft_evaluations(runs_df: pd.DataFrame, run_id: str) -> bool:
    run_row = runs_df[runs_df["run_id"] == run_id]
    if run_row.empty or pd.isna(run_row.iloc[0]["summary"]):
        return False

    try:
        summary = json.loads(run_row.iloc[0]["summary"])

        oe_eval_keys = [
            key for key in summary.keys()
            if key.startswith("oe_eval_metrics/") and not key.endswith("task_config")
        ]

        return len(oe_eval_keys) > 0

    except (json.JSONDecodeError, TypeError):
        return False


def extract_eval_metrics(
    runs_df: pd.DataFrame, plotting_df: pd.DataFrame
) -> pd.DataFrame:
    console.print("Extracting evaluation metrics from runs_df...")

    all_eval_columns = {}

    for _, row in plotting_df.iterrows():
        run_id = row["run_id"]

        run_row = runs_df[runs_df["run_id"] == run_id]
        if run_row.empty or pd.isna(run_row.iloc[0]["summary"]):
            continue

        try:
            summary = json.loads(run_row.iloc[0]["summary"])

            for key, value in summary.items():
                if key.startswith("oe_eval_metrics/") and not key.endswith(
                    "task_config"
                ):
                    if isinstance(value, dict):
                        # Handle nested structure: oe_eval_metrics/task -> {metric: value}
                        task_name = key.replace("oe_eval_metrics/", "")
                        for metric_name, metric_value in value.items():
                            if isinstance(metric_value, (int, float)) and metric_name not in EXCLUDED_METRICS:
                                clean_key = f"{task_name}_{metric_name}".replace(":", "_").replace("/", "_")

                                for old_task, new_task in TASK_NAME_MAPPING.items():
                                    if old_task in clean_key:
                                        clean_key = clean_key.replace(old_task, new_task)

                                for old_metric, new_metric in METRIC_NAME_MAPPING.items():
                                    if old_metric in clean_key:
                                        clean_key = clean_key.replace(old_metric, new_metric)

                                all_eval_columns[clean_key] = None
                    elif isinstance(value, (int, float)) and not any(
                        excluded in key for excluded in EXCLUDED_METRICS
                    ):
                        # Handle flat structure: oe_eval_metrics/task/metric
                        clean_key = (
                            key.replace("oe_eval_metrics/", "")
                            .replace(":", "_")
                            .replace("/", "_")
                        )

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

    for col_name in all_eval_columns:
        all_eval_columns[col_name] = []

    for _, row in plotting_df.iterrows():
        run_id = row["run_id"]

        run_row = runs_df[runs_df["run_id"] == run_id]
        if run_row.empty or pd.isna(run_row.iloc[0]["summary"]):
            for col_name in all_eval_columns:
                all_eval_columns[col_name].append(None)
            continue

        try:
            summary = json.loads(run_row.iloc[0]["summary"])

            for col_name in all_eval_columns:
                found_value = None

                # Search through all oe_eval_metrics keys
                for key, value in summary.items():
                    if key.startswith("oe_eval_metrics/") and not key.endswith("task_config"):
                        if isinstance(value, dict):
                            # Handle nested structure: oe_eval_metrics/task -> {metric: value}
                            task_name = key.replace("oe_eval_metrics/", "")
                            for metric_name, metric_value in value.items():
                                if isinstance(metric_value, (int, float)) and metric_name not in EXCLUDED_METRICS:
                                    clean_key = f"{task_name}_{metric_name}".replace(":", "_").replace("/", "_")

                                    for old_task, new_task in TASK_NAME_MAPPING.items():
                                        if old_task in clean_key:
                                            clean_key = clean_key.replace(old_task, new_task)

                                    for old_metric, new_metric in METRIC_NAME_MAPPING.items():
                                        if old_metric in clean_key:
                                            clean_key = clean_key.replace(old_metric, new_metric)

                                    if clean_key == col_name:
                                        found_value = metric_value
                                        break
                        elif isinstance(value, (int, float)) and not any(
                            excluded in key for excluded in EXCLUDED_METRICS
                        ):
                            # Handle flat structure: oe_eval_metrics/task/metric
                            clean_summary_key = (
                                key.replace("oe_eval_metrics/", "")
                                .replace(":", "_")
                                .replace("/", "_")
                            )

                            for old_task, new_task in TASK_NAME_MAPPING.items():
                                if old_task in clean_summary_key:
                                    clean_summary_key = clean_summary_key.replace(
                                        old_task, new_task
                                    )

                            for old_metric, new_metric in METRIC_NAME_MAPPING.items():
                                if old_metric in clean_summary_key:
                                    clean_summary_key = clean_summary_key.replace(
                                        old_metric, new_metric
                                    )

                            if clean_summary_key == col_name:
                                found_value = value
                                break

                    if found_value is not None:
                        break

                all_eval_columns[col_name].append(found_value)

        except (json.JSONDecodeError, TypeError):
            for col_name in all_eval_columns:
                all_eval_columns[col_name].append(None)

    eval_columns_df = pd.DataFrame(
        {f"ft_{col_name}": values for col_name, values in all_eval_columns.items()}
    )
    plotting_df = pd.concat([plotting_df, eval_columns_df], axis=1)

    return plotting_df


def get_available_task_metric_columns(pretrain_df: pd.DataFrame) -> list[str]:
    all_tasks = OLMES_TASKS
    task_metric_columns = []

    for task in all_tasks:
        for metric in METRIC_NAMES:
            if metric in EXCLUDED_METRICS:
                continue
            col_name = f"{task}_{metric}"
            if col_name in pretrain_df.columns:
                task_metric_columns.append(col_name)

    return task_metric_columns


def extract_pretraining_lists(
    pretrain_df: pd.DataFrame, ckpt_params: str, ckpt_data: str
) -> dict[str, list]:
    matching_data = get_matching_pretrained_data(pretrain_df, ckpt_params, ckpt_data)

    if matching_data.empty:
        console.print(
            f"[yellow]No matching pretraining data for {ckpt_params} / {ckpt_data}[/yellow]"
        )
        return {}

    sorted_data = matching_data.sort_values("step")
    console.print(
        f"Found {len(sorted_data)} pretraining steps for {ckpt_params} / {ckpt_data}"
    )

    result_lists = {}

    task_metric_columns = get_available_task_metric_columns(pretrain_df)
    for col in task_metric_columns:
        if col in sorted_data.columns:
            values = sorted_data[col].dropna().tolist()
            if len(values) == 0:
                console.print(
                    f"[red]Empty list for {col} in {ckpt_params}/{ckpt_data}[/red]"
                )
            result_lists[f"pt_{col}_list"] = values

    for ppl_type in PPL_TYPES:
        if ppl_type in sorted_data.columns:
            values = sorted_data[ppl_type].dropna().tolist()
            if len(values) == 0:
                console.print(
                    f"[red]Empty list for {ppl_type} in {ckpt_params}/{ckpt_data}[/red]"
                )
            result_lists[f"pt_{ppl_type}_list"] = values

    for core_metric in CORE_METRICS:
        if core_metric in sorted_data.columns:
            values = sorted_data[core_metric].dropna().tolist()
            if len(values) == 0:
                console.print(
                    f"[red]Empty list for {core_metric} in {ckpt_params}/{ckpt_data}[/red]"
                )
            result_lists[f"pt_{core_metric}_list"] = values

    return result_lists


def extract_pretraining_checkpoint_values(
    pretrain_df: pd.DataFrame, ckpt_params: str, ckpt_data: str, ckpt_steps: int
) -> dict[str, float | None]:
    matching_data = get_matching_pretrained_data(pretrain_df, ckpt_params, ckpt_data)

    if matching_data.empty:
        return {}

    checkpoint_data = matching_data[matching_data["step"] == ckpt_steps]

    if checkpoint_data.empty:
        return {}

    checkpoint_row = checkpoint_data.iloc[0]
    result_values = {}

    task_metric_columns = get_available_task_metric_columns(pretrain_df)
    for col in task_metric_columns:
        if col in checkpoint_row.index and pd.notna(checkpoint_row[col]):
            result_values[f"pt_{col}"] = checkpoint_row[col]

    for ppl_type in PPL_TYPES:
        if ppl_type in checkpoint_row.index and pd.notna(checkpoint_row[ppl_type]):
            result_values[f"pt_{ppl_type}"] = checkpoint_row[ppl_type]

    for core_metric in CORE_METRICS:
        if core_metric in checkpoint_row.index and pd.notna(
            checkpoint_row[core_metric]
        ):
            result_values[f"pt_{core_metric}"] = checkpoint_row[core_metric]

    return result_values


def add_pretraining_data(
    plotting_df: pd.DataFrame, pretrain_df: pd.DataFrame
) -> pd.DataFrame:
    plotting_df = plotting_df.copy()

    console.print("Adding pretraining data lists and checkpoint values...")

    task_metric_columns = get_available_task_metric_columns(pretrain_df)

    all_list_columns = {}
    all_checkpoint_columns = {}

    for col in task_metric_columns:
        all_list_columns[f"pt_{col}_list"] = []
        all_checkpoint_columns[f"pt_{col}"] = []

    for ppl_type in PPL_TYPES:
        all_list_columns[f"pt_{ppl_type}_list"] = []
        all_checkpoint_columns[f"pt_{ppl_type}"] = []

    for core_metric in CORE_METRICS:
        all_list_columns[f"pt_{core_metric}_list"] = []
        all_checkpoint_columns[f"pt_{core_metric}"] = []

    console.print(
        f"Pre-initialized {len(all_list_columns)} list columns and {len(all_checkpoint_columns)} checkpoint columns for {len(plotting_df)} rows"
    )

    for col_name in all_list_columns:
        all_list_columns[col_name] = [[] for _ in range(len(plotting_df))]

    for col_name in all_checkpoint_columns:
        all_checkpoint_columns[col_name] = [None for _ in range(len(plotting_df))]

    for idx, row in plotting_df.iterrows():
        ckpt_params = row["ckpt_params"]
        ckpt_data = row["ckpt_data"]
        ckpt_steps = row["ckpt_steps"]

        if pd.isna(ckpt_steps) or ckpt_steps == "main":
            continue

        try:
            ckpt_steps_int = int(ckpt_steps)
        except (ValueError, TypeError):
            continue

        list_data = extract_pretraining_lists(pretrain_df, ckpt_params, ckpt_data)
        checkpoint_data = extract_pretraining_checkpoint_values(
            pretrain_df, ckpt_params, ckpt_data, ckpt_steps_int
        )

        for col_name, values in list_data.items():
            if col_name in all_list_columns:
                all_list_columns[col_name][idx] = values

        for col_name, value in checkpoint_data.items():
            if col_name in all_checkpoint_columns:
                all_checkpoint_columns[col_name][idx] = value

    new_columns_data = {**all_list_columns, **all_checkpoint_columns}

    if new_columns_data:
        new_columns_df = pd.DataFrame(new_columns_data)
        plotting_df = pd.concat([plotting_df, new_columns_df], axis=1)

    total_new_cols = len(all_list_columns) + len(all_checkpoint_columns)
    console.print(
        f"Added {total_new_cols} pretraining columns ({len(all_list_columns)} lists, {len(all_checkpoint_columns)} checkpoint values)"
    )

    return plotting_df


def add_plotting_helper_columns(plotting_df: pd.DataFrame) -> pd.DataFrame:
    plotting_df = plotting_df.copy()

    console.print("Adding plotting helper columns...")

    # 1. Delta columns for acc_uncond (ft - pt)
    delta_acc_uncond_cols = []
    for task in OLMES_TASKS:
        ft_col = f"ft_{task}_acc_uncond"
        pt_col = f"pt_{task}_acc_uncond"
        delta_col = f"delta_{task}_acc_uncond"

        if ft_col in plotting_df.columns and pt_col in plotting_df.columns:
            plotting_df[delta_col] = plotting_df[ft_col] - plotting_df[pt_col]
            delta_acc_uncond_cols.append(delta_col)

    # 2. Delta columns for correct_prob (ft - pt)
    delta_correct_prob_cols = []
    for task in OLMES_TASKS:
        ft_col = f"ft_{task}_correct_prob"
        pt_col = f"pt_{task}_correct_prob"
        delta_col = f"delta_{task}_correct_prob"

        if ft_col in plotting_df.columns and pt_col in plotting_df.columns:
            plotting_df[delta_col] = plotting_df[ft_col] - plotting_df[pt_col]
            delta_correct_prob_cols.append(delta_col)

    # 3. Total tokens (pt + ft)
    if "pt_tokens" in plotting_df.columns and "ft_tok" in plotting_df.columns:
        plotting_df["pt_ft_toks"] = plotting_df["pt_tokens"] + plotting_df["ft_tok"]
        console.print("Added pt_ft_toks column")

    # 4. Matched group columns (only for matched runs)
    if ("comparison_metric" in plotting_df.columns and
        "comparison_model_size" in plotting_df.columns):

        # Matched runs: group by comparison model (size + metric + fixed Dolma1.7)
        plotting_df["matched_group_name"] = (
            plotting_df["comparison_model_size"] + "_" +
            plotting_df["comparison_metric"] + "_" +
            "Dolma1.7"
        )

        # Determine sort metric based on comparison_metric
        def get_sort_metric_for_matched_group(group_rows: pd.DataFrame) -> str:
            comparison_metric = group_rows["comparison_metric"].iloc[0]
            if pd.notna(comparison_metric):
                if "c4" in str(comparison_metric).lower():
                    return "pt_c4_en-valppl"
                elif "pile" in str(comparison_metric).lower():
                    return "pt_pile-valppl"
            return "pt_pile-valppl"

        # Create group data for each matched group
        group_to_data = {}
        group_counter = 1

        for group_name in plotting_df["matched_group_name"].unique():
            group_rows = plotting_df[plotting_df["matched_group_name"] == group_name]
            sort_metric = get_sort_metric_for_matched_group(group_rows)

            if sort_metric in plotting_df.columns:
                group_median = group_rows[sort_metric].median()
                group_mean = group_rows[sort_metric].mean()
                group_to_data[group_name] = (group_median, group_mean, sort_metric, group_counter)
                group_counter += 1
            else:
                console.print(f"[yellow]Warning: {sort_metric} not found for {group_name}[/yellow]")
                group_to_data[group_name] = (float('inf'), float('inf'), sort_metric, group_counter)
                group_counter += 1

        # Sort groups by their median metric values and assign final group IDs
        sorted_groups = sorted(group_to_data.items(), key=lambda x: x[1][0])
        final_group_mapping = {}

        for idx, (group_name, (median_val, mean_val, metric_name, _)) in enumerate(sorted_groups):
            group_id = idx + 1
            final_group_mapping[group_name] = {
                'id': group_id,
                'median': median_val,
                'mean': mean_val,
                'metric': metric_name
            }

        # Add all the matched group columns
        plotting_df["matched_group_id"] = plotting_df["matched_group_name"].map(
            lambda x: final_group_mapping[x]['id']
        )
        plotting_df["matched_group_med_metric"] = plotting_df["matched_group_name"].map(
            lambda x: final_group_mapping[x]['median']
        )
        plotting_df["matched_group_avg_metric"] = plotting_df["matched_group_name"].map(
            lambda x: final_group_mapping[x]['mean']
        )

        # Report which metrics were used
        metrics_used = set(data['metric'] for data in final_group_mapping.values())
        console.print(f"Created matched_group_id with {len(final_group_mapping)} groups, sorted by: {sorted(metrics_used)}")

        # 5. Add group statistics delta columns for matched runs
        all_delta_cols = delta_acc_uncond_cols + delta_correct_prob_cols
        group_stats_cols = []

        for delta_col in all_delta_cols:
            if delta_col not in plotting_df.columns:
                continue

            # Group averages
            group_avg_col = f"group_avg_{delta_col}"
            group_averages = plotting_df.groupby("matched_group_name")[delta_col].apply(
                lambda x: x.mean() if len(x.dropna()) > 0 else None
            )
            plotting_df[group_avg_col] = plotting_df["matched_group_name"].map(group_averages)
            group_stats_cols.append(group_avg_col)

            # Group medians
            group_med_col = f"group_med_{delta_col}"
            group_medians = plotting_df.groupby("matched_group_name")[delta_col].apply(
                lambda x: x.median() if len(x.dropna()) > 0 else None
            )
            plotting_df[group_med_col] = plotting_df["matched_group_name"].map(group_medians)
            group_stats_cols.append(group_med_col)

            # Group standard deviations
            group_std_col = f"group_std_{delta_col}"
            group_stds = plotting_df.groupby("matched_group_name")[delta_col].apply(
                lambda x: x.std() if len(x.dropna()) > 1 else None
            )
            plotting_df[group_std_col] = plotting_df["matched_group_name"].map(group_stds)
            group_stats_cols.append(group_std_col)

            # Group median absolute deviations
            group_mad_col = f"group_mad_{delta_col}"
            def calculate_mad_safe(group_data: pd.Series) -> float | None:
                clean_data = group_data.dropna()
                if len(clean_data) == 0:
                    return None
                median = clean_data.median()
                return (clean_data - median).abs().median()

            group_mads = plotting_df.groupby("matched_group_name")[delta_col].apply(calculate_mad_safe)
            plotting_df[group_mad_col] = plotting_df["matched_group_name"].map(group_mads)
            group_stats_cols.append(group_mad_col)

        console.print(f"Added {len(group_stats_cols)} group statistics delta columns (avg, med, std, mad)")

    console.print(f"Added {len(delta_acc_uncond_cols)} delta_acc_uncond columns")
    console.print(f"Added {len(delta_correct_prob_cols)} delta_correct_prob columns")

    return plotting_df
