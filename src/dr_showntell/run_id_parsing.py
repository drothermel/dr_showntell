from __future__ import annotations

import re
from typing import Any

import pandas as pd

DEFAULT_COMPARISON_MODEL_RECIPE = "Dolma1.7"
TIMESTAMP_6 = r"(?P<timestamp>\d{6}-\d{6})"
TIMESTAMP_8 = r"(?P<timestamp>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
EXP_NAME = r"(?P<exp_name>[\w_]+)"
COMPARISON_MODEL_SIZE = r"(?P<comparison_model_size>\d+[MB])"
# COMPARISON_MODEL_RECIPE = r"(?P<comparison_model_recipe>[\w_]+)"
COMPARISON_METRIC = r"(?P<comparison_metric>[\w_]+)"
INITIAL_CHECKPOINT_RECIPE = r"(?P<initial_checkpoint_recipe>[\w_]+)"
INITIAL_CHECKPOINT_RECIPE_DASH = r"(?P<initial_checkpoint_recipe>[\w-]+)"
INITIAL_CHECKPOINT_SIZE = r"(?P<initial_checkpoint_size>\d+[MB])"
INITIAL_CHECKPOINT_STEPS = r"(?P<initial_checkpoint_steps>\d+)"
INITIAL_CHECKPOINT_STEPS_WORD = r"(?P<initial_checkpoint_steps>\w+)"
SEED = r"(?P<seed>\d+)"
LEARNING_RATE = r"(?P<lr>[0-9.e-]+)"
LEARNING_RATE_1 = r"(?P<lr1>[0-9.e-]+)"
LEARNING_RATE_2 = r"(?P<lr2>[0-9.e-]+)"
FINETUNE_TOKENS_EPOCHS_8 = (
    r"(?P<num_finetune_tokens_per_epoch>\d+[MB])tx(?P<num_finetune_epochs>\d+)"
)
FINETUNE_TOKENS_EPOCHS_6 = (
    r"(?P<num_finetune_tokens_per_epoch>\d+[MG])tx(?P<num_finetune_epochs>\d+)"
)
FINETUNE_TOKENS_8 = r"(?P<num_finetune_tokens>\d+[MB])"
FINETUNE_TOKENS_GT = r"(?P<num_finetune_tokens>\d+[MG]t)"
FINETUNE_TOKENS_SIMPLE = r"(?P<num_finetune_tokens>\d+)"
REDUCE_LOSS = r"(?P<reduce_loss>\w+)"

TIMESTAMP_6_EXP_NAME = rf"{TIMESTAMP_6}_{EXP_NAME}"
TIMESTAMP_8_EXP_NAME = rf"{TIMESTAMP_8}_{EXP_NAME}"

LR_SUFFIX = rf"_lr={LEARNING_RATE}"
LEARNING_RATE_FLAG = rf"_--learning_rate={LEARNING_RATE}"
LEARNING_RATE_EQUAL = rf"_learning_rate={LEARNING_RATE}"

FINETUNE_FT = "_finetune_Ft"
FINETUNE_TOKENS_6 = rf"_finetune_{FINETUNE_TOKENS_EPOCHS_6}"

DD_BLOCK_STEPS_WORD = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}"
DD_BLOCK_FULL = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}-{INITIAL_CHECKPOINT_STEPS}-{SEED}"  # noqa: E501
DD_COMPARISON_6 = rf"DD-[\w-]+-{COMPARISON_MODEL_SIZE}"

MATCHED_PREFIX_6 = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}"
MATCHED_PREFIX_WITH_METRIC = (
    rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}_{COMPARISON_METRIC}"
)

FT1_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_{FINETUNE_TOKENS_EPOCHS_8}{LEARNING_RATE_FLAG}$"
)

FT3_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_{FINETUNE_TOKENS_8}_toks{LEARNING_RATE_FLAG}$"
)

FT4_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_Ft{LEARNING_RATE_FLAG}$"
)

FT5_PATTERN = re.compile(
    rf"^{TIMESTAMP_6_EXP_NAME}_DD-{INITIAL_CHECKPOINT_RECIPE_DASH}-{INITIAL_CHECKPOINT_SIZE}_Ft{LEARNING_RATE_EQUAL}$"
)

FT6_PATTERN = re.compile(
    rf"^{TIMESTAMP_6_EXP_NAME}_{FINETUNE_TOKENS_EPOCHS_8}_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

FT7_PATTERN = re.compile(rf"^{TIMESTAMP_6_EXP_NAME}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$")

MATCHED6_PATTERN = re.compile(
    rf"^{TIMESTAMP_6_EXP_NAME}_{DD_COMPARISON_6}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

MATCHED7_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_COMPARISON_6}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

REDUCE_LOSS_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_default_--max_train_samples={FINETUNE_TOKENS_SIMPLE}_--reduce_loss={REDUCE_LOSS}$"
)

DPO1_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_default$"
)

DPO2_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_dd__{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}__{INITIAL_CHECKPOINT_STEPS_WORD}__{FINETUNE_TOKENS_GT}_lr={LEARNING_RATE_1}_default_--learning_rate={LEARNING_RATE_2}$"
)

MATCHED1_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_WITH_METRIC}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}$"
)

MATCHED2_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_WITH_METRIC}{FINETUNE_FT}_{DD_BLOCK_FULL}$"
)

MATCHED3_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_6}{FINETUNE_FT}_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

MATCHED4_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_6}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}$"
)

MATCHED5_PATTERN = re.compile(rf"^{MATCHED_PREFIX_6}{FINETUNE_FT}_{DD_BLOCK_FULL}$")

MATCHED8_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_6}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

MIN_VALID_RUN_ID_SEGMENTS = 2
EXPECTED_DATE_OR_TIME_RAW_LEN = 6
EXPECTED_DATATIME_RAW_LEN = 2 * EXPECTED_DATE_OR_TIME_RAW_LEN + 1
DEFAULT_COMPARISON_MODEL_RECIPE = "Dolma1.7"
ALT_COMPARISON_MODEL_RECIPE_STR = "c4"
ALT_COMPARISON_MODEL_RECIPE = "C4"
FINETUNE_PATTERN = r"(\d+M)tx(\d+)"
DD_PATTERN = r"DD-([^-]+)-(\d+M)-(\d+)-(\d+)"

DEFAULTS = {
    "num_finetune_epochs": "1",
    "initial_checkpoint_steps": "main",
    "comparison_metric": "pile",
}

RECIPE_MAPPING = {
    "d17": "Dolma1.7",
    "d16": "Dolma1.6++",
    "c4": "C4",
    "dclm": "DCLM-Baseline",
    "dclm_qc10p": "DCLM-Baseline (QC 10%)",
    "dclm_qc20p": "DCLM-Baseline (QC 20%)",
    "dclm_qc7p_fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_qc7p_fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm_qcfw10p": "DCLM-Baseline (QC FW 10%)",
    "dclm_qcfw3p": "DCLM-Baseline (QC FW 3%)",
    "dclm25_dolma75": "DCLM-Baseline 25% / Dolma 75%",
    "dclm50_dolma50": "DCLM-Baseline 50% / Dolma 50%",
    "dclm75_dolma25": "DCLM-Baseline 75% / Dolma 25%",
    "dclm_25_d17_75": "DCLM-Baseline 25% / Dolma 75%",
    "dclm_50_d17_50": "DCLM-Baseline 50% / Dolma 50%",
    "dclm_75_d17_25": "DCLM-Baseline 75% / Dolma 25%",
    "falcon": "Falcon",
    "falcon_cc": "Falcon+CC",
    "falcon_cc_qc10p": "Falcon+CC (QC 10%)",
    "falcon_cc_qc20p": "Falcon+CC (QC 20%)",
    "falcon_cc_qcorig10p": "Falcon+CC (QC Orig 10%)",
    "falcon_cc_qctulu10p": "Falcon+CC (QC Tulu 10%)",
    "fineweb_edu": "FineWeb-Edu",
    "fineweb_pro": "FineWeb-Pro",
    "dolma1_7": "Dolma1.7",
    "dclm_qc7fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_qc7fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline": "DCLM-Baseline",
    "dclm-baseline-qc-10p": "DCLM-Baseline (QC 10%)",
    "dclm-baseline-qc-20p": "DCLM-Baseline (QC 20%)",
    "dclm-baseline-qc-7p-fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm-baseline-qc-7p-fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline-qc-fw-10p": "DCLM-Baseline (QC FW 10%)",
    "dclm-baseline-qc-fw-3p": "DCLM-Baseline (QC FW 3%)",
    "Dolma 1.7": "Dolma1.7",
    "Dolma 1.6": "Dolma1.6++",
    "C4": "C4",
    "DCLM": "DCLM-Baseline",
    "DCLM Baseline": "DCLM-Baseline",
    "Falcon": "Falcon",
    "FineWeb": "FineWeb-Edu",
}


def convert_timestamp(ts_str: str) -> pd.Timestamp | None:
    if pd.isna(ts_str):
        return None
    ts_str = str(ts_str)
    if "_" in ts_str:
        try:
            return pd.to_datetime(ts_str, format="%Y_%m_%d-%H_%M_%S")
        except (ValueError, TypeError):
            return None
    else:
        try:
            return pd.to_datetime(ts_str, format="%y%m%d-%H%M%S")
        except (ValueError, TypeError):
            return None


def convert_string_to_number(value_str: str) -> float | None:
    if pd.isna(value_str):
        return None
    value_str = str(value_str).strip().upper()
    if value_str in {"N/A", ""}:
        return None
    try:
        if value_str.endswith("M"):
            return float(value_str[:-1]) * 1_000_000
        elif value_str.endswith(("G", "B")):
            return float(value_str[:-1]) * 1_000_000_000
        elif value_str.endswith("T"):
            return float(value_str[:-2]) * 1_000_000_000_000
        else:
            return float(value_str)
    except (ValueError, TypeError):
        return None


def classify_run_id_type_and_extract(run_id: str) -> tuple[str, dict[str, str | None]]:
    patterns_to_try = [
        (DPO1_PATTERN, "dpo", "DPO1_PATTERN"),
        (DPO2_PATTERN, "dpo", "DPO2_PATTERN"),
        (MATCHED1_PATTERN, "matched", "MATCHED1_PATTERN"),
        (MATCHED2_PATTERN, "matched", "MATCHED2_PATTERN"),
        (MATCHED3_PATTERN, "matched", "MATCHED3_PATTERN"),
        (MATCHED4_PATTERN, "matched", "MATCHED4_PATTERN"),
        (MATCHED5_PATTERN, "matched", "MATCHED5_PATTERN"),
        (MATCHED6_PATTERN, "matched", "MATCHED6_PATTERN"),
        (MATCHED7_PATTERN, "matched", "MATCHED7_PATTERN"),
        (MATCHED8_PATTERN, "matched", "MATCHED8_PATTERN"),
        (REDUCE_LOSS_PATTERN, "reduce_type", "REDUCE_LOSS_PATTERN"),
        (FT1_PATTERN, "simple_ft_vary_tokens", "FT1_PATTERN"),
        (FT3_PATTERN, "simple_ft_vary_tokens", "FT3_PATTERN"),
        (FT6_PATTERN, "simple_ft_vary_tokens", "FT6_PATTERN"),
        (FT4_PATTERN, "simple_ft", "FT4_PATTERN"),
        (FT5_PATTERN, "simple_ft", "FT5_PATTERN"),
        (FT7_PATTERN, "simple_ft", "FT7_PATTERN"),
    ]

    for pattern, run_type, pattern_name in patterns_to_try:
        match = pattern.match(run_id)
        if match:
            extracted_data = match.groupdict()
            extracted_data["pattern_name"] = pattern_name
            return run_type, extracted_data

    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}

    return "other", {}


def parse_and_group_run_ids(
    df: pd.DataFrame, run_id_col: str = "run_id"
) -> dict[str, list[dict]]:
    assert run_id_col in df.columns, f"Column '{run_id_col}' not found in DataFrame"

    all_run_ids = df[run_id_col].tolist()
    type_groups = {}
    type_data = {}

    for run_id in all_run_ids:
        run_type, extracted_data = classify_run_id_type_and_extract(run_id)

        if run_type not in type_groups:
            type_groups[run_type] = []
            type_data[run_type] = []

        type_groups[run_type].append(run_id)

        if run_type != "old":
            extracted_data["run_id"] = run_id
            type_data[run_type].append(extracted_data)

    for run_type in type_data:
        type_data[run_type] = sorted(type_data[run_type], key=lambda x: x["run_id"])

    return type_data


def convert_groups_to_dataframes(
    grouped_data: dict[str, list[dict]],
) -> dict[str, pd.DataFrame]:
    dataframes = {}

    for run_type, data_list in grouped_data.items():
        if data_list:
            df = pd.DataFrame(data_list)

            if "pattern_name" in df.columns:
                df = df.sort_values("pattern_name")

            columns = ["run_id"] + [col for col in df.columns if col != "run_id"]
            df = df[columns]

            dataframes[run_type] = df

    return dataframes


def extract_config_fields(
    runs_df: pd.DataFrame, run_ids: list[str], field_mapping: dict[str, str]
) -> dict[str, Any]:
    import json

    config_data = {}

    for run_id in run_ids:
        run_row = runs_df[runs_df["run_id"] == run_id]
        if run_row.empty:
            continue

        try:
            config = json.loads(run_row.iloc[0]["config"])
            for target_field, config_field in field_mapping.items():
                if config_field in config and config[config_field] is not None:
                    if run_id not in config_data:
                        config_data[run_id] = {}
                    config_data[run_id][target_field] = config[config_field]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        if "summary" in run_row.columns and not run_row["summary"].isna().iloc[0]:
            try:
                summary = json.loads(run_row.iloc[0]["summary"])
                if "total_tokens" in summary and summary["total_tokens"] is not None:
                    if run_id not in config_data:
                        config_data[run_id] = {}
                    config_data[run_id]["num_finetuned_tokens_real"] = summary[
                        "total_tokens"
                    ]
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    return config_data


def apply_processing(
    dataframes: dict[str, pd.DataFrame],
    defaults: dict[str, Any] | None = None,
    column_map: dict[str, str] | None = None,
    runs_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if defaults is None:
        defaults = DEFAULTS
    if column_map is None:
        column_map = {}

    recipe_columns = ["comparison_model_recipe", "initial_checkpoint_recipe"]
    config_field_mapping = {
        "lr": "learning_rate",
        "seed": "seed",
        "num_finetune_epochs": "num_train_epochs",
    }
    processed = {}

    for run_type, df in dataframes.items():
        processed_df = df.copy()

        for old_col, new_col in column_map.items():
            if old_col in processed_df.columns:
                processed_df = processed_df.rename(columns={old_col: new_col})

        for col, default_val in defaults.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(default_val)

        for recipe_col in recipe_columns:
            if recipe_col in processed_df.columns:
                processed_df[recipe_col] = processed_df[recipe_col].map(
                    lambda x: RECIPE_MAPPING.get(x, x) if pd.notna(x) else x
                )

        if runs_df is not None and "run_id" in processed_df.columns:
            run_ids = processed_df["run_id"].tolist()
            config_data = extract_config_fields(runs_df, run_ids, config_field_mapping)

            for run_id, fields in config_data.items():
                run_idx = processed_df[processed_df["run_id"] == run_id].index
                if not run_idx.empty:
                    for field, value in fields.items():
                        if field == "num_finetuned_tokens_real":
                            if field not in processed_df.columns:
                                processed_df[field] = None
                            processed_df.loc[run_idx[0], field] = value
                        elif field in processed_df.columns:
                            # Only fill if current value is null
                            current_val = processed_df.loc[run_idx[0], field]
                            if pd.isna(current_val) or current_val == "N/A":
                                processed_df.loc[run_idx[0], field] = str(value)

        if "timestamp" in processed_df.columns:
            processed_df["timestamp"] = processed_df["timestamp"].apply(
                convert_timestamp
            )

        if "comparison_model_size" in processed_df.columns:
            processed_df["comparison_model_recipe"] = DEFAULT_COMPARISON_MODEL_RECIPE

        if run_type == "matched":
            if "comparison_metric" not in processed_df.columns:
                processed_df["comparison_metric"] = "pile"
            processed_df["comparison_metric"] = processed_df["comparison_metric"].fillna("pile")
            processed_df["comparison_metric"] = processed_df["comparison_metric"].map(
                lambda x: x + "_en-valppl" if x == "c4" else x + "-valppl"
            )

        if "num_finetune_tokens" in processed_df.columns:
            processed_df["num_finetune_tokens"] = processed_df[
                "num_finetune_tokens"
            ].apply(convert_string_to_number)

        if "num_finetune_tokens_per_epoch" in processed_df.columns:
            processed_df["num_finetune_tokens_per_epoch"] = processed_df[
                "num_finetune_tokens_per_epoch"
            ].apply(convert_string_to_number)

        if "num_finetuned_tokens_real" in processed_df.columns:
            processed_df["num_finetuned_tokens_real"] = processed_df[
                "num_finetuned_tokens_real"
            ].apply(convert_string_to_number)

        if (
            "num_finetune_tokens_per_epoch" in processed_df.columns
            and "num_finetune_epochs" in processed_df.columns
        ):
            if "num_finetune_tokens" not in processed_df.columns:
                processed_df["num_finetune_tokens"] = None

            processed_df["num_finetune_epochs"] = pd.to_numeric(
                processed_df["num_finetune_epochs"], errors="coerce"
            )

            mask = (
                processed_df["num_finetune_tokens_per_epoch"].notna()
                & processed_df["num_finetune_epochs"].notna()
                & processed_df["num_finetune_tokens"].isna()
            )

            processed_df.loc[mask, "num_finetune_tokens"] = (
                processed_df.loc[mask, "num_finetune_tokens_per_epoch"]
                * processed_df.loc[mask, "num_finetune_epochs"]
            )

        if (
            "num_finetune_tokens" in processed_df.columns
            and "num_finetuned_tokens_real" in processed_df.columns
        ):
            mask = (
                processed_df["num_finetune_tokens"].notna()
                & processed_df["num_finetuned_tokens_real"].notna()
                & (processed_df["num_finetune_tokens"] != 0)
            )

            processed_df["abs_difference_ft_tokens_pct"] = None
            processed_df.loc[mask, "abs_difference_ft_tokens_pct"] = (
                abs(
                    processed_df.loc[mask, "num_finetune_tokens"]
                    - processed_df.loc[mask, "num_finetuned_tokens_real"]
                )
                / processed_df.loc[mask, "num_finetune_tokens"]
                * 100
            )

        processed[run_type] = processed_df

    return processed
