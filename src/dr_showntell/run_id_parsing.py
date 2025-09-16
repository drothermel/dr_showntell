from __future__ import annotations

import re
from typing import Any

import pandas as pd

TIMESTAMP_6 = r"(?P<timestamp>\d{6}-\d{6})"
TIMESTAMP_8 = r"(?P<timestamp>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
EXP_NAME = r"(?P<exp_name>[\w_]+)"
COMPARISON_MODEL_SIZE = r"(?P<comparison_model_size>\d+[MB])"
COMPARISON_MODEL_RECIPE = r"(?P<comparison_model_recipe>[\w_]+)"
COMPARISON_MODEL_RECIPE_DASH = r"(?P<comparison_model_recipe>[\w-]+)"
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
    r"(?P<num_finetune_tokens>\d+[MB])tx(?P<num_finetune_epochs>\d+)"
)
FINETUNE_TOKENS_EPOCHS_6 = (
    r"(?P<num_finetune_tokens>\d+[MG])tx(?P<num_finetune_epochs>\d+)"
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
DD_BLOCK_FULL = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}-{INITIAL_CHECKPOINT_STEPS}-{SEED}"
DD_COMPARISON_6 = rf"DD-{COMPARISON_MODEL_RECIPE_DASH}-{COMPARISON_MODEL_SIZE}"
DD_COMPARISON_8 = rf"DD-{COMPARISON_MODEL_RECIPE}-{COMPARISON_MODEL_SIZE}"

MATCHED_PREFIX_6 = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}"
MATCHED_PREFIX_WITH_RECIPE = (
    rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}_{COMPARISON_MODEL_RECIPE}"
)

FT1_PATTERN = re.compile(
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_{FINETUNE_TOKENS_8}tx(?P<num_finetune_epochs>\d+){LEARNING_RATE_FLAG}$"
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
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_COMPARISON_8}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$"
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
    rf"^{MATCHED_PREFIX_WITH_RECIPE}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}$"
)

MATCHED2_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_WITH_RECIPE}{FINETUNE_FT}_{DD_BLOCK_FULL}$"
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


def apply_processing(
    dataframes: dict[str, pd.DataFrame],
    defaults: dict[str, Any] | None = None,
    column_map: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    if defaults is None:
        defaults = {}
    if column_map is None:
        column_map = {}

    processed = {}

    for run_type, df in dataframes.items():
        processed_df = df.copy()

        for old_col, new_col in column_map.items():
            if old_col in processed_df.columns:
                processed_df = processed_df.rename(columns={old_col: new_col})

        for col, default_val in defaults.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(default_val)

        processed[run_type] = processed_df

    return processed
