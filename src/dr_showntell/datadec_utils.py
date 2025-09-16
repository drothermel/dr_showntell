from __future__ import annotations

import re

import pandas as pd

# Reusable regex components
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
FINETUNE_TOKENS_EPOCHS_8 = r"(?P<num_finetune_tokens>\d+[MB])tx(?P<num_finetune_epochs>\d+)"
FINETUNE_TOKENS_EPOCHS_6 = r"(?P<num_finetune_tokens>\d+[MG])tx(?P<num_finetune_epochs>\d+)"
FINETUNE_TOKENS_8 = r"(?P<num_finetune_tokens>\d+[MB])"
FINETUNE_TOKENS_GT = r"(?P<num_finetune_tokens>\d+[MG]t)"
FINETUNE_TOKENS_SIMPLE = r"(?P<num_finetune_tokens>\d+)"
REDUCE_LOSS = r"(?P<reduce_loss>\w+)"

# Timestamp + Exp Name combinations
TIMESTAMP_6_EXP_NAME = rf"{TIMESTAMP_6}_{EXP_NAME}"
TIMESTAMP_8_EXP_NAME = rf"{TIMESTAMP_8}_{EXP_NAME}"

# Learning rate variations
LR_SUFFIX = rf"_lr={LEARNING_RATE}"
LEARNING_RATE_FLAG = rf"_--learning_rate={LEARNING_RATE}"
LEARNING_RATE_EQUAL = rf"_learning_rate={LEARNING_RATE}"

# Finetune variations
FINETUNE_FT = "_finetune_Ft"
FINETUNE_TOKENS_6 = rf"_finetune_{FINETUNE_TOKENS_EPOCHS_6}"

# Block components
DD_BLOCK_STEPS_WORD = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}"
DD_BLOCK_FULL = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}-{INITIAL_CHECKPOINT_STEPS}-{SEED}"
DD_COMPARISON_6 = rf"DD-{COMPARISON_MODEL_RECIPE_DASH}-{COMPARISON_MODEL_SIZE}"
DD_COMPARISON_8 = rf"DD-{COMPARISON_MODEL_RECIPE}-{COMPARISON_MODEL_SIZE}"

# Common prefixes for matched patterns
MATCHED_PREFIX_6 = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}"
MATCHED_PREFIX_WITH_RECIPE = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}_{COMPARISON_MODEL_RECIPE}"

# Composed patterns
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

FT7_PATTERN = re.compile(
    rf"^{TIMESTAMP_6_EXP_NAME}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

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

MATCHED5_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_6}{FINETUNE_FT}_{DD_BLOCK_FULL}$"
)

MATCHED8_PATTERN = re.compile(
    rf"^{MATCHED_PREFIX_6}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}{LR_SUFFIX}$"
)

PRETRAINED_DF_PATH = "/Users/daniellerothermel/drotherm/repos/datadec/data/datadecide/mean_eval_melted.parquet"  # noqa: E501
RUNS_DF_PATH = (
    "/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_metadata.parquet"
)
HISTORY_DF_PATH = (
    "/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_history.parquet"
)
MIN_VALID_RUN_ID_SEGMENTS = 2
EXPECTED_DATE_OR_TIME_RAW_LEN = 6
EXPECTED_DATATIME_RAW_LEN = 2 * EXPECTED_DATE_OR_TIME_RAW_LEN + 1
DEFAULT_COMPARISON_MODEL_RECIPE = "Dolma1.7"
ALT_COMPARISON_MODEL_RECIPE_STR = "c4"
ALT_COMPARISON_MODEL_RECIPE = "C4"
FINETUNE_PATTERN = r"(\d+M)tx(\d+)"
DD_PATTERN = r"DD-([^-]+)-(\d+M)-(\d+)-(\d+)"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pretrain_df = pd.read_parquet(PRETRAINED_DF_PATH)
    runs_df = pd.read_parquet(RUNS_DF_PATH)
    history_df = pd.read_parquet(HISTORY_DF_PATH)
    return pretrain_df, runs_df, history_df


def size_match_pretrained_df(
    pretrain_df: pd.DataFrame, model_size: str
) -> pd.DataFrame:
    return pretrain_df[pretrain_df["params"] == model_size]


def recipe_match_pretrained_df(pretrain_df: pd.DataFrame, recipe: str) -> pd.DataFrame:
    return pretrain_df[pretrain_df["data"] == recipe]


def step_match_pretrained_df(pretrain_df: pd.DataFrame, step: int) -> pd.DataFrame:
    return pretrain_df[pretrain_df["step"] == step]


def runids_match_df(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return df[df["run_id"] == run_id]


def match_runids_pattern(df: pd.DataFrame, pattern_match: str) -> list[str]:
    run_ids = df["run_id"].dropna().astype(str).tolist()
    return [rid for rid in run_ids if pattern_match.lower() in rid.lower()]


def parse_datetime(input_str: str) -> str | None:
    if "-" in input_str and len(input_str) >= EXPECTED_DATATIME_RAW_LEN:
        date_part, time_part = input_str.split("-")
        if (
            len(date_part) == EXPECTED_DATE_OR_TIME_RAW_LEN
            and len(time_part) == EXPECTED_DATE_OR_TIME_RAW_LEN
        ):
            year = f"20{date_part[:2]}"
            month = date_part[2:4]
            day = date_part[4:6]
            hour = time_part[:2]
            minute = time_part[2:4]
            second = time_part[4:6]
            return f"{year} {month} {day} {hour} {minute} {second}"
    return None


def parse_exp_type(
    input_strs: list[str], possibile_types: list[str]
) -> tuple[str, int]:
    exp_type = ""
    exp_type_idx = -1
    for i, part in enumerate(input_strs):
        if part in possibile_types:
            exp_type = part
            exp_type_idx = i
            break
    return exp_type, exp_type_idx


def parse_exp_name(input_strs: list[str], exp_type_idx: int) -> str | None:
    if exp_type_idx > 1:
        return "_".join(input_strs[1:exp_type_idx])
    for i, part in enumerate(input_strs[2:], 2):
        if re.search(r"\d+M", part):
            return "_".join(input_strs[1:i])
    return None


def parse_comparison_model_size(exp_name: str) -> str | None:
    model_match = re.search(r"(\d+M)", exp_name)
    if model_match:
        return model_match.group(1)
    return None


def parse_comparison_model_recipe(exp_name: str) -> str | None:
    if ALT_COMPARISON_MODEL_RECIPE_STR in exp_name.lower():
        return ALT_COMPARISON_MODEL_RECIPE
    return DEFAULT_COMPARISON_MODEL_RECIPE


def extract_finetune_tokens_and_epochs(
    parts: list[str],
) -> tuple[str | None, str | None]:
    for part in parts:
        match = re.search(FINETUNE_PATTERN, part)
        if match:
            return match.group(1), match.group(2)
    return None, None


def extract_initial_checkpoint_info(
    parts: list[str],
) -> tuple[str | None, str | None, str | None, str | None]:
    for part in parts:
        match = re.search(DD_PATTERN, part)
        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
    return None, None, None, None


def extract_learning_rate(parts: list[str]) -> str | None:
    for part in parts:
        if part.startswith("lr="):
            return part.replace("lr=", "")
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
            extracted_data['pattern_name'] = pattern_name
            return run_type, extracted_data

    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}

    return "other", {}


def classify_run_id_type(run_id: str) -> str:
    run_type, _ = classify_run_id_type_and_extract(run_id)
    return run_type


def parse_run_id_components(run_id: str) -> dict[str, str]:
    components = {
        "full_id": run_id,
        "run_id_type": None,
        "datetime": None,
        "exp_name": None,
        "exp_type": None,
        "comparison_model_size": None,
        "comparison_model_recipe": None,
        "num_finetune_tokens": None,
        "num_finetune_epochs": None,
        "initial_checkpoint_size": None,
        "initial_checkpoint_recipe": None,
        "initial_checkpoint_steps": None,
        "seed": None,
        "lr": None,
    }

    components["run_id_type"] = classify_run_id_type(run_id)

    parts = run_id.split("_")
    if len(parts) < MIN_VALID_RUN_ID_SEGMENTS:
        return components
    datetime_part = parts[0]
    dt = parse_datetime(datetime_part)
    if dt is not None:
        components["datetime"] = dt
    exp_type, exp_type_idx = parse_exp_type(parts, ["finetune", "pretrain", "train"])
    components["exp_type"] = exp_type
    components["exp_name"] = parse_exp_name(parts, exp_type_idx)
    components["comparison_model_size"] = parse_comparison_model_size(
        components["exp_name"]
    )
    components["comparison_model_recipe"] = parse_comparison_model_recipe(
        components["exp_name"]
    )
    components["num_finetune_tokens"], components["num_finetune_epochs"] = (
        extract_finetune_tokens_and_epochs(parts)
    )
    (
        components["initial_checkpoint_recipe"],
        components["initial_checkpoint_size"],
        components["initial_checkpoint_steps"],
        components["seed"],
    ) = extract_initial_checkpoint_info(parts)
    components["lr"] = extract_learning_rate(parts)
    return components
