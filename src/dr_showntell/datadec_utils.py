from __future__ import annotations

import re

import pandas as pd

PRETRAINED_DF_PATH = "/Users/daniellerothermel/drotherm/repos/datadec/data/datadecide/mean_eval_melted.parquet"
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


def parse_run_id_components(run_id: str) -> dict[str, str]:
    from .run_id_parsing import classify_run_id_type_and_extract

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

    run_type, _ = classify_run_id_type_and_extract(run_id)
    components["run_id_type"] = run_type

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
