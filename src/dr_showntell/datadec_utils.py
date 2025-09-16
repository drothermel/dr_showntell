from __future__ import annotations

import pandas as pd

PRETRAINED_DF_PATH = "/Users/daniellerothermel/drotherm/repos/datadec/data/datadecide/mean_eval_melted.parquet"
RUNS_DF_PATH = "/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_metadata.parquet"
HISTORY_DF_PATH = "/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_history.parquet"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pretrain_df = pd.read_parquet(PRETRAINED_DF_PATH)
    runs_df = pd.read_parquet(RUNS_DF_PATH)
    history_df = pd.read_parquet(HISTORY_DF_PATH)
    return pretrain_df, runs_df, history_df


def filter_by_model_size(pretrain_df: pd.DataFrame, model_size: str) -> pd.DataFrame:
    assert "params" in pretrain_df.columns, "params column not found in dataframe"
    return pretrain_df[pretrain_df["params"] == model_size]


def filter_by_recipe(pretrain_df: pd.DataFrame, recipe: str) -> pd.DataFrame:
    assert "data" in pretrain_df.columns, "data column not found in dataframe"
    return pretrain_df[pretrain_df["data"] == recipe]


def filter_by_step(pretrain_df: pd.DataFrame, step: int) -> pd.DataFrame:
    assert "step" in pretrain_df.columns, "step column not found in dataframe"
    return pretrain_df[pretrain_df["step"] == step]


def filter_by_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    assert "run_id" in df.columns, "run_id column not found in dataframe"
    return df[df["run_id"] == run_id]


def find_runs_matching_pattern(df: pd.DataFrame, pattern: str) -> list[str]:
    assert "run_id" in df.columns, "run_id column not found in dataframe"
    run_ids = df["run_id"].dropna().astype(str).tolist()
    pattern_lower = pattern.lower()
    return [run_id for run_id in run_ids if pattern_lower in run_id.lower()]


# Legacy function names for backward compatibility
size_match_pretrained_df = filter_by_model_size
recipe_match_pretrained_df = filter_by_recipe
step_match_pretrained_df = filter_by_step
runids_match_df = filter_by_run_id
match_runids_pattern = find_runs_matching_pattern
