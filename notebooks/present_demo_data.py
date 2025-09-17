#!/usr/bin/env python3

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console

from dr_showntell.console_components import dataframe_to_fancy_tables

console = Console()

MAX_COLS = 20

def load_combined_data() -> pd.DataFrame:
    data_path = Path("data/combined_plotting_data_matched.pkl")
    assert data_path.exists(), f"Data file not found: {data_path}"

    console.print(f"Loading combined data from: [cyan]{data_path}[/cyan]")
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    console.print(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def display_group_summary(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Matched Group Summary[/bold blue]")

    group_summary = df[['matched_group_name', 'matched_group_id', 'matched_group_med_metric']].drop_duplicates()
    group_summary = group_summary.sort_values('matched_group_id')

    console.print(f"Found {len(group_summary)} unique matched groups")

    tables = dataframe_to_fancy_tables(
        group_summary,
        max_cols_per_table_split=MAX_COLS,
        title="Matched Groups Overview"
    )

    for table in tables:
        console.print(table)

def display_token_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Fine-tuning Token Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ft_tok_real', 'matched_group_name'])
    df_clean = df_clean.copy()
    df_clean['ft_tokens_rounded'] = (df_clean['ft_tok_real'] / 10_000_000).round() * 10

    token_pivot = df_clean.groupby(['matched_group_name', 'ft_tokens_rounded']).size().unstack(fill_value=0)
    token_pivot = token_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    token_pivot.columns = [f"{int(col)}M" for col in token_pivot.columns]

    tables = dataframe_to_fancy_tables(
        token_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Fine-tuning Token Counts by Group (rounded to nearest 10M)"
    )

    for table in tables:
        console.print(table)

def display_checkpoint_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Checkpoint Parameter Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ckpt_params', 'matched_group_name'])

    params_pivot = df_clean.groupby(['matched_group_name', 'ckpt_params']).size().unstack(fill_value=0)
    params_pivot = params_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    tables = dataframe_to_fancy_tables(
        params_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Parameter Counts by Group"
    )

    for table in tables:
        console.print(table)

def display_recipe_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Checkpoint Recipe Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ckpt_data', 'matched_group_name'])

    recipe_pivot = df_clean.groupby(['matched_group_name', 'ckpt_data']).size().unstack(fill_value=0)
    recipe_pivot = recipe_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    tables = dataframe_to_fancy_tables(
        recipe_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Recipe Counts by Group"
    )

    for table in tables:
        console.print(table)

@click.command()
@click.option("--pause", type=int, default=0, help="Pause duration in seconds")
def main(pause: int, **kwargs: Any) -> None:
    console.print("[bold blue]🔬 Combined Data Presentation Tool[/bold blue]")

    df = load_combined_data()
    display_group_summary(df)
    display_token_distribution_by_group(df)
    display_checkpoint_distribution_by_group(df)
    display_recipe_distribution_by_group(df)

    if pause > 0:
        console.print(f"\n[dim]Pausing for {pause} seconds...[/dim]")
        import time
        time.sleep(pause)


if __name__ == "__main__":
    main()